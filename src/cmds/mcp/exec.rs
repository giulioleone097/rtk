//! `ctx_execute` and `ctx_execute_file`: run one snippet in a temporary script
//! and answer with its output, or with the index when the output is large.

use std::collections::hash_map::DefaultHasher;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
// Through rmcp, so the derive and the server agree on one schemars version.
use rmcp::schemars::{self, JsonSchema};
use serde::Deserialize;

use super::store::Store;
use super::tools::{self, timeout_of, Captured, Launch, Renderer, DEFAULT_SECTION_LIMIT};

/// Output up to this many raw bytes travels back whole; past it the index
/// answers instead.
const INLINE_LIMIT_BYTES: usize = 5_000;
/// Head of an indexed output echoed in the response.
const PREVIEW_BYTES: usize = 2048;
/// What the head of an indexed output is called when a chunk repeats it.
const HEAD_LABEL: &str = "the output above";

/// Variables that make an interpreter run something before the caller's script
/// — a startup file, a hook, an extra option line — or change how it runs it:
/// `SHELLOPTS` carries `xtrace`, which echoes the script and the temp path it
/// runs from. The script the caller sent is the only thing that should run, so
/// the child never sees them.
const STARTUP_HOOKS: &[&str] = &[
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
    "NODE_OPTIONS",
    "PERL5OPT",
    "SHELLOPTS",
];

/// Prefix of an exported bash function: `BASH_FUNC_ls%%=() { … }` in the parent
/// makes `ls` in the script run that body instead of the command.
const BASH_FUNCTION: &str = "BASH_FUNC_";

/// Input of the `ctx_execute` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExecuteInput {
    /// Runtime to run the code with: `shell`, `javascript` or `python`.
    pub language: String,
    /// The script itself.
    pub code: String,
    /// Timeout in milliseconds. Defaults to 60000.
    pub timeout_ms: Option<u64>,
    /// What to look for in the output when it is too large to return whole.
    pub intent: Option<String>,
    /// Directory to run in. Defaults to the server's own.
    pub cwd: Option<String>,
}

/// Input of the `ctx_execute_file` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExecuteFileInput {
    /// File the script reads: absolute, or relative to `cwd`.
    pub path: String,
    /// Runtime to run the code with: `shell`, `javascript` or `python`.
    pub language: String,
    /// The script itself. `FILE_PATH` and `FILE_CONTENT` are already set. In
    /// shell FILE_CONTENT loses trailing newlines and NUL bytes; read
    /// "$FILE_PATH" directly for exact bytes.
    pub code: String,
    /// Timeout in milliseconds. Defaults to 60000.
    pub timeout_ms: Option<u64>,
    /// What to look for in the output when it is too large to return whole.
    pub intent: Option<String>,
    /// Directory to run in. Defaults to the server's own.
    pub cwd: Option<String>,
}

/// One supported runtime: what runs the script, what the script is called,
/// the preamble that hands `FILE_CONTENT` to a `ctx_execute_file` script, and
/// the environment the script runs under.
struct Language {
    name: &'static str,
    interpreter: &'static str,
    suffix: &'static str,
    file_preamble: &'static str,
    env: &'static [(&'static str, &'static str)],
}

/// The three runtimes the corpus actually asks for.
const LANGUAGES: &[Language] = &[
    Language {
        name: "shell",
        interpreter: "bash",
        suffix: "sh",
        file_preamble: "FILE_CONTENT=$(cat \"$FILE_PATH\")\n",
        env: &[],
    },
    Language {
        name: "javascript",
        interpreter: "node",
        // Not `js`: a "type": "module" in any package.json above the temp
        // directory would make node read the script as ESM, and `require` fail.
        suffix: "cjs",
        file_preamble: "const FILE_CONTENT = require(\"fs\").readFileSync(process.env.FILE_PATH, \"utf8\");\n",
        env: &[],
    },
    Language {
        name: "python",
        interpreter: "python3",
        suffix: "py",
        file_preamble:
            "import os\nFILE_CONTENT = open(os.environ[\"FILE_PATH\"], encoding=\"utf-8\", errors=\"replace\").read()\n",
        // Python block-buffers stdout on a pipe: a call that hits the capture
        // cap or its timeout would lose the script's last prints.
        env: &[("PYTHONUNBUFFERED", "1")],
    },
];

fn language(name: &str) -> Result<&'static Language> {
    LANGUAGES
        .iter()
        .find(|language| language.name == name)
        .with_context(|| format!("unsupported language: {name} (shell, javascript or python)"))
}

impl Language {
    /// The variables a script under this runtime always runs with; the call's
    /// own (like `FILE_PATH`) come on top.
    fn env(&self) -> Vec<(&'static str, String)> {
        self.env
            .iter()
            .map(|(name, value)| (*name, (*value).to_string()))
            .collect()
    }
}

/// One prepared call: what to run, what the script needs in its environment,
/// and the source its output is indexed under.
struct Job {
    language: &'static Language,
    code: String,
    env: Vec<(&'static str, String)>,
    source: String,
}

/// A job ready to run, or the answer to give when there is nothing to run.
type Prepared = std::result::Result<Job, String>;

impl Job {
    /// A bare snippet, indexed under `execute:<language>:<call>`. The call id
    /// keeps every call in a bucket of its own: under one shared label the
    /// query that reaches "the rest" would answer with another call's output.
    fn snippet(input: &ExecuteInput) -> Result<Self> {
        let language = language(&input.language)?;
        Ok(Job {
            language,
            code: input.code.clone(),
            env: language.env(),
            source: format!("execute:{}:{}", language.name, call_id(&input.code)),
        })
    }

    /// A snippet over one file, indexed under `file:<canonical path>`. `Err` is
    /// what to answer when the caller's path is not a readable file: that is an
    /// answer, not a failure.
    fn file(input: &ExecuteFileInput, cwd: &Path) -> Result<Prepared> {
        let language = language(&input.language)?;
        let path = match resolve(&input.path, cwd) {
            Ok(path) => path,
            Err(answer) => return Ok(Err(answer)),
        };
        let mut env = language.env();
        env.push(("FILE_PATH", path.display().to_string()));
        Ok(Ok(Job {
            language,
            code: format!("{}{}", language.file_preamble, input.code),
            env,
            source: format!("file:{}", path.display()),
        }))
    }

    /// Write the script to a directory of its own, run it there, and take that
    /// directory back down: the script exists for this call and no other.
    async fn run(&self, cwd: PathBuf, timeout: Duration) -> Result<Captured> {
        // Resolved here so a missing runtime is the same spawn failure a missing
        // command is, instead of a shell's own "not found" under exit 127.
        let interpreter = match which::which(self.language.interpreter) {
            Ok(path) => path,
            Err(err) => {
                return Ok(Captured::failed(format!(
                    "failed to spawn {}: {err}",
                    self.language.interpreter
                )))
            }
        };
        let dir = temp_dir()?;
        let script = dir.path().join(format!("script.{}", self.language.suffix));
        write_script(&script, &self.code)?;
        let launch = Launch {
            command: format!("exec {} {}", quote(&interpreter), quote(&script)),
            remove_env: stripped_env(),
            set_env: self.env.clone(),
        };
        Ok(tools::run_command(launch, Some(cwd), timeout).await)
    }
}

/// Eight hex characters no other call shares: the code, the clock, and a
/// counter for two calls landing in the same tick.
fn call_id(code: &str) -> String {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let mut hasher = DefaultHasher::new();
    code.hash(&mut hasher);
    SystemTime::now().hash(&mut hasher);
    CALLS.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    format!("{:08x}", hasher.finish() as u32)
}

/// The names the child must not inherit: the startup hooks, plus every bash
/// function the parent exported. The `BASH_FUNC_*` set is read from the
/// environment because the name carries the function it redefines.
fn stripped_env() -> Vec<OsString> {
    let mut names: Vec<OsString> = STARTUP_HOOKS.iter().map(OsString::from).collect();
    names.extend(
        std::env::vars_os()
            .map(|(name, _)| name)
            .filter(|name| name.to_string_lossy().starts_with(BASH_FUNCTION)),
    );
    names
}

/// The directory the script runs in, absolute against the server's own, or the
/// answer to give when the caller named one that is not there.
fn working_dir(cwd: Option<&str>) -> std::result::Result<PathBuf, String> {
    let named = cwd.unwrap_or(".");
    let missing = || format!("cwd not found: {named}");
    let dir = std::path::absolute(named).map_err(|_| missing())?;
    if !dir.is_dir() {
        return Err(missing());
    }
    Ok(dir)
}

/// The file the caller named, absolute and readable, or the answer to give
/// instead. A relative path resolves against the directory the script runs in,
/// which is where the script itself would look for it.
fn resolve(path: &str, cwd: &Path) -> std::result::Result<PathBuf, String> {
    let missing = || format!("file not found: {path}");
    // A join with an absolute path keeps that path, so both cases are one line.
    let file = std::path::absolute(cwd.join(path)).map_err(|_| missing())?;
    if !file.is_file() {
        return Err(missing());
    }
    // The script would fail on it anyway; failing here says why.
    std::fs::File::open(&file).map_err(|_| format!("cannot read: {path}"))?;
    Ok(file)
}

/// A directory only its owner can enter: it holds the caller's code, and on a
/// shared machine the system temp directory is readable by everyone.
#[cfg(unix)]
fn temp_dir() -> Result<tempfile::TempDir> {
    use std::os::unix::fs::PermissionsExt;
    Ok(tempfile::Builder::new()
        .permissions(std::fs::Permissions::from_mode(0o700))
        .tempdir()?)
}

#[cfg(not(unix))]
fn temp_dir() -> Result<tempfile::TempDir> {
    Ok(tempfile::tempdir()?)
}

/// Write the script readable by its owner alone, in one step: a script created
/// world-readable is world-readable for as long as it exists.
#[cfg(unix)]
fn write_script(path: &Path, code: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?
        .write_all(code.as_bytes())
}

#[cfg(not(unix))]
fn write_script(path: &Path, code: &str) -> std::io::Result<()> {
    std::fs::write(path, code)
}

/// `path` as a single `sh` word.
fn quote(path: &Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', r"'\''"))
}

/// `ctx_execute` against the default index.
pub async fn execute(input: ExecuteInput) -> Result<String> {
    let cwd = match working_dir(input.cwd.as_deref()) {
        Ok(cwd) => cwd,
        Err(answer) => return Ok(answer),
    };
    let job = Job::snippet(&input)?;
    let captured = job.run(cwd, timeout_of(input.timeout_ms)).await?;
    // The index connection is opened only once no await is left: it is not `Sync`,
    // and the tool future must be `Send`.
    let store = Store::open_default()?;
    render(&store, &job, &captured, input.intent.as_deref())
}

/// `ctx_execute_file` against the default index.
pub async fn execute_file(input: ExecuteFileInput) -> Result<String> {
    let cwd = match working_dir(input.cwd.as_deref()) {
        Ok(cwd) => cwd,
        Err(answer) => return Ok(answer),
    };
    let job = match Job::file(&input, &cwd)? {
        Ok(job) => job,
        Err(answer) => return Ok(answer),
    };
    let captured = job.run(cwd, timeout_of(input.timeout_ms)).await?;
    let store = Store::open_default()?;
    render(&store, &job, &captured, input.intent.as_deref())
}

/// Small output goes back whole. Large output is indexed instead, and the
/// response carries only what points at it: the head, the sections answering
/// `intent`, and the query that reaches the rest.
fn render(store: &Store, job: &Job, captured: &Captured, intent: Option<&str>) -> Result<String> {
    // The raw byte count, the one the summary reports: lossy decoding widens
    // invalid bytes, and the caller cannot see that number anywhere.
    if captured.bytes <= INLINE_LIMIT_BYTES {
        return Ok(format!(
            "{}{}",
            tools::summary(captured, None),
            captured.text
        ));
    }
    let indexed = store.index(&job.source, &captured.text)?;
    let mut out = tools::summary(captured, Some(&indexed));
    let head = preview(&captured.text);
    out.push_str(&head);
    out.push('\n');
    let mut renderer = Renderer::default();
    // The head is already in this response: a chunk lying inside it comes back
    // as a back-reference instead of a second copy.
    renderer.shown(HEAD_LABEL, &head);
    if let Some(intent) = intent {
        tools::query_block(
            store,
            &mut renderer,
            intent,
            DEFAULT_SECTION_LIMIT,
            Some(&job.source),
            &mut out,
        )?;
    }
    out.push_str(&format!(
        "Search the rest with ctx_search {{queries: [\"<what you look for>\"], source: \"{}\"}}\n",
        job.source
    ));
    Ok(out)
}

/// The first [`PREVIEW_BYTES`] of `text`, cut on a character boundary.
fn preview(text: &str) -> String {
    let head = tools::cut_at(text, PREVIEW_BYTES);
    if head.len() == text.len() {
        return text.to_string();
    }
    format!("{head}…\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        (dir, store)
    }

    fn snippet(language: &str, code: &str) -> ExecuteInput {
        ExecuteInput {
            language: language.to_string(),
            code: code.to_string(),
            timeout_ms: None,
            intent: None,
            cwd: None,
        }
    }

    fn run_in(job: &Job, cwd: &str) -> Captured {
        let cwd = working_dir(Some(cwd)).expect("the directory is there");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(job.run(cwd, timeout_of(None)))
            .unwrap()
    }

    fn run(job: &Job) -> Captured {
        run_in(job, ".")
    }

    #[test]
    fn every_runtime_answers_with_the_output_of_its_own_script() {
        let (_dir, store) = store();
        for (language, code, want) in [
            ("shell", "printf 'a\\nb\\n'", "a\nb\n"),
            ("python", "print(sum(range(10)))", "45\n"),
            ("javascript", "console.log(process.version[0])", "v\n"),
        ] {
            let job = Job::snippet(&snippet(language, code)).unwrap();
            let captured = run(&job);
            let out = render(&store, &job, &captured, None).unwrap();
            assert_eq!(
                out,
                format!("exit 0, {} bytes\n{want}", want.len()),
                "{out}"
            );
        }
    }

    #[test]
    fn output_past_the_inline_limit_is_indexed_and_answered_by_intent() {
        let (_dir, store) = store();
        let job = Job::snippet(&snippet("shell", "seq 1 4000; echo 'the needle is here'")).unwrap();
        let captured = run(&job);
        assert!(captured.text.len() > INLINE_LIMIT_BYTES, "output too small");

        let out = render(&store, &job, &captured, Some("needle")).unwrap();
        assert!(out.starts_with("exit 0, "), "{out}");
        // The head, and the chunk that answers the intent, not the 19 KB between.
        assert!(out.contains("\n1\n2\n3\n"), "no preview");
        assert!(out.contains("## needle"), "no intent section");
        assert!(out.contains("the needle is here"), "intent not answered");
        assert!(!out.contains("\n2000\n"), "full output returned");
        // The hint is a call ctx_search accepts, against this call's own label.
        assert!(
            out.contains(&format!(
                "Search the rest with ctx_search {{queries: [\"<what you look for>\"], source: \"{}\"}}",
                job.source
            )),
            "{out}"
        );
        assert!(out.len() < 6000, "response of {} bytes", out.len());
    }

    #[test]
    fn a_chunk_lying_inside_the_head_is_not_printed_twice() {
        let (_dir, store) = store();
        let job = Job::snippet(&snippet("shell", "echo 'the marker is here'; seq 1 4000")).unwrap();
        let captured = run(&job);
        let out = render(&store, &job, &captured, Some("marker")).unwrap();

        assert!(out.contains("## marker"), "{out}");
        assert!(
            out.contains(&format!("(already shown above under \"{HEAD_LABEL}\")")),
            "{out}"
        );
        // Once in the head, and not again under the intent.
        assert_eq!(out.matches("the marker is here").count(), 1, "{out}");
    }

    #[test]
    fn two_calls_do_not_share_a_source_label() {
        let input = snippet("shell", "echo same");
        let first = Job::snippet(&input).unwrap().source;
        let second = Job::snippet(&input).unwrap().source;
        assert!(first.starts_with("execute:shell:"), "{first}");
        assert_ne!(first, second);
    }

    #[test]
    fn a_file_script_reads_the_file_it_was_given() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.txt");
        std::fs::write(&path, "x".repeat(123)).unwrap();
        let input = ExecuteFileInput {
            path: path.display().to_string(),
            language: "shell".to_string(),
            code: "printf '%s' \"$FILE_CONTENT\" | wc -c".to_string(),
            timeout_ms: None,
            intent: None,
            cwd: None,
        };

        let here = working_dir(None).unwrap();
        let job = Job::file(&input, &here)
            .unwrap()
            .expect("the file is there");
        assert_eq!(job.source, format!("file:{}", path.display()));
        assert_eq!(run(&job).text.trim(), "123");

        // The javascript preamble reads the file with `require`, which only
        // resolves in a CommonJS script.
        let javascript = ExecuteFileInput {
            path: path.display().to_string(),
            language: "javascript".to_string(),
            code: "console.log(FILE_CONTENT.length)".to_string(),
            timeout_ms: None,
            intent: None,
            cwd: None,
        };
        let job = Job::file(&javascript, &here)
            .unwrap()
            .expect("the file is there");
        assert_eq!(run(&job).text.trim(), "123");

        // A path that is not there is an answer, not a tool failure.
        let missing = ExecuteFileInput {
            path: "no/such/file.txt".to_string(),
            ..input
        };
        assert_eq!(
            Job::file(&missing, &here).unwrap().err(),
            Some("file not found: no/such/file.txt".to_string())
        );
    }

    #[test]
    fn a_relative_path_is_read_from_the_directory_the_script_runs_in() {
        // Under the server's own directory, so `cwd` itself can be relative.
        let dir = tempfile::Builder::new().tempdir_in(".").unwrap();
        let cwd = dir
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("data.txt"), "x".repeat(7)).unwrap();

        let input = ExecuteFileInput {
            path: "sub/data.txt".to_string(),
            language: "shell".to_string(),
            code: "printf '%s' \"$FILE_CONTENT\" | wc -c".to_string(),
            timeout_ms: None,
            intent: None,
            cwd: Some(cwd),
        };
        let here = working_dir(input.cwd.as_deref()).expect("the directory is there");
        let job = Job::file(&input, &here)
            .unwrap()
            .expect("the file is there");

        // FILE_PATH is absolute, so the script does not resolve it a second
        // time against the directory it already runs in.
        assert!(job.source.starts_with("file:/"), "{}", job.source);
        assert_eq!(run_in(&job, input.cwd.as_deref().unwrap()).text.trim(), "7");

        // The same file from the server's own directory: no `cwd`, and the
        // label carries no `.` component from the way it was resolved.
        let from_here = ExecuteFileInput {
            path: format!("{}/sub/data.txt", input.cwd.as_deref().unwrap()),
            cwd: None,
            ..ExecuteFileInput {
                path: String::new(),
                language: "shell".to_string(),
                code: "printf '%s' \"$FILE_CONTENT\" | wc -c".to_string(),
                timeout_ms: None,
                intent: None,
                cwd: None,
            }
        };
        let job = Job::file(&from_here, &working_dir(None).unwrap())
            .unwrap()
            .expect("the file is there");
        assert!(!job.source.contains("/./"), "{}", job.source);
        assert_eq!(run(&job).text.trim(), "7");

        // A directory is not a file to read, and that is an answer too.
        let directory = ExecuteFileInput {
            path: "sub".to_string(),
            ..input
        };
        assert_eq!(
            Job::file(&directory, &here).unwrap().err(),
            Some("file not found: sub".to_string())
        );
    }

    #[test]
    fn a_cwd_that_is_not_there_is_an_answer_from_both_tools() {
        // Answered before anything runs or is indexed, so the child does not
        // fail with a spawn error the caller cannot read.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let code = ExecuteInput {
            cwd: Some("no/such/dir".to_string()),
            ..snippet("shell", "echo hi")
        };
        let file = ExecuteFileInput {
            path: "data.txt".to_string(),
            language: "shell".to_string(),
            code: "echo hi".to_string(),
            timeout_ms: None,
            intent: None,
            cwd: Some("no/such/dir".to_string()),
        };

        assert_eq!(
            runtime.block_on(execute(code)).unwrap(),
            "cwd not found: no/such/dir"
        );
        assert_eq!(
            runtime.block_on(execute_file(file)).unwrap(),
            "cwd not found: no/such/dir"
        );
    }

    #[test]
    fn an_exported_bash_function_does_not_reach_the_script() {
        // A parent that exports `ls` as a function redefines it for every shell
        // below it. `SHELLOPTS=xtrace` is stripped the same way, but setting it
        // here would trace the shells of the tests running alongside this one.
        std::env::set_var("BASH_FUNC_ls%%", "() { echo hijacked; }");
        let job = Job::snippet(&snippet("shell", "env | grep -c BASH_FUNC || true")).unwrap();
        let captured = run(&job);
        std::env::remove_var("BASH_FUNC_ls%%");

        assert_eq!(captured.text.trim(), "0", "{}", captured.text);
    }

    #[cfg(unix)]
    #[test]
    fn the_script_and_its_directory_are_readable_by_their_owner_alone() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir().unwrap();
        let script = dir.path().join("script.sh");
        write_script(&script, "echo hi").unwrap();

        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(dir.path()), 0o700, "the directory is not private");
        assert_eq!(mode(&script), 0o600, "the script is not private");
    }

    #[test]
    fn a_python_script_runs_with_stdout_unbuffered() {
        if which::which("python3").is_err() {
            return;
        }
        let out = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(execute(snippet(
                "python",
                "import os; print(os.environ.get(\"PYTHONUNBUFFERED\"))",
            )))
            .unwrap();
        assert_eq!(out, "exit 0, 2 bytes\n1\n");

        // A `ctx_execute_file` job runs under the same set.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.txt");
        std::fs::write(&path, "x").unwrap();
        let file = ExecuteFileInput {
            path: path.display().to_string(),
            language: "python".to_string(),
            code: "print(FILE_CONTENT)".to_string(),
            timeout_ms: None,
            intent: None,
            cwd: None,
        };
        let job = Job::file(&file, &working_dir(None).unwrap())
            .unwrap()
            .expect("the file is there");
        assert!(job.env.contains(&("PYTHONUNBUFFERED", "1".to_string())));
    }
}
