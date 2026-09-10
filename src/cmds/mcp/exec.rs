//! `ctx_execute` and `ctx_execute_file`: run one snippet in a temporary script
//! and answer with its output, or with the index when the output is large.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
// Through rmcp, so the derive and the server agree on one schemars version.
use rmcp::schemars::{self, JsonSchema};
use serde::Deserialize;

use super::store::Store;
use super::tools::{self, Captured, Launch, Renderer, DEFAULT_SECTION_LIMIT, DEFAULT_TIMEOUT_MS};

/// Output up to this size travels back whole; past it the index answers instead.
const INLINE_LIMIT_BYTES: usize = 5_000;
/// Head of an indexed output echoed in the response.
const PREVIEW_BYTES: usize = 2048;

/// Variables that make an interpreter run something before the caller's script:
/// a startup file, a hook, an extra option line. The script the caller sent is
/// the only thing that should run, so the child never sees them.
const STARTUP_HOOKS: &[&str] = &[
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
    "PYTHONSTARTUP",
    "NODE_OPTIONS",
    "PERL5OPT",
];

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
    /// The script itself. `FILE_PATH` and `FILE_CONTENT` are already set.
    pub code: String,
    /// Timeout in milliseconds. Defaults to 60000.
    pub timeout_ms: Option<u64>,
    /// What to look for in the output when it is too large to return whole.
    pub intent: Option<String>,
    /// Directory to run in. Defaults to the server's own.
    pub cwd: Option<String>,
}

/// One supported runtime: what runs the script, what the script is called, and
/// the preamble that hands `FILE_CONTENT` to a `ctx_execute_file` script.
struct Language {
    name: &'static str,
    interpreter: &'static str,
    suffix: &'static str,
    file_preamble: &'static str,
}

/// The three runtimes the corpus actually asks for.
const LANGUAGES: &[Language] = &[
    Language {
        name: "shell",
        interpreter: "bash",
        suffix: "sh",
        file_preamble: "FILE_CONTENT=$(cat \"$FILE_PATH\")\n",
    },
    Language {
        name: "javascript",
        interpreter: "node",
        suffix: "js",
        file_preamble: "const FILE_CONTENT = require(\"fs\").readFileSync(process.env.FILE_PATH, \"utf8\");\n",
    },
    Language {
        name: "python",
        interpreter: "python3",
        suffix: "py",
        file_preamble:
            "import os\nFILE_CONTENT = open(os.environ[\"FILE_PATH\"], encoding=\"utf-8\", errors=\"replace\").read()\n",
    },
];

fn language(name: &str) -> Result<&'static Language> {
    LANGUAGES
        .iter()
        .find(|language| language.name == name)
        .with_context(|| format!("unsupported language: {name} (shell, javascript or python)"))
}

/// One prepared call: what to run, what the script needs in its environment,
/// and the source its output is indexed under.
struct Job {
    language: &'static Language,
    code: String,
    env: Vec<(&'static str, String)>,
    source: String,
}

impl Job {
    /// A bare snippet, indexed under `execute:<language>`.
    fn snippet(input: &ExecuteInput) -> Result<Self> {
        let language = language(&input.language)?;
        Ok(Job {
            language,
            code: input.code.clone(),
            env: Vec::new(),
            source: format!("execute:{}", language.name),
        })
    }

    /// A snippet over one file, indexed under `file:<path>`. `None` when the
    /// file the caller named is not there: that is an answer, not a failure.
    fn file(input: &ExecuteFileInput) -> Result<Option<Self>> {
        let language = language(&input.language)?;
        let cwd = input.cwd.as_ref().map(PathBuf::from);
        let path = resolve(&input.path, cwd.as_deref());
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(Job {
            language,
            code: format!("{}{}", language.file_preamble, input.code),
            env: vec![("FILE_PATH", path.display().to_string())],
            source: format!("file:{}", input.path),
        }))
    }

    /// Write the script to a directory of its own, run it there, and take that
    /// directory back down: the script exists for this call and no other.
    async fn run(&self, cwd: Option<PathBuf>, timeout: Duration) -> Result<Captured> {
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
        let dir = tempfile::tempdir()?;
        let script = dir.path().join(format!("script.{}", self.language.suffix));
        std::fs::write(&script, &self.code)?;
        let launch = Launch {
            command: format!("exec {} {}", quote(&interpreter), quote(&script)),
            remove_env: STARTUP_HOOKS,
            set_env: self.env.clone(),
        };
        Ok(tools::run_command(launch, cwd, timeout).await)
    }
}

/// `path` as the script will see it: already absolute, or relative to the
/// directory the script runs in.
fn resolve(path: &str, cwd: Option<&Path>) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match cwd {
        Some(dir) => dir.join(path),
        None => std::env::current_dir().unwrap_or_default().join(path),
    }
}

/// `path` as a single `sh` word.
fn quote(path: &Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', r"'\''"))
}

/// `ctx_execute` against the default index.
pub async fn execute(input: ExecuteInput) -> Result<String> {
    let job = Job::snippet(&input)?;
    let captured = job
        .run(cwd_of(&input.cwd), timeout_of(input.timeout_ms))
        .await?;
    // The index connection is opened only once no await is left: it is not `Sync`,
    // and the tool future must be `Send`.
    let store = Store::open_default()?;
    render(&store, &job, &captured, input.intent.as_deref())
}

/// `ctx_execute_file` against the default index.
pub async fn execute_file(input: ExecuteFileInput) -> Result<String> {
    let Some(job) = Job::file(&input)? else {
        return Ok(format!("file not found: {}", input.path));
    };
    let captured = job
        .run(cwd_of(&input.cwd), timeout_of(input.timeout_ms))
        .await?;
    let store = Store::open_default()?;
    render(&store, &job, &captured, input.intent.as_deref())
}

fn cwd_of(cwd: &Option<String>) -> Option<PathBuf> {
    cwd.as_ref().map(PathBuf::from)
}

fn timeout_of(timeout_ms: Option<u64>) -> Duration {
    Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS))
}

/// Small output goes back whole. Large output is indexed instead, and the
/// response carries only what points at it: the head, the sections answering
/// `intent`, and the query that reaches the rest.
fn render(store: &Store, job: &Job, captured: &Captured, intent: Option<&str>) -> Result<String> {
    if captured.text.len() <= INLINE_LIMIT_BYTES {
        return Ok(format!(
            "{}{}",
            tools::summary(captured, None),
            captured.text
        ));
    }
    let indexed = store.index(&job.source, &captured.text)?;
    let mut out = tools::summary(captured, Some(&indexed));
    out.push_str(&preview(&captured.text));
    out.push('\n');
    if let Some(intent) = intent {
        // MERGE POINT (T3): once `tokenaut/fetch` is in, this block searches
        // `job.source` only, through the `source` filter `query_block` gains.
        tools::query_block(
            store,
            &mut Renderer::default(),
            intent,
            DEFAULT_SECTION_LIMIT,
            &mut out,
        )?;
    }
    out.push_str(&format!(
        "Search the rest with ctx_search {{source: \"{}\"}}\n",
        job.source
    ));
    Ok(out)
}

/// The first [`PREVIEW_BYTES`] of `text`, cut on a character boundary.
fn preview(text: &str) -> String {
    if text.len() <= PREVIEW_BYTES {
        return text.to_string();
    }
    let mut end = PREVIEW_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…\n", &text[..end])
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

    fn run(job: &Job) -> Captured {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(job.run(None, timeout_of(None)))
            .unwrap()
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
        assert!(
            out.contains("Search the rest with ctx_search {source: \"execute:shell\"}"),
            "{out}"
        );
        assert!(out.len() < 6000, "response of {} bytes", out.len());
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

        let job = Job::file(&input).unwrap().expect("the file is there");
        assert_eq!(job.source, format!("file:{}", path.display()));
        assert_eq!(run(&job).text.trim(), "123");

        // A path that is not there is an answer, not a tool failure.
        let missing = ExecuteFileInput {
            path: "no/such/file.txt".to_string(),
            ..input
        };
        assert!(Job::file(&missing).unwrap().is_none());
    }
}
