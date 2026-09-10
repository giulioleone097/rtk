//! Tool inputs, command execution and response rendering for `tokenaut mcp`.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
// Through rmcp, so the derive and the server agree on one schemars version.
use rmcp::schemars::{self, JsonSchema};
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::Semaphore;

use super::store::{Indexed, Section, Store};

/// Per-command timeout when the caller does not set one.
pub(super) const DEFAULT_TIMEOUT_MS: u64 = 60_000;
/// Commands running at once; the rest wait for a slot.
const MAX_PARALLEL_COMMANDS: usize = 8;
/// Upper bound on the output captured from a single command.
const MAX_CAPTURE_BYTES: usize = 1024 * 1024;
/// Read size of one bounded read from a child pipe.
const READ_CHUNK_BYTES: usize = 64 * 1024;
/// Lines of raw output echoed back per command.
const PREVIEW_LINES: usize = 10;
/// Byte bound on the same head: ten lines of a one-line megabyte is still a
/// megabyte, and this response is what the caller pays for.
const PREVIEW_BYTES: usize = 2048;
/// Sections per query when the caller does not ask for a different number.
pub(super) const DEFAULT_SECTION_LIMIT: usize = 5;
/// Highest limit a caller can ask for.
const MAX_SEARCH_LIMIT: usize = 20;
/// Once a response passes this size no further section is added.
const MAX_RESPONSE_BYTES: usize = 40 * 1024;

/// The timeout one call runs under: what the caller asked for, or the default.
pub(super) fn timeout_of(timeout_ms: Option<u64>) -> Duration {
    Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS))
}

/// One shell command to run and index.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CommandSpec {
    /// Section header, and the search source of the chunks it produces.
    pub label: String,
    /// Command line, run with `sh -c`.
    pub command: String,
}

/// Input of the `ctx_batch_execute` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchExecuteInput {
    /// Commands to run in parallel.
    pub commands: Vec<CommandSpec>,
    /// Questions answered from the index once every command has finished.
    #[serde(default)]
    pub queries: Vec<String>,
    /// Per-command timeout in milliseconds. Defaults to 60000.
    pub timeout_ms: Option<u64>,
    /// Directory to run the commands in. Defaults to the server's own.
    pub cwd: Option<String>,
}

/// Input of the `ctx_search` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchInput {
    /// Questions to answer from the index.
    pub queries: Vec<String>,
    /// Sections returned per query. Defaults to 5, capped at 20.
    pub limit: Option<usize>,
    /// Only sections from this source label, e.g. `fetch:<label>` or `execute:shell`.
    pub source: Option<String>,
}

/// What one command produced.
pub(super) struct Captured {
    pub exit: i32,
    /// Interleaved stdout and stderr, capped at [`MAX_CAPTURE_BYTES`].
    pub text: String,
    /// Raw bytes read from the command, before any lossy decoding.
    pub bytes: usize,
    /// The command had more to say than the cap allowed.
    pub truncated: bool,
}

impl Captured {
    /// A command that never ran.
    pub(super) fn failed(text: String) -> Self {
        Captured {
            exit: -1,
            text,
            bytes: 0,
            truncated: false,
        }
    }
}

/// Read at most `limit` bytes, then stop the process group so the command
/// cannot block forever on a pipe nobody drains. Returns the bytes and whether
/// more output was left behind.
async fn read_capped<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
    pid: Option<u32>,
) -> (Vec<u8>, bool) {
    let mut buf = Vec::new();
    let mut window = vec![0u8; READ_CHUNK_BYTES];
    loop {
        match reader.read(&mut window).await {
            Ok(0) | Err(_) => return (buf, false),
            Ok(read) => {
                buf.extend_from_slice(&window[..read]);
                if buf.len() > limit {
                    buf.truncate(limit);
                    kill_group(pid);
                    return (buf, true);
                }
            }
        }
    }
}

/// Kill the command and everything it spawned. The child runs in its own
/// process group, so a negative pid reaches background jobs too; without it
/// only `sh` would die and `sleep 600 &` would survive.
fn kill_group(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        #[allow(unsafe_code)]
        // nosemgrep: unsafe-block
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    let _ = pid;
}

/// Wrapper script: the caller's command runs in an inner shell whose stderr is
/// redirected into the outer shell's stdout, so a failure reason lands next to
/// the output that led to it, in the order the child wrote the two. An
/// interpreter that block-buffers stdout to a pipe — python above all — still
/// flushes it after its unbuffered stderr, and then the two arrive out of
/// order. The command travels as an argument, so nothing in it is parsed twice,
/// and the inner shell's own diagnostics are redirected as well.
const MERGE_STREAMS: &str = r#"sh -c "$1" 2>&1"#;

/// One command line plus the environment its interpreter needs. `ctx_execute`
/// runs a script the caller wrote, so it strips the variables that would make
/// an interpreter run something else first and adds the ones the script reads.
pub(super) struct Launch {
    /// Command line, run with `sh -c`.
    pub command: String,
    /// Variables removed from the child environment.
    pub remove_env: Vec<OsString>,
    /// Variables added to it.
    pub set_env: Vec<(&'static str, String)>,
}

impl Launch {
    /// A command run with the server's own environment.
    fn plain(command: String) -> Self {
        Launch {
            command,
            remove_env: Vec::new(),
            set_env: Vec::new(),
        }
    }
}

/// Kills the process group unless the call disarms it first. `kill_on_drop`
/// reaches the direct child only, so a call dropped mid-flight — the client
/// closed stdin and cancelled it — would leave the rest of the group running;
/// this takes the group down on every path out, cancellation included.
struct GroupGuard(Option<u32>);

impl GroupGuard {
    /// The group is down and the child reaped, so its pid must not be signalled
    /// again: by then it can name another process.
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for GroupGuard {
    fn drop(&mut self) {
        kill_group(self.0);
    }
}

/// Run `launch` with `sh -c`, streaming its output under the capture cap.
/// Never fails: a spawn error, a cap or a timeout is reported as output.
pub(super) async fn run_command(
    launch: Launch,
    cwd: Option<PathBuf>,
    timeout: Duration,
) -> Captured {
    let mut builder = tokio::process::Command::new("sh");
    builder
        .arg("-c")
        .arg(MERGE_STREAMS)
        .arg("tokenaut")
        .arg(&launch.command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    for name in &launch.remove_env {
        builder.env_remove(name);
    }
    for (name, value) in &launch.set_env {
        builder.env(name, value);
    }
    if let Some(dir) = &cwd {
        builder.current_dir(dir);
    }
    #[cfg(unix)]
    builder.process_group(0);

    let mut child = match builder.spawn() {
        Ok(child) => child,
        Err(err) => return Captured::failed(format!("failed to spawn: {err}")),
    };
    let pid = child.id();
    // From here on the group goes down on every exit, cancellation included.
    let group = GroupGuard(pid);
    let stdout = child.stdout.take().expect("stdout is piped");
    // The reader runs as a task so a timeout keeps whatever it already read.
    let reading = tokio::spawn(read_capped(stdout, MAX_CAPTURE_BYTES, pid));

    let (exit, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => (exit_code(&status), false),
        Ok(Err(err)) => return Captured::failed(format!("failed to run: {err}")),
        Err(_) => (124, true),
    };
    // Also after a normal exit: a child that inherited the pipe would otherwise
    // hold it open and the reader would never see EOF.
    kill_group(pid);
    let _ = child.start_kill();
    let _ = child.wait().await;
    group.disarm();

    let (out, capped) = reading.await.unwrap_or_default();
    let bytes = out.len();
    let text = String::from_utf8_lossy(&out).into_owned();
    // Invalid bytes decode wider than they arrived, so the cap is checked again.
    let cut = capped || text.len() > MAX_CAPTURE_BYTES;
    let mut text = truncate_to_cap(text);
    if cut {
        text.push_str("\n\n[output truncated at 1 MB]");
    }
    if timed_out {
        text.push_str(&format!("\n\n[timed out after {} ms]", timeout.as_millis()));
    }
    Captured {
        exit,
        text,
        bytes,
        truncated: cut,
    }
}

/// The status a shell would report: `128 + signal` for a command killed by a
/// signal, leaving -1 to mean the command never ran at all.
fn exit_code(status: &ExitStatus) -> i32 {
    #[cfg(unix)]
    if let Some(signal) = std::os::unix::process::ExitStatusExt::signal(status) {
        return 128 + signal;
    }
    status.code().unwrap_or(-1)
}

/// Cut `text` down to [`MAX_CAPTURE_BYTES`] on a character boundary.
fn truncate_to_cap(mut text: String) -> String {
    let end = cut_at(&text, MAX_CAPTURE_BYTES).len();
    text.truncate(end);
    text
}

/// Renders one response body at most once. A command head repeated by another
/// command, or a chunk repeated by another query, is replaced by a
/// back-reference; a chunk keeps its `--- [source | ts] ---` header so the
/// reader still sees every source holding that text.
#[derive(Default)]
pub(super) struct Renderer {
    /// Full captured output -> the command label that rendered its head first.
    heads: HashMap<String, String>,
    /// Chunk content -> the query that rendered it first.
    chunks: HashMap<String, String>,
    /// Text printed outside any section -> what it was called there.
    printed: Vec<(String, String)>,
}

impl Renderer {
    /// The [`PREVIEW_LINES`] head of one command's output, or a back-reference
    /// when another command produced exactly the same output.
    fn head(&mut self, label: &str, text: &str) -> String {
        match remember(&mut self.heads, text, label) {
            Some(first) => format!("(same output as \"{first}\")\n"),
            None => head_preview(text),
        }
    }

    /// Record text this response prints before its sections, so a chunk lying
    /// inside it is answered by a back-reference instead of a second copy.
    pub(super) fn shown(&mut self, label: &str, text: &str) {
        self.printed.push((label.to_string(), text.to_string()));
    }

    fn section(&mut self, query: &str, section: &Section) -> String {
        let header = format!("--- [{} | {}] ---\n", section.source, section.ts);
        let first = self
            .already_printed(&section.content)
            .or_else(|| remember(&mut self.chunks, &section.content, query));
        match first {
            Some(first) => format!("{header}(already shown above under \"{first}\")\n"),
            None => format!("{header}{}\n", section.content),
        }
    }

    /// What `content` was called where this response already carries it whole.
    fn already_printed(&self, content: &str) -> Option<String> {
        let body = content.trim_end();
        if body.is_empty() {
            return None;
        }
        self.printed
            .iter()
            .find(|(_, text)| text.contains(body))
            .map(|(label, _)| label.clone())
    }
}

/// `text` cut down to at most `limit` bytes, on a character boundary.
pub(super) fn cut_at(text: &str, limit: usize) -> &str {
    if text.len() <= limit {
        return text;
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The first [`PREVIEW_LINES`] lines of `text`, cut at [`PREVIEW_BYTES`].
fn head_preview(text: &str) -> String {
    let mut preview = String::new();
    for line in text.lines().take(PREVIEW_LINES) {
        let room = PREVIEW_BYTES.saturating_sub(preview.len());
        if line.len() >= room {
            preview.push_str(cut_at(line, room));
            preview.push_str("…\n");
            return preview;
        }
        preview.push_str(line);
        preview.push('\n');
    }
    preview
}

/// `Some(first)` when `body` was already rendered, otherwise records `name` as
/// its first renderer and returns `None`. Trailing blank lines are not part of
/// the identity.
fn remember(seen: &mut HashMap<String, String>, body: &str, name: &str) -> Option<String> {
    let key = body.trim_end();
    if key.is_empty() {
        return None;
    }
    match seen.get(key) {
        Some(first) => Some(first.clone()),
        None => {
            seen.insert(key.to_string(), name.to_string());
            None
        }
    }
}

/// The line a capture is reported by: what it exited with, how much it wrote,
/// and — when the output was indexed instead of returned — what that produced.
pub(super) fn summary(captured: &Captured, indexed: Option<&Indexed>) -> String {
    let mut line = format!("exit {}, {} bytes", captured.exit, captured.bytes);
    if captured.truncated {
        // The notice sits at the end of the captured text, past the head.
        line.push_str(" (truncated at 1 MB)");
    }
    if let Some(indexed) = indexed {
        line.push_str(&format!(", {} chunks", indexed.chunks));
        if indexed.skipped > 0 {
            line.push_str(&format!(" ({} already indexed)", indexed.skipped));
        }
    }
    line.push('\n');
    line
}

/// Append the `## <query>` block for one query. Returns the sections left out
/// because the response is already at [`MAX_RESPONSE_BYTES`].
pub(super) fn query_block(
    store: &Store,
    renderer: &mut Renderer,
    query: &str,
    limit: usize,
    source: Option<&str>,
    out: &mut String,
) -> Result<usize> {
    out.push_str(&format!("## {query}\n"));
    let sections = store.search(query, limit, source)?;
    if sections.is_empty() {
        out.push_str("No matching sections found.\n");
    }
    for (rendered, section) in sections.iter().enumerate() {
        if out.len() >= MAX_RESPONSE_BYTES {
            return Ok(sections.len() - rendered);
        }
        out.push_str(&renderer.section(query, section));
    }
    out.push('\n');
    Ok(0)
}

/// Append one block per query, then the notice for whatever the response
/// budget left out.
fn query_blocks(
    store: &Store,
    renderer: &mut Renderer,
    queries: &[String],
    limit: usize,
    source: Option<&str>,
    out: &mut String,
) -> Result<()> {
    let mut omitted = 0;
    for query in queries {
        omitted += query_block(store, renderer, query, limit, source, out)?;
    }
    if omitted > 0 {
        out.push_str(&format!(
            "… ({omitted} more sections omitted, narrow the query)\n"
        ));
    }
    Ok(())
}

/// `ctx_batch_execute` against the default index.
pub async fn batch_execute(input: BatchExecuteInput) -> Result<String> {
    let timeout = timeout_of(input.timeout_ms);
    let cwd = input.cwd.as_ref().map(PathBuf::from);
    let captured = run_all(&input.commands, cwd, timeout).await?;
    // The index connection is opened only once no await is left: it is not `Sync`,
    // and the tool future must be `Send`.
    let store = Store::open_default()?;
    render_batch(&store, &input, &captured)
}

/// Run the commands, [`MAX_PARALLEL_COMMANDS`] at a time, keeping the caller's
/// order in the result.
async fn run_all(
    commands: &[CommandSpec],
    cwd: Option<PathBuf>,
    timeout: Duration,
) -> Result<Vec<Captured>> {
    let slots = Arc::new(Semaphore::new(MAX_PARALLEL_COMMANDS));
    let running: Vec<_> = commands
        .iter()
        .map(|spec| {
            let slots = Arc::clone(&slots);
            let command = spec.command.clone();
            let cwd = cwd.clone();
            tokio::spawn(async move {
                // The timeout starts once the command does, not while it waits.
                let _slot = slots.acquire().await.expect("the semaphore stays open");
                run_command(Launch::plain(command), cwd, timeout).await
            })
        })
        .collect();
    let mut captured = Vec::with_capacity(running.len());
    for handle in running {
        captured.push(handle.await?);
    }
    Ok(captured)
}

/// Index every capture and render the per-command blocks plus the query blocks.
fn render_batch(store: &Store, input: &BatchExecuteInput, captured: &[Captured]) -> Result<String> {
    let mut out = String::new();
    let mut renderer = Renderer::default();
    for (spec, result) in input.commands.iter().zip(captured) {
        let indexed = store.index(&spec.label, &result.text)?;
        out.push_str(&format!("### {}\n", spec.label));
        out.push_str(&summary(result, Some(&indexed)));
        out.push_str(&renderer.head(&spec.label, &result.text));
        out.push('\n');
    }

    query_blocks(
        store,
        &mut renderer,
        &input.queries,
        DEFAULT_SECTION_LIMIT,
        None,
        &mut out,
    )?;
    Ok(out)
}

/// `ctx_search` against the default index.
pub fn search(input: SearchInput) -> Result<String> {
    let store = Store::open_default()?;
    search_in(
        &store,
        &input.queries,
        input.limit.unwrap_or(DEFAULT_SECTION_LIMIT),
        input.source.as_deref(),
    )
}

/// `ctx_search` against `store`, optionally within one source label.
pub fn search_in(
    store: &Store,
    queries: &[String],
    limit: usize,
    source: Option<&str>,
) -> Result<String> {
    let mut renderer = Renderer::default();
    let mut out = String::new();
    query_blocks(
        store,
        &mut renderer,
        queries,
        limit.min(MAX_SEARCH_LIMIT),
        source,
        &mut out,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(label: &str, command: &str) -> CommandSpec {
        CommandSpec {
            label: label.to_string(),
            command: command.to_string(),
        }
    }

    fn echo(label: &str) -> CommandSpec {
        spec(label, "printf 'alpha beta\\ngamma\\n'")
    }

    fn capture(commands: &[CommandSpec], timeout_ms: u64) -> Vec<Captured> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_all(commands, None, Duration::from_millis(timeout_ms)))
            .unwrap()
    }

    fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        (dir, store)
    }

    fn batch(commands: Vec<CommandSpec>, queries: Vec<String>) -> BatchExecuteInput {
        BatchExecuteInput {
            commands,
            queries,
            timeout_ms: None,
            cwd: None,
        }
    }

    #[test]
    fn repeated_command_output_is_shown_once_per_response() {
        let input = batch(vec![echo("a"), echo("b")], vec!["alpha".to_string()]);
        let captured = capture(&input.commands, DEFAULT_TIMEOUT_MS);
        let (_dir, store) = store();
        let out = render_batch(&store, &input, &captured).unwrap();

        // Two identical commands: the second head points back at the first.
        assert_eq!(out.matches("(same output as \"a\")").count(), 1, "{out}");
        // One query matching both chunks: the second chunk points back too.
        assert_eq!(
            out.matches("(already shown above under \"alpha\")").count(),
            1,
            "{out}"
        );
    }

    #[test]
    fn different_output_with_the_same_head_is_not_deduplicated() {
        let input = batch(
            vec![spec("short", "seq 1 20"), spec("long", "seq 1 100")],
            Vec::new(),
        );
        let captured = capture(&input.commands, DEFAULT_TIMEOUT_MS);
        let (_dir, store) = store();
        let out = render_batch(&store, &input, &captured).unwrap();

        // Both start with the same ten lines; only the full output decides.
        assert!(!out.contains("same output as"), "{out}");
    }

    #[test]
    fn repeated_section_is_shown_once_per_response() {
        let input = batch(vec![echo("a"), echo("b")], Vec::new());
        let captured = capture(&input.commands, DEFAULT_TIMEOUT_MS);
        let (_dir, store) = store();
        render_batch(&store, &input, &captured).unwrap();

        let queries = ["alpha".to_string(), "gamma".to_string()];
        let out = search_in(&store, &queries, DEFAULT_SECTION_LIMIT, None).unwrap();

        // Two queries x two identical chunks: the body once, a back-reference on
        // each of the other three hits, and a header for both sources.
        assert_eq!(out.matches("alpha beta\ngamma").count(), 1, "{out}");
        assert_eq!(
            out.matches("(already shown above under \"alpha\")").count(),
            3,
            "{out}"
        );
        assert!(out.contains("--- [a | "), "{out}");
        assert!(out.contains("--- [b | "), "{out}");
    }

    #[test]
    fn capture_is_bounded_and_a_timeout_keeps_what_was_read() {
        let flood = spec("flood", "yes abcdefghijklmnopqrstuvwxyz | head -c 3000000");
        let captured = capture(std::slice::from_ref(&flood), DEFAULT_TIMEOUT_MS);
        let text = &captured[0].text;
        assert!(text.contains("[output truncated at 1 MB]"), "no notice");
        assert!(captured[0].truncated, "cap not reported to the caller");
        assert!(
            text.len() < MAX_CAPTURE_BYTES + 1024,
            "kept {} bytes",
            text.len()
        );
        // A megabyte on one line must not become a megabyte of response.
        assert!(
            head_preview(text).len() <= PREVIEW_BYTES + 8,
            "head not bounded"
        );
        // Raw bytes read, and the group killed at the cap, not left running.
        assert_eq!(captured[0].bytes, MAX_CAPTURE_BYTES);
        assert_eq!(captured[0].exit, 128 + libc::SIGKILL);

        let slow = spec("slow", "echo EARLY; sleep 5");
        let captured = capture(std::slice::from_ref(&slow), 300);
        assert_eq!(captured[0].exit, 124);
        assert!(captured[0].text.contains("EARLY"), "{}", captured[0].text);
        assert!(
            captured[0].text.contains("[timed out after 300 ms]"),
            "{}",
            captured[0].text
        );
        // What the command managed to write before the timeout is reported.
        assert_eq!(captured[0].bytes, "EARLY\n".len());
    }

    /// Whether `pid` is still there; a negative `pid` asks about a whole group.
    #[cfg(unix)]
    fn alive(pid: i32) -> bool {
        #[allow(unsafe_code)]
        // nosemgrep: unsafe-block
        unsafe {
            libc::kill(pid, 0) == 0
        }
    }

    #[cfg(unix)]
    fn read_pid(path: &std::path::Path) -> Option<i32> {
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    }

    /// The client closing stdin cancels the call by dropping its future.
    /// `kill_on_drop` would then reach the direct child alone and leave
    /// everything it started running.
    #[cfg(unix)]
    #[test]
    fn a_cancelled_call_takes_its_whole_process_group_down() {
        let dir = tempfile::tempdir().unwrap();
        let group = dir.path().join("pgid");
        let child = dir.path().join("kid");
        let command = format!(
            "ps -o pgid= -p $$ > '{}'; sleep 20 & echo $! > '{}'; wait",
            group.display(),
            child.display()
        );

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let mut call = Box::pin(run_command(
                    Launch::plain(command),
                    None,
                    Duration::from_secs(30),
                ));
                let mut started = None;
                for _ in 0..250 {
                    tokio::select! {
                        _ = &mut call => panic!("the command returned on its own"),
                        _ = tokio::time::sleep(Duration::from_millis(20)) => {}
                    }
                    if let (Some(pgid), Some(kid)) = (read_pid(&group), read_pid(&child)) {
                        started = Some((pgid, kid));
                        break;
                    }
                }
                let (pgid, kid) = started.expect("the command never started");

                drop(call);
                let deadline = std::time::Instant::now() + Duration::from_secs(5);
                while (alive(kid) || alive(-pgid)) && std::time::Instant::now() < deadline {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                assert!(!alive(kid), "sleep {kid} survived the cancelled call");
                assert!(
                    !alive(-pgid),
                    "process group {pgid} survived the cancelled call"
                );
            });
    }

    #[test]
    fn stderr_is_interleaved_and_a_failure_keeps_its_exit_code() {
        let mixed = spec("mixed", "echo out; echo boom >&2; echo done");
        let captured = capture(std::slice::from_ref(&mixed), DEFAULT_TIMEOUT_MS);
        // The reason a command failed has to reach the head, in place.
        assert_eq!(captured[0].text, "out\nboom\ndone\n");

        let killed = spec("killed", "kill -TERM $$");
        let captured = capture(std::slice::from_ref(&killed), DEFAULT_TIMEOUT_MS);
        assert_eq!(captured[0].exit, 128 + libc::SIGTERM);
    }

    #[test]
    fn index_skips_duplicates_and_search_matches_any_term() {
        let (_dir, store) = store();
        let first = store.index("build", "the build error was fatal").unwrap();
        let again = store.index("build", "the build error was fatal").unwrap();
        assert_eq!((first.chunks, first.skipped), (1, 0));
        assert_eq!((again.chunks, again.skipped), (1, 1));

        // OR semantics: a question whose other words are absent still matches.
        let out = search_in(
            &store,
            &["what does the build error say".to_string()],
            5,
            None,
        )
        .unwrap();
        assert!(out.contains("the build error was fatal"), "{out}");

        // An unreachable limit is clamped, so one chunk cannot become 5000 rows.
        let out = search_in(&store, &["error".to_string()], 5000, None).unwrap();
        assert_eq!(out.matches("--- [build | ").count(), 1, "{out}");
    }
}
