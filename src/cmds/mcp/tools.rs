//! Tool inputs, command execution and response rendering for `rtk mcp`.

use std::collections::HashMap;
use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;

use super::store::{Section, Store};

/// Per-command timeout when the caller does not set one.
const DEFAULT_TIMEOUT_MS: u64 = 60_000;
/// Upper bound on the output captured from a single command.
const MAX_CAPTURE_BYTES: usize = 1024 * 1024;
/// Lines of raw output echoed back per command.
const PREVIEW_LINES: usize = 10;
/// Sections per query when `ctx_batch_execute` answers queries.
const BATCH_QUERY_LIMIT: usize = 5;
/// Sections per query when `ctx_search` does not get a limit.
const DEFAULT_SEARCH_LIMIT: usize = 5;

/// One shell command to run and index.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CommandSpec {
    /// Section header, and the search source of the chunks it produces.
    pub label: String,
    /// Command line, run with `sh -c` in the server's working directory.
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
}

/// Input of the `ctx_search` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchInput {
    /// Questions to answer from the index.
    pub queries: Vec<String>,
    /// Sections returned per query. Defaults to 5.
    pub limit: Option<usize>,
}

/// What one command produced.
struct Captured {
    exit: i32,
    /// Merged stdout and stderr, capped at [`MAX_CAPTURE_BYTES`].
    text: String,
    /// Size of the merged output before the cap.
    bytes: usize,
}

/// Run `command` with `sh -c`, merging stdout and stderr. Never fails: a spawn
/// error or a timeout is reported as the command's output.
async fn run_command(command: String, timeout: Duration) -> Captured {
    let spawned = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    let child = match spawned {
        Ok(child) => child,
        Err(err) => {
            return Captured {
                exit: -1,
                text: format!("failed to spawn: {err}"),
                bytes: 0,
            }
        }
    };
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            let bytes = text.len();
            Captured {
                exit: output.status.code().unwrap_or(-1),
                text: cap(text),
                bytes,
            }
        }
        Ok(Err(err)) => Captured {
            exit: -1,
            text: format!("failed to run: {err}"),
            bytes: 0,
        },
        // Dropping the wait future drops the child, and `kill_on_drop` kills it.
        Err(_) => Captured {
            exit: 124,
            text: format!("[timed out after {} ms]", timeout.as_millis()),
            bytes: 0,
        },
    }
}

/// Truncate `text` to [`MAX_CAPTURE_BYTES`], noting the cut.
fn cap(mut text: String) -> String {
    if text.len() <= MAX_CAPTURE_BYTES {
        return text;
    }
    let mut end = MAX_CAPTURE_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str("\n\n[output truncated at 1 MB]");
    text
}

/// Renders one response body at most once. A per-command head or a chunk body
/// already shown in the same response is replaced by a back-reference; a
/// chunk keeps its `--- [source | ts] ---` header so the reader still sees every
/// source holding that text.
#[derive(Default)]
struct Renderer {
    /// Body -> the command label or query that rendered it first.
    seen: HashMap<String, String>,
}

impl Renderer {
    /// The [`PREVIEW_LINES`] head of one command's output, or a back-reference
    /// when the same text was already rendered in this response.
    fn head(&mut self, label: &str, text: &str) -> String {
        let preview: String = text
            .lines()
            .take(PREVIEW_LINES)
            .map(|line| format!("{line}\n"))
            .collect();
        match self.remember(&preview, label) {
            Some(first) => format!("(same output as \"{first}\")\n"),
            None => preview,
        }
    }

    fn section(&mut self, query: &str, section: &Section) -> String {
        let header = format!("--- [{} | {}] ---\n", section.source, section.ts);
        match self.remember(&section.content, query) {
            Some(first) => format!("{header}(already shown above for \"{first}\")\n"),
            None => format!("{header}{}\n", section.content),
        }
    }

    /// `Some(first)` when `body` was already rendered, otherwise records `name`
    /// as its first renderer and returns `None`. Trailing blank lines are not
    /// part of the identity: a head and the chunk cut from it match.
    fn remember(&mut self, body: &str, name: &str) -> Option<String> {
        let key = body.trim_end();
        if key.is_empty() {
            return None;
        }
        match self.seen.get(key) {
            Some(first) => Some(first.clone()),
            None => {
                self.seen.insert(key.to_string(), name.to_string());
                None
            }
        }
    }
}

/// Append the `## <query>` block for one query.
fn query_block(
    store: &Store,
    renderer: &mut Renderer,
    query: &str,
    limit: usize,
    out: &mut String,
) -> Result<()> {
    out.push_str(&format!("## {query}\n"));
    let sections = store.search(query, limit)?;
    if sections.is_empty() {
        out.push_str("No matching sections found.\n");
    }
    for section in &sections {
        out.push_str(&renderer.section(query, section));
    }
    out.push('\n');
    Ok(())
}

/// `ctx_batch_execute` against the default index.
pub async fn batch_execute(input: BatchExecuteInput) -> Result<String> {
    let timeout = Duration::from_millis(input.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS));
    let captured = run_all(&input.commands, timeout).await?;
    // The index connection is opened only once no await is left: it is not `Sync`,
    // and the tool future must be `Send`.
    let store = Store::open_default()?;
    render_batch(&store, &input, &captured)
}

/// Run every command in parallel, keeping the caller's order.
async fn run_all(commands: &[CommandSpec], timeout: Duration) -> Result<Vec<Captured>> {
    let running: Vec<_> = commands
        .iter()
        .map(|spec| tokio::spawn(run_command(spec.command.clone(), timeout)))
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
        let chunks = store.index(&spec.label, &result.text)?;
        out.push_str(&format!("### {}\n", spec.label));
        out.push_str(&format!(
            "exit {}, {} bytes, {} chunks\n",
            result.exit, result.bytes, chunks
        ));
        out.push_str(&renderer.head(&spec.label, &result.text));
        out.push('\n');
    }

    for query in &input.queries {
        query_block(store, &mut renderer, query, BATCH_QUERY_LIMIT, &mut out)?;
    }
    Ok(out)
}

/// `ctx_search` against the default index.
pub fn search(input: SearchInput) -> Result<String> {
    let store = Store::open_default()?;
    search_in(
        &store,
        &input.queries,
        input.limit.unwrap_or(DEFAULT_SEARCH_LIMIT),
    )
}

/// `ctx_search` against `store`.
pub fn search_in(store: &Store, queries: &[String], limit: usize) -> Result<String> {
    let mut renderer = Renderer::default();
    let mut out = String::new();
    for query in queries {
        query_block(store, &mut renderer, query, limit, &mut out)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(label: &str) -> CommandSpec {
        CommandSpec {
            label: label.to_string(),
            command: "printf 'alpha beta\\ngamma\\n'".to_string(),
        }
    }

    fn capture(commands: &[CommandSpec]) -> Vec<Captured> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_all(commands, Duration::from_millis(DEFAULT_TIMEOUT_MS)))
            .unwrap()
    }

    #[test]
    fn repeated_command_output_is_shown_once_per_response() {
        let input = BatchExecuteInput {
            commands: vec![spec("a"), spec("b")],
            queries: vec!["alpha".to_string()],
            timeout_ms: None,
        };
        let captured = capture(&input.commands);

        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let out = render_batch(&store, &input, &captured).unwrap();

        // Two identical commands and one query matching both chunks: the body
        // appears once in the whole response, and the three other places that
        // would repeat it point back at the head that showed it.
        assert_eq!(out.matches("alpha beta\ngamma").count(), 1, "{out}");
        assert_eq!(out.matches("(same output as \"a\")").count(), 1, "{out}");
        assert_eq!(
            out.matches("(already shown above for \"a\")").count(),
            2,
            "{out}"
        );
    }

    #[test]
    fn repeated_section_is_shown_once_per_response() {
        let input = BatchExecuteInput {
            commands: vec![spec("a"), spec("b")],
            queries: Vec::new(),
            timeout_ms: None,
        };
        let captured = capture(&input.commands);

        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        render_batch(&store, &input, &captured).unwrap();

        let queries = ["alpha".to_string(), "gamma".to_string()];
        let out = search_in(&store, &queries, DEFAULT_SEARCH_LIMIT).unwrap();

        // Two queries x two identical chunks: the body once, a back-reference on
        // each of the other three hits, and a header for both sources.
        assert_eq!(out.matches("alpha beta\ngamma").count(), 1, "{out}");
        assert_eq!(
            out.matches("(already shown above for \"alpha\")").count(),
            3,
            "{out}"
        );
        assert!(out.contains("--- [a | "), "{out}");
        assert!(out.contains("--- [b | "), "{out}");
    }
}
