//! `tokenaut bench --history`: does the conversation history resent on every API
//! call outweigh the fresh tool output it carries? The phase-4 compression proxy
//! only pays off when history dominates, so this metric is its activation gate.
//!
//! The metric is an approximation, on purpose:
//! - An "entry" is one JSONL line whose `message.role` is `user` or `assistant`
//!   and whose `message.content` is non-empty: user messages (which is also how
//!   `tool_result` items arrive), assistant messages (which carry `text`,
//!   `thinking` and `tool_use` items). Lines without a `message` — summaries,
//!   snapshots, queue markers — are not sent to the API and do not count.
//! - `size` is the byte length of the compact-JSON serialization of
//!   `message.content`: text lengths plus JSON-structural bytes, including
//!   tool_use inputs and tool_result bodies verbatim. Line metadata (uuid,
//!   timestamp, cwd, …) is excluded — the API never sees it. Not modeled at
//!   all: system prompt, tool schemas, cache breakpoints and result headers —
//!   per-call overhead that does not grow with the conversation.
//! - Each content-bearing entry is treated as one API-call boundary: a request
//!   sends it fresh, and every later request resends it as history. So for
//!   entries `e_0..e_{n-1}`, `fresh = Σ size_i` and
//!   `history = Σ size_i × (n − 1 − i)`, computed per file by keeping each
//!   entry's size and walking once — O(n).

use anyhow::Result;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// History columns are printed in MiB.
const MIB: f64 = 1_048_576.0;
/// The history compression the phase-4 proxy targets; the verdict line prices it.
const PROXY_HISTORY_COMPRESSION: f64 = 0.60;
const DIR_COLUMN: usize = 40;

#[derive(Default)]
struct Measurement {
    sessions: usize,
    entries: u64,
    fresh_bytes: u64,
    history_bytes: u64,
}

impl Measurement {
    fn add(&mut self, other: &Measurement) {
        self.sessions += other.sessions;
        self.entries += other.entries;
        self.fresh_bytes += other.fresh_bytes;
        self.history_bytes += other.history_bytes;
    }

    /// `history / (history + fresh)` — the share of request bytes that is
    /// resent conversation rather than new content.
    fn history_share(&self) -> f64 {
        let total = self.history_bytes + self.fresh_bytes;
        if total == 0 {
            0.0
        } else {
            100.0 * self.history_bytes as f64 / total as f64
        }
    }
}

/// Exit code is always `Ok(0)`: this is a measurement, not a gate — an empty
/// corpus reports zeros, not failure.
pub fn run_history(days: u64, config_dirs: &[PathBuf]) -> Result<i32> {
    let config_dirs = super::resolve_config_dirs(config_dirs);
    let mut total = Measurement::default();
    println!("history vs fresh bytes per transcript (tokens ≈ bytes / 4)");
    println!(
        "{:<width$} {:>8} {:>8} {:>10} {:>11} {:>7}",
        "config dir",
        "sessions",
        "entries",
        "fresh MB",
        "history MB",
        "hist %",
        width = DIR_COLUMN
    );
    for dir in &config_dirs {
        let measured = measure_dir(dir, days);
        total.add(&measured);
        print_row(&dir.display().to_string(), &measured);
    }
    print_row("TOTAL", &total);
    // A proxy compressing history by 60% removes 60% of the history share.
    let cut = PROXY_HISTORY_COMPRESSION * total.history_share();
    let saved_tokens = (PROXY_HISTORY_COMPRESSION * total.history_bytes as f64 / 4.0) as u64;
    println!(
        "proxy at 60% history compression would cut ≈ {cut:.1}% of request bytes \
         (≈ {} tokens)",
        fmt_tokens(saved_tokens)
    );
    Ok(0)
}

fn print_row(label: &str, m: &Measurement) {
    println!(
        "{:<width$.width$} {:>8} {:>8} {:>10.2} {:>11.2} {:>6.1}%",
        label,
        m.sessions,
        m.entries,
        m.fresh_bytes as f64 / MIB,
        m.history_bytes as f64 / MIB,
        m.history_share(),
        width = DIR_COLUMN
    );
}

/// `1234567` → `1.2M`, `12345` → `12.3k`, smaller numbers stay plain.
fn fmt_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1e6)
    } else if tokens >= 10_000 {
        format!("{:.1}k", tokens as f64 / 1e3)
    } else {
        tokens.to_string()
    }
}

/// Every session file `run` discovers under one config dir, measured.
fn measure_dir(config_dir: &Path, days: u64) -> Measurement {
    let mut out = Measurement::default();
    for path in super::transcripts(config_dir, days) {
        out.sessions += 1;
        let file = measure_file(&path);
        out.entries += file.entries;
        out.fresh_bytes += file.fresh_bytes;
        out.history_bytes += file.history_bytes;
    }
    out
}

/// `fresh` is each entry's bytes once; `history` is each entry's bytes once per
/// content-bearing entry that follows it in the same file.
fn measure_file(path: &Path) -> Measurement {
    let mut out = Measurement::default();
    let Ok(file) = File::open(path) else {
        return out;
    };
    let mut reader = BufReader::new(file);
    let mut sizes: Vec<u64> = Vec::new();
    let mut raw = String::new();
    loop {
        raw.clear();
        match reader.read_line(&mut raw) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Ok(entry) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        if let Some(size) = entry_content_size(&entry) {
            sizes.push(size);
        }
    }
    out.entries = sizes.len() as u64;
    out.fresh_bytes = sizes.iter().sum();
    let n = sizes.len() as u64;
    out.history_bytes = sizes
        .iter()
        .enumerate()
        .map(|(i, size)| size * (n - 1 - i as u64))
        .sum();
    out
}

/// The size of a line's serialized content when the line is an entry an API
/// request would carry; `None` for lines the API never sees.
fn entry_content_size(entry: &Value) -> Option<u64> {
    let message = &entry["message"];
    if !matches!(message["role"].as_str(), Some("user") | Some("assistant")) {
        return None;
    }
    let content = &message["content"];
    let empty = match content {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        _ => false,
    };
    if empty {
        return None;
    }
    // A `Value` is already valid JSON, so serialization cannot fail.
    serde_json::to_vec(content)
        .ok()
        .map(|bytes| bytes.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Four content-bearing entries plus two lines that must not count (no
    /// `message` object, and one malformed line). Compact-JSON content sizes:
    /// `"hello"` → 7,
    /// `[{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]` → 98,
    /// `[{"type":"tool_result","tool_use_id":"t1","content":"output text"}]` → 67,
    /// `[{"type":"text","text":"done"}]` → 31.
    /// fresh = 7+98+67+31 = 203; history = 7·3 + 98·2 + 67·1 + 31·0 = 284.
    fn fixture_config_dir() -> (tempfile::TempDir, PathBuf) {
        let temp = tempfile::tempdir().expect("tempdir");
        let config = temp.path().join(".claude-test");
        let project = config.join("projects").join("-tmp-fixture");
        fs::create_dir_all(&project).expect("project dir");
        let lines = [
            r#"{"type":"user","message":{"role":"user","content":"hello"}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]}}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"output text"}]}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}"#,
            r#"{"type":"summary","summary":"checkpoint"}"#,
            "not json at all",
        ];
        fs::write(project.join("s1.jsonl"), lines.join("\n") + "\n").expect("write session");
        (temp, config)
    }

    #[test]
    fn history_is_each_entry_times_the_entries_that_follow_it() {
        let (_temp, config) = fixture_config_dir();
        let measured = measure_dir(&config, 30);
        assert_eq!(measured.sessions, 1);
        assert_eq!(measured.entries, 4);
        assert_eq!(measured.fresh_bytes, 203);
        assert_eq!(measured.history_bytes, 284);
    }

    #[test]
    fn empty_dir_reports_zeros_and_exit_zero() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config = temp.path().join(".claude-empty");
        fs::create_dir_all(config.join("projects")).expect("projects dir");
        let measured = measure_dir(&config, 30);
        assert_eq!(measured.sessions, 0);
        assert_eq!(measured.entries, 0);
        assert_eq!(measured.fresh_bytes, 0);
        assert_eq!(measured.history_bytes, 0);
        assert_eq!(run_history(30, &[config]).expect("run_history"), 0);
    }
}
