//! `tokenaut proxy` compression stage: classify a request content part, crush
//! it by kind, and keep the original retrievable through the context index
//! (CCR — compress, cache, retrieve).
//!
//! Public contract (locked — `src/cmds/proxy` calls only these):
//!
//! - [`Kind`] and [`classify`]: route a text part to its crusher.
//! - [`Crushed`] and [`crush`]: compress one part, deterministically — the same
//!   input and index state must produce the same output, so a request prefix
//!   stays byte-identical between turns and prompt caching keeps working.

mod code;
mod json;
mod log;
mod text;

use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;
use sha2::{Digest, Sha256};

use crate::cmds::mcp::store::Store;
use crate::core::utils;

/// Originals at least this many bytes get a CCR retrieval marker when crushed.
const CCR_MIN_BYTES: usize = 4096;
/// A crushed output at least this many percent of the input is not a win.
const MIN_WIN_PERCENT: usize = 97;

/// One content part already crushed once carries this trailer. Matching it is
/// the idempotency check: re-crushing a marked part would break prefix
/// stability, so the input is returned untouched.
static CCR_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)\[original \d+B → ctx_search \{queries:\["ccr:([0-9a-f]{12})"\]"#).unwrap()
});

/// A line opening with one of these is a strong code signal across the
/// languages the proxy sees (Rust, Python, JS/TS, C/C++, Java, Go, Ruby).
// ceiling: line heuristics, not an AST — a heredoc or markdown list can fake
// the signal. Upgrade trigger = a real parser dep.
static CODE_KEYWORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^\s*(fn|def|class|struct|impl|import|from|package|#include",
        r"|public|private|static|const|let|var|func|interface|enum|type|module|end)\b"
    ))
    .unwrap()
});

/// One line that looks like log or terminal output: timestamps, bracketed or
/// key=value levels, `LEVEL:` prefixes, build-tool banners, separator runs and
/// shell prompts.
// ceiling: tuned on common tool output, not a grammar — a log format the
// alternation misses classifies as Text and gets the conservative crusher.
static LOG_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        // 2024-01-15T10:30:00Z / 2024-01-15 10:30:00
        r"(?i)^\s*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",
        // 10:30:00
        r"|^\s*\d{2}:\d{2}:\d{2}",
        // Jan  5 10:30:00 (syslog)
        r"|^\s*[a-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}",
        // [INFO] / [error] / [WARN 2024-...]
        r"|^\s*\[(?:trace|debug|info|warn(?:ing)?|error|err|fatal|crit(?:ical)?|notice|ok|fail|pass|done)\]",
        // INFO: msg / error | msg
        r"|^\s*(?:trace|debug|info|warn(?:ing)?|error|err|fatal|crit(?:ical)?|notice)\s*[:|]",
        // level=info / severity=ERROR
        r"|\b(?:level|lvl|severity)=(?:trace|debug|info|warn(?:ing)?|err(?:or)?|fatal|crit(?:ical)?|notice)\b",
        // cargo/npm-style banners: "Compiling x", "Finished dev", "Downloading y"
        r"|^\s*(?:compiling|downloading|downloaded|installing|installed|finished|running|updating|locking|building|fetching|checking|linking)\s",
        // separator runs: -----, =====, #####, *****
        r"|^\s*[-=*#_]{3,}\s*$",
        // shell prompt: "$ cargo build"
        r"|^\s*\$\s"
    ))
    .unwrap()
});

/// Content family picked by [`classify`]; each kind has its own crusher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Parseable JSON: API payloads, tool results, structured logs.
    Json,
    /// Source code in any language.
    Code,
    /// Log or terminal output: timestamps, levels, progress noise.
    Log,
    /// Everything else.
    Text,
}

/// Classify `text` for routing to a crusher.
pub fn classify(text: &str) -> Kind {
    if utils::from_json_str::<serde_json::Value>(text).is_ok() {
        return Kind::Json;
    }
    if looks_like_code(text) {
        return Kind::Code;
    }
    if looks_like_log(text) {
        return Kind::Log;
    }
    Kind::Text
}

/// Code heuristic: at least three keyword-opening lines, or at least three
/// lines of which 30%+ end in `;`, `{` or `}`.
fn looks_like_code(text: &str) -> bool {
    let mut non_empty = 0usize;
    let mut keywords = 0usize;
    let mut punct_end = 0usize;
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        non_empty += 1;
        if CODE_KEYWORD.is_match(line) {
            keywords += 1;
        }
        if matches!(t.chars().last(), Some(';' | '{' | '}')) {
            punct_end += 1;
        }
    }
    keywords >= 3 || (punct_end >= 3 && punct_end * 10 >= non_empty * 3)
}

/// Log heuristic: at least half of the non-empty lines match [`LOG_LINE`].
fn looks_like_log(text: &str) -> bool {
    let mut non_empty = 0usize;
    let mut matched = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        non_empty += 1;
        if LOG_LINE.is_match(line) {
            matched += 1;
        }
    }
    non_empty > 0 && matched * 2 >= non_empty
}

/// What [`crush`] returns.
#[derive(Debug)]
pub struct Crushed {
    /// The text that replaces the original inside the request part. When
    /// `ccr` is set it embeds the retrieval marker so the model knows the
    /// full original can be pulled back with `ctx_search`.
    pub text: String,
    /// Original byte count, for stats.
    pub in_bytes: usize,
    /// `ccr:<id>` when the original was indexed for on-demand retrieval.
    pub ccr: Option<String>,
}

/// Compress `text`, indexing the original for retrieval when the win is real.
/// Opens the default context index; a store failure is non-fatal and only
/// disables CCR for this part.
pub fn crush(text: &str) -> Result<Crushed> {
    let store = Store::open_default().ok();
    crush_inner(text, store.as_ref())
}

/// [`crush`] against a caller-provided index — tests keep it on a tempdir.
#[cfg(test)]
pub(crate) fn crush_with(text: &str, store: &Store) -> Result<Crushed> {
    crush_inner(text, Some(store))
}

fn crush_inner(text: &str, store: Option<&Store>) -> Result<Crushed> {
    let in_bytes = text.len();
    if let Some(caps) = CCR_MARKER.captures(text) {
        let id = caps.get(1).map(|g| g.as_str()).unwrap_or_default();
        return Ok(Crushed {
            text: text.to_owned(),
            in_bytes,
            ccr: Some(format!("ccr:{id}")),
        });
    }
    let body = match classify(text) {
        Kind::Json => json::crush_kind(text),
        Kind::Code => code::crush_kind(text),
        Kind::Log => log::crush_kind(text),
        Kind::Text => text::crush_kind(text),
    };
    // Min-win: a marginal rewrite is not worth a changed request prefix.
    let body = if body.len() >= in_bytes.saturating_mul(MIN_WIN_PERCENT) / 100 {
        text.to_owned()
    } else {
        body
    };
    let mut crushed = Crushed {
        text: body,
        in_bytes,
        ccr: None,
    };
    if in_bytes >= CCR_MIN_BYTES && crushed.text.len() < in_bytes {
        if let Some(store) = store {
            let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
            let source = format!("ccr:{}", &digest[..12]);
            // The id heads the indexed payload because the index only
            // searches content, not source labels: without it the marker's
            // `queries:["ccr:<id>"]` would match no chunk. The `:` separator
            // introduces no whitespace, so the store's paragraph packing can't
            // split the id off into a chunk of its own.
            let payload = format!("{source}:{text}");
            if store.index(&source, &payload).is_ok() {
                crushed.text.push_str(&format!(
                    "\n\n[original {in_bytes}B → ctx_search {{queries:[\"{source}\"], source:\"{source}\"}}]"
                ));
                crushed.ccr = Some(source);
            }
        }
    }
    Ok(crushed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        (dir, store)
    }

    #[test]
    fn classify_json_code_log_text() {
        assert_eq!(classify(" {\"a\": 1}"), Kind::Json);
        assert_eq!(classify("[1, 2, 3]"), Kind::Json);
        assert_eq!(
            classify("fn main() {\n    let x = 1;\n    const Y: i32 = 2;\n}\n"),
            Kind::Code
        );
        assert_eq!(
            classify(
                "2024-01-15T10:30:00Z INFO starting\n2024-01-15T10:30:01Z DEBUG connected\n[ERROR] failed\n"
            ),
            Kind::Log
        );
        assert_eq!(
            classify("The quick brown fox\njumps over the lazy dog.\nNothing structured here."),
            Kind::Text
        );
    }

    #[test]
    fn classify_prefers_code_over_logish_lines() {
        // `level=` matches a log pattern, but keyword and punctuation
        // density still say code.
        let code = "struct Cfg {\n    level: u8,\n}\nfn f() {\n    let level = 1;\n}\n";
        assert_eq!(classify(code), Kind::Code);
    }

    #[test]
    fn crush_small_text_is_unchanged() {
        let out = crush_with("plain prose", &temp_store().1).unwrap();
        assert_eq!(out.text, "plain prose");
        assert_eq!(out.in_bytes, 11);
        assert!(out.ccr.is_none());
    }

    #[test]
    fn min_win_returns_original_on_marginal_rewrite() {
        // Only a trailing space to lose: 127/128 bytes ≈ 99% > 97%.
        let input = format!("{}x ", "a".repeat(126));
        let out = crush_with(&input, &temp_store().1).unwrap();
        assert_eq!(out.text, input);
        assert!(out.ccr.is_none());
    }

    #[test]
    fn ccr_indexes_compressible_original_and_marks_output() {
        let (_dir, store) = temp_store();
        let input = format!(
            "[{}]",
            (0..200)
                .map(|i| format!("{{\"name\":\"item {i}\",\"value\":{i},\"extra\":null}}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(input.len() >= CCR_MIN_BYTES);
        let out = crush_with(&input, &store).unwrap();

        let ccr = out.ccr.expect("compressible large input gets a ccr id");
        assert!(ccr.starts_with("ccr:"), "{ccr}");
        assert_eq!(ccr.len(), 4 + 12);
        assert!(out.text.len() < input.len() + 200);
        assert!(
            out.text.contains(&format!(
                "[original {}B → ctx_search {{queries:[\"{ccr}\"], source:\"{ccr}\"}}]",
                out.in_bytes
            )),
            "{}",
            out.text
        );

        // The marker's own query finds a section holding original content.
        let sections = store.search(&ccr, 5, None).unwrap();
        assert!(
            sections
                .iter()
                .any(|s| s.source == ccr && s.content.contains("\"item 0\"")),
            "no section with original content under {ccr}: {} hit(s)",
            sections.len()
        );
    }

    #[test]
    fn ccr_is_skipped_under_the_size_floor() {
        let (_dir, store) = temp_store();
        let input = "line\n\n\n\n\nnext".repeat(10);
        assert!(input.len() < CCR_MIN_BYTES);
        let out = crush_with(&input, &store).unwrap();
        assert!(out.ccr.is_none());
        assert!(store.search("next", 5, None).unwrap().is_empty());
    }

    #[test]
    fn crush_is_idempotent_on_marked_output() {
        let (_dir, store) = temp_store();
        let input = format!(
            "[{}]",
            (0..200)
                .map(|i| format!("{{\"name\":\"item {i}\",\"value\":{i}}}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        // crush() delegates to crush_with once the default store is open; the
        // tempdir store exercises the identical code path hermetically.
        let once = crush_with(&input, &store).unwrap();
        let twice = crush_with(&once.text, &store).unwrap();
        assert_eq!(twice.text, once.text);
        assert_eq!(twice.ccr, once.ccr);
    }

    #[test]
    fn store_failure_leaves_crushed_text_without_ccr() {
        // The contract path for a missing store: crush_inner with None.
        let input = "x\n\n\n\n\ny".repeat(1024);
        let out = crush_inner(&input, None).unwrap();
        assert!(out.ccr.is_none());
        assert!(!out.text.is_empty());
    }
}
