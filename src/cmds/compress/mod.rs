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

use anyhow::Result;

/// Content family picked by [`classify`]; each kind has its own crusher.
// Stub until the real crushers land: allow dead code so the contract compiles.
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn classify(text: &str) -> Kind {
    let t = text.trim_start();
    if t.starts_with('{') || t.starts_with('[') {
        Kind::Json
    } else {
        Kind::Text
    }
}

/// What [`crush`] returns.
#[allow(dead_code)]
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

/// Compress `text`. Identity stub until the real crushers land; callers can
/// wire against this signature already.
#[allow(dead_code)]
pub fn crush(text: &str) -> Result<Crushed> {
    Ok(Crushed {
        in_bytes: text.len(),
        text: text.to_owned(),
        ccr: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crush_stub_is_identity() {
        let out = crush("hello").unwrap();
        assert_eq!(out.text, "hello");
        assert_eq!(out.in_bytes, 5);
        assert!(out.ccr.is_none());
    }

    #[test]
    fn classify_json_vs_text() {
        assert_eq!(classify(" {\"a\":1}"), Kind::Json);
        assert_eq!(classify("[1,2]"), Kind::Json);
        assert_eq!(classify("plain prose"), Kind::Text);
    }
}
