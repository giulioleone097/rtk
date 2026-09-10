//! SQLite FTS5 index backing the `tokenaut mcp` context tools.

use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::Connection;
use sha2::{Digest, Sha256};

/// Upper bound, in characters, of an indexed chunk.
const MAX_CHUNK_CHARS: usize = 1500;
/// Schema of the index. Bumped whenever the tables change shape.
const SCHEMA_VERSION: i64 = 1;

/// What one call to [`Store::index`] wrote.
pub struct Indexed {
    /// Chunks the text was split into.
    pub chunks: usize,
    /// Chunks already present under this source, so not inserted again.
    pub skipped: usize,
}

/// One indexed chunk as returned by a search.
pub struct Section {
    pub source: String,
    pub content: String,
    pub ts: String,
}

/// The FTS5 index. One connection per tool call; nothing is kept open between them.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open the index under the tokenaut config directory, creating both if missing.
    pub fn open_default() -> Result<Self> {
        let dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(crate::core::constants::RTK_DATA_DIR);
        std::fs::create_dir_all(&dir)?;
        Self::open(&dir.join("index.sqlite"))
    }

    /// Open (creating if missing) the index at `path`. `source` is UNINDEXED so
    /// a query that happens to equal a label matches only real content.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version < SCHEMA_VERSION {
            // The index is a cache of fetched and executed output: rebuilding
            // it costs one re-run, so an older schema is dropped, not migrated.
            conn.execute_batch("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS chunk_keys;")?;
        }
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                 source UNINDEXED,
                 content,
                 ts UNINDEXED,
                 tokenize = 'unicode61'
             );
             CREATE TABLE IF NOT EXISTS chunk_keys (key TEXT PRIMARY KEY);
             PRAGMA user_version = {SCHEMA_VERSION};"
        ))?;
        Ok(Self { conn })
    }

    /// Chunk `text` and index every chunk not yet stored under `source`.
    pub fn index(&self, source: &str, text: &str) -> Result<Indexed> {
        let ts = chrono::Local::now().to_rfc3339();
        let chunks = chunk(text);
        let mut claim = self
            .conn
            .prepare("INSERT OR IGNORE INTO chunk_keys(key) VALUES (?1)")?;
        let mut insert = self
            .conn
            .prepare("INSERT INTO chunks(source, content, ts) VALUES (?1, ?2, ?3)")?;
        let mut skipped = 0;
        for content in &chunks {
            // Claiming the key first keeps a re-run of the same command from
            // filling the bm25 slots with copies of itself.
            if claim.execute((chunk_key(source, content),))? == 0 {
                skipped += 1;
                continue;
            }
            insert.execute((source, content, &ts))?;
        }
        Ok(Indexed {
            chunks: chunks.len(),
            skipped,
        })
    }

    /// The `limit` best chunks for `query`, ranked by bm25. `rowid` breaks ties so
    /// two identical queries in one response rank the same rows the same way.
    /// `source`, when set, restricts the match to chunks indexed under that
    /// exact source.
    pub fn search(&self, query: &str, limit: usize, source: Option<&str>) -> Result<Vec<Section>> {
        let expr = fts_query(query);
        if expr.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT source, content, ts FROM chunks
             WHERE chunks MATCH ?1 AND (?2 IS NULL OR source = ?2)
             ORDER BY bm25(chunks), rowid LIMIT ?3",
        )?;
        let sections = stmt
            .query_map((expr, source, limit), |row| {
                Ok(Section {
                    source: row.get(0)?,
                    content: row.get(1)?,
                    ts: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(sections)
    }
}

/// Quote every term so client punctuation cannot be read as FTS5 syntax, and
/// join them with OR so a natural-language question still matches: bm25 then
/// ranks by how many of its terms a chunk covers.
fn fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Identity of a chunk within one source, for the duplicate guard.
fn chunk_key(source: &str, content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update([0u8]);
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Split `text` on blank lines, packing paragraphs into chunks of at most
/// [`MAX_CHUNK_CHARS`]. A paragraph longer than that is cut at the limit.
fn chunk(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in text.split("\n\n") {
        let paragraph = paragraph.trim_start_matches('\n').trim_end();
        if paragraph.is_empty() {
            continue;
        }
        for piece in split_at_limit(paragraph) {
            // +1 for the newline that rejoins two paragraphs in one chunk.
            let fits = current.chars().count() + piece.chars().count() < MAX_CHUNK_CHARS;
            if !current.is_empty() && !fits {
                chunks.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(piece);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Cut `text` into pieces of at most [`MAX_CHUNK_CHARS`] characters, breaking
/// at the last newline (else the last whitespace of any kind, a non-breaking
/// space included) before the limit so no word is split across two chunks and
/// lost to the search; a run with no break at all is cut hard at the limit.
fn split_at_limit(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    loop {
        let Some((limit, _)) = rest.char_indices().nth(MAX_CHUNK_CHARS) else {
            if !rest.is_empty() {
                pieces.push(rest);
            }
            return pieces;
        };
        let window = &rest[..limit];
        let cut = window
            .rfind('\n')
            .or_else(|| window.rfind(char::is_whitespace))
            .filter(|&at| at > 0)
            .unwrap_or(limit);
        let piece = rest[..cut].trim_end();
        if !piece.is_empty() {
            pieces.push(piece);
        }
        rest = rest[cut..].trim_start_matches(char::is_whitespace);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_paragraphs_are_cut_on_line_boundaries() {
        let mut text = String::new();
        for i in 1..=800 {
            text.push_str(&format!("filler line {i} padding padding\n"));
        }
        text.push_str("the needle is here");
        let lines: std::collections::HashSet<&str> = text.lines().collect();
        let chunks = chunk(&text);
        assert!(chunks.len() > 1);
        for piece in &chunks {
            assert!(piece.chars().count() <= MAX_CHUNK_CHARS);
            for line in piece.lines() {
                assert!(lines.contains(line), "split mid-line: {line:?}");
            }
        }
        assert!(chunks.last().unwrap().contains("the needle is here"));
    }

    #[test]
    fn paragraphs_are_cut_on_any_whitespace() {
        // Separated by non-breaking spaces only: no '\n' and no ' ' to break
        // on. The word is seven characters with its separator, so the hard cut
        // at MAX_CHUNK_CHARS lands inside a word rather than between two.
        let mut text = String::new();
        while text.chars().count() < MAX_CHUNK_CHARS + 200 {
            text.push_str("alphas\u{a0}");
        }
        let chunks = chunk(&text);
        assert!(chunks.len() > 1);
        for piece in &chunks {
            for word in piece.split(char::is_whitespace).filter(|w| !w.is_empty()) {
                assert_eq!(word, "alphas", "split mid-word: {word:?}");
            }
        }
    }

    #[test]
    fn a_query_equal_to_a_label_matches_no_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        store
            .index("fetch:zebra", "content without that word")
            .unwrap();

        assert!(store.search("zebra", 10, None).unwrap().is_empty());
        assert_eq!(
            store
                .search("content", 10, Some("fetch:zebra"))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn an_index_from_the_previous_schema_is_rebuilt_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.sqlite");
        let old = Connection::open(&path).unwrap();
        old.execute_batch(
            "CREATE VIRTUAL TABLE chunks USING fts5(
                 source,
                 content,
                 ts UNINDEXED,
                 tokenize = 'unicode61'
             );
             CREATE TABLE chunk_keys (key TEXT PRIMARY KEY);
             INSERT INTO chunks(source, content, ts)
                 VALUES ('fetch:zebra', 'stale row', '2020-01-01T00:00:00+00:00');",
        )
        .unwrap();
        drop(old);

        let store = Store::open(&path).unwrap();
        let version: i64 = store
            .conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        assert!(store.search("stale", 10, None).unwrap().is_empty());
        assert!(store.search("zebra", 10, None).unwrap().is_empty());
    }
}
