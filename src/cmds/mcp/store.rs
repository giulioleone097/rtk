//! SQLite FTS5 index backing the `tokenaut mcp` context tools.

use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::Connection;
use sha2::{Digest, Sha256};

/// Upper bound, in characters, of an indexed chunk.
const MAX_CHUNK_CHARS: usize = 1500;

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

    /// Open (creating if missing) the index at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                 source,
                 content,
                 ts UNINDEXED,
                 tokenize = 'unicode61'
             );
             CREATE TABLE IF NOT EXISTS chunk_keys (key TEXT PRIMARY KEY);",
        )?;
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
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Section>> {
        let expr = fts_query(query);
        if expr.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT source, content, ts FROM chunks WHERE chunks MATCH ?1
             ORDER BY bm25(chunks), rowid LIMIT ?2",
        )?;
        let rows = stmt.query_map((expr, limit), |row| {
            Ok(Section {
                source: row.get(0)?,
                content: row.get(1)?,
                ts: row.get(2)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
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

/// Cut `text` into pieces of at most [`MAX_CHUNK_CHARS`] characters.
fn split_at_limit(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut seen = 0;
    for (idx, _) in text.char_indices() {
        if seen == MAX_CHUNK_CHARS {
            pieces.push(&text[start..idx]);
            start = idx;
            seen = 0;
        }
        seen += 1;
    }
    if start < text.len() {
        pieces.push(&text[start..]);
    }
    pieces
}
