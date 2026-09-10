//! SQLite FTS5 index backing the `rtk mcp` context tools.

use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::Connection;

/// Upper bound, in characters, of an indexed chunk.
const MAX_CHUNK_CHARS: usize = 1500;

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
    /// Open the index under the rtk config directory, creating both if missing.
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
                 session UNINDEXED,
                 tokenize = 'unicode61'
             );",
        )?;
        Ok(Self { conn })
    }

    /// Chunk `text` and index every chunk under `source`. Returns the chunk count.
    pub fn index(&self, source: &str, text: &str) -> Result<usize> {
        let ts = chrono::Local::now().to_rfc3339();
        let session = std::env::var("CLAUDE_SESSION_ID").unwrap_or_default();
        let chunks = chunk(text);
        let mut stmt = self
            .conn
            .prepare("INSERT INTO chunks(source, content, ts, session) VALUES (?1, ?2, ?3, ?4)")?;
        for content in &chunks {
            stmt.execute((source, content, &ts, &session))?;
        }
        Ok(chunks.len())
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

/// Quote every term so client punctuation cannot be read as FTS5 syntax.
/// Terms stay implicitly ANDed, which is FTS5's own default.
fn fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
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
