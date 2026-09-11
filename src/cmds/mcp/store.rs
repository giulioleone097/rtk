//! SQLite FTS5 index backing the `tokenaut mcp` context tools.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::Connection;
use sha2::{Digest, Sha256};

/// Upper bound, in characters, of an indexed chunk.
const MAX_CHUNK_CHARS: usize = 1500;
/// Schema of the index. Bumped whenever the tables change shape.
const SCHEMA_VERSION: i64 = 2;
/// Days a chunk stays indexed before `open` prunes it;
/// `TOKENAUT_INDEX_MAX_AGE_DAYS` overrides.
const DEFAULT_MAX_AGE_DAYS: i64 = 30;
/// Upper bound on indexed chunks — newest by insertion order are kept;
/// `TOKENAUT_INDEX_MAX_CHUNKS` overrides.
const DEFAULT_MAX_CHUNKS: i64 = 200_000;

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

    /// Open (creating if missing) the index at `path`, then prune it. `source`
    /// is UNINDEXED so a query that happens to equal a label matches only real
    /// content; `key` is UNINDEXED too — it dedups, it should not be searched.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        // WAL so a writer does not block readers; NORMAL is the standard
        // pairing since WAL already checksums every commit frame.
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")?;
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
                 key UNINDEXED,
                 tokenize = 'unicode61'
             );
             CREATE TABLE IF NOT EXISTS chunk_keys (key TEXT PRIMARY KEY);
             PRAGMA user_version = {SCHEMA_VERSION};"
        ))?;
        let store = Self { conn };
        store.prune(
            env_or("TOKENAUT_INDEX_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS),
            env_or("TOKENAUT_INDEX_MAX_CHUNKS", DEFAULT_MAX_CHUNKS),
        )?;
        Ok(store)
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
            .prepare("INSERT INTO chunks(source, content, ts, key) VALUES (?1, ?2, ?3, ?4)")?;
        let mut skipped = 0;
        for content in &chunks {
            // Claiming the key first keeps a re-run of the same command from
            // filling the bm25 slots with copies of itself. The key is stored
            // on the row too so pruning can drop it with its chunk.
            let key = chunk_key(source, content);
            if claim.execute((&key,))? == 0 {
                skipped += 1;
                continue;
            }
            insert.execute((source, content, &ts, &key))?;
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

    /// Drop chunks past the age and count caps, along with their dedup keys,
    /// then incrementally merge the FTS5 index. Runs once per `open` — the
    /// caps bound the scan, and the index is a rebuildable cache anyway.
    fn prune(&self, max_age_days: i64, max_chunks: i64) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("prune: begin transaction")?;
        // ceiling: `ts` is `Local::now().to_rfc3339()`, so the cutoff is built
        // the same way and the lexicographic compare stays in the same offset
        // family. Chunks written under a different local offset can sort
        // slightly off — acceptable for a cache; store `ts` in UTC if the
        // index ever crosses time zones.
        // ceiling: the days cap is clamped so `TimeDelta::days` cannot panic
        // on an absurd override; ~1000 years ago reads as "keep everything".
        let days = max_age_days.min(365_000);
        let cutoff = (chrono::Local::now() - chrono::TimeDelta::days(days)).to_rfc3339();
        // Each cap deletes the doomed rows' keys first: a key outliving its
        // chunk would make a later `index` call "skip" content that is no
        // longer searchable, so it could never be re-indexed.
        tx.execute(
            "DELETE FROM chunk_keys WHERE key IN (
                 SELECT key FROM chunks WHERE ts < ?1)",
            (&cutoff,),
        )
        .context("prune: keys of aged chunks")?;
        let mut pruned = tx
            .execute("DELETE FROM chunks WHERE ts < ?1", (&cutoff,))
            .context("prune: aged chunks")?;
        tx.execute(
            "DELETE FROM chunk_keys WHERE key IN (
                 SELECT key FROM chunks WHERE rowid NOT IN (
                     SELECT rowid FROM chunks ORDER BY rowid DESC LIMIT ?1))",
            (max_chunks,),
        )
        .context("prune: keys of over-cap chunks")?;
        pruned += tx
            .execute(
                "DELETE FROM chunks WHERE rowid NOT IN (
                     SELECT rowid FROM chunks ORDER BY rowid DESC LIMIT ?1)",
                (max_chunks,),
            )
            .context("prune: over-cap chunks")?;
        if pruned > 0 {
            tx.execute("INSERT INTO chunks(chunks) VALUES('optimize')", [])
                .context("prune: fts5 optimize")?;
        }
        tx.commit().context("prune: commit")?;
        Ok(())
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

/// `name`d env var as a non-negative integer, else `default`. Read at call
/// time — never baked into a `const` — so tests can override it per run. Only
/// the count-cap test sets `TOKENAUT_INDEX_MAX_CHUNKS`, with a cap above every
/// other test's chunk count, so a store another test opens concurrently while
/// the var is set still prunes nothing.
fn env_or(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|&value| value >= 0)
        .unwrap_or(default)
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

    #[test]
    fn the_index_opens_in_wal_mode() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let mode: String = store
            .conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
    }

    #[test]
    fn aged_chunks_and_their_keys_are_pruned_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.sqlite");
        Store::open(&path)
            .unwrap()
            .index("exec:old", "the stale needle")
            .unwrap();
        {
            let store = Store::open(&path).unwrap();
            store
                .conn
                .execute("UPDATE chunks SET ts = '2000-01-01T00:00:00+00:00'", [])
                .unwrap();
        }
        // The row is older than the default age cap: reopening prunes it along
        // with its dedup key, so a re-index inserts instead of skipping.
        let store = Store::open(&path).unwrap();
        assert!(store.search("stale", 10, None).unwrap().is_empty());
        let indexed = store.index("exec:old", "the stale needle").unwrap();
        assert_eq!(indexed.skipped, 0);
        assert_eq!(indexed.chunks, 1);
        assert_eq!(store.search("stale", 10, None).unwrap().len(), 1);
    }

    #[test]
    fn chunks_past_the_count_cap_are_pruned_oldest_first() {
        // `Store::open` reads TOKENAUT_INDEX_MAX_CHUNKS at call time, so this
        // test — the only one that mutates it — sets a cap above every other
        // test's chunk count: a store opened concurrently mid-test still
        // prunes nothing.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.sqlite");
        std::env::set_var("TOKENAUT_INDEX_MAX_CHUNKS", "3");
        {
            let store = Store::open(&path).unwrap();
            for i in 0..5 {
                store
                    .index("exec:big", &format!("chunk number {i} unique{i}"))
                    .unwrap();
            }
        }
        let store = Store::open(&path).unwrap();
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        let oldest_gone = store.search("unique0", 10, None).unwrap().is_empty()
            && store.search("unique1", 10, None).unwrap().is_empty();
        let newest_kept = store.search("unique4", 10, None).unwrap().len();
        let mut skipped = 0;
        for i in 0..5 {
            skipped += store
                .index("exec:big", &format!("chunk number {i} unique{i}"))
                .unwrap()
                .skipped;
        }
        std::env::remove_var("TOKENAUT_INDEX_MAX_CHUNKS");
        assert_eq!(count, 3);
        assert!(oldest_gone);
        assert_eq!(newest_kept, 1);
        // The pruned rows' keys went with them, so those two chunks
        // re-insert; the three survivors still dedup.
        assert_eq!(skipped, 3);
    }
}
