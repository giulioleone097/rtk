//! `ctx_fetch_and_index`: fetch one or more URLs with `curl`, convert HTML or
//! JSON bodies to plain text, and index the result for `ctx_search`.

use anyhow::Result;
// Through rmcp, so the derive and the server agree on one schemars version.
use rmcp::schemars::{self, JsonSchema};
use serde::Deserialize;

use super::store::Store;

/// The `curl` binary invoked for every fetch. An absolute path: this crate's
/// own build hooks intercept the bare word `curl` in shell commands, which
/// must not apply to this subprocess.
const CURL_BIN: &str = "/usr/bin/curl";
/// Per-request timeout handed to `curl --max-time`.
const FETCH_TIMEOUT_SECS: &str = "30";
/// Body size cap handed to `curl --max-filesize`, 50 MiB.
const MAX_FETCH_BYTES: &str = "52428800";
/// `curl --user-agent`.
const USER_AGENT: &str = "tokenaut/0.2.0";
/// Preview length, in characters, when the caller fetched exactly one URL.
const SINGLE_PREVIEW_CHARS: usize = 3072;
/// Preview length, in characters, once more than one URL was requested.
const MULTI_PREVIEW_CHARS: usize = 384;

/// One URL to fetch, with its own optional index label.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FetchRequest {
    /// URL to fetch. `curl` decides the scheme; `file://` works for tests.
    pub url: String,
    /// Index label for this URL's chunks. Defaults to the tool-wide `source`,
    /// then to the URL's host and path.
    pub source: Option<String>,
}

/// Input of the `ctx_fetch_and_index` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FetchInput {
    /// A single URL to fetch. Ignored when `requests` is set.
    pub url: Option<String>,
    /// Multiple URLs to fetch, each with its own label.
    pub requests: Option<Vec<FetchRequest>>,
    /// Index label applied to every request that does not set its own.
    pub source: Option<String>,
}

/// `ctx_fetch_and_index` against the default index.
pub fn fetch_and_index(input: FetchInput) -> Result<String> {
    let store = Store::open_default()?;
    fetch_and_index_with(&store, input)
}

/// `ctx_fetch_and_index` against `store`. Split out so tests can fetch into a
/// throwaway index instead of the caller's real one.
fn fetch_and_index_with(store: &Store, input: FetchInput) -> Result<String> {
    let global_source = input.source;
    let requests = match input.requests {
        Some(requests) if !requests.is_empty() => requests,
        _ => match input.url {
            Some(url) => vec![FetchRequest { url, source: None }],
            None => Vec::new(),
        },
    };
    let preview_chars = if requests.len() > 1 {
        MULTI_PREVIEW_CHARS
    } else {
        SINGLE_PREVIEW_CHARS
    };

    let mut out = String::new();
    for request in &requests {
        match curl_fetch(&request.url) {
            Ok((content_type, body)) => {
                let text = to_text(&content_type, &body);
                let label = label_for(request, &global_source);
                let source = format!("fetch:{label}");
                let indexed = store.index(&source, &text)?;
                out.push_str(&format!("### {source}\n"));
                out.push_str(&format!("{} bytes, {} chunks", body.len(), indexed.chunks));
                if indexed.skipped > 0 {
                    out.push_str(&format!(" ({} already indexed)", indexed.skipped));
                }
                out.push('\n');
                out.push_str(&preview(&text, preview_chars));
                out.push('\n');
                out.push_str(&format!(
                    "Search with ctx_search {{source: \"{source}\"}}\n"
                ));
            }
            Err(exit) => {
                out.push_str(&format!("fetch failed (curl exit {exit})\n"));
            }
        }
        out.push('\n');
    }
    Ok(out)
}

/// The label a request's chunks are indexed under: the request's own
/// `source`, else the tool-wide `source`, else the URL's host and path.
fn label_for(request: &FetchRequest, global_source: &Option<String>) -> String {
    request
        .source
        .clone()
        .or_else(|| global_source.clone())
        .unwrap_or_else(|| host_and_path(&request.url))
}

/// `<host><path>` with the scheme and any query or fragment stripped.
fn host_and_path(url: &str) -> String {
    let without_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    without_scheme
        .split(['?', '#'])
        .next()
        .unwrap_or(without_scheme)
        .to_string()
}

/// The first `chars` characters of `text`.
fn preview(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
}

/// Fetch `url` with `curl`, returning its content type and raw body, or the
/// exit code on failure (non-zero exit, spawn error or an empty body).
fn curl_fetch(url: &str) -> std::result::Result<(String, Vec<u8>), i32> {
    let body_file = tempfile::NamedTempFile::new().map_err(|_| -1)?;
    let body_path = body_file.path();
    let output = std::process::Command::new(CURL_BIN)
        .args([
            "-sL",
            "--max-time",
            FETCH_TIMEOUT_SECS,
            "--max-filesize",
            MAX_FETCH_BYTES,
            "--user-agent",
            USER_AGENT,
            "-o",
        ])
        .arg(body_path)
        .args(["-w", "%{content_type}", url])
        .output()
        .map_err(|_| -1)?;
    let exit = output.status.code().unwrap_or(-1);
    let body = std::fs::read(body_path).unwrap_or_default();
    if exit != 0 || body.is_empty() {
        return Err(exit);
    }
    Ok((String::from_utf8_lossy(&output.stdout).into_owned(), body))
}

/// Convert a fetched body to plain text: HTML is stripped to text, JSON is
/// pretty-printed, anything else passes through unchanged.
fn to_text(content_type: &str, body: &[u8]) -> String {
    let raw = String::from_utf8_lossy(body).into_owned();
    if looks_like_html(content_type, &raw) {
        html_to_text(&raw)
    } else if looks_like_json(content_type, &raw) {
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(value) => serde_json::to_string_pretty(&value).unwrap_or(raw),
            Err(_) => raw,
        }
    } else {
        raw
    }
}

fn looks_like_html(content_type: &str, body: &str) -> bool {
    if content_type.to_ascii_lowercase().contains("html") {
        return true;
    }
    if !content_type.is_empty() {
        return false;
    }
    body.trim_start().starts_with('<')
}

fn looks_like_json(content_type: &str, body: &str) -> bool {
    if content_type.to_ascii_lowercase().contains("json") {
        return true;
    }
    if !content_type.is_empty() {
        return false;
    }
    let trimmed = body.trim_start();
    if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(body).is_ok()
}

/// Strip `html` down to its text: `<script>`, `<style>` and comments are
/// dropped with their content, a fixed set of block-level closing tags (and
/// `<br>`) become newlines, every other tag is removed, entities are decoded
/// and runs of blank lines collapse to one.
fn html_to_text(html: &str) -> String {
    let no_script = replace_block(html, "script");
    let no_style = replace_block(&no_script, "style");
    let no_comments = strip_between(&no_style, "<!--", "-->");
    let with_breaks = newline_tags(&no_comments);
    let no_tags = strip_tags(&with_breaks);
    let decoded = decode_entities(&no_tags);
    collapse_blank_lines(&decoded)
}

/// Remove every `<tag ...>...</tag>` element, content included, matched
/// case-insensitively without regard to attributes on the opening tag.
fn replace_block(html: &str, tag: &str) -> String {
    let open_needle = format!("<{tag}");
    let close_needle = format!("</{tag}>");
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut pos = 0;
    while let Some(open_rel) = lower[pos..].find(&open_needle) {
        let open_start = pos + open_rel;
        out.push_str(&html[pos..open_start]);
        match lower[open_start..].find(&close_needle) {
            Some(close_rel) => pos = open_start + close_rel + close_needle.len(),
            None => {
                // Unterminated block: drop the rest of the document with it.
                pos = html.len();
            }
        }
    }
    out.push_str(&html[pos..]);
    out
}

/// Remove every `start ... end` span, including both markers.
fn strip_between(text: &str, start: &str, end: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pos = 0;
    while let Some(open_rel) = text[pos..].find(start) {
        let open_start = pos + open_rel;
        out.push_str(&text[pos..open_start]);
        match text[open_start..].find(end) {
            Some(close_rel) => pos = open_start + close_rel + end.len(),
            None => pos = text.len(),
        }
    }
    out.push_str(&text[pos..]);
    out
}

/// Turn `</p>`, `<br>`/`<br/>`, `</div>`, `</li>`, `</h1>`..`</h6>` and
/// `</tr>` into newlines, whatever attributes or spacing they carry.
fn newline_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut chars = html.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if ch != '<' {
            out.push(ch);
            continue;
        }
        let rest = &html[idx..];
        let end = match rest.find('>') {
            Some(end) => end,
            None => {
                out.push_str(rest);
                break;
            }
        };
        let tag = &rest[1..end];
        let lower = tag.trim().trim_end_matches('/').trim().to_ascii_lowercase();
        let is_newline_tag = lower == "br"
            || lower == "/p"
            || lower == "/div"
            || lower == "/li"
            || lower == "/tr"
            || matches!(
                lower.as_str(),
                "/h1" | "/h2" | "/h3" | "/h4" | "/h5" | "/h6"
            );
        out.push_str(if is_newline_tag { "\n" } else { "" });
        if !is_newline_tag {
            out.push_str(&rest[..=end]);
        }
        for _ in 0..end {
            chars.next();
        }
    }
    out
}

/// Remove every remaining `<...>` tag.
fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut depth = 0u32;
    for ch in html.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Decode `&amp; &lt; &gt; &quot; &#39; &nbsp;` and numeric character
/// references (`&#39;`, `&#x27;`).
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if ch != '&' {
            out.push(ch);
            continue;
        }
        let rest = &text[idx..];
        if let Some(end) = rest.find(';') {
            if end <= 12 {
                let entity = &rest[1..end];
                if let Some(decoded) = decode_one_entity(entity) {
                    out.push(decoded);
                    for _ in 0..end {
                        chars.next();
                    }
                    continue;
                }
            }
        }
        out.push('&');
    }
    out
}

fn decode_one_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "nbsp" => Some('\u{a0}'),
        _ => {
            let digits = entity.strip_prefix('#')?;
            let code = if let Some(hex) = digits
                .strip_prefix('x')
                .or_else(|| digits.strip_prefix('X'))
            {
                u32::from_str_radix(hex, 16).ok()?
            } else {
                digits.parse::<u32>().ok()?
            };
            char::from_u32(code)
        }
    }
}

/// Collapse runs of two or more blank lines to one, trimming trailing
/// whitespace from every line.
fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_blank = false;
    for line in text.lines() {
        let trimmed = line.trim_end();
        let is_blank = trimmed.trim().is_empty();
        if is_blank && prev_blank {
            continue;
        }
        out.push_str(trimmed);
        out.push('\n');
        prev_blank = is_blank;
    }
    out.trim_matches('\n').to_string()
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
    fn html_is_converted_to_text_dropping_script_style_and_decoding_entities() {
        let html = r#"<html><head>
<style>body { color: red; }</style>
<script>alert(1);</script>
</head>
<body>
<!-- a comment -->
<h1>Title</h1>
<p>Hello &amp; welcome, &quot;friend&quot;</p>
<br>
<p>Second &#39;paragraph&#39;</p>
</body></html>"#;

        let text = html_to_text(html);
        assert!(!text.contains("color: red"), "{text}");
        assert!(!text.contains("alert(1)"), "{text}");
        assert!(!text.contains('<'), "{text}");
        assert!(text.contains("Title"), "{text}");
        assert!(text.contains("Hello & welcome, \"friend\""), "{text}");
        assert!(text.contains("Second 'paragraph'"), "{text}");

        // <br> put the two paragraphs on separate lines.
        let hello_line = text.lines().find(|l| l.contains("Hello")).unwrap();
        assert!(!hello_line.contains("Second"), "{text}");
    }

    #[test]
    fn fetch_indexes_html_under_its_label_and_source_filters_to_it() {
        let source_dir = tempfile::tempdir().unwrap();
        let path = source_dir.path().join("page.html");
        std::fs::write(
            &path,
            "<html><body><p>Example content alpha beta</p></body></html>",
        )
        .unwrap();
        let url = format!("file://{}", path.display());

        let (_store_dir, store) = temp_store();
        let input = FetchInput {
            url: None,
            requests: Some(vec![
                FetchRequest {
                    url: url.clone(),
                    source: Some("mine".to_string()),
                },
                FetchRequest {
                    url,
                    source: Some("other".to_string()),
                },
            ]),
            source: None,
        };
        let out = fetch_and_index_with(&store, input).unwrap();
        assert!(out.contains("### fetch:mine"), "{out}");
        assert!(out.contains("### fetch:other"), "{out}");
        assert!(out.contains("Example content alpha"), "{out}");
        assert!(
            out.contains("Search with ctx_search {source: \"fetch:mine\"}"),
            "{out}"
        );

        let mine = store.search("alpha", 10, Some("fetch:mine")).unwrap();
        assert_eq!(mine.len(), 1, "expected one chunk, got {}", mine.len());
        assert_eq!(mine[0].source, "fetch:mine");

        let both = store.search("alpha", 10, None).unwrap();
        assert_eq!(both.len(), 2, "expected both sources unfiltered");
    }
}
