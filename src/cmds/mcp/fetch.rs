//! `ctx_fetch_and_index`: fetch one or more URLs with `curl`, convert HTML or
//! JSON bodies to plain text, and index the result for `ctx_search`.

use anyhow::Result;
// Through rmcp, so the derive and the server agree on one schemars version.
use rmcp::schemars::{self, JsonSchema};
use serde::Deserialize;
use url::Url;

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
/// URLs fetched per call; the rest are reported and dropped.
const MAX_REQUESTS: usize = 8;
/// Longest label derived from a URL, in characters.
const MAX_LABEL_CHARS: usize = 80;
/// Longest entity `decode_entities` will look ahead for, in characters.
const MAX_ENTITY_CHARS: usize = 12;
/// Prefix every fetched source label carries.
const SOURCE_PREFIX: &str = "fetch:";

/// One URL to fetch, with its own optional index label.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FetchRequest {
    /// URL to fetch. `http`, `https` and `file` only; `file://` works for tests.
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
    let mut requests = match input.requests {
        Some(requests) if !requests.is_empty() => requests,
        _ => match input.url {
            Some(url) => vec![FetchRequest { url, source: None }],
            None => return Ok("nothing to fetch: pass url or requests\n".to_string()),
        },
    };
    let dropped = requests.len().saturating_sub(MAX_REQUESTS);
    requests.truncate(MAX_REQUESTS);
    let preview_chars = if requests.len() > 1 {
        MULTI_PREVIEW_CHARS
    } else {
        SINGLE_PREVIEW_CHARS
    };

    let mut out = String::new();
    for request in &requests {
        let source = source_for(request, &global_source);
        match fetch_one(&request.url) {
            Ok((bytes, text)) => {
                let indexed = store.index(&source, &text)?;
                out.push_str(&format!("### {source}\n"));
                out.push_str(&format!("{bytes} bytes, {} chunks", indexed.chunks));
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
            Err(reason) => {
                out.push_str(&format!(
                    "fetch failed ({reason}): {source} {}\n",
                    request.url
                ));
            }
        }
        out.push('\n');
    }
    if dropped > 0 {
        out.push_str(&format!(
            "{dropped} further request(s) not fetched: at most {MAX_REQUESTS} per call\n"
        ));
    }
    Ok(out)
}

/// Fetch `url` and convert it to indexable text, or the reason it failed.
fn fetch_one(url: &str) -> std::result::Result<(usize, String), String> {
    let checked = checked_url(url)?;
    let (content_type, body) = curl_fetch(&checked)?;
    if let Some(reason) = binary_reason(&content_type, &body) {
        return Err(reason);
    }
    Ok((body.len(), to_text(&content_type, &body)))
}

/// The URL `curl` may be handed: parsed, so it can never be read as an option,
/// and limited to the schemes this tool supports. `file` stays allowed on
/// purpose — indexing a local file is part of the context-mode contract.
fn checked_url(raw: &str) -> std::result::Result<String, String> {
    let parsed = Url::parse(raw).map_err(|_| "invalid url".to_string())?;
    match parsed.scheme() {
        "http" | "https" | "file" => Ok(parsed.to_string()),
        scheme => Err(format!("unsupported scheme {scheme}")),
    }
}

/// The index label for a request: its own `source`, else the tool-wide
/// `source`, else the URL's host and path — prefixed with `fetch:` unless the
/// caller already spelled the prefix out.
fn source_for(request: &FetchRequest, global_source: &Option<String>) -> String {
    let label = request
        .source
        .clone()
        .or_else(|| global_source.clone())
        .unwrap_or_else(|| host_and_path(&request.url));
    if label.starts_with(SOURCE_PREFIX) {
        label
    } else {
        format!("{SOURCE_PREFIX}{label}")
    }
}

/// `<host><path>` with the scheme, any query or fragment and a trailing slash
/// stripped, capped at [`MAX_LABEL_CHARS`] characters.
fn host_and_path(url: &str) -> String {
    let without_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let path = without_scheme
        .split(['?', '#'])
        .next()
        .unwrap_or(without_scheme);
    let trimmed = path.trim_end_matches('/');
    // A URL that is nothing but a host keeps its slash rather than no label.
    let kept = if trimmed.is_empty() { path } else { trimmed };
    kept.chars().take(MAX_LABEL_CHARS).collect()
}

/// The first `chars` characters of `text`.
fn preview(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
}

/// Fetch `url` with `curl`, returning its content type and raw body, or the
/// reason it failed. `--proto` and `--proto-redir` keep a redirect from
/// switching to a scheme the caller could not have asked for.
fn curl_fetch(url: &str) -> std::result::Result<(String, Vec<u8>), String> {
    let body_file = tempfile::NamedTempFile::new().map_err(|err| format!("temp file: {err}"))?;
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
            "--proto",
            "=http,https,file",
            "--proto-redir",
            "=http,https",
            "-w",
            "%{content_type}\n%{http_code}",
            "-o",
        ])
        .arg(body_path)
        .args(["--url", url])
        .output()
        .map_err(|err| format!("curl: {err}"))?;
    let body = std::fs::read(body_path).unwrap_or_default();
    classify(
        output.status.code().unwrap_or(-1),
        &String::from_utf8_lossy(&output.stdout),
        body,
    )
}

/// Turn one `curl` run into its content type and body, or the reason it
/// failed: a non-zero exit, an HTTP error status, or an empty body.
fn classify(
    exit: i32,
    write_out: &str,
    body: Vec<u8>,
) -> std::result::Result<(String, Vec<u8>), String> {
    if exit != 0 {
        return Err(format!("curl exit {exit}"));
    }
    let (content_type, status) = split_write_out(write_out);
    if status >= 400 {
        return Err(format!("HTTP {status}"));
    }
    if body.is_empty() {
        return Err("empty body".to_string());
    }
    Ok((content_type, body))
}

/// Split `--write-out '%{content_type}\n%{http_code}'`. A protocol with no
/// status, `file://` above all, reports 000.
fn split_write_out(write_out: &str) -> (String, u32) {
    let mut lines = write_out.lines();
    let content_type = lines.next().unwrap_or_default().trim().to_string();
    let status = lines
        .next()
        .unwrap_or_default()
        .trim()
        .parse::<u32>()
        .unwrap_or(0);
    (content_type, status)
}

/// The reason a body must not be indexed, when it is not text: a binary
/// content type, or a NUL byte anywhere in it.
fn binary_reason(content_type: &str, body: &[u8]) -> Option<String> {
    let mime = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let binary_type = mime.starts_with("image/")
        || mime.starts_with("audio/")
        || mime.starts_with("video/")
        || matches!(
            mime.as_str(),
            "application/octet-stream" | "application/pdf" | "application/zip"
        );
    if !binary_type && !body.contains(&0) {
        return None;
    }
    let shown = if mime.is_empty() { "unknown" } else { &mime };
    Some(format!("binary body, content-type {shown}"))
}

/// Convert a fetched body to plain text: HTML is stripped to text, JSON is
/// pretty-printed, anything else passes through unchanged.
fn to_text(content_type: &str, body: &[u8]) -> String {
    let raw = String::from_utf8_lossy(body).into_owned();
    let mime = content_type.to_ascii_lowercase();
    let sniff = mime.is_empty();
    if mime.contains("html") || (sniff && raw.trim_start().starts_with('<')) {
        return html_to_text(&raw);
    }
    if mime.contains("json") || (sniff && raw.trim_start().starts_with(['{', '['])) {
        // Parsed once: the shape check and the pretty-print are the same pass.
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Ok(pretty) = serde_json::to_string_pretty(&value) {
                return pretty;
            }
        }
    }
    raw
}

/// Strip `html` down to its text in one linear pass: comments, `<script>` and
/// `<style>` bodies are dropped, block-level tags become newlines, every other
/// tag vanishes, and a `<` that opens nothing stays text. Entities are decoded
/// and runs of spaces and of blank lines collapse afterwards.
fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut i = 0;
    while i < html.len() {
        let rest = &html[i..];
        if let Some(after) = rest.strip_prefix("<!--") {
            i += "<!--".len() + skip_past(after, "-->");
            continue;
        }
        if let Some(after) = rest.strip_prefix("<![CDATA[") {
            let end = after.find("]]>").unwrap_or(after.len());
            out.push_str(&after[..end]);
            i += "<![CDATA[".len() + after.len().min(end + "]]>".len());
            continue;
        }
        if opens_tag(rest) {
            let tag = read_tag(rest);
            i += tag.len;
            if is_block_break(&tag.name) {
                out.push('\n');
            }
            if tag.name == "script" || tag.name == "style" {
                i += skip_raw(&html[i..], &tag.name);
            }
            continue;
        }
        let ch = rest.chars().next().unwrap_or('<');
        out.push(ch);
        i += ch.len_utf8();
    }
    collapse_whitespace(&decode_entities(&out))
}

/// Bytes of `rest` up to and including the next `end` marker, or all of it.
fn skip_past(rest: &str, end: &str) -> usize {
    match rest.find(end) {
        Some(at) => at + end.len(),
        None => rest.len(),
    }
}

/// Whether `rest` starts a tag: a `<` followed by a name, `/`, `!` or `?`.
/// Anything else — `5 < 10` — is text.
fn opens_tag(rest: &str) -> bool {
    let mut chars = rest.chars();
    chars.next() == Some('<')
        && matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '/' || ch == '!' || ch == '?')
}

/// One tag: how many bytes it spans, and its lowercased name with the closing
/// slash of `</p>` kept as part of it.
struct Tag {
    len: usize,
    name: String,
}

/// Scan the tag at the start of `rest`, quote-aware so a `>` inside an
/// attribute value does not end it early. An unterminated tag spans the rest
/// of the input.
fn read_tag(rest: &str) -> Tag {
    let mut name = String::new();
    let mut in_name = true;
    let mut quote: Option<char> = None;
    let mut len = rest.len();
    for (idx, ch) in rest.char_indices().skip(1) {
        if let Some(open) = quote {
            if ch == open {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => {
                quote = Some(ch);
                in_name = false;
            }
            '>' => {
                len = idx + 1;
                break;
            }
            _ if in_name => {
                if ch.is_whitespace() || (ch == '/' && !name.is_empty()) {
                    in_name = false;
                } else {
                    name.push(ch.to_ascii_lowercase());
                }
            }
            _ => {}
        }
    }
    Tag { len, name }
}

/// Whether a tag ends a block, so its place in the text is a newline.
fn is_block_break(name: &str) -> bool {
    matches!(
        name,
        "/p" | "/div"
            | "/li"
            | "/tr"
            | "/h1"
            | "/h2"
            | "/h3"
            | "/h4"
            | "/h5"
            | "/h6"
            | "/section"
            | "/article"
            | "/table"
            | "/pre"
            | "/blockquote"
            | "br"
            | "hr"
            | "p"
            | "li"
            | "tr"
    )
}

/// Bytes of `rest` holding the body of a raw element (`script`, `style`), up
/// to and including its closing tag.
fn skip_raw(rest: &str, name: &str) -> usize {
    let needle = format!("</{name}");
    match find_ignore_case(rest, &needle) {
        Some(at) => at + read_tag(&rest[at..]).len,
        None => rest.len(),
    }
}

/// The first case-insensitive match of an ASCII `needle` in `haystack`, as a
/// byte offset. `needle` starts with `<`, so the offset is a char boundary.
fn find_ignore_case(haystack: &str, needle: &str) -> Option<usize> {
    let (hay, needle) = (haystack.as_bytes(), needle.as_bytes());
    (0..=hay.len().checked_sub(needle.len())?)
        .find(|&at| hay[at..at + needle.len()].eq_ignore_ascii_case(needle))
}

/// Decode `&amp; &lt; &gt; &quot; &apos; &nbsp;` and numeric character
/// references (`&#39;`, `&#x27;`).
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        if rest.starts_with('&') {
            let end = rest
                .char_indices()
                .take(MAX_ENTITY_CHARS)
                .find(|(_, ch)| *ch == ';')
                .map(|(at, _)| at);
            if let Some(end) = end {
                if let Some(decoded) = decode_one_entity(&rest[1..end]) {
                    out.push(decoded);
                    i += end + 1;
                    continue;
                }
            }
        }
        let ch = rest.chars().next().unwrap_or('&');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn decode_one_entity(entity: &str) -> Option<char> {
    let decoded = match entity {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        _ => {
            let digits = entity.strip_prefix('#')?;
            let code = match digits.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => digits.parse::<u32>().ok()?,
            };
            char::from_u32(code)?
        }
    };
    // A non-breaking space is a plain space to a search index.
    Some(if decoded == '\u{a0}' { ' ' } else { decoded })
}

/// Collapse runs of spaces and tabs within a line to one space, drop leading
/// and trailing spaces, and collapse runs of blank lines to one.
fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_blank = false;
    for line in text.lines() {
        let mut squeezed = String::with_capacity(line.len());
        let mut prev_space = true;
        for ch in line.chars() {
            let is_space = ch == ' ' || ch == '\t';
            if !is_space {
                squeezed.push(ch);
            } else if !prev_space {
                squeezed.push(' ');
            }
            prev_space = is_space;
        }
        let trimmed = squeezed.trim_end();
        let is_blank = trimmed.is_empty();
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

    fn one(url: String) -> FetchInput {
        FetchInput {
            url: Some(url),
            requests: None,
            source: None,
        }
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
    fn html_conversion_keeps_text_around_attributes_bare_angles_and_comments() {
        // A multi-byte attribute value used to eat the first character.
        assert_eq!(html_to_text(r#"<p title="é">HELLO</p>"#), "HELLO");
        // A `<` that opens no tag is text, and used to swallow the rest.
        assert_eq!(html_to_text("<p>5 < 10 and 3 > 1</p>"), "5 < 10 and 3 > 1");
        // A `>` inside a quoted attribute used to end the tag early.
        assert_eq!(html_to_text(r#"<p title="a>b">kept</p>"#), "kept");
        // A commented-out `<script>` used to drop the rest of the document.
        assert_eq!(
            html_to_text("<!-- <script> --><p>after the comment</p>"),
            "after the comment"
        );
        // Raw elements end at their own closing tag, whatever its case, and
        // the `<` inside one does not leak into the text.
        assert_eq!(
            html_to_text("<p>a</p><SCRIPT>var x = 1 < 2;</SCRIPT><p>b</p>"),
            "a\n\nb"
        );
    }

    #[test]
    fn nothing_to_fetch_without_a_url_or_requests() {
        let (_dir, store) = temp_store();
        let input = FetchInput {
            url: None,
            requests: None,
            source: None,
        };
        let out = fetch_and_index_with(&store, input).unwrap();
        assert_eq!(out, "nothing to fetch: pass url or requests\n");
    }

    #[test]
    fn a_curl_option_is_not_fetched_as_a_url() {
        // A positional URL used to be read as an option, so `-K<file>` loaded
        // an attacker's curl config: any header, any proxy, any file written.
        let source_dir = tempfile::tempdir().unwrap();
        let page = source_dir.path().join("page.html");
        std::fs::write(&page, "<p>bait</p>").unwrap();
        let written = source_dir.path().join("written-by-curl.txt");
        let conf = source_dir.path().join("evil.conf");
        std::fs::write(
            &conf,
            format!(
                "url = \"file://{}\"\ntrace = \"{}\"\n",
                page.display(),
                written.display()
            ),
        )
        .unwrap();

        let (_store_dir, store) = temp_store();
        let out = fetch_and_index_with(&store, one(format!("-K{}", conf.display()))).unwrap();
        assert!(out.contains("fetch failed (invalid url)"), "{out}");
        assert!(!out.contains("bait"), "{out}");
        assert!(!written.exists(), "curl ran the injected config: {out}");
    }

    #[test]
    fn only_http_https_and_file_are_fetched() {
        assert_eq!(
            checked_url("ftp://example.com/x").unwrap_err(),
            "unsupported scheme ftp"
        );
        assert_eq!(
            checked_url("javascript:alert(1)").unwrap_err(),
            "unsupported scheme javascript"
        );
        assert!(checked_url("https://example.com/a").is_ok());
        assert!(checked_url("file:///tmp/a.html").is_ok());
    }

    #[test]
    fn http_errors_and_curl_exits_are_reported_instead_of_indexed() {
        assert_eq!(
            classify(0, "text/html\n404", b"<h1>Not Found</h1>".to_vec()).unwrap_err(),
            "HTTP 404"
        );
        assert_eq!(
            classify(37, "\n000", Vec::new()).unwrap_err(),
            "curl exit 37"
        );
        assert_eq!(classify(0, "\n000", Vec::new()).unwrap_err(), "empty body");
        // file:// reports no status, and is still fetched.
        assert_eq!(classify(0, "\n000", b"body".to_vec()).unwrap().1, b"body");
    }

    #[test]
    fn binary_bodies_are_refused() {
        assert!(binary_reason("image/png", b"text").is_some());
        assert!(binary_reason("application/pdf; charset=binary", b"text").is_some());
        assert!(binary_reason("text/plain", b"plain text").is_none());
        assert!(binary_reason("text/plain", b"has a \0 byte").is_some());
    }

    #[test]
    fn a_binary_file_is_not_indexed() {
        let source_dir = tempfile::tempdir().unwrap();
        let path = source_dir.path().join("blob.bin");
        std::fs::write(&path, b"needle\0\x01\x02binary".as_slice()).unwrap();

        let (_store_dir, store) = temp_store();
        let out = fetch_and_index_with(&store, one(format!("file://{}", path.display()))).unwrap();
        assert!(out.contains("fetch failed (binary body"), "{out}");
        assert!(
            store.search("needle", 10, None).unwrap().is_empty(),
            "{out}"
        );
    }

    #[test]
    fn labels_drop_the_query_and_trailing_slash_and_never_double_the_prefix() {
        let derived = FetchRequest {
            url: "https://example.com/docs/guide/?q=1#top".to_string(),
            source: None,
        };
        assert_eq!(source_for(&derived, &None), "fetch:example.com/docs/guide");

        let prefixed = FetchRequest {
            url: "https://example.com".to_string(),
            source: Some("fetch:mine".to_string()),
        };
        assert_eq!(source_for(&prefixed, &None), "fetch:mine");

        let long = FetchRequest {
            url: format!("https://example.com/{}", "a".repeat(200)),
            source: None,
        };
        assert_eq!(
            source_for(&long, &None).chars().count(),
            SOURCE_PREFIX.len() + MAX_LABEL_CHARS
        );
    }

    #[test]
    fn at_most_eight_requests_are_fetched_per_call() {
        let (_dir, store) = temp_store();
        let input = FetchInput {
            url: None,
            requests: Some(
                (0..10)
                    .map(|i| FetchRequest {
                        url: format!("ftp://example.com/{i}"),
                        source: None,
                    })
                    .collect(),
            ),
            source: None,
        };
        let out = fetch_and_index_with(&store, input).unwrap();
        assert_eq!(out.matches("fetch failed").count(), MAX_REQUESTS, "{out}");
        assert!(out.contains("2 further request(s) not fetched"), "{out}");
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
