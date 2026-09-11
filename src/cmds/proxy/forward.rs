//! Forward one client request to the upstream Anthropic/OpenAI API and
//! stream the response back. Everything is byte-exact passthrough except
//! POST `/v1/messages`, `/v1/messages/count_tokens`, `/v1/responses` and
//! `/v1/responses/compact`, whose bodies go through [`pipeline`] first.

use std::io::{self, Read, Write};

use anyhow::{Context, Result};
use tiny_http::{Header, Request, Response, StatusCode};

use super::pipeline;
use super::ProxyConfig;

/// Headers never forwarded in either direction: RFC 7230 hop-by-hop fields,
/// plus `host`/`content-length` (recomputed per hop), anything `proxy-*`,
/// and `expect` — tiny_http owns the client-side `100-continue` dance.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "expect",
];

/// Whether `name` survives the hop. `proxy-*` fields are per-hop by
/// convention; everything else — `x-api-key`, `authorization`,
/// `anthropic-version`, `anthropic-beta`, `content-type`, `accept` —
/// forwards untouched.
fn forwardable(name: &str) -> bool {
    let name = name.trim().to_ascii_lowercase();
    !name.starts_with("proxy-") && !HOP_BY_HOP.contains(&name.as_str())
}

/// The body-bearing request paths eligible for compression: Anthropic
/// Messages plus the OpenAI surfaces (Codex CLI routes `{base_url}/responses`;
/// chat-completions clients get the same `messages[]` walker, which skips
/// `system`/`developer` roles inside the array).
fn compressible_path(path: &str) -> bool {
    matches!(
        path,
        "/v1/messages"
            | "/v1/messages/count_tokens"
            | "/v1/responses"
            | "/v1/responses/compact"
            | "/v1/chat/completions"
    )
}

/// Path plus query of a request target: origin-form is already that; an
/// absolute-form URI (`GET http://host/path?q`) keeps only its path+query so
/// the request still lands on `cfg.upstream`.
fn path_and_query(target: &str) -> String {
    if target.starts_with("http://") || target.starts_with("https://") {
        if let Ok(url) = url::Url::parse(target) {
            return match url.query() {
                Some(q) => format!("{}?{q}", url.path()),
                None => url.path().to_string(),
            };
        }
    }
    target.to_string()
}

/// The path component alone, for the compression decision.
fn path_of(target: &str) -> &str {
    target.split(['?', '#']).next().unwrap_or(target)
}

/// Read, maybe compress, forward, stream back, log one stats line.
pub fn handle(
    mut request: Request,
    cfg: &ProxyConfig,
    agent: &ureq::Agent,
    crush: pipeline::Crusher,
) -> Result<()> {
    let method = request.method().as_str().to_string();
    let target = request.url().to_string();
    let headers: Vec<(String, String)> = request
        .headers()
        .iter()
        .map(|h| {
            (
                h.field.as_str().as_str().to_string(),
                h.value.as_str().to_string(),
            )
        })
        .collect();
    let mut body = Vec::new();
    request
        .as_reader()
        .read_to_end(&mut body)
        .context("read client request body")?;

    let in_bytes = body.len();
    let target = path_and_query(&target);
    let path = path_of(&target).to_string();
    let (out_body, parts) = if method == "POST" && compressible_path(&path) {
        pipeline::process_body(&body, cfg.min_bytes, crush)
    } else {
        (body, 0)
    };
    log_stats(&method, &path, in_bytes, out_body.len(), parts);

    let url = format!("{}{}", cfg.upstream.trim_end_matches('/'), target);
    let mut outgoing = agent.request(&method, &url);
    for (name, value) in &headers {
        if forwardable(name) {
            outgoing = outgoing.set(name, value);
        }
    }
    let sent = if out_body.is_empty() {
        outgoing.call()
    } else {
        outgoing.send_bytes(&out_body)
    };
    let upstream = match sent {
        Ok(response) => response,
        // 4xx/5xx still carry a real response: forward it verbatim.
        Err(ureq::Error::Status(_, response)) => response,
        Err(err) => {
            let _ = request.respond(
                Response::from_string("tokenaut proxy: upstream unreachable\n")
                    .with_status_code(StatusCode(502)),
            );
            return Err(err).context("upstream request failed");
        }
    };

    let status = upstream.status();
    let status_text = upstream.status_text().to_string();
    let mut out_headers = Vec::new();
    for name in upstream.headers_names() {
        if forwardable(&name) {
            if let Some(value) = upstream.header(&name) {
                if let Ok(header) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
                    out_headers.push(header);
                }
            }
        }
    }
    // Known length → Content-Length, none → chunked (SSE streams). Responses
    // that can never carry a body get an empty one.
    let data_length = match upstream
        .header("content-length")
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(n) => Some(n),
        None if status == 204 || status == 304 || method == "HEAD" => Some(0),
        None => None,
    };
    respond_streaming(
        request,
        status,
        &status_text,
        &out_headers,
        upstream,
        data_length,
    )
}

/// Write the response head and stream the body onto the connection.
///
/// The body is framed by hand rather than through `tiny_http::Response`:
/// tiny_http's chunked encoder batches 8 KiB before emitting, which would
/// hold Anthropic's SSE events hostage until the buffer fills — a poison
/// chokepoint for a streaming API. Chunked mode here writes and flushes one
/// frame per upstream read, so events reach the client as they arrive.
/// `connection: close` is sent because the socket is surrendered to us via
/// `Request::into_writer`; a fresh connection per request is free on
/// localhost.
fn respond_streaming(
    request: Request,
    status: u16,
    status_text: &str,
    headers: &[Header],
    upstream: ureq::Response,
    data_length: Option<usize>,
) -> Result<()> {
    let mut writer = request.into_writer();
    write!(writer, "HTTP/1.1 {status} {status_text}\r\n").context("write status line")?;
    for h in headers {
        write!(writer, "{h}\r\n").context("write response header")?;
    }
    match data_length {
        Some(n) => write!(writer, "content-length: {n}\r\n"),
        None => write!(writer, "transfer-encoding: chunked\r\n"),
    }
    .and_then(|_| write!(writer, "connection: close\r\n\r\n"))
    .and_then(|_| writer.flush())
    .context("write response head")?;
    stream_body(&mut writer, upstream.into_reader(), data_length).context("stream response body")
}

/// Copy `reader` onto `writer`, chunked-framed when `data_length` is unknown.
fn stream_body(
    writer: &mut impl Write,
    reader: impl Read,
    data_length: Option<usize>,
) -> io::Result<()> {
    match data_length {
        Some(n) => {
            io::copy(&mut reader.take(n as u64), writer)?;
            writer.flush()
        }
        None => {
            let mut reader = reader;
            let mut buf = [0u8; 16 * 1024];
            loop {
                let n = reader.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                write!(writer, "{n:x}\r\n")?;
                writer.write_all(&buf[..n])?;
                writer.write_all(b"\r\n")?;
                // Each upstream read hits the wire now — SSE events must not
                // wait for a full buffer.
                writer.flush()?;
            }
            writer.write_all(b"0\r\n\r\n")?;
            writer.flush()
        }
    }
}

/// One stderr line per proxied call — `in`/`out` are request-body bytes, so
/// pass-throughs log `saved=0%`: the lever's coverage is visible either way.
fn log_stats(method: &str, path: &str, in_bytes: usize, out_bytes: usize, parts: usize) {
    let saved = in_bytes
        .saturating_sub(out_bytes)
        .checked_mul(100)
        .and_then(|n| n.checked_div(in_bytes))
        .unwrap_or(0);
    eprintln!("proxy {method} {path} in={in_bytes} out={out_bytes} saved={saved}% parts={parts}");
}
