//! `tokenaut api-proxy`: a transparent HTTP/1.1 forward proxy for the
//! Anthropic and OpenAI Responses APIs. Claude Code pointed at
//! `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` — or Codex pointed at the
//! same URL through a `[model_providers]` `base_url` — gets identical API
//! behavior while old, large `messages[]`/`input[]` content is compressed
//! through [`crate::cmds::compress`] before it goes upstream.
//!
//! Cache alignment invariant: `compress::crush` is deterministic, so a part
//! compressed in turn N compresses to the same bytes in turn N+1 — the
//! request prefix reaching upstream stays byte-stable and prompt caching
//! survives. To preserve it the pipeline never reorders `messages`/`input`,
//! never touches `system`, `instructions`, `tools` or `cache_control`, and
//! only rewrites eligible text fields in place.
//!
//! Everything besides POST `/v1/messages`, `/v1/messages/count_tokens`,
//! `/v1/responses` and `/v1/responses/compact` is byte-exact passthrough:
//! headers minus hop-by-hop, status and body streamed back incrementally —
//! `text/event-stream` responses are chunk-copied upstream→client, never
//! buffered whole.
mod forward;
pub(crate) mod pipeline;

use std::net::SocketAddr;
use std::thread;
use std::time::Duration;

use anyhow::Result;

/// What the CLI passes in.
pub struct ProxyConfig {
    /// Address the proxy listens on, e.g. `127.0.0.1:8787`.
    pub listen: String,
    /// Upstream base URL requests forward to, e.g. `https://api.anthropic.com`.
    pub upstream: String,
    /// Smallest part length in bytes eligible for compression.
    pub min_bytes: usize,
}

/// `tokenaut api-proxy` entry point — blocking.
pub fn run(cfg: ProxyConfig) -> Result<()> {
    let (server, addr) = listen(&cfg.listen)?;
    eprintln!(
        "tokenaut proxy listening on http://{addr}, upstream {}",
        cfg.upstream
    );
    serve(server, cfg)
}

/// Bind on `listen` and report the address actually taken — `127.0.0.1:0`
/// picks a free port, which is how tests get a stable endpoint.
pub fn listen(listen: &str) -> Result<(tiny_http::Server, SocketAddr)> {
    let server = tiny_http::Server::http(listen)
        .map_err(|err| anyhow::anyhow!("proxy cannot listen on {listen}: {err}"))?;
    match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => Ok((server, addr)),
        other => anyhow::bail!("unexpected listen address {other:?}"),
    }
}

/// Blocking accept loop. One thread per connection: a localhost agent proxy
/// sees a handful of concurrent connections, and sync I/O keeps the no-async
/// rule. Upstream timeouts: 10s connect, 300s per read — LLM SSE streams are
/// long but should never leave a request hanging forever.
pub fn serve(server: tiny_http::Server, cfg: ProxyConfig) -> Result<()> {
    serve_with(server, cfg, crate::cmds::compress::crush)
}

/// `serve` with the crusher injectable — tests substitute a fake to prove
/// the walk end-to-end; production always gets the shared contract's.
fn serve_with(server: tiny_http::Server, cfg: ProxyConfig, crush: pipeline::Crusher) -> Result<()> {
    let cfg = std::sync::Arc::new(cfg);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(300))
        .build();
    for request in server.incoming_requests() {
        let cfg = cfg.clone();
        let agent = agent.clone();
        thread::spawn(move || {
            if let Err(err) = forward::handle(request, &cfg, &agent, crush) {
                eprintln!("proxy: request failed: {err:#}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::sync::mpsc::{channel, Receiver, Sender};
    use tiny_http::{Response, StatusCode};

    use crate::cmds::compress::Crushed;

    /// What the fake upstream observed; `stream` feeds `/stream` bodies.
    struct Recorded {
        method: String,
        target: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        /// Pushes chunks into the upstream's response on `/stream`; dropped
        /// means EOF. Unused for other paths.
        stream: Sender<Vec<u8>>,
    }

    /// Spawn a `tiny_http` fake upstream: records every request on the
    /// channel, then responds — echo of the request body (empty → `ok`),
    /// hand-framed `text/event-stream` fed by `Recorded.stream` on `/stream`,
    /// 418 on `/teapot`.
    fn spawn_upstream() -> (SocketAddr, Receiver<Recorded>) {
        let (server, addr) = listen("127.0.0.1:0").expect("bind upstream");
        let (tx, rx) = channel::<Recorded>();
        thread::spawn(move || {
            for mut request in server.incoming_requests() {
                let method = request.method().as_str().to_string();
                let target = request.url().to_string();
                let headers = request
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
                let _ = request.as_reader().read_to_end(&mut body);
                let (stx, srx) = channel::<Vec<u8>>();
                tx.send(Recorded {
                    method,
                    target: target.clone(),
                    headers,
                    body: body.clone(),
                    stream: stx,
                })
                .expect("record request");
                if target.starts_with("/stream") {
                    // Hand-framed chunked response: tiny_http's own chunked
                    // encoder batches 8 KiB, which would defeat the
                    // incremental-streaming assertion under test. Every
                    // write needs an explicit flush — `into_writer` sits on
                    // a 1 KiB BufWriter, and dropping it only releases a
                    // shared Arc (no flush-on-drop), so an unflushed
                    // terminator never reaches the client.
                    let mut w = request.into_writer();
                    w.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n")
                        .expect("stream head");
                    w.flush().expect("head flush");
                    while let Ok(chunk) = srx.recv() {
                        write!(w, "{:x}\r\n", chunk.len()).expect("chunk size");
                        w.write_all(&chunk).expect("chunk body");
                        w.write_all(b"\r\n").expect("chunk tail");
                        w.flush().expect("chunk flush");
                    }
                    w.write_all(b"0\r\n\r\n").expect("stream terminator");
                    w.flush().expect("terminator flush");
                    continue;
                }
                let response = if target.starts_with("/teapot") {
                    Response::from_string("teapot\n").with_status_code(StatusCode(418))
                } else {
                    Response::from_data(if body.is_empty() {
                        b"ok".to_vec()
                    } else {
                        body
                    })
                };
                let _ = request.respond(response);
            }
        });
        (addr, rx)
    }

    /// Spawn the proxy under test against `upstream`; returns its address.
    fn spawn_proxy(upstream: SocketAddr) -> SocketAddr {
        spawn_proxy_with(upstream, crate::cmds::compress::crush)
    }

    fn spawn_proxy_with(upstream: SocketAddr, crush: pipeline::Crusher) -> SocketAddr {
        let (server, addr) = listen("127.0.0.1:0").expect("bind proxy");
        let cfg = ProxyConfig {
            listen: addr.to_string(),
            upstream: format!("http://{upstream}"),
            min_bytes: 2048,
        };
        thread::spawn(move || {
            let _ = serve_with(server, cfg, crush);
        });
        addr
    }

    fn client() -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout_read(Duration::from_secs(15))
            .build()
    }

    fn recorded_header<'a>(rec: &'a Recorded, name: &str) -> Option<&'a str> {
        rec.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Test crusher: quarter the text and tag it, so the walk's effect is
    /// visible in the bytes the upstream receives.
    fn quarter(text: &str) -> Result<Crushed> {
        Ok(Crushed {
            text: format!("ccr:deadbeef {}", &text[..text.len() / 4]),
            in_bytes: text.len(),
            ccr: Some("ccr:deadbeef".to_string()),
        })
    }

    #[test]
    fn test_get_passthrough_is_byte_exact() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy(up_addr);
        let resp = client()
            .get(&format!("http://{proxy}/v1/models?beta=x"))
            .set("x-api-key", "sk-test")
            .set("anthropic-version", "2023-06-01")
            .set("x-custom-thing", "kept")
            .call()
            .expect("GET through proxy");
        let rec = rx.recv().expect("upstream saw the request");
        assert_eq!(rec.method, "GET");
        assert_eq!(rec.target, "/v1/models?beta=x");
        assert_eq!(recorded_header(&rec, "x-api-key"), Some("sk-test"));
        assert_eq!(
            recorded_header(&rec, "anthropic-version"),
            Some("2023-06-01")
        );
        assert_eq!(recorded_header(&rec, "x-custom-thing"), Some("kept"));
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.into_string().unwrap(), "ok");
    }

    #[test]
    fn test_post_messages_stub_crusher_passes_bytes_through() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy(up_addr);
        let body = serde_json::json!({
            "model": "claude-x",
            "messages": [
                // 900B stays under CCR_MIN_BYTES so the real crusher is a
                // no-op here and the body round-trips byte-exact.
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x".repeat(900)}]},
                {"role": "user", "content": "fresh turn"}
            ]
        })
        .to_string();
        let resp = client()
            .post(&format!("http://{proxy}/v1/messages"))
            .set("content-type", "application/json")
            .set("anthropic-version", "2023-06-01")
            .send_bytes(body.as_bytes())
            .expect("POST through proxy");
        assert_eq!(resp.status(), 200);
        // Part under the CCR floor: nothing smaller to write, verbatim out.
        let rec = rx.recv().expect("upstream saw the request");
        assert_eq!(rec.method, "POST");
        assert_eq!(rec.target, "/v1/messages");
        assert_eq!(rec.body, body.as_bytes(), "request body must round-trip");
        assert_eq!(resp.into_string().unwrap(), body);
    }

    #[test]
    fn test_post_messages_crushes_eligible_parts_only() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy_with(up_addr, quarter);
        let fresh = format!("fresh {}", "u".repeat(3000));
        let body = serde_json::json!({
            "system": "sys",
            "messages": [
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x".repeat(3000)}]},
                {"role": "user", "content": fresh}
            ]
        })
        .to_string();
        client()
            .post(&format!("http://{proxy}/v1/messages"))
            .set("content-type", "application/json")
            .send_bytes(body.as_bytes())
            .expect("POST through proxy");
        let rec = rx.recv().expect("upstream saw the request");
        let doc: serde_json::Value =
            serde_json::from_slice(&rec.body).expect("upstream body is valid JSON");
        let tool_text = doc["messages"][0]["content"][0]["content"]
            .as_str()
            .expect("tool_result content");
        assert!(
            tool_text.starts_with("ccr:deadbeef xxxx"),
            "old tool_result crushed: {tool_text}"
        );
        assert_eq!(
            doc["messages"][1]["content"], fresh,
            "last user message stays verbatim"
        );
        assert_eq!(doc["system"], "sys");
    }

    #[test]
    fn test_count_tokens_compressible_other_paths_passthrough() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy_with(up_addr, quarter);
        let big = "y".repeat(3000);
        let body = serde_json::json!({"messages": [
            {"role": "assistant", "content": big},
            {"role": "user", "content": "hi"}
        ]})
        .to_string();
        client()
            .post(&format!("http://{proxy}/v1/messages/count_tokens"))
            .send_bytes(body.as_bytes())
            .expect("count_tokens through proxy");
        let rec = rx.recv().expect("upstream saw count_tokens");
        let doc: serde_json::Value = serde_json::from_slice(&rec.body).unwrap();
        assert!(
            doc["messages"][0]["content"]
                .as_str()
                .unwrap()
                .starts_with("ccr:deadbeef"),
            "count_tokens body went through the pipeline"
        );
        client()
            .post(&format!("http://{proxy}/v1/other"))
            .send_bytes(body.as_bytes())
            .expect("other path through proxy");
        let rec = rx.recv().expect("upstream saw other");
        assert_eq!(
            rec.body,
            body.as_bytes(),
            "non-messages paths pass through byte-exact"
        );
    }

    #[test]
    fn test_post_responses_crushes_eligible_parts_only() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy_with(up_addr, quarter);
        let fresh = format!("fresh {}", "u".repeat(3000));
        let body = serde_json::json!({
            "instructions": "sys",
            "input": [
                {"type": "function_call_output", "call_id": "c1", "output": "x".repeat(3000)},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": fresh}]},
                {"type": "function_call_output", "call_id": "c2", "output": "y".repeat(3000)}
            ]
        })
        .to_string();
        client()
            .post(&format!("http://{proxy}/v1/responses"))
            .set("content-type", "application/json")
            .send_bytes(body.as_bytes())
            .expect("POST through proxy");
        let rec = rx.recv().expect("upstream saw the request");
        let doc: serde_json::Value =
            serde_json::from_slice(&rec.body).expect("upstream body is valid JSON");
        assert!(
            doc["input"][0]["output"]
                .as_str()
                .unwrap()
                .starts_with("ccr:deadbeef xxxx"),
            "old function_call_output crushed: {}",
            doc["input"][0]["output"]
        );
        // The freshest turn — the last user message and the tool output
        // trailing it — stays verbatim; `instructions` untouched.
        assert_eq!(doc["input"][1]["content"][0]["text"], fresh);
        assert_eq!(doc["input"][2]["output"], "y".repeat(3000));
        assert_eq!(doc["instructions"], "sys");
    }

    #[test]
    fn test_responses_compact_compressible_chat_completions_passthrough() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy_with(up_addr, quarter);
        let body = serde_json::json!({"input": [
            {"type": "function_call_output", "call_id": "c", "output": "y".repeat(3000)},
            {"type": "message", "role": "user", "content": "hi"}
        ]})
        .to_string();
        client()
            .post(&format!("http://{proxy}/v1/responses/compact"))
            .send_bytes(body.as_bytes())
            .expect("compact through proxy");
        let rec = rx.recv().expect("upstream saw compact");
        let doc: serde_json::Value = serde_json::from_slice(&rec.body).unwrap();
        assert!(
            doc["input"][0]["output"]
                .as_str()
                .unwrap()
                .starts_with("ccr:deadbeef"),
            "compact body went through the pipeline"
        );
        // Chat Completions bodies carry `system`/`developer` inside
        // `messages[]`; the walker spares those roles and compresses the rest.
        let body = serde_json::json!({"messages": [
            {"role": "system", "content": "y".repeat(3000)},
            {"role": "assistant", "content": "z".repeat(3000)},
            {"role": "tool", "tool_call_id": "c1", "content": "w".repeat(3000)},
            {"role": "user", "content": "hi"}
        ]})
        .to_string();
        client()
            .post(&format!("http://{proxy}/v1/chat/completions"))
            .send_bytes(body.as_bytes())
            .expect("chat/completions through proxy");
        let rec = rx.recv().expect("upstream saw chat/completions");
        let doc: serde_json::Value = serde_json::from_slice(&rec.body).unwrap();
        assert_eq!(doc["messages"][0]["content"], "y".repeat(3000));
        assert!(doc["messages"][1]["content"]
            .as_str()
            .unwrap()
            .starts_with("ccr:deadbeef"));
        assert!(doc["messages"][2]["content"]
            .as_str()
            .unwrap()
            .starts_with("ccr:deadbeef"));
    }

    #[test]
    fn test_garbage_body_passthrough() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy(up_addr);
        let garbage = b"{\x00\xff not json, never parses";
        client()
            .post(&format!("http://{proxy}/v1/messages"))
            .set("content-type", "application/json")
            .send_bytes(garbage)
            .expect("garbage through proxy");
        let rec = rx.recv().expect("upstream saw the request");
        assert_eq!(rec.body, garbage, "unparseable bodies forward byte-exact");
    }

    #[test]
    fn test_sse_streams_incrementally() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy(up_addr);
        // The client reads its response on a thread so the test can feed the
        // upstream's stream in steps.
        let (done_tx, done_rx) = channel::<Vec<u8>>();
        let url = format!("http://{proxy}/stream");
        thread::spawn(move || {
            let resp = client().get(&url).call().expect("stream call");
            let mut reader = resp.into_reader();
            let mut got = Vec::new();
            let mut buf = [0u8; 64];
            // Read once and report: if the proxy buffered the whole body this
            // read stalls until the upstream closes.
            match reader.read(&mut buf) {
                Ok(n) => got.extend(&buf[..n]),
                Err(e) => panic!("stream read: {e}"),
            }
            done_tx.send(got.clone()).expect("first chunk");
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => got.extend(&buf[..n]),
                }
            }
            done_tx.send(got).expect("rest of stream");
        });
        let rec = rx.recv().expect("upstream saw the request");
        rec.stream
            .send(b"data: first\n\n".to_vec())
            .expect("push chunk 1");
        let first = done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("first chunk must arrive while the stream is still open");
        assert_eq!(first, b"data: first\n\n", "SSE must not be buffered whole");
        rec.stream
            .send(b"data: second\n\n".to_vec())
            .expect("push chunk 2");
        drop(rec.stream);
        let all = done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("stream must finish once the sender drops");
        assert_eq!(all, b"data: first\n\ndata: second\n\n");
    }

    #[test]
    fn test_upstream_error_status_forwards_verbatim() {
        let (up_addr, rx) = spawn_upstream();
        let proxy = spawn_proxy(up_addr);
        let err = client()
            .get(&format!("http://{proxy}/teapot"))
            .call()
            .expect_err("418 surfaces as Error::Status");
        let ureq::Error::Status(code, resp) = err else {
            panic!("expected status error, got {err:?}")
        };
        assert_eq!(code, 418);
        assert_eq!(resp.into_string().unwrap(), "teapot\n");
        let _ = rx.recv().expect("upstream saw the request");
    }

    #[test]
    fn test_unreachable_upstream_is_502_not_panic() {
        // Port 1 (discard) is never listening → connect refused.
        let (server, addr) = listen("127.0.0.1:0").expect("bind proxy");
        let cfg = ProxyConfig {
            listen: addr.to_string(),
            upstream: "http://127.0.0.1:1".to_string(),
            min_bytes: 2048,
        };
        thread::spawn(move || {
            let _ = serve(server, cfg);
        });
        let err = client()
            .get(&format!("http://{addr}/v1/models"))
            .call()
            .expect_err("unreachable upstream must fail");
        let ureq::Error::Status(code, resp) = err else {
            panic!("expected 502 status, got {err:?}")
        };
        assert_eq!(code, 502);
        assert!(resp.into_string().unwrap().contains("upstream unreachable"));
    }
}
