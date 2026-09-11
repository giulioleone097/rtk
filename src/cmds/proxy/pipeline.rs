//! Request-body compression stage: find compressible `messages[]` parts and
//! route each through [`crate::cmds::compress::crush`].
//!
//! Cache alignment: `crush` is deterministic, so a part compressed in turn N
//! produces identical bytes in turn N+1 — the request prefix that reaches
//! upstream stays byte-stable and prompt caching keeps working. To preserve
//! the invariant this module only rewrites eligible text fields in place: it
//! never reorders `messages` and never touches `system`, `tools` or
//! `cache_control`. A part already carrying a `ccr:` retrieval marker is
//! skipped, so a compressed part round-trips unchanged.

use anyhow::Result;
use serde_json::Value;

#[cfg(test)]
use crate::cmds::compress;
use crate::cmds::compress::Crushed;

/// Signature shared by the real crusher and test fakes.
pub type Crusher = fn(&str) -> Result<Crushed>;

/// One replaced part's sizes: `(original bytes, emitted bytes)`.
pub type CrushedPart = (usize, usize);

/// JSON-pointer paths of every compressible text field in the request body.
///
/// Eligible: `tool_result` content (string or `{type:"text"}` sub-parts) and
/// `text` parts whose text is at least `min_bytes` — anywhere inside
/// `messages[]` except the last `role=="user"` message, which is the freshest
/// human turn and stays verbatim.
pub fn eligible_parts(body: &Value, min_bytes: usize) -> Vec<String> {
    let mut paths = Vec::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return paths;
    };
    let last_user = messages
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"));
    for (i, message) in messages.iter().enumerate() {
        if last_user == Some(i) {
            continue;
        }
        match message.get("content") {
            Some(Value::String(text)) => {
                collect(
                    &mut paths,
                    format!("/messages/{i}/content"),
                    text,
                    min_bytes,
                );
            }
            Some(Value::Array(parts)) => {
                for (j, part) in parts.iter().enumerate() {
                    let base = format!("/messages/{i}/content/{j}");
                    collect_part(&mut paths, &base, part, min_bytes);
                }
            }
            _ => {}
        }
    }
    paths
}

/// Crush the text field at `path` (a JSON pointer minted by
/// [`eligible_parts`]), replacing it only when the crusher's output is
/// strictly shorter. `Err` from the crusher keeps the original — compressing
/// is best-effort and must never break the request. Returns the part's
/// `(in, out)` sizes when a replacement happened.
pub fn crush_at(body: &mut Value, path: &str, crush: Crusher) -> Option<CrushedPart> {
    let slot = match body.pointer_mut(path)? {
        Value::String(s) => s,
        _ => return None,
    };
    let in_bytes = slot.len();
    let out = crush(slot).ok()?;
    if out.text.len() >= in_bytes {
        return None;
    }
    *slot = out.text;
    Some((in_bytes, slot.len()))
}

/// Run the full pipeline over a request body: parse, walk, crush. The body
/// comes back byte-identical whenever parsing fails or no part got smaller —
/// upstream must never see a mangled request. Returns the body to forward and
/// the number of parts actually replaced.
pub fn process_body(body: &[u8], min_bytes: usize, crush: Crusher) -> (Vec<u8>, usize) {
    let Ok(mut doc) = serde_json::from_slice::<Value>(body) else {
        return (body.to_vec(), 0);
    };
    let mut replaced = 0;
    for path in eligible_parts(&doc, min_bytes) {
        if crush_at(&mut doc, &path, crush).is_some() {
            replaced += 1;
        }
    }
    if replaced == 0 {
        return (body.to_vec(), 0);
    }
    match serde_json::to_vec(&doc) {
        Ok(bytes) => (bytes, replaced),
        Err(_) => (body.to_vec(), 0),
    }
}

/// One message part: `text` parts and `tool_result` content.
fn collect_part(paths: &mut Vec<String>, base: &str, part: &Value, min_bytes: usize) {
    match part.get("type").and_then(Value::as_str) {
        Some("text") => {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                collect(paths, format!("{base}/text"), text, min_bytes);
            }
        }
        Some("tool_result") => match part.get("content") {
            Some(Value::String(text)) => {
                collect(paths, format!("{base}/content"), text, min_bytes);
            }
            Some(Value::Array(subs)) => {
                for (k, sub) in subs.iter().enumerate() {
                    if sub.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(text) = sub.get("text").and_then(Value::as_str) {
                            collect(paths, format!("{base}/content/{k}/text"), text, min_bytes);
                        }
                    }
                }
            }
            _ => {}
        },
        _ => {}
    }
}

/// Push `path` when `text` is large enough and not already compressed.
fn collect(paths: &mut Vec<String>, path: String, text: &str, min_bytes: usize) {
    if text.len() >= min_bytes && !has_ccr_marker(text) {
        paths.push(path);
    }
}

/// Whether the part already carries a `ccr:<id>` retrieval marker. Substring
/// match deliberately covers any placement the crushers pick; a false
/// positive merely leaves that part uncompressed, never corrupts it.
fn has_ccr_marker(text: &str) -> bool {
    text.contains("ccr:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn big(tag: &str) -> String {
        format!("{tag}:{}", "x".repeat(3000))
    }

    /// system + old tool_results (string and sub-part forms) + assistant text
    /// + a fresh user turn that must stay verbatim.
    fn fixture() -> Value {
        json!({
            "system": [{"type": "text", "text": big("system")}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": big("tr1")},
                    {"type": "tool_result", "tool_use_id": "t2", "content": [
                        {"type": "text", "text": big("tr2a")},
                        {"type": "image", "source": {"data": big("img")}},
                        {"type": "text", "text": "short"}
                    ]},
                    {"type": "text", "text": big("tr-text")}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": big("assist")},
                    {"type": "tool_use", "id": "t3", "input": {"cmd": big("tooluse")}}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": format!("done ccr:{}", "a".repeat(8))},
                    {"type": "text", "text": big("last-user")}
                ]}
            ],
            "tools": [{"name": "bash"}]
        })
    }

    #[test]
    fn test_eligible_parts_fixture() {
        let paths = eligible_parts(&fixture(), 2048);
        // The two old tool_result texts and the old assistant text. The image
        // data, the short text, the tool_use input, `system`, and the whole
        // last user message are untouched.
        assert_eq!(
            paths,
            vec![
                "/messages/0/content/0/content",
                "/messages/0/content/1/content/0/text",
                "/messages/0/content/2/text",
                "/messages/1/content/0/text",
            ]
        );
    }

    #[test]
    fn test_eligible_parts_string_content_and_min_bytes() {
        let body = json!({"messages": [
            {"role": "user", "content": "tiny"},
            {"role": "assistant", "content": big("old-string-content")},
            {"role": "user", "content": big("fresh")}
        ]});
        assert_eq!(eligible_parts(&body, 2048), vec!["/messages/1/content"]);
        // A lone earlier user message is fair game — only the LAST user
        // message is protected.
        let body = json!({"messages": [
            {"role": "user", "content": big("old-question")},
            {"role": "user", "content": big("fresh")}
        ]});
        assert_eq!(eligible_parts(&body, 2048), vec!["/messages/0/content"]);
    }

    #[test]
    fn test_eligible_parts_no_messages_or_marked() {
        assert!(eligible_parts(&json!({}), 1).is_empty());
        assert!(eligible_parts(&json!({"messages": "not-array"}), 1).is_empty());
        // A compressed part re-sent upstream must not be crushed again.
        let marked = json!({"messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "ccr:9f1a 2000 bytes indexed"}]},
            {"role": "user", "content": "hi"}
        ]});
        assert!(eligible_parts(&marked, 1).is_empty());
    }

    fn half_crusher(text: &str) -> Result<Crushed> {
        Ok(Crushed {
            text: format!("ccr:deadbeef {}", &text[..text.len() / 4]),
            in_bytes: text.len(),
            ccr: Some("ccr:deadbeef".to_string()),
        })
    }

    #[test]
    fn test_crush_at_replaces_and_reports() {
        let mut body = fixture();
        let replaced = crush_at(&mut body, "/messages/0/content/0/content", half_crusher);
        assert_eq!(replaced, Some((3004, 3004 / 4 + 13)));
        let text = body
            .pointer("/messages/0/content/0/content")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(text.starts_with("ccr:deadbeef tr1:"));
        // Missing path / non-string slot: nothing happens.
        assert_eq!(
            crush_at(&mut body, "/messages/9/content", half_crusher),
            None
        );
        assert_eq!(crush_at(&mut body, "/tools/0/name", half_crusher), None);
    }

    #[test]
    fn test_crush_at_keeps_original_when_not_smaller() {
        let mut body = fixture();
        let before = serde_json::to_vec(&body).unwrap();
        assert_eq!(
            crush_at(&mut body, "/messages/0/content/0/content", compress::crush),
            None
        );
        assert_eq!(serde_json::to_vec(&body).unwrap(), before);
    }

    #[test]
    fn test_process_body_end_to_end_with_fake_crusher() {
        let body = json!({"messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": big("tr"), "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": big("fresh")}
        ]});
        let (out, parts) = process_body(&serde_json::to_vec(&body).unwrap(), 2048, half_crusher);
        assert_eq!(parts, 1);
        let doc: Value = serde_json::from_slice(&out).unwrap();
        assert!(doc["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with("ccr:deadbeef"));
        // cache_control and the fresh user turn untouched.
        assert_eq!(
            doc["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(doc["messages"][1]["content"], big("fresh"));
    }

    #[test]
    fn test_process_body_garbage_passthrough() {
        let garbage = b"this is not json at all \x00\xff";
        let (out, parts) = process_body(garbage, 2048, half_crusher);
        assert_eq!(parts, 0);
        assert_eq!(out, garbage);
    }

    #[test]
    fn test_process_body_stub_is_byte_identical() {
        // The identity crusher yields nothing smaller: body forwarded as-is.
        let body = serde_json::to_vec(&fixture()).unwrap();
        let (out, parts) = process_body(&body, 2048, compress::crush);
        assert_eq!(parts, 0);
        assert_eq!(out, body);
    }
}
