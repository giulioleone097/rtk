//! Request-body compression stage: find compressible `messages[]`/`input[]`
//! parts and route each through [`crate::cmds::compress::crush`].
//!
//! Cache alignment: `crush` is deterministic, so a part compressed in turn N
//! produces identical bytes in turn N+1 — the request prefix that reaches
//! upstream stays byte-stable and prompt caching keeps working. To preserve
//! the invariant this module only rewrites eligible text fields in place: it
//! never reorders `messages`/`input` and never touches `system`,
//! `instructions`, `tools` or `cache_control`. A part already carrying a
//! `ccr:` retrieval marker is skipped, so a compressed part round-trips
//! unchanged.

use anyhow::Result;
use serde_json::Value;

use crate::cmds::compress::Crushed;

/// Signature shared by the real crusher and test fakes.
pub type Crusher = fn(&str) -> Result<Crushed>;

/// One replaced part's sizes: `(original bytes, emitted bytes)`.
pub type CrushedPart = (usize, usize);

/// JSON-pointer paths of every compressible text field in the request body.
///
/// Two API shapes dispatch on their body field: Anthropic `messages[]`
/// ([`anthropic_parts`]) and OpenAI Responses `input[]` ([`responses_parts`]).
/// A `messages` field wins when both are present — Anthropic-shaped bodies
/// walk exactly as before. A body matching neither shape yields no paths,
/// so [`process_body`] forwards it byte-identical.
pub fn eligible_parts(body: &Value, min_bytes: usize) -> Vec<String> {
    if body.get("messages").and_then(Value::as_array).is_some() {
        anthropic_parts(body, min_bytes)
    } else {
        responses_parts(body, min_bytes)
    }
}

/// Anthropic Messages shape. Eligible: `tool_result` content (string or
/// `{type:"text"}` sub-parts) and `text` parts whose text is at least
/// `min_bytes` — anywhere inside `messages[]` except the last
/// `role=="user"` message, which is the freshest human turn and stays
/// verbatim.
fn anthropic_parts(body: &Value, min_bytes: usize) -> Vec<String> {
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
    let out = crush(slot).ok()?;
    if out.text.len() >= slot.len() {
        return None;
    }
    let in_bytes = out.in_bytes;
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

/// OpenAI Responses shape (`POST /v1/responses`, `/v1/responses/compact`):
/// the conversation lives in a top-level `input[]` of typed items. Eligible:
/// `function_call_output` items' string `output` and `message` items' text
/// content parts at least `min_bytes` long. Two carve-outs mirror the
/// Anthropic walker's invariants:
///
/// - Everything from the last `role=="user"` message item to the end of the
///   array stays verbatim. In this shape the freshest turn spans items —
///   `function_call_output`s trailing that message are this turn's tool
///   results, the same thing the Anthropic rule protects inside the last
///   user message.
/// - `system`/`developer` messages are never touched: the Anthropic side
///   keeps `system` safe as a top-level field, and `instructions` is the
///   equivalent top-level field here — input messages carrying those roles
///   get the same protection. `instructions`, `tools` and every other
///   top-level field are never visited by construction.
// ceiling: `function_call_output.output` only in its string form — the
// list-of-parts form is legal but rare (Codex sends a string), and those
// items just pass through uncompressed.
fn responses_parts(body: &Value, min_bytes: usize) -> Vec<String> {
    let mut paths = Vec::new();
    let Some(items) = body.get("input").and_then(Value::as_array) else {
        return paths;
    };
    let last_user = items.iter().rposition(|item| {
        item.get("type").and_then(Value::as_str) == Some("message")
            && item.get("role").and_then(Value::as_str) == Some("user")
    });
    for (i, item) in items.iter().enumerate() {
        if last_user.is_some_and(|lu| i >= lu) {
            break;
        }
        match item.get("type").and_then(Value::as_str) {
            Some("function_call_output") => {
                if let Some(text) = item.get("output").and_then(Value::as_str) {
                    collect(&mut paths, format!("/input/{i}/output"), text, min_bytes);
                }
            }
            Some("function_call") => {
                if let Some(text) = item.get("arguments").and_then(Value::as_str) {
                    collect(&mut paths, format!("/input/{i}/arguments"), text, min_bytes);
                }
            }
            Some("reasoning") => {
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    for (j, part) in summary.iter().enumerate() {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            collect(
                                &mut paths,
                                format!("/input/{i}/summary/{j}/text"),
                                text,
                                min_bytes,
                            );
                        }
                    }
                }
            }
            Some("message") => {
                if !matches!(
                    item.get("role").and_then(Value::as_str),
                    Some("user" | "assistant")
                ) {
                    continue;
                }
                match item.get("content") {
                    Some(Value::String(text)) => {
                        collect(&mut paths, format!("/input/{i}/content"), text, min_bytes);
                    }
                    Some(Value::Array(parts)) => {
                        for (j, part) in parts.iter().enumerate() {
                            if matches!(
                                part.get("type").and_then(Value::as_str),
                                Some("input_text" | "output_text" | "text")
                            ) {
                                if let Some(text) = part.get("text").and_then(Value::as_str) {
                                    collect(
                                        &mut paths,
                                        format!("/input/{i}/content/{j}/text"),
                                        text,
                                        min_bytes,
                                    );
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    paths
}

/// One message part, walked by field name rather than part `type` so every
/// text-bearing part is covered: `text`/`thinking` strings, `content`
/// (string or `{type:"text"}` sub-parts — tool_result, web_search results,
/// anything else using that shape), and `input` object's string leaves
/// (`tool_use`/`server_tool_use`: `Write.file_text` et al carry whole file
/// bodies). Binary payloads are safe: `image`/`document` keep data under
/// `source`, which this never visits.
fn collect_part(paths: &mut Vec<String>, base: &str, part: &Value, min_bytes: usize) {
    // `thinking` deliberately excluded: its `signature` is integrity-bound to
    // the thinking text, so a crushed thinking block fails upstream.
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        collect(paths, format!("{base}/text"), text, min_bytes);
    }
    match part.get("content") {
        Some(Value::String(text)) => {
            collect(paths, format!("{base}/content"), text, min_bytes);
        }
        Some(Value::Array(subs)) => {
            for (k, sub) in subs.iter().enumerate() {
                if let Some(text) = sub.get("text").and_then(Value::as_str) {
                    collect(paths, format!("{base}/content/{k}/text"), text, min_bytes);
                }
            }
        }
        _ => {}
    }
    if let Some(input) = part.get("input") {
        collect_input_strings(paths, &format!("{base}/input"), input, min_bytes);
    }
}

/// Recurse an `input`/`arguments` JSON tree collecting every string leaf
/// (`MultiEdit.edits[].new_string`, nested tool payloads, …). Keys needing
/// JSON-pointer escaping are rare; `crush_at` resolves what it can and a
/// dangling path simply stays uncompressed.
fn collect_input_strings(paths: &mut Vec<String>, base: &str, value: &Value, min_bytes: usize) {
    match value {
        Value::String(text) => collect(paths, base.to_owned(), text, min_bytes),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                collect_input_strings(paths, &format!("{base}/{i}"), item, min_bytes);
            }
        }
        Value::Object(map) => {
            for (key, val) in map {
                let key = key.replace('~', "~0").replace('/', "~1");
                collect_input_strings(paths, &format!("{base}/{key}"), val, min_bytes);
            }
        }
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
        // The two old tool_result texts, the old assistant text and the
        // tool_use input field. The image data, the short text, `system`, and
        // the whole last user message are untouched.
        assert_eq!(
            paths,
            vec![
                "/messages/0/content/0/content",
                "/messages/0/content/1/content/0/text",
                "/messages/0/content/2/text",
                "/messages/1/content/0/text",
                "/messages/1/content/1/input/cmd",
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

    /// OpenAI Responses shape: `instructions`/`tools` untouched, old
    /// `function_call_output` and message text parts eligible, and the tail
    /// starting at the last `role=="user"` message — the freshest turn —
    /// stays verbatim.
    fn responses_fixture() -> Value {
        json!({
            "instructions": big("instr"),
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": big("dev")}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": big("u1")}]},
                {"type": "function_call_output", "call_id": "c1", "output": big("fco")},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": big("a1")}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": big("u2")}]},
                {"type": "function_call_output", "call_id": "c2", "output": big("tail-fco")},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": big("sum")}]}
            ],
            "tools": [{"type": "function", "name": "exec"}]
        })
    }

    #[test]
    fn test_eligible_parts_responses_fixture() {
        let paths = eligible_parts(&responses_fixture(), 2048);
        // The old user text, the old function_call_output and the old
        // assistant text. The developer message, `instructions`, `tools`,
        // the reasoning summary, and the whole tail from the last user
        // message on (including `tail-fco`) are untouched.
        assert_eq!(
            paths,
            vec![
                "/input/1/content/0/text",
                "/input/2/output",
                "/input/3/content/0/text",
            ]
        );
    }

    #[test]
    fn test_eligible_parts_responses_string_content_and_min_bytes() {
        let body = json!({"input": [
            {"type": "message", "role": "user", "content": big("old-question")},
            {"type": "function_call_output", "call_id": "c", "output": "tiny"},
            {"type": "message", "role": "user", "content": big("fresh")}
        ]});
        assert_eq!(eligible_parts(&body, 2048), vec!["/input/0/content"]);
        // A `function_call_output` ending the array is the current turn's
        // freshest tool result: it stays verbatim like an Anthropic
        // tool_result inside the last user message.
        let body = json!({"input": [
            {"type": "function_call_output", "call_id": "old", "output": big("stale")},
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call_output", "call_id": "new", "output": big("just-ran")}
        ]});
        assert_eq!(eligible_parts(&body, 2048), vec!["/input/0/output"]);
    }

    #[test]
    fn test_eligible_parts_responses_skips() {
        // `input` as a plain string (single-prompt form) and bodies matching
        // neither shape produce no paths.
        assert!(eligible_parts(&json!({"input": "hi"}), 1).is_empty());
        assert!(eligible_parts(&json!({"foo": 1}), 1).is_empty());
        // system/developer messages stay verbatim like Anthropic `system`.
        let body = json!({"input": [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": big("sys")}]},
            {"type": "message", "role": "user", "content": "hi"}
        ]});
        assert!(eligible_parts(&body, 1).is_empty());
        // Already-compressed parts are not re-crushed.
        let marked = json!({"input": [
            {"type": "function_call_output", "call_id": "c", "output": "ccr:9f1a 2000 bytes indexed"},
            {"type": "message", "role": "user", "content": "hi"}
        ]});
        assert!(eligible_parts(&marked, 1).is_empty());
        // Anthropic shape still wins when both fields are present.
        let both = json!({
            "messages": [{"role": "assistant", "content": big("m")}, {"role": "user", "content": "x"}],
            "input": [{"type": "function_call_output", "call_id": "c", "output": big("i")}]
        });
        assert_eq!(eligible_parts(&both, 2048), vec!["/messages/0/content"]);
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

    /// Crusher that never wins — the no-op contract stays testable without
    /// depending on how `compress::crush` treats these fixtures.
    fn identity_crusher(text: &str) -> Result<Crushed> {
        Ok(Crushed {
            text: text.to_owned(),
            in_bytes: text.len(),
            ccr: None,
        })
    }

    #[test]
    fn test_crush_at_keeps_original_when_not_smaller() {
        let mut body = fixture();
        let before = serde_json::to_vec(&body).unwrap();
        assert_eq!(
            crush_at(&mut body, "/messages/0/content/0/content", identity_crusher),
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
        let (out, parts) = process_body(&body, 2048, identity_crusher);
        assert_eq!(parts, 0);
        assert_eq!(out, body);
    }

    #[test]
    fn test_crush_at_responses_paths() {
        let mut body = responses_fixture();
        let replaced = crush_at(&mut body, "/input/2/output", half_crusher);
        assert_eq!(replaced, Some((3004, 3004 / 4 + 13)));
        assert!(body["input"][2]["output"]
            .as_str()
            .unwrap()
            .starts_with("ccr:deadbeef fco:"));
        assert_eq!(
            crush_at(&mut body, "/input/1/content/0/text", half_crusher),
            Some((3003, 3003 / 4 + 13))
        );
        // Missing path / non-string slot: nothing happens.
        assert_eq!(crush_at(&mut body, "/input/9/output", half_crusher), None);
        assert_eq!(crush_at(&mut body, "/tools/0/name", half_crusher), None);
    }

    #[test]
    fn test_process_body_responses_end_to_end() {
        let body = serde_json::to_vec(&responses_fixture()).unwrap();
        let (out, parts) = process_body(&body, 2048, half_crusher);
        assert_eq!(parts, 3);
        let doc: Value = serde_json::from_slice(&out).unwrap();
        assert!(doc["input"][2]["output"]
            .as_str()
            .unwrap()
            .starts_with("ccr:deadbeef fco:"));
        // instructions, the freshest turn and untouched top-level fields.
        assert_eq!(doc["instructions"], big("instr"));
        assert_eq!(doc["input"][4]["content"][0]["text"], big("u2"));
        assert_eq!(doc["input"][5]["output"], big("tail-fco"));
        assert_eq!(doc["tools"][0]["name"], "exec");
    }

    #[test]
    fn test_process_body_neither_shape_is_byte_identical() {
        // Parses fine but matches neither `messages` nor `input` shape.
        for body in [
            serde_json::to_vec(&json!({"foo": {"bar": big("x")}})).unwrap(),
            serde_json::to_vec(&json!({"input": "single string prompt"})).unwrap(),
            serde_json::to_vec(&responses_fixture()).unwrap(),
        ] {
            let (out, parts) = process_body(&body, 2048, identity_crusher);
            assert_eq!(parts, 0);
            assert_eq!(out, body);
        }
    }
}
