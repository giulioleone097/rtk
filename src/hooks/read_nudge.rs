//! Claude Code PreToolUse nudge for the `Read` tool: when the target is a
//! regular file above 50 KiB, point the agent at `ctx_execute_file` (or
//! `offset`/`limit`). `additionalContext` is advisory only, so a missing file,
//! a directory or a small file all resolve to "say nothing".

use super::constants::PRE_TOOL_USE_KEY;
use serde_json::{json, Value};

const LARGE_FILE_THRESHOLD_BYTES: u64 = 51200;

/// PreToolUse response for a `Read` payload, or `None` when no nudge is due.
pub fn read_hook_output(v: &Value) -> Option<Value> {
    let path = v.pointer("/tool_input/file_path")?.as_str()?;
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() <= LARGE_FILE_THRESHOLD_BYTES {
        return None;
    }
    let kb = metadata.len() / 1024;
    Some(json!({
        "hookSpecificOutput": {
            "hookEventName": PRE_TOOL_USE_KEY,
            "additionalContext": format!(
                "File is {kb} KB: read it with ctx_execute_file (tokenaut) and print only what you need, or pass offset/limit to Read."
            )
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn payload_for(path: &str) -> Value {
        json!({ "tool_name": "Read", "tool_input": { "file_path": path }, "session_id": "s" })
    }

    #[test]
    fn large_file_yields_additional_context() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![b'a'; 61440]).unwrap();
        let output = read_hook_output(&payload_for(file.path().to_str().unwrap())).unwrap();
        assert_eq!(
            output["hookSpecificOutput"]["hookEventName"],
            PRE_TOOL_USE_KEY
        );
        let context = output["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap();
        assert!(context.contains("ctx_execute_file"));
        assert!(context.len() <= 200);
    }

    #[test]
    fn small_file_yields_no_output() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![b'a'; 1024]).unwrap();
        assert!(read_hook_output(&payload_for(file.path().to_str().unwrap())).is_none());
    }

    #[test]
    fn missing_file_or_directory_yields_no_output() {
        assert!(read_hook_output(&payload_for("/nonexistent/rtk-read-nudge-test")).is_none());
        let dir = tempfile::tempdir().unwrap();
        assert!(read_hook_output(&payload_for(dir.path().to_str().unwrap())).is_none());
        assert!(read_hook_output(&json!({ "tool_name": "Read", "tool_input": {} })).is_none());
    }
}
