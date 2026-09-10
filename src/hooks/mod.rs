//! Hook installation and lifecycle management for AI coding agents.

pub mod constants;
pub mod hook_audit_cmd;
pub mod hook_check;
#[deny(clippy::print_stdout, clippy::print_stderr)]
pub mod hook_cmd;
pub mod init;
pub mod integrity;
pub mod permissions;
mod read_nudge;
pub mod rewrite_cmd;
pub mod trust;
pub mod verify_cmd;

/// Both binary names are accepted: a settings file written by upstream rtk still
/// registers a working hook here, so `hook check` must report it as installed
/// instead of prompting for a second, duplicate entry.
pub fn is_claude_hook_command(command: &str) -> bool {
    use crate::core::constants::{BIN, LEGACY_BIN};

    let parts = crate::discover::lexer::shell_split(command);
    let [binary, hook, claude] = parts.as_slice() else {
        return false;
    };

    let binary_name = binary.rsplit(['/', '\\']).next().unwrap_or(binary);

    (binary_name == BIN || binary_name == LEGACY_BIN) && hook == "hook" && claude == "claude"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_hook_command_matches_bare_and_absolute_rtk() {
        assert!(is_claude_hook_command("rtk hook claude"));
        assert!(is_claude_hook_command("/opt/homebrew/bin/rtk hook claude"));
        assert!(is_claude_hook_command(
            "\"/opt/homebrew/bin/rtk\" hook claude"
        ));
    }

    #[test]
    fn claude_hook_command_rejects_other_commands() {
        assert!(!is_claude_hook_command("not-rtk hook claude"));
        assert!(!is_claude_hook_command("/opt/homebrew/bin/rtk hook cursor"));
        assert!(!is_claude_hook_command("echo rtk hook claude"));
    }
}
