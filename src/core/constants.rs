/// Binary name of this fork. It is the first word of every rewritten command and
/// of the hook command written into an agent's settings, and it names the state
/// directory under the platform config and data directories.
pub const BIN: &str = "tokenaut";
/// Upstream binary name. Kept so an `rtk hook claude` already installed in a
/// settings file is still recognised as this fork's hook, and so the state
/// directory upstream wrote can be migrated on first run.
pub const LEGACY_BIN: &str = "rtk";

pub const RTK_DATA_DIR: &str = BIN;
pub const LEGACY_DATA_DIR: &str = LEGACY_BIN;
pub const HISTORY_DB: &str = "history.db";
pub const CONFIG_TOML: &str = "config.toml";
pub const FILTERS_TOML: &str = "filters.toml";
pub const TRUSTED_FILTERS_JSON: &str = "trusted_filters.json";
pub const DEFAULT_HISTORY_DAYS: i64 = 90;

/// RTK-only subcommands that should never fall back to raw execution.
/// When adding a new RTK-only subcommand to `Commands`, add its clap name here.
pub const RTK_META_COMMANDS: &[&str] = &[
    "gain",
    "discover",
    "learn",
    "init",
    "config",
    "proxy",
    "run",
    "hook",
    "hook-audit",
    "pipe",
    "cc-economics",
    "verify",
    "trust",
    "untrust",
    "session",
    "rewrite",
    "telemetry",
    "smart",
    "deps",
    "json",
    "bench",
];
