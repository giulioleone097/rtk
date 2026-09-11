//! `tokenaut bench`: replay the candidate filters over recent Claude Code transcripts
//! and prove that neither a token cited later in the session nor a block of text a
//! later step reproduced verbatim is removed.

/// `bench --compress`: measure the proxy pipeline over a request-body corpus.
pub mod compress;
mod filters;
/// `bench --history`: history-vs-fresh request bytes (the proxy gate).
pub mod history;

use regex::Regex;
use serde_json::Value;

/// context-mode search tools measured as their own classes.
pub const CTX_PREFIXES: [&str; 2] = [
    "mcp__plugin_context-mode_context-mode__ctx_search",
    "mcp__plugin_context-mode_context-mode__ctx_batch_execute",
];
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Prefix of a Claude Code config directory in the home dir. Claude Code itself
/// only ever creates `.claude`, but a second instance is normally run by pointing
/// `$CLAUDE_CONFIG_DIR` at a sibling, so the whole family is scanned.
const CONFIG_DIR_PREFIX: &str = ".claude";
const CLASSES: [&str; 3] = ["bash-pipe", "ctx-search", "ctx-batch"];
const BASH_PIPE: usize = 0;
const CTX_SEARCH: usize = 1;
const CTX_BATCH: usize = 2;
const MAX_EXAMPLES: usize = 10;
const TOP_PRODUCERS: usize = 15;
/// Tools whose input carries text copied out of an earlier tool result.
const EDIT_TOOLS: [&str; 3] = ["Edit", "MultiEdit", "Write"];
const MIN_QUOTE_BYTES: usize = 20;
const MIN_QUOTE_LINES: usize = 2;

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[A-Za-z_][A-Za-z0-9_./-]{5,}|[0-9]{3,}").expect("static token pattern")
    })
}

/// ANSI bytes are terminal control, not content, so they are stripped before every
/// token comparison: `\x1b[33mWARNING` must not register as the token `mWARNING`.
fn tokens(text: &str) -> HashSet<String> {
    token_re()
        .find_iter(&filters::strip_ansi(text))
        .map(|m| m.as_str().to_string())
        .collect()
}

#[derive(Default, Clone, Copy)]
struct Stats {
    results: usize,
    bytes_in: u64,
    bytes_out: u64,
    cited_lost: usize,
    exact_lost: usize,
}

impl Stats {
    fn saved_pct(&self) -> f64 {
        if self.bytes_in == 0 {
            0.0
        } else {
            100.0 * (self.bytes_in - self.bytes_out) as f64 / self.bytes_in as f64
        }
    }
}

#[derive(Default)]
struct Totals {
    classes: [Stats; CLASSES.len()],
    producers: HashMap<String, Stats>,
    examples: Vec<String>,
    exact_examples: Vec<String>,
}

/// A filtered result waiting for the tokens and the quotes that follow it in the
/// same file. Both texts are kept ANSI-free so every later comparison sees content.
struct Pending {
    line: usize,
    class: usize,
    candidates: Vec<String>,
    original: String,
    filtered: String,
}

/// A block of text a later step reproduced verbatim: an edit payload or a fenced
/// code block in assistant prose. A filter that rewrites whitespace cannot be caught
/// by the token criterion, but it does break a quote byte for byte.
struct Quote {
    line: usize,
    tool: &'static str,
    text: String,
}

/// Exit code: non-zero when the bench found a citation or a verbatim quote that a
/// candidate filter had removed.
pub fn run(days: u64, config_dirs: &[PathBuf], gaps: bool) -> i32 {
    let config_dirs = resolve_config_dirs(config_dirs);
    if gaps {
        return gaps::run(days, &config_dirs);
    }
    let mut totals = Totals::default();
    let mut files = 0usize;
    for dir in &config_dirs {
        for path in transcripts(dir, days) {
            files += 1;
            scan_file(&path, &mut totals);
        }
    }

    println!(
        "files: {files}  days: {days}  config dirs: {}",
        config_dirs.len()
    );
    println!(
        "{:<12} {:>8} {:>12} {:>12} {:>10} {:>11} {:>11}",
        "class", "results", "bytes_in", "bytes_out", "saved_pct", "cited_lost", "exact_lost"
    );
    for (index, name) in CLASSES.iter().enumerate() {
        let stat = totals.classes[index];
        println!(
            "{:<12} {:>8} {:>12} {:>12} {:>9.1}% {:>11} {:>11}",
            name,
            stat.results,
            stat.bytes_in,
            stat.bytes_out,
            stat.saved_pct(),
            stat.cited_lost,
            stat.exact_lost
        );
    }
    print_producers(&totals);

    print_examples("cited tokens lost", &totals.examples);
    print_examples("exact quotes lost", &totals.exact_examples);
    let lost: usize = totals
        .classes
        .iter()
        .map(|stat| stat.cited_lost + stat.exact_lost)
        .sum();
    i32::from(lost != 0)
}

fn print_examples(title: &str, examples: &[String]) {
    if examples.is_empty() {
        return;
    }
    println!("\n{title} (first {MAX_EXAMPLES}):");
    for example in examples.iter().take(MAX_EXAMPLES) {
        println!("{example}");
    }
}

/// Which commands produce the bash-pipe bytes, so the next filters can target them.
fn print_producers(totals: &Totals) {
    let mut ranked: Vec<(&String, &Stats)> = totals.producers.iter().collect();
    ranked.sort_by(|a, b| b.1.bytes_in.cmp(&a.1.bytes_in).then_with(|| a.0.cmp(b.0)));
    println!("\ntop {TOP_PRODUCERS} bash-pipe producers by bytes_in:");
    println!(
        "{:<24} {:>8} {:>12} {:>12} {:>10}",
        "producer", "results", "bytes_in", "bytes_out", "saved_pct"
    );
    for (name, stat) in ranked.iter().take(TOP_PRODUCERS) {
        println!(
            "{:<24.24} {:>8} {:>12} {:>12} {:>9.1}%",
            name,
            stat.results,
            stat.bytes_in,
            stat.bytes_out,
            stat.saved_pct()
        );
    }
}

/// The directories to scan: what `--config-dir` named, or the default pair under
/// the home directory when it named none.
/// The config directories to scan: `--config-dir` when given, else `$CLAUDE_CONFIG_DIR`
/// plus every `~/.claude*` directory that actually holds transcripts. Discovered
/// rather than listed, because the second config dir's name is a local convention
/// and a hardcoded one measures the wrong machine.
fn resolve_config_dirs(requested: &[PathBuf]) -> Vec<PathBuf> {
    if !requested.is_empty() {
        return requested.to_vec();
    }
    let mut dirs_found: Vec<PathBuf> = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .filter(|dir| dir.join("projects").is_dir())
        .into_iter()
        .collect();
    if let Some(home) = dirs::home_dir() {
        for entry in std::fs::read_dir(&home).into_iter().flatten().flatten() {
            let path = entry.path();
            let named_claude = entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(CONFIG_DIR_PREFIX));
            if named_claude && path.join("projects").is_dir() && !dirs_found.contains(&path) {
                dirs_found.push(path);
            }
        }
    }
    dirs_found.sort();
    dirs_found
}

/// Every `<config-dir>/projects/**/*.jsonl` modified in the last `days`, through
/// the walk `discover` uses. A project directory also holds `subagents/`
/// transcripts one level deeper, and a shallow read of it would measure a smaller
/// corpus than the audit reports on.
fn transcripts(config_dir: &Path, days: u64) -> Vec<PathBuf> {
    let mut paths = crate::discover::provider::ClaudeProvider::discover_sessions_in_projects_dir(
        &config_dir.join("projects"),
        None,
        Some(days),
    )
    .unwrap_or_default();
    paths.sort();
    paths
}

fn scan_file(path: &Path, totals: &mut Totals) {
    let Ok(file) = File::open(path) else {
        return;
    };
    let mut reader = BufReader::new(file);
    let mut tool_uses: HashMap<String, (usize, String)> = HashMap::new();
    let mut cited: Vec<(usize, HashSet<String>)> = Vec::new();
    let mut quotes: Vec<Quote> = Vec::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut raw = String::new();
    let mut line_number = 0usize;

    loop {
        raw.clear();
        match reader.read_line(&mut raw) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        line_number += 1;
        let Ok(entry) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let message = &entry["message"];
        let is_assistant = message["role"].as_str() == Some("assistant");
        let Some(items) = message["content"].as_array() else {
            continue;
        };
        for item in items {
            match item["type"].as_str() {
                Some("tool_use") => {
                    if let (Some(id), Some(name)) = (item["id"].as_str(), item["name"].as_str()) {
                        if let Some(class) = classify(name, &item["input"]) {
                            let producer = match class {
                                BASH_PIPE => {
                                    producer(item["input"]["command"].as_str().unwrap_or(""))
                                }
                                _ => String::new(),
                            };
                            tool_uses.insert(id.to_string(), (class, producer));
                        }
                        if let Some(tool) = EDIT_TOOLS.iter().find(|known| **known == name) {
                            for text in edit_payloads(&item["input"]) {
                                push_quote(line_number, tool, &text, &mut quotes);
                            }
                        }
                    }
                    cited.push((line_number, tokens(&item["input"].to_string())));
                }
                Some("tool_result") => {
                    let Some(id) = item["tool_use_id"].as_str() else {
                        continue;
                    };
                    let Some((class, producer)) = tool_uses.get(id).cloned() else {
                        continue;
                    };
                    let Some(text) = result_text(&item["content"]) else {
                        continue;
                    };
                    if let Some(record) = measure(line_number, class, &producer, &text, totals) {
                        pending.push(record);
                    }
                }
                Some("text") if is_assistant => {
                    if let Some(text) = item["text"].as_str() {
                        cited.push((line_number, tokens(text)));
                        for block in fenced_blocks(text) {
                            push_quote(line_number, "text", &block, &mut quotes);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    attribute_cited_losses(path, &mut pending, &mut cited, totals);
    attribute_exact_losses(path, &mut pending, &mut quotes, totals);
}

/// The input fields of Edit, MultiEdit and Write that carry copied text.
fn edit_payloads(input: &Value) -> Vec<String> {
    let mut out: Vec<String> = ["old_string", "new_string", "content"]
        .iter()
        .filter_map(|key| input[*key].as_str().map(str::to_string))
        .collect();
    for edit in input["edits"].as_array().into_iter().flatten() {
        for key in ["old_string", "new_string"] {
            if let Some(text) = edit[key].as_str() {
                out.push(text.to_string());
            }
        }
    }
    out
}

/// The bodies of ``` fences in assistant prose: the other place a session repeats
/// tool output verbatim.
fn fenced_blocks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut body: Option<Vec<&str>> = None;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            match body.take() {
                Some(lines) => out.push(lines.join("\n")),
                None => body = Some(Vec::new()),
            }
        } else if let Some(lines) = body.as_mut() {
            lines.push(line);
        }
    }
    out
}

/// Keep a quote only when it is big enough to be a real copy, normalised like the
/// outputs it will be compared against.
fn push_quote(line: usize, tool: &'static str, text: &str, out: &mut Vec<Quote>) {
    let text = filters::strip_ansi(text);
    if text.len() >= MIN_QUOTE_BYTES && text.lines().count() >= MIN_QUOTE_LINES {
        out.push(Quote { line, tool, text });
    }
}

fn classify(name: &str, input: &Value) -> Option<usize> {
    if name == "Bash" {
        let command = input["command"].as_str()?;
        return filters::has_top_level_pipe(command).then_some(BASH_PIPE);
    }
    if name.starts_with(CTX_PREFIXES[0]) {
        return Some(CTX_SEARCH);
    }
    if name.starts_with(CTX_PREFIXES[1]) {
        return Some(CTX_BATCH);
    }
    None
}

/// Wrapper words a producer walk steps over before reaching the real command.
const WRAPPERS: [&str; 5] = ["sudo", "time", "nohup", "command", "env"];
/// Keywords that open a compound statement: the command runs in its body, so the
/// segment they head never names the producer (`for f in a b; do node run.js`).
const BODY_OPENERS: [&str; 6] = ["for", "while", "until", "if", "case", "select"];

/// The command that produced the bytes: the first top-level segment that is
/// neither a `cd`, a bare `NAME=value` assignment nor a compound-statement head,
/// reduced to the first word of its first pipeline stage past any assignment,
/// wrapper or `timeout <duration>`. A newline and a `;` separate segments exactly
/// like `&&` does. Segments come from the quote-aware scanner, so a newline or a
/// pipe inside a quoted script is not a break.
fn producer(command: &str) -> String {
    for segment in filters::top_level_segments(command) {
        let trimmed = segment.trim();
        let mut words = trimmed.split_whitespace().peekable();
        let Some(&first) = words.peek() else {
            continue;
        };
        if first == "cd" || BODY_OPENERS.contains(&first) {
            continue;
        }
        if words.clone().all(|word| assignment_value(word).is_some()) {
            continue;
        }
        return stage_producer(&filters::first_pipeline_stage(trimmed));
    }
    "(none)".to_string()
}

/// The producer inside a single stage: skip leading assignments, wrappers and
/// body openers, then take the next word.
fn stage_producer(stage: &str) -> String {
    let mut words = stage.split_whitespace().peekable();
    while let Some(&word) = words.peek() {
        if let Some(value) = assignment_value(word) {
            // `n=$(grep ... | cut ...)` pipes inside the substitution: name what runs there.
            match value.strip_prefix("$(").or_else(|| value.strip_prefix('`')) {
                Some(inner) if !inner.is_empty() => return inner.to_string(),
                _ => {
                    words.next();
                    continue;
                }
            }
        }
        if word == "timeout" {
            words.next();
            if words.peek().is_some_and(|arg| is_timeout_duration(arg)) {
                words.next();
            }
            continue;
        }
        if WRAPPERS.contains(&word) || matches!(word, "do" | "then") {
            words.next();
            continue;
        }
        break;
    }
    match words.next() {
        Some(word) => word.trim_start_matches(['(', '{', '`', '!']).to_string(),
        None => "(none)".to_string(),
    }
}

/// The value of a leading `NAME=value` assignment, if the word is one.
fn assignment_value(word: &str) -> Option<&str> {
    let (name, value) = word.split_once('=')?;
    let named = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    named.then_some(value)
}

/// `30s`, `30`, `1m`: a leading run of digits followed by an optional unit.
fn is_timeout_duration(word: &str) -> bool {
    let digits = word.trim_end_matches(['s', 'm', 'h', 'd']);
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

fn result_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            if items
                .iter()
                .any(|item| item["type"].as_str() == Some("image"))
            {
                return None;
            }
            let parts: Vec<&str> = items
                .iter()
                .filter_map(|item| item["text"].as_str())
                .collect();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        _ => None,
    }
}

/// Apply the candidate filter of the result's class, account its bytes, and keep the
/// tokens the filter dropped so they can be checked against later citations.
fn measure(
    line: usize,
    class: usize,
    producer: &str,
    original: &str,
    totals: &mut Totals,
) -> Option<Pending> {
    if original.is_empty() {
        return None;
    }
    let filtered = filters::clean_bash_stdout(original);
    let applied = filtered.len() < original.len();
    let bytes_out = if applied {
        filtered.len()
    } else {
        original.len()
    } as u64;
    let class_stat = &mut totals.classes[class];
    class_stat.results += 1;
    class_stat.bytes_in += original.len() as u64;
    class_stat.bytes_out += bytes_out;
    if !producer.is_empty() {
        let producer_stat = totals.producers.entry(producer.to_string()).or_default();
        producer_stat.results += 1;
        producer_stat.bytes_in += original.len() as u64;
        producer_stat.bytes_out += bytes_out;
    }
    if !applied {
        return None;
    }
    // A token whose neighbours changed is retokenised, not lost, so the survivor test
    // is substring containment. The token set is the prefilter that keeps that scan
    // off the hot path. Both sides are ANSI-free: `tokens` strips, so must the text.
    let plain = filters::strip_ansi(&filtered);
    let kept = tokens(&filtered);
    let candidates: Vec<String> = tokens(original)
        .into_iter()
        .filter(|token| !kept.contains(token) && !plain.contains(token.as_str()))
        .collect();
    Some(Pending {
        line,
        class,
        candidates,
        original: filters::strip_ansi(original),
        filtered: plain,
    })
}

/// Walk the file backwards so the citation corpus is built once: at each result
/// `later` holds the tokens of exactly the assistant text and tool_use inputs that
/// follow it. A citation is a token occurrence, so `maximumError` does not cite `mError`.
fn attribute_cited_losses(
    path: &Path,
    pending: &mut [Pending],
    cited: &mut [(usize, HashSet<String>)],
    totals: &mut Totals,
) {
    pending.sort_by_key(|record| record.line);
    cited.sort_by_key(|(line, _)| *line);
    let mut later: HashSet<String> = HashSet::new();
    let mut index = cited.len();
    let mut found: Vec<String> = Vec::new();
    for record in pending.iter().rev() {
        while index > 0 && cited[index - 1].0 > record.line {
            index -= 1;
            later.extend(cited[index].1.iter().cloned());
        }
        for token in &record.candidates {
            if later.contains(token) {
                totals.classes[record.class].cited_lost += 1;
                found.push(format!(
                    "{}:{} {} {}",
                    path.display(),
                    record.line,
                    CLASSES[record.class],
                    token
                ));
            }
        }
    }
    found.reverse();
    totals.examples.extend(found);
}

/// The same reverse walk over quotes: a quote that sits inside the original output
/// but not inside the filtered one is text the session pasted byte for byte and the
/// candidate filter would have rewritten under it.
fn attribute_exact_losses(
    path: &Path,
    pending: &mut [Pending],
    quotes: &mut [Quote],
    totals: &mut Totals,
) {
    pending.sort_by_key(|record| record.line);
    quotes.sort_by_key(|quote| quote.line);
    let quotes: &[Quote] = quotes;
    let mut later: Vec<&Quote> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    let mut index = quotes.len();
    let mut found: Vec<String> = Vec::new();
    for record in pending.iter().rev() {
        while index > 0 && quotes[index - 1].line > record.line {
            index -= 1;
            if seen.insert(quotes[index].text.as_str()) {
                later.push(&quotes[index]);
            }
        }
        for quote in &later {
            let lost = quote.text.len() <= record.original.len()
                && record.original.contains(&quote.text)
                && !record.filtered.contains(&quote.text);
            if lost {
                totals.classes[record.class].exact_lost += 1;
                found.push(format!(
                    "{}:{} {} {} {}",
                    path.display(),
                    record.line,
                    CLASSES[record.class],
                    quote.tool,
                    first_line(&quote.text)
                ));
            }
        }
    }
    found.reverse();
    totals.exact_examples.extend(found);
}

/// The first line of a quote, short enough to read in the report.
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    match line.char_indices().nth(80) {
        Some((cut, _)) => format!("{}...", &line[..cut]),
        None => line.to_string(),
    }
}

/// `tokenaut bench --gaps`: which not-rewritten Bash commands cost the most bytes.
/// Walks the same corpus as the module above, but every Bash command counts (piped
/// or not) and the question per unique command text is binary: does the rewrite
/// engine `hook check` calls return a rewrite for it.
mod gaps {
    use super::{producer, result_text, transcripts, HashMap};
    use std::path::{Path, PathBuf};

    /// A command this long is never rewritten in practice, so it is counted
    /// not-rewritten without asking the engine.
    const MAX_CHECKED_COMMAND_BYTES: usize = 4000;
    const TOP_PRODUCERS: usize = 20;
    const EXAMPLE_CHARS: usize = 70;

    #[derive(Default)]
    struct CommandStats {
        bytes: u64,
        results: usize,
    }

    #[derive(Default)]
    struct Producer {
        bytes: u64,
        results: usize,
        example: String,
        example_bytes: u64,
    }

    pub fn run(days: u64, config_dirs: &[PathBuf]) -> i32 {
        let mut commands: HashMap<String, CommandStats> = HashMap::new();
        for dir in config_dirs {
            for path in transcripts(dir, days) {
                scan_file(&path, &mut commands);
            }
        }

        let keys: Vec<String> = commands.keys().cloned().collect();
        let rewritten = check_rewrites(&keys);

        let unique_commands = commands.len();
        let total_results: usize = commands.values().map(|entry| entry.results).sum();
        let total_bytes: u64 = commands.values().map(|entry| entry.bytes).sum();
        let mut rewritten_bytes = 0u64;
        let mut producers: HashMap<String, Producer> = HashMap::new();
        for key in &keys {
            let entry = &commands[key];
            if rewritten[key] {
                rewritten_bytes += entry.bytes;
                continue;
            }
            let slot = producers.entry(producer(key)).or_default();
            slot.bytes += entry.bytes;
            slot.results += entry.results;
            if entry.bytes > slot.example_bytes {
                slot.example = key.clone();
                slot.example_bytes = entry.bytes;
            }
        }

        let rewritten_pct = pct(rewritten_bytes, total_bytes);
        println!(
            "unique commands={unique_commands} results={total_results} bytes={total_bytes} \
             rewritten_bytes={rewritten_bytes} ({rewritten_pct:.1}%)"
        );
        println!(
            "{:<24} {:>8} {:>12} {:>7} example",
            "producer", "results", "bytes", "share"
        );
        let mut ranked: Vec<(&String, &Producer)> = producers.iter().collect();
        ranked.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes).then_with(|| a.0.cmp(b.0)));
        for (name, stat) in ranked.iter().take(TOP_PRODUCERS) {
            println!(
                "{:<24.24} {:>8} {:>12} {:>6.1}% {}",
                name,
                stat.results,
                stat.bytes,
                pct(stat.bytes, total_bytes),
                truncate(&stat.example, EXAMPLE_CHARS)
            );
        }
        0
    }

    fn pct(part: u64, whole: u64) -> f64 {
        if whole == 0 {
            0.0
        } else {
            100.0 * part as f64 / whole as f64
        }
    }

    fn truncate(text: &str, max_chars: usize) -> String {
        match text.char_indices().nth(max_chars) {
            Some((cut, _)) => text[..cut].to_string(),
            None => text.to_string(),
        }
    }

    /// Every Bash command in the file, keyed by its exact text, with the summed
    /// byte size and count of the results it produced across the whole corpus.
    fn scan_file(path: &Path, commands: &mut HashMap<String, CommandStats>) {
        use serde_json::Value;
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let Ok(file) = File::open(path) else {
            return;
        };
        let mut reader = BufReader::new(file);
        let mut pending: HashMap<String, String> = HashMap::new();
        let mut raw = String::new();
        loop {
            raw.clear();
            match reader.read_line(&mut raw) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let Ok(entry) = serde_json::from_str::<Value>(&raw) else {
                continue;
            };
            let Some(items) = entry["message"]["content"].as_array() else {
                continue;
            };
            for item in items {
                match item["type"].as_str() {
                    Some("tool_use") if item["name"].as_str() == Some("Bash") => {
                        if let (Some(id), Some(command)) =
                            (item["id"].as_str(), item["input"]["command"].as_str())
                        {
                            pending.insert(id.to_string(), command.to_string());
                        }
                    }
                    Some("tool_result") => {
                        let Some(id) = item["tool_use_id"].as_str() else {
                            continue;
                        };
                        let Some(command) = pending.remove(id) else {
                            continue;
                        };
                        let Some(text) = result_text(&item["content"]) else {
                            continue;
                        };
                        if text.is_empty() {
                            continue;
                        }
                        let entry = commands.entry(command).or_default();
                        entry.bytes += text.len() as u64;
                        entry.results += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Asks the rewrite engine directly, in this process, with the same exclusion
    /// and transparent-prefix configuration `hook check` reads: a `Some` is the
    /// rewrite whose absence `hook check` reports as `No rewrite`.
    fn check_rewrites(keys: &[String]) -> HashMap<String, bool> {
        let (excluded, transparent_prefixes) = crate::core::config::hook_rewrite_params();
        keys.iter()
            .map(|command| {
                let rewritten = command.len() <= MAX_CHECKED_COMMAND_BYTES
                    && crate::discover::registry::rewrite_command(
                        command,
                        &excluded,
                        &transparent_prefixes,
                    )
                    .is_some();
                (command.clone(), rewritten)
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn producer_skips_cd_assignment_and_wrapper_across_segments() {
            assert_eq!(
                producer("cd /tmp\nFOO=1 timeout 30s sed -n '1,5p' file"),
                "sed"
            );
            assert_eq!(producer("git status"), "git");
            assert_eq!(producer("cd /tmp && ls -la | wc -l"), "ls");
            assert_eq!(producer("FOO=1 BAR=2\nnode run.js"), "node");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifies_only_piped_bash_and_context_mode() {
        assert_eq!(
            classify("Bash", &json!({"command": "ls | head"})),
            Some(BASH_PIPE)
        );
        assert_eq!(classify("Bash", &json!({"command": "ls"})), None);
        assert_eq!(
            classify(
                "mcp__plugin_context-mode_context-mode__ctx_batch_execute",
                &json!({})
            ),
            Some(CTX_BATCH)
        );
        assert_eq!(classify("Read", &json!({})), None);
    }

    #[test]
    fn names_the_producer_of_the_first_pipeline_stage() {
        assert_eq!(producer("grep -rn foo . | head -5"), "grep");
        assert_eq!(producer("cd /tmp && rg --json bar | jq ."), "rg");
        assert_eq!(
            producer("RUST_LOG=debug sudo env cargo test | tail"),
            "cargo"
        );
        assert_eq!(producer("cd /x; ls -la | wc -l"), "ls");
        assert_eq!(
            producer("d=/repo\ncd $d\ngit diff --name-only | grep .ts"),
            "git"
        );
        assert_eq!(producer("for f in a b; do node run.js | tee log"), "node");
        assert_eq!(
            producer("n=$(grep -n x f | cut -d: -f1); sed -n 1p f"),
            "grep"
        );
        assert_eq!(producer("echo 'a | b' | cat"), "echo");
    }

    #[test]
    fn reads_string_and_array_result_content() {
        assert_eq!(result_text(&json!("plain")).as_deref(), Some("plain"));
        assert_eq!(
            result_text(&json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]))
                .as_deref(),
            Some("a\nb")
        );
        assert!(result_text(&json!([{"type": "image"}])).is_none());
    }

    #[test]
    fn an_ansi_terminator_is_not_a_cited_token() {
        let mut totals = Totals::default();
        let original = "\x1b[33mWARNING: keep this\nrepeated-line-here\nrepeated-line-here\n";
        let record = measure(3, BASH_PIPE, "grep", original, &mut totals).expect("filter applied");
        assert!(
            record.candidates.is_empty(),
            "unexpected candidates: {:?}",
            record.candidates
        );
    }

    #[test]
    fn a_dropped_token_cited_later_is_counted() {
        let mut totals = Totals::default();
        let mut pending = [Pending {
            line: 3,
            class: CTX_SEARCH,
            candidates: vec!["dropped-identifier".to_string()],
            original: String::new(),
            filtered: String::new(),
        }];
        let mut cited = vec![
            (2, tokens("dropped-identifier before")),
            (9, tokens("dropped-identifier after")),
        ];
        attribute_cited_losses(
            Path::new("/t/x.jsonl"),
            &mut pending,
            &mut cited,
            &mut totals,
        );
        assert_eq!(totals.classes[CTX_SEARCH].cited_lost, 1);
        assert_eq!(
            totals.examples,
            vec!["/t/x.jsonl:3 ctx-search dropped-identifier".to_string()]
        );
    }

    #[test]
    fn quotes_come_from_edit_payloads_and_fenced_blocks() {
        let input = json!({"content": "one", "edits": [{"old_string": "a", "new_string": "b"}]});
        assert_eq!(edit_payloads(&input), vec!["one", "a", "b"]);
        assert_eq!(
            fenced_blocks("intro\n```sh\nline one\nline two\n```\nafter"),
            vec!["line one\nline two"]
        );
        let mut quotes = Vec::new();
        push_quote(1, "Edit", "too short\nq", &mut quotes);
        push_quote(1, "Edit", "one line long enough to pass", &mut quotes);
        push_quote(1, "Edit", "\x1b[31mline one is long\nline two", &mut quotes);
        assert_eq!(quotes.len(), 1);
        assert_eq!(quotes[0].text, "line one is long\nline two");
    }

    #[test]
    fn a_rewritten_block_quoted_later_is_counted() {
        let mut totals = Totals::default();
        let mut pending = [Pending {
            line: 3,
            class: BASH_PIPE,
            candidates: Vec::new(),
            original: "header line kept here\nrepeated-line-here\nrepeated-line-here\ntail line\n"
                .to_string(),
            filtered: "header line kept here\nrepeated-line-here  (x2)\ntail line\n".to_string(),
        }];
        let mut quotes = vec![
            Quote {
                line: 1,
                tool: "Edit",
                text: "repeated-line-here\nrepeated-line-here".to_string(),
            },
            Quote {
                line: 7,
                tool: "Edit",
                text: "repeated-line-here\nrepeated-line-here".to_string(),
            },
        ];
        attribute_exact_losses(
            Path::new("/t/x.jsonl"),
            &mut pending,
            &mut quotes,
            &mut totals,
        );
        assert_eq!(totals.classes[BASH_PIPE].exact_lost, 1);
        assert_eq!(
            totals.exact_examples,
            vec!["/t/x.jsonl:3 bash-pipe Edit repeated-line-here".to_string()]
        );
    }
}
