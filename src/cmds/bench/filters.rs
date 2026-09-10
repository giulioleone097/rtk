//! Candidate text filters measured by the bench harness. None ships as a hook yet.
//! Every function maps `&str` to `String` and performs no I/O.

use regex::Regex;
use std::sync::OnceLock;

fn ansi_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\x1b(?:\[[0-9;:?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x40-\x5f])")
            .expect("static ANSI pattern")
    })
}

/// Remove ANSI escape sequences (CSI, OSC and two-byte escapes).
pub fn strip_ansi(text: &str) -> String {
    ansi_re().replace_all(text, "").into_owned()
}

/// The Bash stdout candidate: ANSI stripping only. Trailing-whitespace stripping,
/// blank-line collapsing and repeated-line folding were measured and rejected on
/// 2026-09-10 (see README): they break verbatim pastes into Edit/Write.
pub fn clean_bash_stdout(text: &str) -> String {
    strip_ansi(text)
}

/// A top-level command separator: where it starts, where the next stage resumes,
/// and its kind (`|`, `;`, `&` for `&&`, `\n` for a newline or a heredoc end).
struct Break {
    at: usize,
    resume: usize,
    kind: char,
}

/// True when the command contains a `|` outside single quotes, double quotes and heredoc bodies.
pub fn has_top_level_pipe(command: &str) -> bool {
    top_level_breaks(command)
        .iter()
        .any(|separator| separator.kind == '|')
}

/// The stage that produced the output: everything between the last separator and
/// the first top-level pipe, or the whole command when it has no pipe.
pub fn first_pipeline_stage(command: &str) -> String {
    let chars: Vec<char> = command.chars().collect();
    let breaks = top_level_breaks(command);
    let end = breaks
        .iter()
        .find(|b| b.kind == '|')
        .map_or(chars.len(), |b| b.at);
    let start = breaks
        .iter()
        .filter(|b| b.kind != '|' && b.at < end)
        .map(|b| b.resume)
        .max()
        .unwrap_or(0);
    chars[start..end].iter().collect()
}

/// Top-level segments of `command`: split on `;`, newline, `&&` and `||`, outside
/// quotes and heredoc bodies. A lone `|` stays inside its segment: it names a
/// pipeline stage, not a new command.
pub fn top_level_segments(command: &str) -> Vec<String> {
    let chars: Vec<char> = command.chars().collect();
    let breaks = top_level_breaks(command);
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut i = 0;
    while i < breaks.len() {
        let current = &breaks[i];
        if current.kind == '|' {
            match breaks.get(i + 1) {
                Some(next) if next.kind == '|' && next.at == current.at + 1 => {
                    segments.push(chars[start..current.at].iter().collect());
                    start = next.resume;
                    i += 2;
                }
                _ => i += 1,
            }
            continue;
        }
        segments.push(chars[start..current.at].iter().collect());
        start = current.resume;
        i += 1;
    }
    segments.push(chars[start..].iter().collect());
    segments
}

fn top_level_breaks(command: &str) -> Vec<Break> {
    let chars: Vec<char> = command.chars().collect();
    let mut breaks = Vec::new();
    let mut heredocs: Vec<(String, bool)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => i += 2,
            '\'' => {
                i += 1;
                while i < chars.len() && chars[i] != '\'' {
                    i += 1;
                }
                i += 1;
            }
            '"' => {
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
            }
            '|' | ';' => {
                breaks.push(Break {
                    at: i,
                    resume: i + 1,
                    kind: chars[i],
                });
                i += 1;
            }
            '&' if chars.get(i + 1) == Some(&'&') => {
                breaks.push(Break {
                    at: i,
                    resume: i + 2,
                    kind: '&',
                });
                i += 2;
            }
            '<' if chars.get(i + 1) == Some(&'<') => {
                if chars.get(i + 2) == Some(&'<') {
                    i += 3; // here-string, not a heredoc
                    continue;
                }
                let mut j = i + 2;
                let dashed = chars.get(j) == Some(&'-');
                if dashed {
                    j += 1;
                }
                while matches!(chars.get(j), Some(' ') | Some('\t')) {
                    j += 1;
                }
                let mut tag = String::new();
                match chars.get(j) {
                    Some(&quote @ ('\'' | '"')) => {
                        j += 1;
                        while j < chars.len() && chars[j] != quote {
                            tag.push(chars[j]);
                            j += 1;
                        }
                        j += 1;
                    }
                    _ => {
                        while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_') {
                            tag.push(chars[j]);
                            j += 1;
                        }
                    }
                }
                if !tag.is_empty() {
                    heredocs.push((tag, dashed));
                }
                i = j;
            }
            '\n' => {
                breaks.push(Break {
                    at: i,
                    resume: i + 1,
                    kind: '\n',
                });
                i += 1;
                for (tag, dashed) in std::mem::take(&mut heredocs) {
                    loop {
                        if i >= chars.len() {
                            break;
                        }
                        let start = i;
                        while i < chars.len() && chars[i] != '\n' {
                            i += 1;
                        }
                        let line: String = chars[start..i].iter().collect();
                        if i < chars.len() {
                            i += 1;
                        }
                        let terminator = if dashed {
                            line.trim_start_matches('\t') == tag
                        } else {
                            line == tag
                        };
                        if terminator {
                            breaks.push(Break {
                                at: i - 1,
                                resume: i,
                                kind: '\n',
                            });
                            break;
                        }
                    }
                }
            }
            _ => i += 1,
        }
    }
    breaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_sequences() {
        assert_eq!(strip_ansi("\x1b[32mok\x1b[0m done"), "ok done");
        assert_eq!(strip_ansi("\x1b]0;title\x07plain"), "plain");
    }

    #[test]
    fn detects_pipes_outside_quotes_and_heredocs() {
        assert!(has_top_level_pipe("a | b"));
        assert!(!has_top_level_pipe("echo 'a | b'"));
        assert!(has_top_level_pipe("echo \"x|y\" | head"));
        assert!(!has_top_level_pipe("cat <<EOF\nvalue | other\nEOF\n"));
        assert!(has_top_level_pipe("cat <<EOF | head\nvalue | other\nEOF\n"));
        assert!(!has_top_level_pipe("grep -n 'x' file"));
        assert!(!has_top_level_pipe("echo \\| literal"));
    }

    #[test]
    fn names_the_stage_that_produced_the_output() {
        assert_eq!(first_pipeline_stage("ls -la | wc -l"), "ls -la ");
        assert_eq!(first_pipeline_stage("cd /x && git log | head"), " git log ");
        assert_eq!(
            first_pipeline_stage("d=/x\ncd $d\ngit diff | grep ts"),
            "git diff "
        );
        assert_eq!(
            first_pipeline_stage("sed -n '1p;\n5p' f | head"),
            "sed -n '1p;\n5p' f "
        );
        assert_eq!(first_pipeline_stage("ls -la"), "ls -la");
        assert_eq!(
            first_pipeline_stage("cat <<EOF | head\na;b\nEOF\n"),
            "cat <<EOF "
        );
    }

    #[test]
    fn splits_top_level_segments_on_double_ampersand_and_double_pipe() {
        assert_eq!(
            top_level_segments("cd /tmp && ls -la"),
            vec!["cd /tmp ", " ls -la"]
        );
        assert_eq!(
            top_level_segments("test -f x || echo missing"),
            vec!["test -f x ", " echo missing"]
        );
        assert_eq!(
            top_level_segments("grep -n foo x | head -5"),
            vec!["grep -n foo x | head -5"]
        );
    }
}
