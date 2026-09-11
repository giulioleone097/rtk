//! CodeCompressor: drops comment-only lines and whitespace noise.
//!
//! A line is dropped only when every non-whitespace character sits inside a
//! comment: `//` and `#` line comments plus `/* … */` blocks tracked by a
//! small line state machine. `#` lines that are preprocessor directives
//! (`#include`, `#if`, …), shebangs (`#!`), attributes (`#[`) or string
//! interpolation (`#{`) are code and stay. `code(); // tail` keeps its whole
//! line — a line with any code on it is never removed.
//!
//! ceiling: not string-literal aware — a line inside a multiline string that
//! opens with `//`, `#` or `/*` is still dropped. Upgrade trigger = a real
//! parser dep.

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

use super::fold;

/// `#`-lines that are code: preprocessor directives.
static PREPROC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^#\s*(include|define|pragma|if|ifdef|ifndef|elif|else|endif|undef|error|warning|import|line|using|region|endregion)\b",
    )
    .unwrap()
});

/// True when a trimmed `#`-line is a comment rather than a shebang, an
/// attribute, a directive or an interpolation.
fn is_hash_comment(t: &str) -> bool {
    t.starts_with('#')
        && !t.starts_with("#!")
        && !t.starts_with("#[")
        && !t.starts_with("#{")
        && !PREPROC.is_match(t)
}

/// Strip comment-only lines, collapse runs of blank lines to one, and trim
/// trailing whitespace.
pub(crate) fn crush_kind(text: &str) -> String {
    let mut kept: Vec<Cow<'_, str>> = Vec::new();
    let mut in_block = false;
    let mut blanks = 0usize;
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            blanks += 1;
            if blanks == 1 {
                kept.push(Cow::Borrowed(""));
            }
            continue;
        }
        let indent = &line[..line.len() - line.trim_start().len()];
        let mut rest = line;
        let mut consumed = false;
        loop {
            let t = rest.trim_start();
            if in_block {
                match t.find("*/") {
                    Some(p) => {
                        in_block = false;
                        rest = &t[p + 2..];
                        consumed = true;
                        continue;
                    }
                    None => rest = "",
                }
            } else if t.starts_with("//") || is_hash_comment(t) {
                rest = "";
            } else if let Some(after) = t.strip_prefix("/*") {
                match after.find("*/") {
                    Some(p) => {
                        rest = &after[p + 2..];
                        consumed = true;
                        continue;
                    }
                    None => {
                        in_block = true;
                        rest = "";
                    }
                }
            }
            break;
        }
        if rest.trim().is_empty() {
            // A dropped comment line is removed outright, not blanked, and the
            // blank run across it still collapses.
            continue;
        }
        blanks = 0;
        // Once a comment was eaten the gap between it and the code is noise;
        // the line's own indent is still the right one to keep.
        kept.push(if consumed {
            Cow::Owned(format!("{indent}{}", rest.trim_start()))
        } else {
            Cow::Borrowed(line)
        });
    }
    let mut out = kept.join("\n");
    if text.ends_with('\n') && !out.is_empty() {
        out.push('\n');
    }
    fold::fold_similar(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_line_comments_keeps_code() {
        let input = "// header\nfn main() {\n    # python comment\n    let x = 1;\n}\n";
        assert_eq!(crush_kind(input), "fn main() {\n    let x = 1;\n}\n");
    }

    #[test]
    fn drops_block_comments_across_lines() {
        let input = "/* start\nmiddle\nend */\ncode();\n";
        assert_eq!(crush_kind(input), "code();\n");
    }

    #[test]
    fn code_after_a_block_comment_survives() {
        assert_eq!(crush_kind("/* note */ run();\n"), "run();\n");
        assert_eq!(crush_kind("x(); /* tail */\n"), "x(); /* tail */\n");
        assert_eq!(crush_kind("x(); // tail\n"), "x(); // tail\n");
    }

    #[test]
    fn hash_directives_are_code() {
        let input = "#include <stdio.h>\n#define N 1\n#[derive(Debug)]\n#!shebang\n";
        assert_eq!(crush_kind(input), input);
    }

    #[test]
    fn collapses_blank_runs_to_one() {
        assert_eq!(crush_kind("a\n\n\n\nb\n"), "a\n\nb\n");
        assert_eq!(crush_kind("a\n\nb\n"), "a\n\nb\n");
    }

    #[test]
    fn blank_run_across_a_dropped_comment_still_collapses() {
        assert_eq!(crush_kind("a\n\n// gone\n\nb\n"), "a\n\nb\n");
    }

    #[test]
    fn trims_trailing_whitespace() {
        assert_eq!(crush_kind("a   \nb\t\n"), "a\nb\n");
    }

    #[test]
    fn keeps_indentation() {
        let input = "def f():\n    # gone\n    return 1\n";
        assert_eq!(crush_kind(input), "def f():\n    return 1\n");
    }
}
