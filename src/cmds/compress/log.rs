//! LogCrusher: strips terminal noise and folds repeats.
//!
//! ANSI escapes come off first ([`crate::core::utils::strip_ansi`]). Inside a
//! line a carriage return keeps only the last non-empty segment — a progress
//! bar redraws over itself and the final frame is the one that mattered. A
//! run of identical consecutive lines then collapses into `<line>  (xN)`, the
//! same fold the PostToolUse filters emit; blank lines never fold.

use crate::core::utils;

/// The part of `line` a terminal would still show after its carriage returns.
fn last_frame(line: &str) -> &str {
    line.rsplit('\r').find(|s| !s.is_empty()).unwrap_or("")
}

/// Append `line` `run` times folded: `line` once, or `line  (xN)`.
fn flush(out: &mut String, line: &str, run: usize) {
    match run {
        0 => {}
        1 => {
            out.push_str(line);
            out.push('\n');
        }
        _ => {
            out.push_str(&format!("{line}  (x{run})"));
            out.push('\n');
        }
    }
}

pub(crate) fn crush_kind(text: &str) -> String {
    let stripped = utils::strip_ansi(text);
    let mut out = String::with_capacity(stripped.len());
    let mut run_line = "";
    let mut run = 0usize;
    for raw in stripped.lines() {
        let line = last_frame(raw);
        if !line.is_empty() && line == run_line {
            run += 1;
            continue;
        }
        flush(&mut out, run_line, run);
        run_line = line;
        run = 1;
    }
    flush(&mut out, run_line, run);
    if !text.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_escapes() {
        assert_eq!(crush_kind("\x1b[31mERR\x1b[0m failed\n"), "ERR failed\n");
    }

    #[test]
    fn folds_identical_line_runs() {
        assert_eq!(crush_kind("spin\nspin\nspin\n"), "spin  (x3)\n");
        assert_eq!(crush_kind("a\nb\nb\nc\n"), "a\nb  (x2)\nc\n");
    }

    #[test]
    fn blank_lines_do_not_fold() {
        assert_eq!(crush_kind("a\n\n\nb\n"), "a\n\n\nb\n");
    }

    #[test]
    fn carriage_return_keeps_the_last_frame() {
        assert_eq!(
            crush_kind("progress 10%\rprogress 20%\rprogress 30%\n"),
            "progress 30%\n"
        );
        assert_eq!(crush_kind("work done\r\n"), "work done\n");
    }

    #[test]
    fn distinct_lines_pass_through() {
        let input = "INFO start\nWARN careful\nERROR stop\n";
        assert_eq!(crush_kind(input), input);
    }
}
