//! TextCrusher: deliberately conservative. Trailing whitespace is trimmed and
//! a run of three or more blank lines collapses to one; nothing else moves —
//! prose has no safe rewrite.

/// Crush prose: trim line tails, collapse blank runs of 3+ to a single line.
pub(crate) fn crush_kind(text: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    let mut blanks = 0usize;
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.is_empty() {
            blanks += 1;
            continue;
        }
        kept.extend(std::iter::repeat_n(
            "",
            if blanks >= 3 { 1 } else { blanks },
        ));
        blanks = 0;
        kept.push(line);
    }
    kept.extend(std::iter::repeat_n(
        "",
        if blanks >= 3 { 1 } else { blanks },
    ));
    let mut out = kept.join("\n");
    if text.ends_with('\n') && !out.is_empty() {
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_three_or_more_blanks_to_one() {
        assert_eq!(crush_kind("a\n\n\n\nb\n"), "a\n\nb\n");
        assert_eq!(crush_kind("a\n\n\n\n\n\n\nb\n"), "a\n\nb\n");
    }

    #[test]
    fn two_blanks_are_kept() {
        assert_eq!(crush_kind("a\n\n\nb\n"), "a\n\n\nb\n");
        assert_eq!(crush_kind("a\n\nb\n"), "a\n\nb\n");
    }

    #[test]
    fn trims_trailing_whitespace() {
        assert_eq!(crush_kind("one  \ntwo\t\n"), "one\ntwo\n");
    }

    #[test]
    fn clean_prose_is_unchanged() {
        let input = "alpha\nbeta\ngamma\n";
        assert_eq!(crush_kind(input), input);
    }
}
