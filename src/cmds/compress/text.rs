//! TextCrusher: mostly conservative. Trailing whitespace is trimmed and a run
//! of three or more blank lines collapses to one; prose itself is never
//! reworded. The one structural pass is [`fold`]: numbered near-duplicate
//! lines (generated listings, tables, test names) share a digit-normalized
//! shape, and a shape repeated [`MIN_TOTAL`]-plus times is gist, not prose —
//! the original still sits behind `ccr:<id>` for retrieval.

use super::fold;

/// Crush prose: trim line tails, collapse blank runs of 3+ to a single line,
/// then fold digit-normalized near-duplicate lines.
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
    fold::fold_similar(&out)
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
