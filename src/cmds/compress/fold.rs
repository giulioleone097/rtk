//! Near-duplicate line folding: the pass the exact-match fold cannot do.
//!
//! A line's *shape* is the line with every run of digits replaced by `#`
//! (`test tests::case_42 ... ok` → `test tests::case_# ... ok`). Cargo,
//! pytest, generated code and logs emit hundreds of lines that differ only in
//! counters. When one shape appears at least [`MIN_TOTAL`] times, its first
//! [`KEEP`] lines and its final line stay verbatim, the middle lines drop,
//! and one summary line per folded shape is appended:
//! `… "<shape>" ×N`.
//!
//! The dropped content is still retrievable: parts large enough carry a
//! `ccr:<id>` marker into the FTS5 index, so folding trades verbatim
//! repetition for shape + count — the model can pull the original back.
//!
//! ceiling: interleaved shapes fold too (Compiling/test lines), and the fold
//! position is not marked inline — the summary block lands at the end.
//! Upgrade trigger = a corpus where position-in-stream matters to the model.

use std::collections::HashMap;

/// Shapes below this many total lines are never folded.
const MIN_TOTAL: usize = 8;
/// Leading lines of a folded shape kept verbatim (the last is kept too).
const KEEP: usize = 4;
/// Distinct shapes tracked before giving up on new ones (bound on memory).
const MAX_SHAPES: usize = 256;

/// Digit-normalized shape of a line, or `None` for blank lines.
fn shape_of(line: &str) -> Option<String> {
    if line.trim().is_empty() {
        return None;
    }
    let mut shape = String::with_capacity(line.len());
    let mut digits = 0usize;
    for ch in line.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
        } else {
            if digits > 0 {
                shape.push('#');
                digits = 0;
            }
            shape.push(ch);
        }
    }
    if digits > 0 {
        shape.push('#');
    }
    Some(shape)
}

/// Fold near-duplicate lines in `text`; returns `text` unchanged when no
/// shape repeats enough to be worth a summary block.
pub(crate) fn fold_similar(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut totals: HashMap<String, usize> = HashMap::new();
    for line in &lines {
        if let Some(shape) = shape_of(line) {
            if totals.len() < MAX_SHAPES || totals.contains_key(&shape) {
                *totals.entry(shape).or_default() += 1;
            }
        }
    }
    if totals.values().all(|&n| n < MIN_TOTAL) {
        return text.to_owned();
    }
    // Index of each folded shape's last line — it stays verbatim.
    let mut last_at: HashMap<String, usize> = HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(shape) = shape_of(line) {
            if totals.get(&shape).copied().unwrap_or(0) >= MIN_TOTAL {
                last_at.insert(shape, i);
            }
        }
    }
    let mut emitted: HashMap<String, usize> = HashMap::new();
    let mut out = String::with_capacity(text.len());
    for (i, line) in lines.iter().enumerate() {
        let folded =
            shape_of(line).filter(|shape| totals.get(shape).copied().unwrap_or(0) >= MIN_TOTAL);
        match folded {
            Some(shape) => {
                let n = emitted.entry(shape.clone()).or_default();
                *n += 1;
                if *n <= KEEP || last_at.get(&shape).copied() == Some(i) {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    let mut dropped_any = false;
    for (shape, &total) in &totals {
        if total < MIN_TOTAL {
            continue;
        }
        let dropped = total.saturating_sub(KEEP + 1);
        if dropped == 0 {
            continue;
        }
        dropped_any = true;
        out.push_str(&format!("… \"{}\" ×{dropped} more\n", shape.trim_end()));
    }
    if !dropped_any {
        return text.to_owned();
    }
    if !text.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_numbered_repeats() {
        let text: String = (0..20).map(|i| format!("test case_{i} ... ok\n")).collect();
        let out = fold_similar(&text);
        assert!(out.contains("test case_0"));
        assert!(out.contains("test case_19"));
        assert!(out.contains("×15 more"));
        assert!(!out.contains("test case_10"));
    }

    #[test]
    fn folds_interleaved_shapes() {
        let text: String = (0..10)
            .map(|i| format!("Compiling dep{i} v1.{i}.0\ntest tests::case_{i} ... ok\n"))
            .collect();
        let out = fold_similar(&text);
        assert!(out.contains("Compiling dep# v#.#.#"));
        assert!(out.contains("case_#"));
    }

    #[test]
    fn sparse_shapes_pass_through() {
        let text = "alpha one\nbeta two\ngamma three\n";
        assert_eq!(fold_similar(text), text);
    }

    #[test]
    fn folds_few_lines_untouched() {
        let text: String = (0..5).map(|i| format!("line {i}\n")).collect();
        assert_eq!(fold_similar(&text), text);
    }
}
