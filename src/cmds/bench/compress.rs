//! `tokenaut bench --compress` — measure the api-proxy compression pipeline on
//! a corpus of captured request bodies.
//!
//! Corpus: each argument is a path — a `.jsonl` file (one request body per
//! line, the body itself or `{"body": ...}`) or a `.json` file holding one
//! body. Reports per-file and total bytes in/out, saved %, parts crushed.
//! Exit 0 always: this is a measurement, not a gate.
//!
//! Corpus builder: `scripts/gen-compress-corpus.sh` (synthetic), or capture real
//! traffic by pointing a host at `tokenaut api-proxy` and dumping bodies.

use std::fs;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use serde_json::Value;

use crate::cmds::compress;
use crate::cmds::compress::Crushed;
use crate::cmds::proxy::pipeline;

/// Index-free crusher for corpus measurement (CCR needs a live index; it only
/// adds savings on top of what this measures).
fn measure_crush(raw: &str) -> Result<Crushed> {
    compress::crush(raw)
}

pub fn run(paths: &[PathBuf], min_bytes: usize) -> Result<i32> {
    let mut total_in = 0usize;
    let mut total_out = 0usize;
    let mut total_parts = 0usize;
    let mut files = 0usize;
    for path in paths {
        let (fin, fout, parts) =
            process_path(path, min_bytes).with_context(|| format!("read {}", path.display()))?;
        files += 1;
        total_in += fin;
        total_out += fout;
        total_parts += parts;
        let saved = if fin > 0 {
            (1.0 - fout as f64 / fin as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<44} {:>10} {:>10} {:>7.1}% parts={}",
            path.display().to_string(),
            fin,
            fout,
            saved,
            parts
        );
    }
    println!("{}", "-".repeat(76));
    let saved = if total_in > 0 {
        (1.0 - total_out as f64 / total_in as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "TOTAL files={files} in={total_in} out={total_out} saved={saved:.1}% parts={total_parts} min_bytes={min_bytes}"
    );
    Ok(0)
}

fn process_path(path: &Path, min_bytes: usize) -> Result<(usize, usize, usize)> {
    if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
        let file = fs::File::open(path)?;
        let mut rin = 0usize;
        let mut rout = 0usize;
        let mut parts = 0usize;
        for line in BufReader::new(file).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line)
                .with_context(|| format!("invalid JSONL line in {}", path.display()))?;
            let body = value.get("body").cloned().unwrap_or(value);
            let bytes = serde_json::to_vec(&body)?;
            rin += bytes.len();
            let (out, n) = pipeline::process_body(&bytes, min_bytes, measure_crush);
            rout += out.len();
            parts += n;
        }
        return Ok((rin, rout, parts));
    }
    let bytes = fs::read(path)?;
    let (out, n) = pipeline::process_body(&bytes, min_bytes, measure_crush);
    Ok((bytes.len(), out.len(), n))
}
