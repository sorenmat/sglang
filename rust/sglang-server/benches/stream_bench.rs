//! M9 microbenchmark (plan.md §8 gate: Rust ≤ ½ Python p50 at B=256):
//! the Rust stream-frame builder — column assembly, msgpack header, and
//! egress framing — over the canonical batch shapes B=1/16/64/256, with
//! the logprob/hidden families off and all 7 families + hidden rows on.
//!
//! The Python baseline (the current `rust_server.py:push_generation`
//! flatten + msgpack pack, and `output_streamer.py` payload build) is
//! measured in the CI parity test `test/registered/rust/
//! test_rust_stream_parity.py`, which imports the same extension and
//! compares end-to-end.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use sglang_server::{HiddenRow, StreamColumns, build_stream_frame, frame_decode_batch_cols};

const TOKS: usize = 32; // new tokens per request per step

fn rid(i: usize) -> String {
    format!("req-{i:08}")
}

/// A B-request decode batch. `logprobs` switches all 7 families and the
/// hidden rows on (the worst case the egress ring has to ship).
fn columns(b: usize, logprobs: bool) -> StreamColumns {
    let rids: Vec<String> = (0..b).map(rid).collect();
    let mut output_ids = Vec::with_capacity(b * TOKS);
    for r in 0..b {
        for t in 0..TOKS {
            output_ids.push((r * 1000 + t) as i32);
        }
    }
    let c = StreamColumns {
        rids,
        finish_reasons: vec![None; b],
        prompt_tokens: vec![128; b],
        tok_lens: vec![TOKS as u32; b],
        output_ids,
        ..Default::default()
    };
    if !logprobs {
        return c;
    }
    StreamColumns {
        out_lp_vals: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| Some(-((r * TOKS + t) as f32) / 100.0))
                        .collect()
                })
                .collect(),
        ),
        out_lp_idxs: Some(
            (0..b)
                .map(|r| (0..TOKS).map(|t| (r * TOKS + t) as i32).collect())
                .collect(),
        ),
        in_lp_vals: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| (t > 0).then_some(-((r * TOKS + t) as f32) / 300.0))
                        .collect()
                })
                .collect(),
        ),
        in_lp_idxs: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| 100_000 + r as i32 * TOKS as i32 + t as i32)
                        .collect()
                })
                .collect(),
        ),
        out_top_vals: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| Some(vec![-((r * 5 + t) as i32) as f32]))
                        .collect()
                })
                .collect(),
        ),
        out_top_idxs: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| Some(vec![r as i32 * 5 + t as i32]))
                        .collect()
                })
                .collect(),
        ),
        in_top_vals: Some(
            (0..b)
                .map(|_r| (0..TOKS).map(|t| Some(vec![t as f32 / 7.0])).collect())
                .collect(),
        ),
        in_top_idxs: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| Some(vec![50_000 + r as i32 + t as i32]))
                        .collect()
                })
                .collect(),
        ),
        out_tid_vals: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| Some(vec![(r + t) as f32 * 0.25]))
                        .collect()
                })
                .collect(),
        ),
        out_tid_idxs: Some(
            (0..b)
                .map(|_r| (0..TOKS).map(|t| Some(vec![t as i32])).collect())
                .collect(),
        ),
        in_tid_vals: Some(
            (0..b)
                .map(|_r| (0..TOKS).map(|t| Some(vec![-(t as f32) * 0.1])).collect())
                .collect(),
        ),
        in_tid_idxs: Some(
            (0..b)
                .map(|r| (0..TOKS).map(|_t| Some(vec![r as i32])).collect())
                .collect(),
        ),
        hidden_rows: Some(
            (0..b)
                .map(|r| {
                    (0..TOKS)
                        .map(|t| {
                            if t == 0 {
                                HiddenRow::L(vec![HiddenRow::F(0.5), HiddenRow::F(0.25)])
                            } else {
                                HiddenRow::F(t as f32 * r as f32)
                            }
                        })
                        .collect()
                })
                .collect(),
        ),
        ..c
    }
}

/// The full Rust-side cost: columns → header + data buffers → framed bytes.
fn frame_bytes(c: &StreamColumns) -> bytes::Bytes {
    let f = build_stream_frame(c);
    let cols: Vec<&[u8]> = f.cols.iter().map(|c| c.as_slice()).collect();
    frame_decode_batch_cols(&f.header, &cols)
}

fn bench_batch(c: &mut Criterion, b: usize, logprobs: bool) {
    let name = if logprobs {
        format!("stream_frame_B{b}_logprobs")
    } else {
        format!("stream_frame_B{b}")
    };
    // The columns are collected once, outside the timed loop — Python's
    // `push_generation` does the per-request collection in Python too, and
    // the M9 gate compares the two packing stages from that same point.
    let cols = columns(b, logprobs);
    c.bench_function(&name, |bencher| {
        bencher.iter(|| black_box(frame_bytes(&cols)))
    });
}

fn criterion_bench(c: &mut Criterion) {
    // M9 shapes: B=1/16/64/256 × logprobs off/on.
    for b in [1usize, 16, 64, 256] {
        bench_batch(c, b, false);
        bench_batch(c, b, true);
    }
}

criterion_group!(benches, criterion_bench);
criterion_main!(benches);
