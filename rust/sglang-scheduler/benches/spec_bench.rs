//! Spec-v2 accept-run resolution microbenchmarks (plan §9 / M6).
//!
//! Shapes mirror the spec decode hot path on a Qwen3-27B NVFP4 server:
//!
//! - **resolve** at B = 1/16/64/256, stride 2 (MTP-style, one draft +
//!   bonus) and 64 (EAGLE-style long draft chain), grammar-truncated and
//!   plain, with and without the MTP block/cap columns;
//! - **counters** — one `SpecCounters::update` per req per spec step
//!   (histogram grow + counter bumps), the per-iteration bookkeeping the
//!   core folds in `apply_result`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use sglang_scheduler::{SpecCounters, SpecRow, resolve_spec_runs};

fn rows(b: usize, stride: u32, grammar: bool, block_cap: bool) -> Vec<SpecRow> {
    (0..b)
        .map(|i| SpecRow {
            accept_len: ((i as u32 % stride) + 1).max(1),
            retracted: i == b.saturating_sub(1) && b > 3, // one unsettled tail
            finished: false,
            grammar_retained: grammar.then(|| vec![7, 8]),
            block_accept_len: block_cap.then_some(2),
            cap_len: block_cap.then_some(3),
        })
        .collect()
}

fn bench_resolve(c: &mut Criterion) {
    for b in [1usize, 16, 64, 256] {
        for (name, stride) in [("mtp s2", 2u32), ("eagle s64", 64u32)] {
            for (grammar, block_cap) in [(false, false), (true, false), (true, true)] {
                let rs = rows(b, stride, grammar, block_cap);
                let buf: Vec<i64> = (0..(b as u32 * stride) as i64).collect();
                let label = format!("resolve B{b} {name} grammar={grammar} block_cap={block_cap}");
                c.bench_function(&label, |bench| {
                    bench.iter(|| {
                        let r =
                            resolve_spec_runs(black_box(&buf), black_box(stride), black_box(&rs))
                                .unwrap();
                        black_box(r.num_correct_drafts)
                    })
                });
            }
        }
    }
}

fn bench_counters(c: &mut Criterion) {
    for b in [1usize, 16, 64, 256] {
        let label = format!("counters update B{b}");
        c.bench_function(&label, |bench| {
            bench.iter(|| {
                let mut counters = vec![SpecCounters::default(); b];
                for (i, c) in counters.iter_mut().enumerate() {
                    c.update((i as u32) % 4, Some(2), Some(3 + i as u32 % 5));
                }
                black_box(counters[0].spec_num_correct_drafts)
            })
        });
    }
}

criterion_group!(
    name = spec_benches;
    config = Criterion::default().sample_size(50);
    targets = bench_resolve, bench_counters
);
criterion_main!(spec_benches);
