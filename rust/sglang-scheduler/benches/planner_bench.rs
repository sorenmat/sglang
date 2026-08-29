//! M5–M7: microbenchmarks for the pure planner engine
//! (`sglang-scheduler`).
//!
//! Shapes mirror the 16-concurrent-coding-agent workload on a
//! Qwen3-27B NVFP4 server (fast prefill, long decodes, heavy shared
//! system-prefix traffic):
//!
//! - **M5 prefill-burst**: a prefill pass over a waiting queue with
//!   LPM scoring snapshots (fresh + fully-cached + partial prefixes),
//!   the hot `plan_next_batch` prefill arm.
//! - **M6 decode-steady**: the decode arm over a 16-req running batch
//!   (page-crossing mix), the per-iteration hot path.
//! - **M7 mixed**: prefill pass with a parked chunked request + running
//!   decode offset (chunked continuation + admission in one call).
//!
//! All inputs are prebuilt once (the bench measures the decision, not the
//! snapshot marshalling); the PyO3 facade benches (Phase 4) measure the
//! full FFI round-trip on top of these.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use sglang_scheduler::{plan_next_batch, Config, PlanReq, StepEnv};
use sglang_scheduler::ntr::Ntr;

/// Waiting-queue snapshot: 64 reqs, coding-agent shape — half share the
/// system prompt prefix, quarter are fully cached, quarter are fresh.
fn waiting_burst() -> (Vec<PlanReq>, Vec<u32>, Vec<bool>) {
    let mut waiting = Vec::with_capacity(64);
    let mut scores = Vec::with_capacity(64);
    let deprio = vec![false; 64];
    for i in 0..64u32 {
        let (prefix, score) = match i % 4 {
            0 => (0, 0u32), // fresh
            1 => (4096, 4096), // fully cached
            2 => (2048, 2048), // partial
            _ => (1024, 1024), // shared system prefix
        };
        waiting.push(PlanReq {
            pool_idx: i,
            origin_len: 8192,
            out_len: 0,
            committed_len: 8192,
            prefix_len: prefix,
            last_node: u32::MAX,
            priority: 0,
            arrival_seq: i as u64,
            max_new_tokens: 2048,
            routing_key: 0,
            ignore_eos: false,
            finished: false,
            retracted_stain: i % 16 == 0,
            host_hit_length: 0,
        });
        scores.push(score);
    }
    (waiting, scores, deprio)
}

/// 16 running decode reqs with a page-crossing mix at page_size 64.
fn running_decode() -> Vec<PlanReq> {
    (0..16u32)
        .map(|i| {
            // committed = 64*i - (i%4==0 ? 0 : 1) -> one page-crosser per 4
            let committed = 64 * i - if i % 4 == 0 { 0 } else { 1 };
            PlanReq {
                pool_idx: 1000 + i,
                origin_len: 8192,
                out_len: (committed - 8192).max(1),
                committed_len: committed,
                prefix_len: 8192,
                last_node: 1,
                priority: 0,
                arrival_seq: i as u64,
                max_new_tokens: 2048,
                routing_key: 0,
                ignore_eos: false,
                finished: i == 15, // one finished req to exercise the filter
                retracted_stain: false,
                host_hit_length: 0,
            }
        })
        .collect()
}

fn env_open() -> StepEnv {
    StepEnv {
        allocator_avail_tokens: 1_000_000,
        tree_evictable_tokens: 100_000,
        num_allocatable_reqs: u32::MAX,
        batch_is_full: false,
        mixed_chunk_allowed: true,
    }
}

fn bench_m5_prefill_burst(c: &mut Criterion) {
    let cfg = Config::default();
    let ntr = Ntr::from_config(&cfg);
    let (waiting, scores, deprio) = waiting_burst();
    let running = running_decode();
    let env = env_open();

    c.bench_function("M5 prefill burst c64 (LPM, chunked off)", |b| {
        b.iter(|| {
            let plan = plan_next_batch(
                black_box(&cfg),
                black_box(&ntr),
                black_box(&waiting),
                black_box(&running),
                None,
                black_box(&scores),
                black_box(&deprio),
                black_box(&env),
                0,
            );
            black_box(plan.extend_tokens())
        })
    });
}

fn bench_m6_decode_steady(c: &mut Criterion) {
    let cfg = Config {
        page_size: 64,
        ..Config::default()
    };
    let ntr = Ntr::from_config(&cfg);
    let running = running_decode();
    let env = env_open();

    c.bench_function("M6 decode steady c16 (p64)", |b| {
        b.iter(|| {
            let plan = plan_next_batch(
                black_box(&cfg),
                black_box(&ntr),
                &[],
                black_box(&running),
                None,
                &[],
                &[],
                black_box(&env),
                1,
            );
            black_box(plan.mode)
        })
    });
}

fn bench_m7_chunked_mixed(c: &mut Criterion) {
    let cfg = Config {
        page_size: 64,
        chunked_prefill_size: Some(8192),
        ..Config::default()
    };
    let ntr = Ntr::from_config(&cfg);
    let (waiting, scores, deprio) = waiting_burst();
    let running = running_decode();
    let env = env_open();
    let chunked = PlanReq {
        pool_idx: 9999,
        origin_len: 32_768,
        out_len: 0,
        committed_len: 32_768,
        prefix_len: 8192,
        last_node: 1,
        max_new_tokens: 2048,
        ..Default::default()
    };

    c.bench_function("M7 chunked continuation + c64 waiting", |b| {
        b.iter(|| {
            let plan = plan_next_batch(
                black_box(&cfg),
                black_box(&ntr),
                black_box(&waiting),
                black_box(&running),
                Some(black_box(&chunked)),
                black_box(&scores),
                black_box(&deprio),
                black_box(&env),
                2,
            );
            black_box(plan.extend_tokens())
        })
    });
}

criterion_group!(
    name = planner_benches;
    config = Criterion::default().sample_size(50);
    targets = bench_m5_prefill_burst, bench_m6_decode_steady, bench_m7_chunked_mixed
);
criterion_main!(planner_benches);
