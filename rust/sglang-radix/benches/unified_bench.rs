//! Microbenchmarks for M2/1e: the unified multi-pool radix tree
//! (`UnifiedRadixTree`), FULL-only shape, page_size=1 — stepped insert over
//! the coding-agent shape (one long shared prefix, many private tails),
//! full/partial match, full device-drain (write-through delete and
//! write-back backup+demote), insert_host + drive_host_eviction, lock-ref
//! walks, the KV-canary walk, and tree cloning.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use sglang_radix::{
    EvictionPolicy, UnifiedRadixTree, UConfig, UDecLockParams, UInsertParams, CT_FULL,
};

const SHARED: usize = 8192; // shared system prompt
const TAIL: usize = 1024; // per-agent private tokens

fn config(write_back: bool) -> UConfig {
    UConfig {
        page_size: 1,
        is_eagle: false,
        sliding_window_size: 0,
        mamba_checkpoint_grid: 0,
        mamba_max_states_per_path: -1,
        eviction_policy: EvictionPolicy::Lru,
        write_through_threshold: 256,
        is_write_back: write_back,
        has_swa_host_pool: false,
        enable_session_radix_cache: false,
    }
}

fn shared_tokens() -> Vec<i64> {
    (0..SHARED).map(|i| ((i * 7919) % 100_000) as i64).collect()
}

/// Coding-agent shape: `agents` requests sharing `SHARED` tokens, each with a
/// `TAIL`-token private tail.
fn build_agent_tree(agents: usize, write_back: bool) -> (UnifiedRadixTree, Vec<Vec<i64>>) {
    let mut tree = UnifiedRadixTree::new(config(write_back));
    let params = UInsertParams::default();
    let shared = shared_tokens();
    let mut keys = Vec::new();
    for a in 0..agents {
        let mut k = shared.clone();
        let mut v: Vec<i64> = (0..SHARED).map(|i| 100_000 + i as i64).collect();
        for j in 0..TAIL {
            let tok = a as i64 * 10_000 + j as i64;
            k.push(tok);
            v.push(5_000_000 + tok);
        }
        keys.push(k.clone());
        let mut step = tree.begin_insert(0, &k, Some(v), &params);
        while step.result.is_none() {
            step = tree.resume_insert();
        }
        let _ = tree.end_insert();
    }
    (tree, keys)
}

fn pump_insert(tree: &mut UnifiedRadixTree, key: &[i64], value: Vec<i64>) {
    let params = UInsertParams::default();
    let mut step = tree.begin_insert(0, key, Some(value), &params);
    while step.result.is_none() {
        step = tree.resume_insert();
    }
    let _ = tree.end_insert();
}

/// Drive the facade's FULL device-eviction loop to completion (or quota);
/// write-back leaves get a deterministic fake D->H backup before the demote.
fn drain_device(tree: &mut UnifiedRadixTree, write_back: bool, quota: i64) -> i64 {
    let mut freed = 0i64;
    tree.evict_device_start(CT_FULL, quota);
    loop {
        let step = tree.evict_device_next_node(CT_FULL, freed);
        for (_, n) in &step.tracker {
            freed += n;
        }
        let Some(node) = step.node_id else {
            if !step.made_progress {
                break;
            }
            continue;
        };
        let out = tree.evict_device_leaf(node, write_back);
        for (_, n) in &out.tracker {
            freed += n;
        }
        if let Some(ids) = out.backup_kv {
            for id in ids {
                let (dev, xfers) = tree.build_backup_spec(id);
                let host = (0..dev.len()).map(|i| 123_000_000 + i as i64).collect::<Vec<_>>();
                tree.commit_backup(id, &host, &xfers);
                let dem = tree.demote_node(id);
                for (_, n) in &dem.tracker {
                    freed += n;
                }
            }
        }
    }
    tree.evict_device_end(CT_FULL);
    freed
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_insert");
    let (tree, keys) = build_agent_tree(256, false);
    // Fresh agent: 8k shared + 1024 private.
    let mut fresh = keys[0].clone();
    fresh.extend((1_000_000..1_001_024).map(|i| i as i64));
    let fresh_v: Vec<i64> = (0..fresh.len()).map(|i| 8_000_000 + i as i64).collect();
    group.bench_function("agent_9k_high_overlap", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            pump_insert(black_box(&mut t), black_box(&fresh), fresh_v.clone());
        })
    });
    // Completely new key (no overlap).
    let novel: Vec<i64> = (0..781).map(|i| 400_000_000 + i as i64).collect();
    let novel_v: Vec<i64> = (0..781).map(|i| 9_000_000 + i as i64).collect();
    group.bench_function("novel_781", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            pump_insert(black_box(&mut t), black_box(&novel), novel_v.clone());
        })
    });
    group.finish();
}

fn bench_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_match");
    let (tree, keys) = build_agent_tree(256, false);
    let full = keys[0].clone();
    group.bench_function("agent_9k_full_hit", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.match_prefix(0, black_box(&full)));
        })
    });
    // Mid-node split: 8k shared + 512 of the first agent's 1024-token tail.
    let split = &keys[0][..SHARED + 512];
    group.bench_function("agent_8k512_split", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.match_prefix(0, black_box(split)));
        })
    });
    // 32k-token key with 8k shared + 24k unseen (miss tail).
    let mut long_probe = full.clone();
    long_probe.extend((0..24_576).map(|i| 90_000_000 + i as i64));
    group.bench_function("agent_32k_partial_hit", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.match_prefix(0, black_box(&long_probe)));
        })
    });
    group.finish();
}

fn bench_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_evict");
    let (tree, _keys) = build_agent_tree(256, false);
    let total: i64 = 256 * (SHARED + TAIL) as i64;
    group.bench_function("write_through_full_drain", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(drain_device(black_box(&mut t), false, i64::MAX));
        })
    });
    for (frac, label) in [(10, "10pct"), (1, "1pct")] {
        let quota = total / frac;
        group.bench_function(format!("write_through_{label}"), |b| {
            b.iter(|| {
                let mut t = tree.clone();
                let _ = black_box(drain_device(black_box(&mut t), false, quota));
            })
        });
    }
    // Write-back: every leaf backs up (fake D->H) and demotes.
    let wb_tree = build_agent_tree(256, true).0;
    group.bench_function("write_back_full_drain", |b| {
        b.iter(|| {
            let mut t = wb_tree.clone();
            let _ = black_box(drain_device(black_box(&mut t), true, i64::MAX));
        })
    });
    group.finish();
}

fn bench_hicache(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_hicache");
    let (tree, keys) = build_agent_tree(256, true);
    let key = keys[0].clone();
    let host_value: Vec<i64> = (0..key.len()).map(|i| 700_000 + i as i64).collect();
    let hashes: Vec<String> = vec!["h".to_string(); key.len()];
    group.bench_function("insert_host_9k", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.insert_host(
                t.root,
                0,
                black_box(&key),
                host_value.clone(),
                Some(hashes.clone()),
            ));
        })
    });
    // Host-pressure drain over a fully backed-up tree.
    let (mut backed, _) = build_agent_tree(256, true);
    let _ = drain_device(&mut backed, true, i64::MAX);
    group.bench_function("drive_host_eviction_full", |b| {
        b.iter(|| {
            let mut t = backed.clone();
            let out = t.drive_host_eviction(CT_FULL, i64::MAX);
            black_box(out)
        })
    });
    group.finish();
}

fn bench_lock_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_lock_ref");
    let (tree, keys) = build_agent_tree(256, false);
    let mut probe = tree.clone();
    let leaf = probe.match_prefix(0, &keys[0]).best_match_node;
    group.bench_function("agent_9k_inc_dec", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = t.inc_lock_ref(black_box(leaf), &[]);
            let params = UDecLockParams {
                swa_uuid_for_lock: None,
                swa_uuid_for_host_lock: None,
                skip_lock_node_ids: Vec::new(),
            };
            t.dec_lock_ref(black_box(leaf), &params, false);
        })
    });
    // Wide: lock 128 distinct one-token leaves.
    let mut wide = UnifiedRadixTree::new(config(false));
    let mut leaves = Vec::new();
    for i in 0..128u32 {
        let k = vec![1, i as i64 * 1000 + 7];
        let v = vec![i as i64];
        let mut step = wide.begin_insert(0, &k, Some(v), &UInsertParams::default());
        while step.result.is_none() {
            step = wide.resume_insert();
        }
        leaves.push(step.result.unwrap().last_device_node);
        let _ = wide.end_insert();
    }
    group.bench_function("depth-2-x128", |b| {
        b.iter(|| {
            let mut t = wide.clone();
            for &l in &leaves {
                t.inc_lock_ref(black_box(l), &[]);
            }
            let params = UDecLockParams {
                swa_uuid_for_lock: None,
                swa_uuid_for_host_lock: None,
                skip_lock_node_ids: Vec::new(),
            };
            for &l in &leaves {
                t.dec_lock_ref(black_box(l), &params, false);
            }
        })
    });
    group.finish();
}

fn bench_canary_walk(c: &mut Criterion) {
    let (tree, _keys) = build_agent_tree(256, false);
    let mut group = c.benchmark_group("unified_kv_canary_walk");
    group.bench_function("agent-256", |b| {
        b.iter(|| {
            let t = tree.clone();
            black_box(t.walk_for_kv_canary(false, false))
        })
    });
    group.finish();
}

fn bench_clone_tree(c: &mut Criterion) {
    let (tree, _k) = build_agent_tree(256, false);
    let mut group = c.benchmark_group("unified_clone");
    group.bench_function("agent-256", |b| b.iter(|| tree.clone()));
    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_match,
    bench_evict,
    bench_hicache,
    bench_lock_ref,
    bench_canary_walk,
    bench_clone_tree
);
criterion_main!(benches);
