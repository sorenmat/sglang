//! Microbenchmarks M1–M4 from the migration plan: `match_prefix`,
//! `insert`, `evict(n)`, and lock-ref walks, over realistic coding-agent
//! shapes (one long shared prefix, many private tails). Plus the M2/1b
//! SWA dual-counter tree and the M2/1c Mamba (full + SSM) tree over the
//! same shapes.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use sglang_radix::{EvictionPolicy, MambaRadixTree, RadixKey, RadixTree, SWARadixTree};

const SHARED: usize = 8192; // shared system prompt
const TAIL: usize = 1024; // per-agent private tokens

/// Build a coding-agent-shaped tree: `agents` requests sharing `SHARED`
/// tokens, each with a `TAIL`-token private tail.
fn build_agent_tree(agents: usize) -> (RadixTree, Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut tree = RadixTree::new(1, false, EvictionPolicy::Lru);
    let shared: Vec<i64> = (0..SHARED).map(|i| ((i * 7919) % 100_000) as i64).collect();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for a in 0..agents {
        let mut k = shared.clone();
        let mut v = (0..SHARED).map(|i| 100_000 + i as i64).collect::<Vec<_>>();
        for j in 0..TAIL {
            let tok = a as i64 * 10_000 + j as i64;
            k.push(tok);
            v.push(5_000_000 + tok);
        }
        keys.push(k.clone());
        values.push(v);
        tree.insert(&RadixKey::new(&k), &values[a], 0, false);
    }
    (tree, keys, values)
}

/// Random-ish distinct trees of `total` tokens across `branches` leaves of
/// ~equal depth.
fn build_random_tree(total: usize, branches: usize) -> (RadixTree, Vec<Vec<i64>>) {
    let mut tree = RadixTree::new(1, false, EvictionPolicy::Lru);
    let depth = total / branches;
    let mut keys = Vec::new();
    for b in 0..branches {
        let k: Vec<i64> = (0..depth)
            .map(|i| (b as i64 * 1_000_003 + (i as i64) * 31) % 500_000)
            .collect();
        let v: Vec<i64> = (0..depth).map(|i| b as i64 * 1_000_003 + i as i64).collect();
        tree.insert(&RadixKey::new(&k), &v, 0, false);
        keys.push(k);
    }
    (tree, keys)
}

fn bench_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("match_prefix");
    // Agent shape: 256 agents, full-length hits (deep, long shared prefix).
    let (tree, keys, _values) = build_agent_tree(256);
    let probe = keys[0].clone();
    let probe_r = RadixKey::new(&probe);
    group.bench_function("agent_full_hit", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.match_prefix(black_box(&probe_r)));
        })
    });
    // 32k-token key with 8k shared + 24k unseen (miss tail).
    let mut long_probe: Vec<i64> = probe.clone();
    long_probe.extend((0..24_576).map(|i| 90_000_000 + i as i64));
    let long_r = RadixKey::new(&long_probe);
    group.bench_function("agent_32k_partial_hit", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.match_prefix(black_box(&long_r)));
        })
    });

    for (total, branches, label) in [(100_000, 128, "random-100k"), (10_000, 64, "random-10k")] {
        let (tree, keys) = build_random_tree(total, branches);
        let probe = keys[branches / 2].clone();
        let probe_r = RadixKey::new(&probe);
        group.bench_function(format!("{label}_full_hit"), |b| {
            b.iter(|| {
                let mut t = tree.clone();
                let _ = black_box(t.match_prefix(black_box(&probe_r)));
            })
        });
    }
    group.finish();
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    let (tree, keys, _values) = build_agent_tree(256);
    // Fresh agent: 8k shared + 1024 private.
    let mut fresh = keys[0].clone();
    fresh.extend((1_000_000..1_001_024).map(|i| i as i64));
    let fresh_r = RadixKey::new(&fresh);
    let fresh_v: Vec<i64> = (0..fresh.len()).map(|i| 8_000_000 + i as i64).collect();
    group.bench_function("agent_9k_high_overlap", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.insert(black_box(&fresh_r), &fresh_v, 0, false));
        })
    });

    let (tree, _keys) = build_random_tree(100_000, 128);
    // Completely new key (no overlap): 781 tokens.
    let novel: Vec<i64> = (0..781).map(|i| 400_000_000 + i as i64).collect();
    let novel_r = RadixKey::new(&novel);
    let novel_v: Vec<i64> = (0..novel.len()).map(|i| 9_000_000 + i as i64).collect();
    group.bench_function("random_781_novel", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.insert(black_box(&novel_r), &novel_v, 0, false));
        })
    });
    group.finish();
}

fn bench_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("evict");
    for (total, branches, label) in [
        (100_000, 128, "random-100k"),
        (1_000_000, 1024, "random-1M"),
    ] {
        let (tree, _keys) = build_random_tree(total, branches);
        for (frac, label2) in [(0.01, "1pct"), (0.1, "10pct")] {
            let n = (total as f64 * frac) as usize;
            group.bench_with_input(
                format!("{label}_{label2}"),
                &n,
                |b, &n| {
                    b.iter(|| {
                        let mut t = tree.clone();
                        let _ = black_box(t.evict(black_box(n)));
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_lock_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_ref_walk");
    // Deep chain: one node per level (worst-case walk depth).
    let mut tree = RadixTree::new(1, false, EvictionPolicy::Lru);
    let mut prefix: Vec<i64> = Vec::new();
    let mut leaf: u32 = 0;
    for level in 0..256usize {
        prefix.push(level as i64 * 17 + 1);
        // One KV index per token of the (growing) prefix.
        let v: Vec<i64> = (0..=level).map(|i| level as i64 * 1_000_000 + i as i64).collect();
        let key = RadixKey::new(&prefix);
        let r = tree.insert(&key, &v, 0, false);
        leaf = r.last_node;
    }
    group.bench_function("depth-256", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = t.inc_lock_ref(black_box(leaf));
            let _ = t.dec_lock_ref(black_box(leaf));
        })
    });
    // Shallow wide: 128 siblings under root.
    let mut wide = RadixTree::new(1, false, EvictionPolicy::Lru);
    let mut leaves = Vec::new();
    for i in 0..128u32 {
        let k = vec![1, i as i64 * 1000 + 7];
        let v = vec![i as i64];
        leaves.push(wide.insert(&RadixKey::new(&k), &v, 0, false).last_node);
    }
    group.bench_function("depth-2-x128", |b| {
        b.iter(|| {
            let mut t = wide.clone();
            for &l in &leaves {
                t.inc_lock_ref(black_box(l));
            }
            for &l in &leaves {
                t.dec_lock_ref(black_box(l));
            }
        })
    });
    group.finish();
}

fn bench_clone_tree(c: &mut Criterion) {
    // The benches above clone the tree per iteration (cheap enough? measure
    // it — if clone cost dominates, switch to per-iteration reset).
    let (tree, _k, _v) = build_agent_tree(256);
    let mut group = c.benchmark_group("tree_clone");
    group.bench_function("agent-256", |b| b.iter(|| tree.clone()));
    group.finish();
}

// ------------------------------------------------------------- SWA (M2/1b)

const SWA_WINDOW: usize = 4096; // typical SWA sliding-window size

/// SWA coding-agent shape: `agents` requests sharing `SHARED` tokens, each
/// with a `TAIL`-token private tail; window < shared prefix so matches
/// span the window boundary.
fn build_swa_agent_tree(agents: usize) -> (SWARadixTree, Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut tree = SWARadixTree::new(1, false, SWA_WINDOW);
    let shared: Vec<i64> = (0..SHARED).map(|i| ((i * 7919) % 100_000) as i64).collect();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for a in 0..agents {
        let mut k = shared.clone();
        let mut v = (0..SHARED).map(|i| 100_000 + i as i64).collect::<Vec<_>>();
        for j in 0..TAIL {
            let tok = a as i64 * 10_000 + j as i64;
            k.push(tok);
            v.push(5_000_000 + tok);
        }
        keys.push(k.clone());
        values.push(v);
        tree.insert(&RadixKey::new(&k), &values[a], 0, 0);
    }
    (tree, keys, values)
}

fn bench_swa_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("swa_match_prefix");
    let (tree, keys, _values) = build_swa_agent_tree(256);
    let probe = keys[0].clone();
    let probe_r = RadixKey::new(&probe);
    group.bench_function("agent_full_hit", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.match_prefix(black_box(&probe_r)));
        })
    });
    group.finish();
}

fn bench_swa_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("swa_insert");
    let (tree, keys, _values) = build_swa_agent_tree(256);
    let mut fresh = keys[0].clone();
    fresh.extend((2_000_000..2_001_024).map(|i| i as i64));
    let fresh_r = RadixKey::new(&fresh);
    let fresh_v: Vec<i64> = (0..fresh.len()).map(|i| 8_000_000 + i as i64).collect();
    group.bench_function("agent_9k_high_overlap", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.insert(black_box(&fresh_r), &fresh_v, 0, 0));
        })
    });
    group.finish();
}

fn bench_swa_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("swa_evict");
    let (tree, _keys, _values) = build_swa_agent_tree(256);
    let total = SHARED * 256 + TAIL * 256;
    for (frac, label) in [(0.01, "1pct"), (0.1, "10pct")] {
        let n = (total as f64 * frac) as usize;
        group.bench_with_input(format!("agent-256_swa-{label}"), &n, |b, &n| {
            b.iter(|| {
                let mut t = tree.clone();
                let _ = black_box(t.evict(0, n));
            })
        });
    }
    group.finish();
}

fn bench_swa_lock_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("swa_lock_ref_walk");
    // Deep chain: one node per level; window smaller than the chain so
    // the uuid boundary lands mid-walk.
    let mut tree = SWARadixTree::new(1, false, 32);
    let mut prefix: Vec<i64> = Vec::new();
    let mut leaf: u32 = 0;
    for level in 0..256usize {
        prefix.push(level as i64 * 17 + 1);
        let v: Vec<i64> = (0..=level).map(|i| level as i64 * 1_000_000 + i as i64).collect();
        let key = RadixKey::new(&prefix);
        let r = tree.insert(&key, &v, 0, 0);
        leaf = r.last_node;
    }
    group.bench_function("depth-256-window-32", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let (uuid, _) = t.inc_lock_ref(black_box(leaf));
            t.dec_lock_ref(black_box(leaf), uuid, false);
        })
    });
    group.finish();
}

// ------------------------------------------------------------- Mamba (M2/1c)

const MAMBA_CHUNK: usize = 64; // FLA chunk size (default mamba_cache_chunk_size)

/// Mamba coding-agent shape: same topology as the SWA shape, one mamba
/// state per inserted leaf (the shared prefix collapses to one node).
fn build_mamba_agent_tree(agents: usize) -> (MambaRadixTree, Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut tree = MambaRadixTree::new(1, false, MAMBA_CHUNK);
    let shared: Vec<i64> = (0..SHARED).map(|i| ((i * 7919) % 100_000) as i64).collect();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for a in 0..agents {
        let mut k = shared.clone();
        let mut v = (0..SHARED).map(|i| 100_000 + i as i64).collect::<Vec<_>>();
        for j in 0..TAIL {
            let tok = a as i64 * 10_000 + j as i64;
            k.push(tok);
            v.push(5_000_000 + tok);
        }
        keys.push(k.clone());
        values.push(v);
        let slot = (a as i64 + 1) * 3;
        tree.insert(&RadixKey::new(&k), &values[a], &[slot], 0);
    }
    (tree, keys, values)
}

fn bench_mamba_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba_match_prefix");
    let (tree, keys, _values) = build_mamba_agent_tree(256);
    let probe = keys[0].clone();
    let probe_r = RadixKey::new(&probe);
    group.bench_function("agent_full_hit", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.match_prefix(black_box(&probe_r)));
        })
    });
    group.finish();
}

fn bench_mamba_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba_insert");
    let (tree, keys, _values) = build_mamba_agent_tree(256);
    let mut fresh = keys[0].clone();
    fresh.extend((2_000_000..2_001_024).map(|i| i as i64));
    let fresh_r = RadixKey::new(&fresh);
    let fresh_v: Vec<i64> = (0..fresh.len()).map(|i| 8_000_000 + i as i64).collect();
    group.bench_function("agent_9k_high_overlap", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.insert(black_box(&fresh_r), &fresh_v, &[42], 0));
        })
    });
    group.finish();
}

fn bench_mamba_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba_evict");
    let (tree, _keys, _values) = build_mamba_agent_tree(256);
    let total = SHARED * 256 + TAIL * 256;
    for (frac, label) in [(0.01, "full-1pct"), (0.1, "full-10pct")] {
        let n = (total as f64 * frac) as usize;
        group.bench_with_input(format!("agent-256_{label}"), &n, |b, &n| {
            b.iter(|| {
                let mut t = tree.clone();
                let _ = black_box(t.evict(n, 0));
            })
        });
    }
    // 1 shared internal state + 255 tail states.
    let m = (256usize as f64 * 0.1) as usize;
    group.bench_with_input("agent-256_mamba-10pct", &m, |b, &m| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = black_box(t.evict(0, m));
        })
    });
    group.finish();
}

fn bench_mamba_lock_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba_lock_ref_walk");
    // Deep chain: one node per level; the full walk covers the whole
    // depth and the mamba lock lands on the leaf alone.
    let mut tree = MambaRadixTree::new(1, false, MAMBA_CHUNK);
    let mut prefix: Vec<i64> = Vec::new();
    let mut leaf: u32 = 0;
    for level in 0..256usize {
        prefix.push(level as i64 * 17 + 1);
        let v: Vec<i64> = (0..=level).map(|i| level as i64 * 1_000_000 + i as i64).collect();
        let key = RadixKey::new(&prefix);
        let r = tree.insert(&key, &v, &[level as i64], 0);
        leaf = r.last_node;
    }
    group.bench_function("depth-256", |b| {
        b.iter(|| {
            let mut t = tree.clone();
            let _ = t.inc_lock_ref(black_box(leaf));
            let _ = t.dec_lock_ref(black_box(leaf));
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_match,
    bench_insert,
    bench_evict,
    bench_lock_ref,
    bench_clone_tree,
    bench_swa_match,
    bench_swa_insert,
    bench_swa_evict,
    bench_swa_lock_ref,
    bench_mamba_match,
    bench_mamba_insert,
    bench_mamba_evict,
    bench_mamba_lock_ref
);
criterion_main!(benches);
