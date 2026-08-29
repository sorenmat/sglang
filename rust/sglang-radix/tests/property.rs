//! Property tests: randomized operation sequences against structural
//! invariants, plus a determinism (replay) check. No external RNG crate —
//! a SplitMix64 keeps the sequences reproducible by construction.

use std::collections::HashSet;

use sglang_radix::{EvictionPolicy, RadixKey, RadixTree, ROOT};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + splitmix64(&mut self.0) % (hi - lo)
    }
}

/// The full observable history of one run, for the determinism check.
#[derive(Clone, Debug, PartialEq)]
struct History {
    match_hits: Vec<(Vec<i64>, u32)>,
    evicted: Vec<Vec<Vec<i64>>>,
    prefix_lens: Vec<usize>,
    sizes: Vec<(i64, i64, i64)>,
}

#[derive(Debug)]
struct Op {
    kind: u8, // 0 insert, 1 match, 2 evict, 3 lock, 4 unlock, 5 reset-free
    req: usize,
    n: usize,
}

/// Build the operation list for a scenario (shared across two runs for the
/// determinism check).
fn build_ops(rng_seed: u64, ops: usize, reqs: usize, vocab: u64, max_len: usize) -> Vec<Op> {
    let mut r = Rng(rng_seed);
    let mut out = Vec::with_capacity(ops);
    for _ in 0..ops {
        let kind = match r.range(0, 100) {
            0..=45 => 0, // insert
            46..=75 => 1, // match
            76..=88 => 2, // evict
            89..=94 => 3, // lock
            _ => 4, // unlock
        };
        out.push(Op {
            kind,
            req: r.range(0, reqs.max(1) as u64) as usize,
            n: r.range(1, 64) as usize,
        });
    }
    // keep vocab/max_len deterministic per scenario via the seed as well
    let _ = (vocab, max_len);
    out
}

/// One full scenario: returns the recorded history.
fn run_scenario(
    seed: u64,
    ops: &Vec<Op>,
    reqs: usize,
    vocab: u64,
    max_len: usize,
    page_size: usize,
    is_eagle: bool,
) -> History {
    let mut tree = RadixTree::new(page_size, is_eagle, EvictionPolicy::Lru);
    // Per-request token streams, from a small vocab so prefixes collide.
    let mut rr = Rng(seed.wrapping_add(0xA5A5));
    let streams: Vec<Vec<i64>> = (0..reqs)
        .map(|_| {
            let len = (rr.range(1, max_len as u64) as usize)
                .div_ceil(page_size)
                * page_size;
            (0..len).map(|_| rr.range(0, vocab) as i64).collect()
        })
        .collect();
    // Unique value per (req, pos).
    let values: Vec<Vec<i64>> = streams
        .iter()
        .enumerate()
        .map(|(qi, s)| s.iter().enumerate().map(|(j, _)| qi as i64 * 1_000_000 + j as i64).collect())
        .collect();

    let mut history = History {
        match_hits: Vec::new(),
        evicted: Vec::new(),
        prefix_lens: Vec::new(),
        sizes: Vec::new(),
    };
    let mut locked: HashSet<u32> = HashSet::new();
    let issued_values: HashSet<i64> = values
        .iter()
        .flatten()
        .copied()
        .collect();

    for op in ops {
        let req = op.req % reqs;
        let tokens = &streams[req];
        let vals = &values[req];
        match op.kind {
            0 => {
                // Insert a random prefix of the request stream.
                let k = ((op.n % max_len) + 1).min(tokens.len());
                let aligned = k / page_size * page_size;
                if aligned == 0 {
                    continue;
                }
                let key = RadixKey::new(&tokens[..aligned.max(1)]);
                let val = if is_eagle {
                    // bigram: one KV index per bigram unit = aligned - 1
                    &vals[..aligned.max(1).saturating_sub(1)]
                } else {
                    &vals[..aligned]
                };
                if val.is_empty() {
                    continue;
                }
                let ir = tree.insert(&key, val, 0, op.bool_());
                history.prefix_lens.push(ir.prefix_len);
                // Invariant: prefix_len <= logical key length.
                assert!(
                    ir.prefix_len <= if is_eagle { aligned - 1 } else { aligned },
                    "prefix_len {} exceeds key len",
                    ir.prefix_len
                );
            }
            1 => {
                let k = ((op.n % max_len) + 1).min(tokens.len());
                let key = RadixKey::new(&tokens[..k.max(1)]);
                let m = tree.match_prefix(&key);
                // Invariant: matched length is page-aligned and <= key len.
                assert_eq!(m.indices.len() % page_size, 0);
                assert!(m.indices.len() <= (if is_eagle {
                    tokens.len().saturating_sub(1)
                } else {
                    tokens.len()
                }));
                // Invariant: every matched index is a value we issued.
                assert!(
                    m.indices.iter().all(|x| issued_values.contains(x)),
                    "match returned unknown values {:#?}",
                    m.indices
                );
                // Monotonicity: a shorter query matches a prefix of this.
                if m.indices.len() >= page_size {
                    let short = RadixKey::new(&tokens[..page_size]);
                    let ms = tree.match_prefix(&short);
                    assert_eq!(&m.indices[..ms.indices.len()], ms.indices, "prefix mismatch");
                }
                history.match_hits.push((m.indices, m.last_node));
            }
            2 => {
                // Protection invariant: locked live nodes must survive
                // evict with byte-identical values (evict never touches
                // protected nodes). Values are reusable over time — the
                // test re-caches the same req streams — so this
                // before/after snapshot is the precise invariant, not a
                // "never evicted historically" check.
                let locked_snap: Vec<(u32, Vec<i64>)> = locked
                    .iter()
                    .filter_map(|item| tree.node_value(*item).map(|v| (*item, v.to_vec())))
                    .collect();
                let ev_before = tree.evictable_size();
                let r = tree.evict(op.n);
                for (n, v) in &locked_snap {
                    assert_eq!(
                        tree.node_value(*n),
                        Some(v.as_slice()),
                        "locked node {n} changed by evict"
                    );
                }
                // Invariants:
                // - never evict more live unlocked tokens than existed;
                // - if the budget fit inside the evictable set, it is met
                //   (the budget-crossing pop may overshoot by one node);
                // - otherwise everything evictable is evicted (cascades
                //   keep pushing until only locked nodes remain).
                assert!(
                    (r.num_tokens_evicted as i64) <= ev_before,
                    "evicted {} > evictable {ev_before}",
                    r.num_tokens_evicted
                );
                if ev_before >= op.n as i64 {
                    assert!(
                        r.num_tokens_evicted >= op.n,
                        "budget {op:?} not met: {}",
                        r.num_tokens_evicted
                    );
                } else {
                    assert_eq!(
                        r.num_tokens_evicted,
                        ev_before as usize,
                        "incomplete drain of evictable set"
                    );
                }
                history.evicted.push(r.evicted_values);
            }
            3 => {
                let candidate = history
                    .match_hits
                    .iter()
                    .rev()
                    .find(|item| tree.is_live_node(item.1))
                    .map(|item| item.1)
                    .filter(|&node| node != ROOT);
                if let Some(node) = candidate {
                    tree.inc_lock_ref(node);
                    locked.insert(node);
                }
            }
            4 => {
                // Deterministic pick: smallest locked id. (HashSet iteration
                // order is per-instance random and would break the replay
                // comparison below.)
                let mut v: Vec<u32> = locked.iter().copied().collect();
                v.sort_unstable();
                if let Some(node) = v.into_iter().next() {
                    tree.dec_lock_ref(node);
                    locked.remove(&node);
                }
            }
            _ => {}
        }

        // Structural invariants after every op.
        let live_logical: i64 = {
            let mut stack = vec![ROOT];
            let mut total = 0i64;
            while let Some(id) = stack.pop() {
                if let Some(v) = tree.node_value(id) {
                    total += v.len() as i64;
                }
                stack.extend(tree.node_children(id));
            }
            total
        };
        assert_eq!(
            live_logical, tree.total_size(), "total_size drifted after op {op:?}"
        );
        let (ev, prot, total) = (tree.evictable_size(), tree.protected_size(), live_logical);
        assert!(ev >= 0 && prot >= 0, "negative size bookkeeping");
        assert_eq!(ev + prot, total, "evictable+protected != live tokens");
        history.sizes.push((ev, prot, total));
    }

    // Locked nodes must still be live: protected throughout the run and
    // evict never touches locked nodes.
    for &node in &locked {
        assert!(tree.is_live_node(node), "locked node {node} was evicted");
    }

    history
}

impl Op {
    fn bool_(&self) -> bool {
        self.kind % 2 == 1
    }
}

#[test]
fn invariants_plain() {
    let ops = build_ops(42, 3000, 64, 200, 512);
    let _ = run_scenario(42, &ops, 64, 200, 512, 1, false);
}

#[test]
fn invariants_paged() {
    let ops = build_ops(7, 2000, 48, 128, 384);
    let _ = run_scenario(7, &ops, 48, 128, 384, 2, false);
}

#[test]
fn invariants_bigram() {
    let ops = build_ops(99, 2000, 32, 200, 512);
    let _ = run_scenario(99, &ops, 32, 200, 512, 1, true);
}

#[test]
fn determinism_replay() {
    let ops = build_ops(1234, 4000, 40, 256, 256);
    let a = run_scenario(1234, &ops, 40, 256, 256, 1, false);
    let b = run_scenario(1234, &ops, 40, 256, 256, 1, false);
    assert_eq!(a.match_hits, b.match_hits, "match results diverged");
    assert_eq!(a.evicted, b.evicted, "eviction sequences diverged");
    assert_eq!(a.prefix_lens, b.prefix_lens);
    assert_eq!(a.sizes, b.sizes);
}

#[test]
fn heavy_shared_prefix_coding_agent_shape() {
    // 8k shared system prefix + 256 agents with 1k private tails: the
    // coding-agent workload shape from the migration plan.
    let shared: Vec<i64> = (0..8192).map(|i| (i * 7919) % 100_000).collect();
    let mut tree = RadixTree::new(1, false, EvictionPolicy::Lru);
    let mut agent_keys = Vec::new();
    let mut agent_values = Vec::new();
    for a in 0..256u32 {
        let mut k = shared.clone();
        let mut v = (0..8192).map(|i| 1_000_000 + i as i64).collect::<Vec<_>>();
        for j in 0..1024 {
            let tok = a as i64 * 10_000 + j as i64;
            k.push(tok);
            v.push(9_000_000 + tok);
        }
        agent_keys.push(k);
        agent_values.push(v);
    }
    for (i, (k, v)) in agent_keys.iter().zip(agent_values.iter()).enumerate() {
        let r = tree.insert(&RadixKey::new(k), v, 0, i % 512 == 0);
        // Agent 0 primes the tree; every later agent hits the shared 8k
        // prefix (up to a split-boundary off-by-page).
        assert!(
            i == 0 || r.prefix_len >= 8192 - 4,
            "shared prefix should mostly hit: {} at agent {i}",
            r.prefix_len
        );
    }
    // Full-length match for agent 0.
    let m = tree.match_prefix(&RadixKey::new(&agent_keys[0]));
    assert_eq!(m.indices.len(), 9216);
    assert_eq!(m.indices[..8192], agent_values[0][..8192]);
    // Evict half the tree; shared prefix must survive if agents are locked.
    let shared_node = tree.match_prefix(&RadixKey::new(&shared)).last_node;
    tree.inc_lock_ref(shared_node);
    let r = tree.evict(400_000);
    // The locked shared prefix (8192 tokens) must remain.
    let m2 = tree.match_prefix(&RadixKey::new(&shared));
    assert_eq!(m2.indices.len(), 8192, "locked shared prefix was evicted");
    let _ = r;
    tree.dec_lock_ref(shared_node);
}
