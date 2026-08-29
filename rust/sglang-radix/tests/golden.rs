//! Golden tests: hand-computed expectations that pin the exact semantics of
//! the base RadixCache (split points, prefix lengths, eviction order,
//! lock-ref size bookkeeping, page alignment, bigram view, namespaces).

use sglang_radix::{EvictionPolicy, InsertResult, RadixKey, RadixTree, ROOT};

fn tree() -> RadixTree {
    RadixTree::new(1, false, EvictionPolicy::Lru)
}

fn key(t: &[i64]) -> RadixKey<'_> {
    RadixKey::new(t)
}

fn insert(tree: &mut RadixTree, t: &[i64], v: &[i64]) -> usize {
    tree.insert(&key(t), v, 0, false).prefix_len
}

fn match_(tree: &mut RadixTree, t: &[i64]) -> (Vec<i64>, u32) {
    let r = tree.match_prefix(&key(t));
    (r.indices, r.last_node)
}

#[test]
fn basic_insert_match_split() {
    let mut t = tree();
    // Mirrors the __main__ demo in radix_cache.py.
    assert_eq!(insert(&mut t, &[1, 2, 3], &[10, 11, 12]), 0);
    // [1,2] already present -> split at 2; new leaf [4,5].
    assert_eq!(insert(&mut t, &[1, 2, 4, 5], &[20, 21, 22, 23]), 2);
    assert_eq!(insert(&mut t, &[1, 2, 4, 5, 6, 7], &[30, 31, 32, 33, 34, 35]), 4);

    assert_eq!(match_(&mut t, &[1, 2, 3, 13, 14]).0, vec![10, 11, 12]);
    assert_eq!(match_(&mut t, &[1, 2]).0, vec![10, 11]);
    // Match ending mid-node [4,5] -> split at 1.
    assert_eq!(match_(&mut t, &[1, 2, 4]).0, vec![10, 11, 22]);
    assert_eq!(
        match_(&mut t, &[1, 2, 4, 5, 6, 7]).0,
        vec![10, 11, 22, 23, 34, 35]
    );
    // Re-inserting a fully cached key reports the full prefix.
    assert_eq!(insert(&mut t, &[1, 2, 3], &[90, 91, 92]), 3);

    assert_eq!(t.total_size(), 7); // 2 + 1 + 2 + 2
    assert_eq!(t.evictable_size(), 7);
    assert_eq!(t.protected_size(), 0);
}

#[test]
fn lock_ref_and_protected_eviction() {
    let mut t = tree();
    assert_eq!(insert(&mut t, &[1, 2, 4, 5], &[10, 11, 20, 21]), 0);
    let leaf = t.match_prefix(&key(&[1, 2])).last_node; // node [1,2]
    let (indices, leaf45) = match_(&mut t, &[1, 2, 4, 5]);
    assert_eq!(indices, vec![10, 11, 20, 21]);

    // Lock the [4,5] leaf: 2 tokens protected, walk covers [4,5] and [1,2].
    let delta = t.inc_lock_ref(leaf45);
    assert_eq!(delta, -4);
    assert_eq!(t.evictable_size(), 0);
    assert_eq!(t.protected_size(), 4);
    // Nothing unlocked -> nothing evictable.
    assert_eq!(t.evict(100).num_tokens_evicted, 0);

    // Unlock again.
    let delta = t.dec_lock_ref(leaf45);
    assert_eq!(delta, 4);
    assert_eq!(t.evictable_size(), 4);
    assert_eq!(t.protected_size(), 0);

    // Now evict everything: LRU order from the last walk above.
    let r = t.evict(100);
    assert_eq!(r.num_tokens_evicted, 4);
    assert!(r.evicted_values.iter().all(|v| v
        .iter()
        .all(|x| [10, 11, 20, 21].contains(x))));
    assert_eq!(t.total_size(), 0);
    // `leaf` handle is now dead.
    assert!(!t.is_live_node(leaf));
}

#[test]
fn lru_eviction_order_and_cascade() {
    let mut t = tree();
    assert_eq!(insert(&mut t, &[1, 2, 3], &[10, 11, 12]), 0);
    assert_eq!(insert(&mut t, &[1, 2, 4, 5], &[20, 21, 22, 23]), 2);
    assert_eq!(insert(&mut t, &[1, 2, 4, 5, 6, 7], &[30, 31, 32, 33, 34, 35]), 4);
    match_(&mut t, &[1, 2, 3, 13, 14]); // touches [1,2] and [3]
    match_(&mut t, &[1, 2]); // touches [1,2]
    match_(&mut t, &[1, 2, 4]); // split [4,5] -> [4],[5]
    match_(&mut t, &[1, 2, 4, 5, 6, 7]); // touches [1,2],[4],[5],[6,7]

    let (idx, node_3) = match_(&mut t, &[1, 2, 3]);
    assert_eq!(idx, vec![10, 11, 12]);
    // Lock [3]: the walk covers [3] AND its parent [1,2] -> 3 tokens
    // protected, 4 evictable remain (the [4],[5],[6,7] chain).
    let delta = t.inc_lock_ref(node_3);
    assert_eq!(delta, -3);
    assert_eq!(t.evictable_size(), 4);
    assert_eq!(t.protected_size(), 3);

    let r = t.evict(100);
    // Only [6,7] is an evictable leaf at this point (its ancestors [5]
    // and [4] still have live children). Eviction cascades up:
    //   [6,7](2) -> [5](1) -> [4](1). Locked [3]/[1,2] are untouched.
    assert_eq!(r.num_tokens_evicted, 4);
    let flat: Vec<i64> = r.evicted_values.iter().flatten().copied().collect();
    assert_eq!(
        flat.iter().filter(|&&x| [22, 23, 34, 35].contains(&x)).count(),
        4,
        "evicted values were {flat:?}"
    );
    // [3] and [1,2] survive; [3] stays locked.
    let (idx, _) = match_(&mut t, &[1, 2, 3]);
    assert_eq!(idx, vec![10, 11, 12]);
    assert_eq!(t.total_size(), 3);

    t.dec_lock_ref(node_3);
    let r = t.evict(100);
    // Everything remaining: [3](1) -> [1,2](2) plus the now-evicted
    // leaves already gone. LRU order: [4]/[5] were evicted above, so the
    // live leaves are [3] (touched most recently) and nothing else:
    // the [1,2] node cascades in after [3]. Total 3 tokens.
    assert_eq!(r.num_tokens_evicted, 3);
    assert_eq!(t.total_size(), 0);
}

#[test]
fn page_size_two_alignment() {
    let mut t = RadixTree::new(2, false, EvictionPolicy::Lru);
    assert_eq!(insert(&mut t, &[1, 2, 3, 4, 5, 6], &[1, 2, 3, 4, 5, 6]), 0);
    // Match key longer than the tree: aligned 8 -> matches all 6.
    assert_eq!(
        match_(&mut t, &[1, 2, 3, 4, 5, 6, 7, 8]).0,
        vec![1, 2, 3, 4, 5, 6]
    );
    // 3 tokens -> aligned to 2 -> matches [1,2] and splits.
    assert_eq!(match_(&mut t, &[1, 2, 3]).0, vec![1, 2]);
    assert_eq!(match_(&mut t, &[1, 2, 3, 4]).0, vec![1, 2, 3, 4]);
    assert_eq!(insert(&mut t, &[1, 2, 9, 8], &[7, 8, 9, 10]), 2);

    // LRU: the [5,6] tail is the oldest leaf (its last access predates the
    // [3,4] split node and the [9,8] leaf); one 2-token leaf meets the
    // budget exactly (Python `evict` stops at `num_evicted >= num_tokens`).
    let r = t.evict(2);
    assert_eq!(r.num_tokens_evicted, 2);
    assert_eq!(r.evicted_values.iter().flatten().copied().collect::<Vec<_>>(), vec![5, 6]);
    assert_eq!(t.total_size(), 6); // 8 inserted - 2 evicted
}

#[test]
fn bigram_eagle_view() {
    let mut t = RadixTree::new(1, true, EvictionPolicy::Lru);
    // raw [1,2,3,4] = 3 bigram units; 3 KV indices.
    assert_eq!(
        t.insert(&RadixKey::new(&[1, 2, 3, 4]), &[100, 101, 102], 0, false).prefix_len,
        0
    );
    // raw [1,2,3,5] = units (1,2),(2,3),(3,5): 2 shared -> split at unit 2.
    assert_eq!(
        t.match_prefix(&RadixKey::new(&[1, 2, 3, 5])).indices,
        vec![100, 101]
    );
    assert_eq!(
        t.match_prefix(&RadixKey::new(&[1, 2, 3, 4])).indices,
        vec![100, 101, 102]
    );
    // Extend the raw sequence by one token: one more bigram unit.
    assert_eq!(
        t.insert(&RadixKey::new(&[1, 2, 3, 4, 5]), &[200, 201, 202, 203], 0, false).prefix_len,
        3
    );
    assert_eq!(
        t.match_prefix(&RadixKey::new(&[1, 2, 3, 4, 5])).indices,
        vec![100, 101, 102, 203]
    );
    assert_eq!(t.total_size(), 4);
}

#[test]
fn namespaces_are_disjoint() {
    let mut t = tree();
    let mut k_a = key(&[1, 2]);
    k_a.extra_key = Some("A");
    t.insert(&k_a, &[1, 2], 0, false);
    let mut k_b = key(&[1, 2]);
    k_b.extra_key = Some("B");
    t.insert(&k_b, &[9, 8], 0, false);
    t.insert(&key(&[1, 2]), &[5, 6], 0, false);
    let mut k_s = key(&[1, 2]);
    k_s.cache_salt = Some("s");
    t.insert(&k_s, &[7, 7], 0, false);

    let mut q = key(&[1, 2]);
    q.extra_key = Some("A");
    assert_eq!(t.match_prefix(&q).indices, vec![1, 2]);
    let mut q = key(&[1, 2]);
    q.extra_key = Some("B");
    assert_eq!(t.match_prefix(&q).indices, vec![9, 8]);
    assert_eq!(t.match_prefix(&key(&[1, 2])).indices, vec![5, 6]);
    let mut q = key(&[1, 2]);
    q.cache_salt = Some("s");
    assert_eq!(t.match_prefix(&q).indices, vec![7, 7]);
    // extra_key + salt together is its own namespace.
    let mut q = key(&[1, 2]);
    q.extra_key = Some("A");
    q.cache_salt = Some("s");
    assert_eq!(t.match_prefix(&q).indices, Vec::<i64>::new());

    assert_eq!(t.total_size(), 8);
}

#[test]
fn limit_caps_the_key() {
    let mut t = tree();
    t.insert(&key(&[1, 2, 3, 4, 5, 6]), &[1, 2, 3, 4, 5, 6], 0, false);
    let mut q = key(&[1, 2, 3, 4, 5, 6, 9, 9, 9]);
    q.limit = Some(4);
    assert_eq!(t.match_prefix(&q).indices, vec![1, 2, 3, 4]);
}

#[test]
fn disabled_tree_is_a_noop() {
    let mut t = tree().with_disable(true);
    assert_eq!(
        t.insert(&key(&[1, 2, 3]), &[1, 2, 3], 0, false),
        InsertResult {
            prefix_len: 0,
            last_node: ROOT
        }
    );
    assert_eq!(t.match_prefix(&key(&[1, 2, 3])).indices, Vec::<i64>::new());
    assert_eq!(t.evict(10).num_tokens_evicted, 0);
    assert_eq!(t.inc_lock_ref(ROOT), 0); // root guard is bypassed when disabled
}

#[test]
fn reset_rebuilds_root() {
    let mut t = tree();
    insert(&mut t, &[1, 2, 3], &[10, 11, 12]);
    t.inc_lock_ref(1);
    t.reset();
    assert_eq!(t.total_size(), 0);
    assert_eq!(t.evictable_size(), 0);
    assert_eq!(t.protected_size(), 0);
    assert!(t.is_live_node(ROOT));
    assert!(!t.is_live_node(1));
}

#[test]
fn priority_policy_evicts_low_priority_first() {
    let mut t = RadixTree::new(1, false, EvictionPolicy::Priority);
    // Two sibling leaves with different priorities (distinct heads:
    // [1,2] and [3,4] stay separate children of the root).
    t.insert(&key(&[1, 2]), &[10, 11], 5, false);
    t.insert(&key(&[3, 4]), &[12, 13], 1, false);
    let r = t.evict(1);
    // Priority policy: lower `priority` value evicts first -> [3,4] node.
    assert_eq!(r.num_tokens_evicted, 2);
    assert_eq!(
        r.evicted_values.iter().flatten().copied().collect::<Vec<_>>(),
        vec![12, 13]
    );
}

#[test]
fn lfu_policy_prefers_cold_hits() {
    let mut t = RadixTree::new(1, false, EvictionPolicy::Lfu);
    t.insert(&key(&[1, 2, 3]), &[10, 11, 12], 0, false);
    t.insert(&key(&[4, 5, 6]), &[20, 21, 22], 0, false);
    // Hit [1,2,3] twice more: its node hit_count climbs above [4,5,6]'s.
    t.match_prefix(&key(&[1, 2, 3]));
    t.match_prefix(&key(&[1, 2, 3]));
    // Insert-only hit counting: both leaves have hit_count 1 from insert;
    // matches do NOT bump hit_count in the Python implementation.
    // LFU tie -> last_access decides: [4,5,6] is older (smaller clock).
    let r = t.evict(2);
    assert_eq!(r.num_tokens_evicted, 3);
    assert_eq!(
        r.evicted_values.iter().flatten().copied().collect::<Vec<_>>(),
        vec![20, 21, 22]
    );
}
