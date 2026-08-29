//! The Mamba (hybrid full + Mamba/SSM) radix tree — a faithful port of
//! the tree semantics in `python/sglang/srt/mem_cache/mamba_radix_cache.py`
//! (`MambaRadixCache`).
//!
//! Differences from the Python implementation, deliberately:
//! - The two LRU lists are intrusive linked lists over `NodeId`s (same
//!   design as `SWARadixTree`; see `lru.rs`). The FULL list holds every
//!   non-root node — mamba tombstones included, since their full KV is
//!   still live; the MAMBA list holds only nodes with a live
//!   `mamba_value`.
//! - The allocator does not live inside the tree. `free_segment(run,
//!   start_pos)` calls come back as `(run, start_pos)` pairs
//!   (`MambaFreeOps.kv`) and `_free_mamba_value` calls as runs
//!   (`MambaFreeOps.mamba`), both in call order. Which mamba pool a
//!   freed value goes to (active vs. int8 checkpoint) is a caller-side
//!   routing decision, as in Python's `_free_mamba_value`.
//! - The deferred-COW part of `_match_post_processor` (`cow_mamba` +
//!   the request-side slot allocation/evict/lock dance) is a request
//!   sidecar and stays on the caller; the tree reports
//!   `mamba_branching_seqlen`, the only match-result field the Python
//!   tree computes from tree state.
//! - The int8 checkpoint pool, the ping-pong extra buffer, ReplaySSM
//!   cursor math and KV event recording are request/pool-side
//!   machinery, not tree state.
//!
//! Lock model (invariant: `full_lock_ref >= mamba_lock_ref` on every
//! node): `inc_lock_ref(node)` locks the FULL refs on `[node, root)`
//! (exclusive) and the MAMBA ref on `node` alone, when it holds a
//! mamba value. A mamba state is one pool slot per node (Python
//! asserts `len(mamba_value) == 1`), so the mamba counters count
//! states, not tokens.

use std::collections::HashMap;

use crate::key::{common_prefix_len, RadixKey};
use crate::lru::{Lru, LRUList};
use crate::tree::{ChildKey, Head, NodeId, ROOT};

/// One Mamba tree node. `key` is flattened (one element per token, or
/// two per bigram); `value` has one KV index per logical unit.
#[derive(Debug, Clone)]
pub struct MambaNode {
    pub children: HashMap<ChildKey, NodeId>,
    pub parent: Option<NodeId>,
    pub key: Vec<i64>,
    /// `None` = the node was deleted from the tree by an evict.
    pub value: Option<Vec<i64>>,
    /// The node's mamba state slot (Python: a length-1 tensor);
    /// `None` = mamba tombstone (state evicted, full KV still live).
    pub mamba_value: Option<Vec<i64>>,
    /// Invariant: `full_lock_ref >= mamba_lock_ref` always.
    pub full_lock_ref: u32,
    pub mamba_lock_ref: u32,
    /// Sanity-check ticks (mirror the Python `last_access_time` /
    /// `mamba_last_access_time` floats); the LRU lists themselves order
    /// the eviction.
    pub last_access: u64,
    pub mamba_last_access: u64,
    pub id: NodeId,
}

/// Allocator-free bookkeeping of the runs an op wants released.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MambaFreeOps {
    /// `free_segment(run, start_pos)` on the paged KV allocator, in
    /// call order. `start_pos` is the run's offset within the request's
    /// KV row (page alignment for the paged allocator); evict paths
    /// always report 0.
    pub kv: Vec<(Vec<i64>, usize)>,
    /// `free_mamba(run)` on the mamba pool (active or int8 checkpoint —
    /// the routing is caller-side), in call order.
    pub mamba: Vec<Vec<i64>>,
}

/// Port of `match_prefix`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MambaMatchResult {
    /// The reusable KV prefix (the Python `device_indices`).
    pub indices: Vec<i64>,
    /// The node the caller locks (`best_last_node` in the Python tree).
    pub last_node: NodeId,
    /// `mamba_branching_seqlen`: the chunk-aligned total length of the
    /// whole matched run list when it extends past the last
    /// mamba-holding node, else `None`.
    pub mamba_branching_seqlen: Option<usize>,
}

/// Port of `insert`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MambaInsertResult {
    /// Logical tokens already present in the tree.
    pub prefix_len: usize,
    /// The deepest node the walk touched (the new leaf when one was
    /// created).
    pub last_node: NodeId,
    /// A mamba value already existed at the deepest node (or the key
    /// was empty): the caller must free the incoming mamba value.
    pub mamba_exist: bool,
    pub free: MambaFreeOps,
}

/// Port of `evict` (dual full/mamba budgets).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MambaEvictResult {
    /// Full KV tokens freed in the full phase, including tombstone
    /// cascades. Mamba-phase leaf deletions free full KV too, but do
    /// NOT count here (faithful to `MambaRadixCache.evict`).
    pub full_num_evicted: usize,
    /// Mamba states evicted in the mamba phase (full-phase leaf
    /// deletions free mamba states too, but they do not count here).
    pub mamba_num_evicted: usize,
    pub free: MambaFreeOps,
}

#[derive(Clone)]
pub struct MambaRadixTree {
    nodes: Vec<MambaNode>,
    page_size: usize,
    /// Flattened elements per logical unit: 1 (plain) or 2 (bigram).
    unit_size: usize,
    is_eagle: bool,
    disable: bool,
    /// `mamba_cache_chunk_size`: the granularity at which mamba states
    /// are cached (Python: `max(model chunk size, page_size)`).
    mamba_cache_chunk_size: usize,
    full_evictable_size: i64,
    mamba_evictable_size: i64,
    full_protected_size: i64,
    mamba_protected_size: i64,
    full_lru: LRUList,
    mamba_lru: LRUList,
    /// Sanity-check walk clock.
    clock: u64,
    ns_map: HashMap<(String, String), u32>,
    ns_list: Vec<(String, String)>,
}

impl MambaRadixTree {
    pub fn new(page_size: usize, is_eagle: bool, mamba_cache_chunk_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        assert!(
            mamba_cache_chunk_size >= 1,
            "mamba_cache_chunk_size must be >= 1"
        );
        let mut t = Self {
            nodes: Vec::new(),
            page_size,
            unit_size: if is_eagle { 2 } else { 1 },
            is_eagle,
            disable: false,
            mamba_cache_chunk_size,
            full_evictable_size: 0,
            mamba_evictable_size: 0,
            full_protected_size: 0,
            mamba_protected_size: 0,
            full_lru: LRUList::default(),
            mamba_lru: LRUList::default(),
            clock: 0,
            ns_map: HashMap::new(),
            ns_list: Vec::new(),
        };
        t.reset();
        t
    }

    /// `reset` — recreate the root and clear all bookkeeping.
    pub fn reset(&mut self) {
        let root = MambaNode {
            children: HashMap::new(),
            parent: None,
            key: vec![],
            value: Some(vec![]),
            mamba_value: None,
            full_lock_ref: 1,
            mamba_lock_ref: 1,
            last_access: 0,
            mamba_last_access: 0,
            id: ROOT,
        };
        self.nodes = vec![root];
        self.full_evictable_size = 0;
        self.mamba_evictable_size = 0;
        self.full_protected_size = 0;
        self.mamba_protected_size = 0;
        self.full_lru = LRUList::default();
        self.mamba_lru = LRUList::default();
        self.full_lru.grow(1);
        self.mamba_lru.grow(1);
        self.clock = 0;
        self.ns_map.clear();
        self.ns_list.clear();
    }

    pub fn with_disable(mut self, disable: bool) -> Self {
        self.disable = disable;
        self
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn mamba_cache_chunk_size(&self) -> usize {
        self.mamba_cache_chunk_size
    }

    pub fn full_evictable_size(&self) -> i64 {
        self.full_evictable_size
    }

    pub fn mamba_evictable_size(&self) -> i64 {
        self.mamba_evictable_size
    }

    pub fn full_protected_size(&self) -> i64 {
        self.full_protected_size
    }

    pub fn mamba_protected_size(&self) -> i64 {
        self.mamba_protected_size
    }

    /// Port of `total_size` / `_total_size_helper`: `(full, mamba)`.
    pub fn total_size(&self) -> (i64, i64) {
        let mut full = 0i64;
        let mut mamba = 0i64;
        let mut stack = vec![ROOT];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id as usize];
            if let Some(v) = &n.value {
                full += v.len() as i64;
            }
            if n.mamba_value.is_some() {
                mamba += 1;
            }
            for &child in n.children.values() {
                if self.nodes[child as usize].value.is_some() {
                    stack.push(child);
                }
            }
        }
        (full, mamba)
    }

    /// Live children of a node (debug/parity tooling).
    pub fn node_children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .get(id as usize)
            .map(|n| n.children.values().copied().collect())
            .unwrap_or_default()
    }

    /// `true` when the node's mamba state was evicted (tombstone).
    pub fn node_mamba_tombstone(&self, id: NodeId) -> bool {
        self.nodes
            .get(id as usize)
            .is_some_and(|n| n.mamba_value.is_none())
    }

    pub fn node_mamba_value(&self, id: NodeId) -> Option<Vec<i64>> {
        self.nodes
            .get(id as usize)
            .and_then(|n| n.mamba_value.clone())
    }

    pub fn node_full_lock_ref(&self, id: NodeId) -> u32 {
        self.nodes
            .get(id as usize)
            .map(|n| n.full_lock_ref)
            .unwrap_or(u32::MAX)
    }

    pub fn node_mamba_lock_ref(&self, id: NodeId) -> u32 {
        self.nodes
            .get(id as usize)
            .map(|n| n.mamba_lock_ref)
            .unwrap_or(u32::MAX)
    }

    /// Port of `match_prefix` (`_match_prefix_helper` +
    /// `_match_post_processor` folded in; the deferred COW is
    /// caller-side).
    pub fn match_prefix(&mut self, key: &RadixKey) -> MambaMatchResult {
        let is_bigram = if self.is_eagle {
            true
        } else {
            key.is_bigram
        };
        let key = RadixKey {
            is_bigram,
            ..*key
        };
        if self.disable || key.logical_len() == 0 {
            return MambaMatchResult::default();
        }
        let flat = key.flatten_page_aligned(self.page_size);
        if flat.is_empty() {
            return MambaMatchResult::default();
        }

        // `_match_prefix_helper`: `best_value_len` is a RUN count, not
        // tokens — the Python tree stores `len(value)` (the per-node
        // tensor list) and the post-processor slices it.
        let mut node = ROOT;
        let mut value: Vec<Vec<i64>> = Vec::new();
        let mut best_value_len = 0usize;
        let mut best_last_node = ROOT;
        let mut remaining: &[i64] = &flat;
        let mut child_key = self.child_key_for(&key, remaining);
        while !remaining.is_empty() {
            let Some(&child) = self.nodes[node as usize].children.get(&child_key) else {
                break;
            };
            // The Python helper checks the CURRENT node (before
            // descending into the child): a mamba-holding node becomes
            // the best candidate with the runs accumulated so far.
            if self.nodes[node as usize].mamba_value.is_some() {
                best_value_len = value.len();
                best_last_node = node;
            }
            let c = &self.nodes[child as usize];
            let m_flat = common_prefix_len(remaining, &c.key);
            let m = m_flat / self.unit_size;
            let child_logical = c.key.len() / self.unit_size;
            if m < child_logical {
                // Match ends inside the child: split, take the front
                // half (always a mamba tombstone).
                let front = self.split_node(child, m * self.unit_size);
                value.push(
                    self.nodes[front as usize]
                        .value
                        .clone()
                        .unwrap_or_default(),
                );
                node = front;
                break;
            } else {
                value.push(c.value.clone().unwrap_or_default());
                node = child;
                remaining = &remaining[m * self.unit_size..];
                if !remaining.is_empty() {
                    child_key = self.child_key_for(&key, remaining);
                }
            }
        }
        // The case where the last node is fully matched.
        if self.nodes[node as usize].mamba_value.is_some() {
            best_value_len = value.len();
            best_last_node = node;
        }

        // `_match_post_processor`: the FULL LRU refreshes the whole
        // matched chain; the MAMBA LRU refreshes only the single state
        // actually consumed (`best_last_node` itself).
        self.full_chain_mru(best_last_node, ROOT);
        if best_last_node != ROOT
            && self.nodes[best_last_node as usize].mamba_value.is_some()
        {
            self.mamba_reset_mru(best_last_node);
        }
        // Re-stamp `last_access_time` for every node on the chain,
        // deepest first (sanity-check only).
        let mut tick = self.tick();
        let mut cur = Some(best_last_node);
        while let Some(id) = cur {
            self.nodes[id as usize].last_access = tick;
            tick = tick.saturating_sub(1);
            cur = self.nodes[id as usize].parent;
        }

        // The branching point: the last chunk-aligned position that does
        // not have a mamba value, when the match extends past it.
        let mamba_branching_seqlen = if value.len() > best_value_len {
            let total: usize = value.iter().map(|v| v.len()).sum();
            let chunk = self.mamba_cache_chunk_size;
            let chunk_aligned = total / chunk * chunk;
            if chunk_aligned > 0 {
                Some(chunk_aligned)
            } else {
                None
            }
        } else {
            None
        };

        // `value[:best_value_len]` in the Python post-processor slices
        // the per-NODE run list (not a token slice).
        let mut indices = Vec::with_capacity(best_value_len);
        for run in value.iter().take(best_value_len) {
            indices.extend_from_slice(run);
        }
        MambaMatchResult {
            indices,
            last_node: best_last_node,
            mamba_branching_seqlen,
        }
    }

    /// Port of `insert` + `_insert_helper`.
    pub fn insert(
        &mut self,
        key: &RadixKey,
        value: &[i64],
        mamba_value: &[i64],
        prev_prefix_len: usize,
    ) -> MambaInsertResult {
        let mut res = MambaInsertResult::default();
        if self.disable {
            return res;
        }
        let is_bigram = if self.is_eagle {
            true
        } else {
            key.is_bigram
        };
        let key = RadixKey {
            is_bigram,
            ..*key
        };
        let flat = key.flatten_page_aligned(self.page_size);
        if flat.is_empty() {
            // Python `_insert_helper` returns `(0, True)` for an empty
            // key: nothing was inserted, the caller frees the incoming
            // mamba value.
            res.mamba_exist = true;
            return res;
        }
        let logical = flat.len() / self.unit_size;
        let value = &value[..logical.min(value.len())];
        debug_assert_eq!(
            value.len(),
            logical,
            "insert value must have one KV index per logical unit"
        );
        let ns = self.intern_ns(key.extra_key, key.cache_salt);

        let (prefix_len, last_node, mamba_exist) = self.insert_helper(
            ns,
            key.is_bigram,
            &flat,
            value,
            mamba_value,
            prev_prefix_len,
            &mut res.free,
        );
        res.prefix_len = prefix_len;
        res.last_node = last_node;
        res.mamba_exist = mamba_exist;
        res
    }

    /// Port of `_insert_helper`. Returns
    /// `(total_prefix_length, deepest_node, mamba_value_exist)`.
    fn insert_helper(
        &mut self,
        ns: u32,
        is_bigram: bool,
        flat: &[i64],
        value: &[i64],
        mamba_value: &[i64],
        prev_prefix_len: usize,
        free: &mut MambaFreeOps,
    ) -> (usize, NodeId, bool) {
        // Refresh the full LRU for the start of the walk; the mamba
        // states of existing nodes were not recomputed by this insert,
        // so the mamba LRU is left untouched here.
        self.nodes[ROOT as usize].last_access = self.tick();
        let mut remaining: &[i64] = flat;
        let mut val: &[i64] = value;
        if remaining.is_empty() {
            return (0, ROOT, true);
        }

        let child_key = |t: &mut Self, remaining: &[i64]| -> ChildKey {
            ChildKey {
                ns,
                head: t.head_from_flat(remaining, is_bigram),
            }
        };

        let mut node = ROOT;
        let mut total_prefix_length = 0usize;
        while !remaining.is_empty() {
            let ck = child_key(self, remaining);
            let Some(&child) = self.nodes[node as usize].children.get(&ck) else {
                break;
            };
            node = child;
            self.nodes[node as usize].last_access = self.tick();
            self.full_lru.reset_mru(node);

            let m_flat = common_prefix_len(remaining, &self.nodes[node as usize].key);
            let prefix_len = m_flat / self.unit_size;

            if prev_prefix_len < total_prefix_length + prefix_len {
                // `value` sits at offset `total_prefix_length` of the
                // KV row; match() rounds to page multiples, so frees
                // never share a page.
                let start = prev_prefix_len.saturating_sub(total_prefix_length);
                free.kv
                    .push((val[start..prefix_len].to_vec(), total_prefix_length + start));
            }

            total_prefix_length += prefix_len;
            remaining = &remaining[prefix_len * self.unit_size..];
            val = &val[prefix_len..];

            if prefix_len < self.nodes[node as usize].key.len() / self.unit_size {
                let front = self.split_node(node, prefix_len * self.unit_size);
                node = front;
            }
        }

        let mut mamba_value_exist = false;
        if !remaining.is_empty() {
            node = self.add_new_node(node, ns, is_bigram, remaining, val, mamba_value);
        } else if self.nodes[node as usize].mamba_value.is_none() {
            // Revive a mamba tombstone: attach the incoming state. The
            // full KV already counts, only the mamba side changes.
            self.nodes[node as usize].mamba_value = Some(mamba_value.to_vec());
            self.full_lru.reset_mru(node);
            self.mamba_insert_mru(node);
            self.mamba_evictable_size += mamba_value.len() as i64;
            self.nodes[node as usize].last_access = self.tick();
        } else {
            // The mamba value already exists: the caller frees the
            // incoming one.
            mamba_value_exist = true;
            self.full_lru.reset_mru(node);
            self.nodes[node as usize].last_access = self.tick();
        }

        (total_prefix_length, node, mamba_value_exist)
    }

    /// Port of `evict` (dual full/mamba budgets, two phases).
    pub fn evict(&mut self, full_num_tokens: usize, mamba_num: usize) -> MambaEvictResult {
        let mut res = MambaEvictResult::default();
        if self.disable {
            return res;
        }
        let mut full_num_evicted = 0usize;
        let mut mamba_num_evicted = 0usize;

        // Phase 1: `evict_full` (full LRU, leaf-only).
        if full_num_tokens > 0 {
            let mut x = self.full_get_leaf_lru_no_lock();
            while full_num_evicted < full_num_tokens {
                let Some(xid) = x else {
                    break;
                };
                assert!(
                    xid != ROOT,
                    "root node should not exist in the full lru list"
                );
                let (fe, _, x_after, x_next) =
                    self.evict_leaf_node(xid, false, &mut res.free);
                full_num_evicted += fe;

                // If the parent has no more children it is a leaf now
                // and may itself be the LRU: restart the scan.
                let parent = self.nodes[x_after as usize].parent.unwrap_or(ROOT);
                if self.nodes[parent as usize].children.is_empty() {
                    x = self.full_get_leaf_lru_no_lock();
                } else {
                    x = x_next;
                }
            }
        }

        // Phase 2: `evict_mamba` (mamba LRU, any node) with its OWN
        // budget — full-side leaf deletions free mamba states too, but
        // they do not count toward the mamba budget (faithful to
        // `MambaRadixCache.evict`).
        if mamba_num > 0 {
            let mut x = self.mamba_get_lru_no_lock();
            let mut m_evicted = 0usize;
            while m_evicted < mamba_num {
                let Some(xid) = x else {
                    break;
                };
                let mval = self
                    .nodes[xid as usize]
                    .mamba_value
                    .clone()
                    .unwrap_or_default();
                assert!(!mval.is_empty(), "node has no mamba value, id={xid}");
                assert_eq!(mval.len(), 1, "node has abnormal mamba length, id={xid}");
                assert!(xid != ROOT, "root node is not evictable");
                assert!(
                    self.nodes[xid as usize].mamba_lock_ref == 0,
                    "node is in use by mamba kv indices, id={xid}"
                );

                if !self.nodes[xid as usize].children.is_empty() {
                    // Internal node: free the mamba state only, keep
                    // the full KV.
                    res.free.mamba.push(mval);
                    m_evicted += 1;
                    let x_next = self.mamba_next_no_lock(xid);
                    self.mamba_lru.remove(xid);
                    self.tombstone_internal_node(xid);
                    x = x_next;
                } else {
                    // Leaf: free full + mamba, delete (the cascade's
                    // full tokens are NOT counted — faithful).
                    let (_, me, _, x_next) = self.evict_leaf_node(xid, true, &mut res.free);
                    m_evicted += me;
                    x = x_next;
                }
            }
            mamba_num_evicted = m_evicted;
        }

        res.full_num_evicted = full_num_evicted;
        res.mamba_num_evicted = mamba_num_evicted;
        res
    }

    /// Port of `inc_lock_ref`: full lock on `[node, root)` exclusive;
    /// mamba lock on `node` alone, when it holds a state. Returns
    /// `(full_delta, mamba_delta)` — units moved evictable ->
    /// protected (<= 0).
    pub fn inc_lock_ref(&mut self, node: NodeId) -> (i64, i64) {
        if self.disable {
            return (0, 0);
        }
        let mut full_delta = 0i64;
        let mut mamba_delta = 0i64;
        if self.nodes[node as usize].mamba_value.is_some() {
            if self.nodes[node as usize].mamba_lock_ref == 0 {
                let l = self.nodes[node as usize]
                    .mamba_value
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0) as i64;
                self.mamba_evictable_size -= l;
                self.mamba_protected_size += l;
                mamba_delta -= l;
            }
            self.nodes[node as usize].mamba_lock_ref += 1;
        }
        let mut cur = node;
        while cur != ROOT {
            {
                let n = &mut self.nodes[cur as usize];
                if n.full_lock_ref == 0 {
                    let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0) as i64;
                    self.full_evictable_size -= l;
                    self.full_protected_size += l;
                    full_delta -= l;
                }
                n.full_lock_ref += 1;
            }
            cur = self.nodes[cur as usize].parent.expect("walk reaches root");
        }
        (full_delta, mamba_delta)
    }

    /// Port of `dec_lock_ref`. Returns `(full_delta, mamba_delta)`
    /// (>= 0).
    pub fn dec_lock_ref(&mut self, node: NodeId) -> (i64, i64) {
        if self.disable {
            return (0, 0);
        }
        let mut full_delta = 0i64;
        let mut mamba_delta = 0i64;
        if self.nodes[node as usize].mamba_value.is_some() {
            assert!(
                self.nodes[node as usize].mamba_lock_ref > 0,
                "dec_lock_ref on node {node} with mamba_lock_ref=0"
            );
            if self.nodes[node as usize].mamba_lock_ref == 1 {
                let l = self.nodes[node as usize]
                    .mamba_value
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0) as i64;
                self.mamba_evictable_size += l;
                self.mamba_protected_size -= l;
                mamba_delta += l;
            }
            self.nodes[node as usize].mamba_lock_ref -= 1;
        }
        let mut cur = node;
        while cur != ROOT {
            {
                let n = &mut self.nodes[cur as usize];
                assert!(
                    n.full_lock_ref > 0,
                    "dec_lock_ref on node {cur} with full_lock_ref=0"
                );
                if n.full_lock_ref == 1 {
                    let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0) as i64;
                    self.full_evictable_size += l;
                    self.full_protected_size -= l;
                    full_delta += l;
                }
                n.full_lock_ref -= 1;
            }
            cur = self.nodes[cur as usize].parent.expect("walk reaches root");
        }
        (full_delta, mamba_delta)
    }

    // ---- internals ----

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// `reset_node_and_parents_mru` on the FULL list (no skips: every
    /// non-root node is a member, mamba tombstones included).
    fn full_chain_mru(&mut self, mut node: NodeId, root: NodeId) {
        let mut prev_node = Lru::Head;
        while node != root {
            debug_assert!(
                self.full_lru.in_list(node),
                "full_chain_mru: node {node} not in the full list"
            );
            self.full_lru.remove(node);
            self.full_lru.add_after(prev_node, node);
            prev_node = Lru::Node(node);
            node = self.nodes[node as usize]
                .parent
                .expect("walk reaches the root");
        }
    }

    /// `reset_node_mru` on the mamba list, with the mamba-side access
    /// stamp (the Python `LRUList` stamps `mamba_last_access_time` on
    /// these two ops for the mamba list).
    fn mamba_reset_mru(&mut self, id: NodeId) {
        if !self.mamba_lru.in_list(id) {
            return;
        }
        self.nodes[id as usize].mamba_last_access = self.tick();
        self.mamba_lru.reset_mru(id);
    }

    /// `insert_mru` on the mamba list, with the mamba-side access stamp.
    fn mamba_insert_mru(&mut self, id: NodeId) {
        debug_assert!(
            !self.mamba_lru.in_list(id),
            "mamba_insert_mru: node {id} already in the mamba list"
        );
        debug_assert!(
            self.nodes[id as usize].mamba_value.is_some(),
            "inserting a mamba tombstone node in the mamba lru list: {id}"
        );
        self.nodes[id as usize].mamba_last_access = self.tick();
        self.mamba_lru.insert_mru(id);
    }

    /// Port of `get_lru_no_lock` on the mamba list.
    fn mamba_get_lru_no_lock(&self) -> Option<NodeId> {
        let mut x = self.mamba_lru.predecessor(Lru::Tail);
        while let Lru::Node(id) = x {
            if self.nodes[id as usize].mamba_lock_ref == 0 {
                return Some(id);
            }
            x = self.mamba_lru.predecessor(Lru::Node(id));
        }
        None
    }

    /// Port of `get_prev_no_lock(from)` on the mamba list.
    fn mamba_next_no_lock(&self, from: NodeId) -> Option<NodeId> {
        let mut x = self.mamba_lru.predecessor(Lru::Node(from));
        while let Lru::Node(id) = x {
            if self.nodes[id as usize].mamba_lock_ref == 0 {
                return Some(id);
            }
            x = self.mamba_lru.predecessor(Lru::Node(id));
        }
        None
    }

    /// Port of `get_leaf_lru_no_lock` on the full list.
    fn full_get_leaf_lru_no_lock(&self) -> Option<NodeId> {
        let mut x = self.full_lru.predecessor(Lru::Tail);
        while let Lru::Node(id) = x {
            let n = &self.nodes[id as usize];
            if n.full_lock_ref == 0 && n.children.is_empty() {
                return Some(id);
            }
            x = self.full_lru.predecessor(Lru::Node(id));
        }
        None
    }

    /// Port of `get_prev_leaf_no_lock(from)`: walk the full list from
    /// `from`'s predecessor (more recent) toward the head.
    fn full_next_leaf_no_lock(&self, from: NodeId) -> Option<NodeId> {
        let mut x = self.full_lru.predecessor(Lru::Node(from));
        while let Lru::Node(id) = x {
            let n = &self.nodes[id as usize];
            if n.full_lock_ref == 0 && n.children.is_empty() {
                return Some(id);
            }
            x = self.full_lru.predecessor(Lru::Node(id));
        }
        None
    }

    /// Port of `_evict_leaf_node`. Returns `(full_evicted,
    /// mamba_evicted, x_after_cascade, x_next)`.
    fn evict_leaf_node(
        &mut self,
        xid: NodeId,
        is_evict_mamba: bool,
        free: &mut MambaFreeOps,
    ) -> (usize, usize, NodeId, Option<NodeId>) {
        assert!(
            self.nodes[xid as usize].full_lock_ref == 0
                && self.nodes[xid as usize].mamba_lock_ref == 0,
            "evict leaf node invalid with id={xid} full={} mamba={}",
            self.nodes[xid as usize].full_lock_ref,
            self.nodes[xid as usize].mamba_lock_ref
        );
        assert!(
            self.nodes[xid as usize].mamba_value.is_some(),
            "leaf node mamba value is None, id={xid}"
        );

        // 1. free the full KV and the mamba state.
        let kv = self
            .nodes[xid as usize]
            .value
            .clone()
            .unwrap_or_default();
        let mval = self
            .nodes[xid as usize]
            .mamba_value
            .clone()
            .unwrap_or_default();
        free.kv.push((kv.clone(), 0));
        free.mamba.push(mval.clone());
        let full_num_evicted = kv.len();
        let mamba_num_evicted = mval.len();

        // 2. the next candidate, before detaching the node.
        let x_next = if is_evict_mamba {
            self.mamba_next_no_lock(xid)
        } else {
            self.full_next_leaf_no_lock(xid)
        };
        self.full_lru.remove(xid);
        self.mamba_lru.remove(xid);

        // 3. delete the leaf.
        self.delete_leaf(xid);

        // 4. iteratively delete tombstone parents that lost their last
        //    child (invariant: leaf nodes are not tombstones).
        let (x_after, leaf_full) = self.iteratively_delete_tombstone_leaf(xid, free);
        (full_num_evicted + leaf_full, mamba_num_evicted, x_after, x_next)
    }

    /// Port of `_add_new_node` (the insert new-leaf creation).
    fn add_new_node(
        &mut self,
        parent: NodeId,
        ns: u32,
        is_bigram: bool,
        key: &[i64],
        value: &[i64],
        mamba_value: &[i64],
    ) -> NodeId {
        assert!(!key.is_empty(), "key should not be empty");
        let new_id = self.new_node_slot();
        let new_last_access = self.tick();
        let new_mamba_last_access = self.tick();
        {
            let n = &mut self.nodes[new_id as usize];
            n.parent = Some(parent);
            n.key = key.to_vec();
            n.value = Some(value.to_vec());
            n.mamba_value = Some(mamba_value.to_vec());
            n.last_access = new_last_access;
            n.mamba_last_access = new_mamba_last_access;
        }
        let ck = ChildKey {
            ns,
            head: self.head_from_flat(key, is_bigram),
        };
        self.nodes[parent as usize].children.insert(ck, new_id);
        self.full_lru.insert_mru(new_id);
        self.mamba_lru.insert_mru(new_id);
        self.full_evictable_size += value.len() as i64;
        self.mamba_evictable_size += mamba_value.len() as i64;
        new_id
    }

    /// Port of `_split_node(key, child, split_len)`: split `child`
    /// after `split_flat` flattened elements. The new front node is
    /// ALWAYS a mamba tombstone ("mamba cache can not be split") and
    /// carries the full lock ref with zero mamba lock; the `child`
    /// record is mutated in place into the tail (as in the Python
    /// tree, where the original node object becomes the tail). Only
    /// the FULL LRU reorders — the mamba LRU is left untouched because
    /// the live state (the tail's) is unchanged.
    fn split_node(&mut self, child: NodeId, split_flat: usize) -> NodeId {
        let (child_key, gp, full_lock_ref) = {
            let c = &self.nodes[child as usize];
            (
                c.key.clone(),
                c.parent.expect("split target must have a parent"),
                c.full_lock_ref,
            )
        };
        let ns = self
            .nodes[gp as usize]
            .children
            .iter()
            .find(|item| item.1 == &child)
            .map(|(k, _)| k.ns)
            .expect("split target missing from parent's children");

        let is_bigram = self.unit_size == 2;
        let front_key = child_key[..split_flat].to_vec();
        let tail_key = child_key[split_flat..].to_vec();
        let split_logical = split_flat / self.unit_size;
        let (front_value, tail_value) = {
            let v = self
                .nodes[child as usize]
                .value
                .clone()
                .expect("live node has a value");
            (v[..split_logical].to_vec(), v[split_logical..].to_vec())
        };
        // Pre-compute the heads and the re-parent entries so the
        // children-map mutations below hold no live helper borrows.
        let front_ck = ChildKey {
            ns,
            head: self.head_from_flat(&front_key, is_bigram),
        };
        let tail_ck = ChildKey {
            ns,
            head: self.head_from_flat(&tail_key, is_bigram),
        };
        let old_ck = ChildKey {
            ns,
            head: self.head_from_flat(&child_key, is_bigram),
        };

        // Detach the child from the FULL list only; the mamba list
        // keeps it in place (its state and stamp are unchanged).
        self.full_lru.remove(child);

        let new_id = self.new_node_slot();
        let new_last_access = self.tick();
        // The tail's stamp must be fresher than the front's (the
        // Python split re-stamps `child.last_access_time` after
        // construction).
        let tail_last_access = self.tick();
        {
            let mut children = HashMap::with_capacity(1);
            children.insert(tail_ck, child);
            let n = &mut self.nodes[new_id as usize];
            *n = MambaNode {
                children,
                parent: Some(gp),
                key: front_key,
                value: Some(front_value),
                mamba_value: None, // mamba cache can not be split
                full_lock_ref,
                mamba_lock_ref: 0,
                last_access: new_last_access,
                mamba_last_access: 0,
                id: new_id,
            };
        }
        // Mutate the tail node in place; it keeps its mamba state and
        // locks and gets a fresher stamp than the front node.
        {
            let c = &mut self.nodes[child as usize];
            c.key = tail_key;
            c.value = Some(tail_value);
            c.parent = Some(new_id);
            c.last_access = tail_last_access;
        }
        // Re-parent at the grandparent.
        {
            let g = &mut self.nodes[gp as usize];
            let removed = g.children.remove(&old_ck);
            assert_eq!(removed, Some(child), "split: parent lost its child");
            g.children.insert(front_ck, new_id);
        }

        // Parent first so that the child ends up more recently used.
        self.full_lru.insert_mru(new_id);
        self.full_lru.insert_mru(child);
        new_id
    }

    /// Port of `_delete_leaf` (the node holds a live mamba state).
    fn delete_leaf(&mut self, node: NodeId) {
        assert!(
            self.nodes[node as usize].children.is_empty(),
            "leaf node has children, id={node}"
        );
        assert!(
            self.nodes[node as usize].mamba_value.is_some(),
            "invariant violated: leaf node is a tombstone, id={node}"
        );
        let key = self.nodes[node as usize].key.clone();
        let parent = self.nodes[node as usize].parent.unwrap_or(ROOT);
        let ns = self
            .nodes[parent as usize]
            .children
            .iter()
            .find(|item| item.1 == &node)
            .map(|(k, _)| k.ns)
            .expect("leaf missing from parent's children");
        let ck = ChildKey {
            ns,
            head: self.head_from_flat(&key, self.unit_size == 2),
        };
        let removed = self.nodes[parent as usize].children.remove(&ck);
        assert_eq!(removed, Some(node), "parent does not have child key");
        let logical = key.len() as i64 / self.unit_size as i64;
        self.full_evictable_size -= logical;
        let mamba_len = self.nodes[node as usize]
            .mamba_value
            .as_ref()
            .map(|v| v.len() as i64)
            .unwrap_or(0);
        self.mamba_evictable_size -= mamba_len;
    }

    /// Port of `_tombstone_internal_node`.
    fn tombstone_internal_node(&mut self, node: NodeId) {
        assert!(
            !self.nodes[node as usize].children.is_empty(),
            "cannot tombstone a leaf node, id={node}"
        );
        let l = self.nodes[node as usize]
            .mamba_value
            .as_ref()
            .map(|v| v.len() as i64)
            .unwrap_or(0);
        self.mamba_evictable_size -= l;
        self.nodes[node as usize].mamba_value = None;
    }

    /// Port of `_delete_tombstone_leaf`.
    fn delete_tombstone_leaf(&mut self, node: NodeId) {
        assert!(
            self.nodes[node as usize].mamba_value.is_none(),
            "deleting an unexpected non-tombstone leaf, id={node}"
        );
        assert!(
            self.nodes[node as usize].children.is_empty(),
            "leaf node has children, id={node}"
        );
        let key = self.nodes[node as usize].key.clone();
        let parent = self.nodes[node as usize].parent.unwrap_or(ROOT);
        let ns = self
            .nodes[parent as usize]
            .children
            .iter()
            .find(|item| item.1 == &node)
            .map(|(k, _)| k.ns)
            .expect("leaf missing from parent's children");
        let ck = ChildKey {
            ns,
            head: self.head_from_flat(&key, self.unit_size == 2),
        };
        let removed = self.nodes[parent as usize].children.remove(&ck);
        assert_eq!(removed, Some(node), "parent does not have child key");
        self.full_evictable_size -= key.len() as i64 / self.unit_size as i64;
    }

    /// Port of `_iteratively_delete_tombstone_leaf`. Returns the new
    /// position (`node`) and the full tokens evicted by the cascade.
    fn iteratively_delete_tombstone_leaf(
        &mut self,
        mut node: NodeId,
        free: &mut MambaFreeOps,
    ) -> (NodeId, usize) {
        let mut full_num_evicted = 0usize;
        loop {
            let Some(parent) = self.nodes[node as usize].parent else {
                break;
            };
            let p = &self.nodes[parent as usize];
            if p.mamba_value.is_some() || !p.children.is_empty() {
                break;
            }
            // The root node is not evictable.
            if parent == ROOT {
                break;
            }
            // If locked, the node is in use: skip.
            if p.full_lock_ref > 0 {
                break;
            }
            assert!(
                p.mamba_lock_ref == 0,
                "tombstone mamba_lock_ref should always be 0, full={}",
                p.full_lock_ref
            );
            let kv = p.value.clone().unwrap_or_default();
            free.kv.push((kv.clone(), 0));
            full_num_evicted += kv.len();
            self.full_lru.remove(parent);
            self.delete_tombstone_leaf(parent);
            node = parent;
        }
        (node, full_num_evicted)
    }

    fn intern_ns(&mut self, extra_key: Option<&str>, cache_salt: Option<&str>) -> u32 {
        let k = (
            extra_key.unwrap_or("").to_string(),
            cache_salt.unwrap_or("").to_string(),
        );
        match self.ns_map.entry(k) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let id = self.ns_list.len() as u32;
                self.ns_list.push(v.key().clone());
                *v.insert(id)
            }
        }
    }

    fn head_from_flat(&self, flat: &[i64], is_bigram: bool) -> Head {
        if self.page_size == 1 {
            if is_bigram {
                Head::Bigram(flat[0], flat[1])
            } else {
                Head::Token(flat[0])
            }
        } else {
            let n = (self.page_size * self.unit_size).min(flat.len());
            Head::Tokens(flat[..n].to_vec())
        }
    }

    fn child_key_for(&mut self, key: &RadixKey, remaining: &[i64]) -> ChildKey {
        let ns = self.intern_ns(key.extra_key, key.cache_salt);
        ChildKey {
            ns,
            head: self.head_from_flat(remaining, key.is_bigram),
        }
    }

    /// Push a fresh node record and grow the LRU arrays with it.
    fn new_node_slot(&mut self) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(MambaNode {
            children: HashMap::new(),
            parent: None,
            key: vec![],
            value: Some(vec![]),
            mamba_value: None,
            full_lock_ref: 0,
            mamba_lock_ref: 0,
            last_access: 0,
            mamba_last_access: 0,
            id,
        });
        self.full_lru.grow(self.nodes.len());
        self.mamba_lru.grow(self.nodes.len());
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(n: usize, off: i64) -> Vec<i64> {
        (0..n as i64).map(|i| i + off).collect()
    }

    #[test]
    fn insert_match_lock_basic() {
        let mut t = MambaRadixTree::new(1, false, 64);
        let ids: Vec<i64> = (0..10).collect();
        let r = t.insert(&RadixKey::new(&ids), &vals(10, 1000), &[77], 0);
        assert_eq!(r.prefix_len, 0);
        assert!(!r.mamba_exist);
        assert_eq!(t.full_evictable_size(), 10);
        assert_eq!(t.mamba_evictable_size(), 1);

        let m = t.match_prefix(&RadixKey::new(&ids));
        assert_eq!(m.indices.len(), 10);
        assert_eq!(m.last_node, r.last_node);
        assert_eq!(m.mamba_branching_seqlen, None);
        // match does not lock
        assert_eq!(t.full_protected_size(), 0);
        assert_eq!(t.mamba_protected_size(), 0);

        let (fd, md) = t.inc_lock_ref(r.last_node);
        assert_eq!(fd, -10);
        assert_eq!(md, -1);
        assert_eq!(t.full_protected_size(), 10);
        assert_eq!(t.mamba_protected_size(), 1);
        assert_eq!(t.node_full_lock_ref(r.last_node), 1);
        assert_eq!(t.node_mamba_lock_ref(r.last_node), 1);
        assert_eq!(t.dec_lock_ref(r.last_node), (10, 1));
        assert_eq!(t.full_protected_size(), 0);
        assert_eq!(t.mamba_protected_size(), 0);
    }

    #[test]
    fn evict_full_leaf_overfills_to_node_boundary() {
        let mut t = MambaRadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..10).collect();
        let b: Vec<i64> = (100..110).collect();
        t.insert(&RadixKey::new(&a), &vals(10, 1000), &[1], 0);
        t.insert(&RadixKey::new(&b), &vals(10, 2000), &[2], 0);

        // Full-side evict of 5: the LRU leaf is A (inserted first),
        // overfilling to its 10-token boundary; the mamba state goes
        // with it but does NOT count toward the mamba budget.
        let r = t.evict(5, 0);
        assert_eq!(r.full_num_evicted, 10);
        assert_eq!(r.mamba_num_evicted, 0);
        assert_eq!(r.free.kv, vec![(vals(10, 1000), 0)]);
        assert_eq!(r.free.mamba, vec![vec![1]]);
        assert_eq!(t.full_evictable_size(), 10);
        assert_eq!(t.mamba_evictable_size(), 1);
        assert_eq!(t.node_children(ROOT).len(), 1);
    }

    #[test]
    fn evict_mamba_internal_tombstones_then_cascades() {
        // root -> A(10) -> B(10).
        let mut t = MambaRadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..10).collect();
        t.insert(&RadixKey::new(&a), &vals(10, 1000), &[1], 0);
        let ab: Vec<i64> = (0..20).collect();
        t.insert(&RadixKey::new(&ab), &vals(20, 1000), &[2], 0);

        // Mamba evict of 5: the LRU mamba node is A (internal) -> free
        // its state, tombstone it (1). The scan continues from B
        // (leaf, unlocked) -> free full+mamba, delete; deleting B
        // cascades: A is a tombstone without children and unlocked, so
        // its full KV is freed and it is deleted too.
        let r = t.evict(0, 5);
        // Phase 2 only: no full phase ran, so the cascade's 10 tokens
        // are NOT in full_num_evicted (faithful discard).
        assert_eq!(r.full_num_evicted, 0);
        assert_eq!(r.mamba_num_evicted, 2);
        assert_eq!(r.free.mamba, vec![vec![1], vec![2]]);
        // B holds the tail of the second insert's value.
        assert_eq!(
            r.free.kv,
            vec![(vals(20, 1000)[10..].to_vec(), 0), (vals(10, 1000), 0)]
        );
        assert!(t.node_children(ROOT).is_empty());
        assert_eq!(t.total_size(), (0, 0));
        assert_eq!(t.full_evictable_size(), 0);
        assert_eq!(t.mamba_evictable_size(), 0);
    }

    #[test]
    fn match_past_mamba_tombstone_keeps_full_prefix() {
        // root -> A(100, mamba) -> B(100, mamba). Evict A's state only.
        let mut t = MambaRadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..100).collect();
        t.insert(&RadixKey::new(&a), &vals(100, 1000), &[1], 0);
        let ab: Vec<i64> = (0..200).collect();
        t.insert(&RadixKey::new(&ab), &vals(200, 2000), &[2], 0);
        let r = t.evict(0, 1);
        assert_eq!(r.mamba_num_evicted, 1);
        assert!(t.node_mamba_tombstone(1)); // A
        assert!(!t.node_mamba_tombstone(2)); // B

        // Matching the full 200: best node is B (it holds a state), so
        // the whole full KV is reusable even though A's state is gone.
        let m = t.match_prefix(&RadixKey::new(&ab));
        assert_eq!(m.indices.len(), 200);
        assert_eq!(m.last_node, 2);
        assert_eq!(m.mamba_branching_seqlen, None);

        // Matching 150 tokens ends inside B: the split front is a
        // mamba tombstone and A's state is gone, so NO node on the
        // path holds a live state -> the match falls back to the root
        // (empty prefix) and reports the chunk-aligned branching point
        // of the whole 150-token run (150 // 64 * 64 = 128), where the
        // caller's forward pass will place the new checkpoint. The
        // split itself did happen: B is now B1(50, tombstone) -> B2.
        let m = t.match_prefix(&RadixKey {
            tokens: &ab[..150],
            ..RadixKey::new(&[])
        });
        assert!(m.indices.is_empty());
        assert_eq!(m.last_node, ROOT);
        assert_eq!(m.mamba_branching_seqlen, Some(128));
        // B (id 2) became the split tail; its new front (id 3) is a
        // mamba tombstone holding the state-less first 50 tokens.
        assert!(t.node_mamba_tombstone(3));
        assert!(!t.node_mamba_tombstone(2));
        assert_eq!(t.node_children(3).len(), 1);
    }

    #[test]
    fn insert_revives_mamba_tombstone() {
        // root -> A(10) -> B(10); tombstone A's state; re-insert A.
        let mut t = MambaRadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..10).collect();
        t.insert(&RadixKey::new(&a), &vals(10, 1000), &[1], 0);
        let ab: Vec<i64> = (0..20).collect();
        t.insert(&RadixKey::new(&ab), &vals(20, 1000), &[2], 0);
        t.evict(0, 1);
        assert!(t.node_mamba_tombstone(1));
        assert_eq!(t.mamba_evictable_size(), 1);

        let r = t.insert(&RadixKey::new(&a), &vals(10, 5000), &[9], 0);
        assert_eq!(r.prefix_len, 10);
        assert!(!r.mamba_exist); // the state was attached, not duplicated
        assert!(!t.node_mamba_tombstone(1));
        assert_eq!(t.node_mamba_value(1), Some(vec![9]));
        // The incoming full KV overlapped the tree: freed, not kept.
        assert_eq!(r.free.kv, vec![(vals(10, 5000), 0)]);
        assert_eq!(t.mamba_evictable_size(), 2);
        // The tree keeps ITS own A value.
        let m = t.match_prefix(&RadixKey::new(&a));
        assert_eq!(m.indices, vals(10, 1000));
    }

    #[test]
    fn insert_prev_prefix_len_partial_free() {
        let mut t = MambaRadixTree::new(1, false, 64);
        let ids: Vec<i64> = (0..10).collect();
        t.insert(&RadixKey::new(&ids), &vals(10, 1000), &[1], 0);

        // The first 4 tokens of the incoming value are already locked
        // (prev_prefix_len=4): only [4, 10) is freed, positioned at
        // start_pos 4. The mamba value already exists -> the caller
        // frees the incoming one.
        let r = t.insert(&RadixKey::new(&ids), &vals(10, 4000), &[8], 4);
        assert_eq!(r.prefix_len, 10);
        assert!(r.mamba_exist);
        assert_eq!(r.free.kv, vec![(vals(10, 4000)[4..].to_vec(), 4)]);
        assert!(r.free.mamba.is_empty());
        // The tree's own value and state are untouched.
        assert_eq!(t.node_mamba_value(1), Some(vec![1]));
    }

    #[test]
    fn insert_empty_key_reports_mamba_exist() {
        let mut t = MambaRadixTree::new(1, false, 64);
        let ids: [i64; 0] = [];
        let r = t.insert(&RadixKey::new(&ids), &[], &[3], 0);
        assert_eq!(r.prefix_len, 0);
        assert!(r.mamba_exist); // Python `_insert_helper` returns (0, True)
        assert_eq!(r.last_node, ROOT);
        assert!(r.free.kv.is_empty());
    }

    #[test]
    fn evict_full_counts_cascade() {
        // root -> A(10, tombstone) -> B(10): the full-side evict of the
        // leaf B cascades into deleting A, and (unlike the SWA phase-2
        // quirk) the cascade's full tokens ARE counted.
        let mut t = MambaRadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..10).collect();
        t.insert(&RadixKey::new(&a), &vals(10, 1000), &[1], 0);
        let ab: Vec<i64> = (0..20).collect();
        t.insert(&RadixKey::new(&ab), &vals(20, 1000), &[2], 0);
        t.evict(0, 1); // tombstone A's state
        assert!(t.node_mamba_tombstone(1));

        let r = t.evict(1, 0);
        assert_eq!(r.full_num_evicted, 20); // B(10) + cascade A(10)
        assert_eq!(r.mamba_num_evicted, 0);
        assert_eq!(
            r.free.kv,
            vec![(vals(20, 1000)[10..].to_vec(), 0), (vals(10, 1000), 0)]
        );
        assert_eq!(r.free.mamba, vec![vec![2]]);
        assert!(t.node_children(ROOT).is_empty());
        assert_eq!(t.total_size(), (0, 0));
    }

    #[test]
    fn split_front_node_is_mamba_tombstone_with_full_lock() {
        let mut t = MambaRadixTree::new(1, false, 64);
        let ids: Vec<i64> = (0..20).collect();
        let leaf = t.insert(&RadixKey::new(&ids), &vals(20, 1000), &[5], 0).last_node;
        let (fd, md) = t.inc_lock_ref(leaf);
        assert_eq!((fd, md), (-20, -1));

        // Match the first 10 tokens: splits the node at 10. The front
        // keeps the full lock ref, loses the mamba lock and state; the
        // tail keeps the mamba lock and state. The split front is a
        // mamba tombstone and the root holds no state, so the match
        // itself falls back to the root (empty prefix).
        let m = t.match_prefix(&RadixKey {
            tokens: &ids[..10],
            ..RadixKey::new(&[])
        });
        assert!(m.indices.is_empty());
        assert_eq!(m.last_node, ROOT);
        let front = t.node_children(ROOT).pop().unwrap();
        let tail = t.node_children(front).pop().unwrap();
        assert_eq!(front, 2); // the new front node
        assert_eq!(tail, leaf); // the original leaf became the tail
        assert!(t.node_mamba_tombstone(front));
        assert_eq!(t.node_full_lock_ref(front), 1);
        assert_eq!(t.node_mamba_lock_ref(front), 0);
        assert!(!t.node_mamba_tombstone(tail));
        assert_eq!(t.node_full_lock_ref(tail), 1);
        assert_eq!(t.node_mamba_lock_ref(tail), 1);
        assert_eq!(t.node_mamba_value(tail), Some(vec![5]));
        assert_eq!(t.full_protected_size(), 20);
        assert_eq!(t.mamba_protected_size(), 1);

        // Release from the original leaf id (now the tail).
        assert_eq!(t.dec_lock_ref(leaf), (20, 1));
    }

    #[test]
    fn mamba_evict_respects_locks() {
        let mut t = MambaRadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..10).collect();
        let b: Vec<i64> = (100..110).collect();
        let la = t.insert(&RadixKey::new(&a), &vals(10, 1000), &[1], 0).last_node;
        let lb = t.insert(&RadixKey::new(&b), &vals(10, 2000), &[2], 0).last_node;
        t.inc_lock_ref(la); // lock A (LRU)

        // Mamba evict of 10: A is locked -> skip; B is free -> evict
        // both its states... one state per node, so exactly 1.
        let r = t.evict(0, 10);
        assert_eq!(r.mamba_num_evicted, 1);
        assert_eq!(r.free.mamba, vec![vec![2]]);
        assert!(!t.node_mamba_tombstone(la));
        assert!(t.node_children(ROOT).len() == 1);

        t.dec_lock_ref(la);
        let r = t.evict(0, 10);
        assert_eq!(r.mamba_num_evicted, 1);
        assert_eq!(r.free.mamba, vec![vec![1]]);
        assert_eq!(r.free.kv, vec![(vals(10, 1000), 0)]);
        assert_eq!(t.total_size(), (0, 0));
    }
}
