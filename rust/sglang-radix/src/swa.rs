//! The SWA (sliding-window-attention) dual-counter radix tree — a faithful
//! port of the tree semantics in
//! `python/sglang/srt/mem_cache/swa_radix_cache.py` (`SWARadixCache`).
//!
//! Differences from the Python implementation, deliberately:
//! - The two LRU lists are intrusive linked lists over `NodeId`s (dummy
//!   head/tail sentinels, exactly like the Python `LRUList`). Order is
//!   maintained structurally by the same move-to-MRU calls the Python tree
//!   makes on every match/insert walk, so the eviction order is
//!   deterministic without a wall clock. A `last_access` tick is kept per
//!   node for the same sanity-check role the Python float counter plays,
//!   but it orders nothing.
//! - The allocator does not live inside the tree. Every allocator call the
//!   Python tree makes (`free` / `free_full` / `free_swa` and the
//!   tombstone-recovery mapping ops) is returned to the caller as a run
//!   list on the result type, in call order.
//! - The match result truncates the *node-run list* to `best_value_len`
//!   entries before concatenation — exactly what the Python post-processor
//!   does with `value[:best_value_len]` on its per-node tensor list (a
//!   list slice, not a token slice). Kept faithful, quirk included.
//! - The Python-only optimizations behind
//!   `SGLANG_OPT_SWA_RADIX_CACHE_COMPACT` and
//!   `SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT` (both off by default, both
//!   flagged FIXME upstream) are not ported.
//! - `swa_reprefill_tail_tokens` (DeepSeek-V4 unified-KV ring guard) is a
//!   policy-side helper, not tree state.

use std::collections::HashMap;

use crate::key::{common_prefix_len, RadixKey};
use crate::lru::{Lru, LRUList};
use crate::tree::{ChildKey, Head, NodeId, ROOT};

/// One SWA tree node. `key` is flattened (one element per token, or two
/// per bigram); `value` has one KV index per logical unit.
#[derive(Debug, Clone)]
pub struct SWANode {
    pub children: HashMap<ChildKey, NodeId>,
    pub parent: Option<NodeId>,
    pub key: Vec<i64>,
    /// `None` = the node was deleted from the tree by a full-side evict.
    pub value: Option<Vec<i64>>,
    /// Invariant: `full_lock_ref >= swa_lock_ref` always.
    pub full_lock_ref: u32,
    pub swa_lock_ref: u32,
    /// The SWA pool slots of this node are freed while the full KV stays
    /// alive (leaf early-release or an internal-node SWA evict).
    pub swa_tombstone: bool,
    /// Marks the node where a request's SWA lock stops (the `inc_lock_ref`
    /// window boundary); the split front node inherits it, the tail loses
    /// it.
    pub swa_uuid: Option<u64>,
    /// Sanity-check tick (mirrors the Python `last_access_time` float).
    pub last_access: u64,
    pub id: NodeId,
}

/// Allocator-free bookkeeping of the KV runs an op wants released.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FreeOps {
    /// `allocator.free(run)` — full + mapped SWA slots.
    pub kv: Vec<Vec<i64>>,
    /// `allocator.free_full(run)` — full slots only.
    pub full: Vec<Vec<i64>>,
    /// `allocator.free_swa(run)` — SWA slots only.
    pub swa: Vec<Vec<i64>>,
}

/// Tombstone recovery on a node whose full KV is locked by a running
/// request: keep `tree_value`, re-point its full→SWA mapping at
/// `incoming`. The caller performs:
/// ```text
/// swa = allocator.translate_loc_from_full_to_swa(incoming)
/// allocator.set_full_to_swa_mapping(tree_value, swa)
/// allocator.clear_full_to_swa_mapping(incoming)
/// ```
/// (the `free_full(incoming)` call the Python tree makes is already
/// reported in `FreeOps.full`, in walk order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SWARecover {
    pub tree_value: Vec<i64>,
    pub incoming: Vec<i64>,
}

/// Port of `match_prefix` (`_match_prefix_helper` +
/// `_match_post_processor` folded in).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SWAMatchResult {
    /// The reusable KV prefix (the Python `device_indices`).
    pub indices: Vec<i64>,
    /// The node the caller locks (`best_last_node` in the Python tree).
    pub last_node: NodeId,
}

/// Port of `insert` (`_insert_helper` folded in).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SWAInsertResult {
    /// Logical tokens already present in the tree.
    pub prefix_len: usize,
    /// The deepest node the walk touched (the new leaf when one was
    /// created) — the Python tree re-derives this via a follow-up
    /// `match_prefix`; returned here for call-site convenience.
    pub last_node: NodeId,
    pub free: FreeOps,
    pub recover_locked_full: Vec<SWARecover>,
}

/// Port of `evict` (dual full/SWA budgets).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SWAEvictResult {
    pub full_num_evicted: usize,
    pub swa_num_evicted: usize,
    pub free: FreeOps,
}

/// Port of `dec_swa_lock_only`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SWADecResult {
    /// SWA runs the caller must free (leaf early-release).
    pub free_swa: Vec<Vec<i64>>,
}

#[derive(Clone)]
pub struct SWARadixTree {
    nodes: Vec<SWANode>,
    page_size: usize,
    /// Flattened elements per logical unit: 1 (plain) or 2 (bigram).
    unit_size: usize,
    is_eagle: bool,
    disable: bool,
    sliding_window_size: usize,
    full_evictable_size: i64,
    swa_evictable_size: i64,
    full_protected_size: i64,
    swa_protected_size: i64,
    full_lru: LRUList,
    swa_lru: LRUList,
    /// `gen_swa_uuid` counter (Python: starts at 1, post-increment, so the
    /// first issued uuid is 2).
    swa_uuid_counter: u64,
    /// Sanity-check walk clock.
    clock: u64,
    ns_map: HashMap<(String, String), u32>,
    ns_list: Vec<(String, String)>,
}

impl SWARadixTree {
    pub fn new(page_size: usize, is_eagle: bool, sliding_window_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        let mut t = Self {
            nodes: Vec::new(),
            page_size,
            unit_size: if is_eagle { 2 } else { 1 },
            is_eagle,
            disable: false,
            sliding_window_size,
            full_evictable_size: 0,
            swa_evictable_size: 0,
            full_protected_size: 0,
            swa_protected_size: 0,
            full_lru: LRUList::default(),
            swa_lru: LRUList::default(),
            swa_uuid_counter: 1,
            clock: 0,
            ns_map: HashMap::new(),
            ns_list: Vec::new(),
        };
        t.reset();
        t
    }

    /// `reset` — recreate the root and clear all bookkeeping.
    pub fn reset(&mut self) {
        let root = SWANode {
            children: HashMap::new(),
            parent: None,
            key: vec![],
            value: Some(vec![]),
            full_lock_ref: 1,
            swa_lock_ref: 1,
            swa_tombstone: false,
            swa_uuid: None,
            last_access: 0,
            id: ROOT,
        };
        self.nodes = vec![root];
        self.full_evictable_size = 0;
        self.swa_evictable_size = 0;
        self.full_protected_size = 0;
        self.swa_protected_size = 0;
        self.full_lru = LRUList::default();
        self.swa_lru = LRUList::default();
        self.full_lru.grow(1);
        self.swa_lru.grow(1);
        self.swa_uuid_counter = 1;
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

    pub fn sliding_window_size(&self) -> usize {
        self.sliding_window_size
    }

    pub fn full_evictable_size(&self) -> i64 {
        self.full_evictable_size
    }

    pub fn swa_evictable_size(&self) -> i64 {
        self.swa_evictable_size
    }

    pub fn full_protected_size(&self) -> i64 {
        self.full_protected_size
    }

    pub fn swa_protected_size(&self) -> i64 {
        self.swa_protected_size
    }

    /// Port of `total_size` / `_total_size_helper`: `(full, swa)`.
    pub fn total_size(&self) -> (i64, i64) {
        let mut full = 0i64;
        let mut swa = 0i64;
        let mut stack = vec![ROOT];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id as usize];
            if let Some(v) = &n.value {
                full += v.len() as i64;
                if !n.swa_tombstone {
                    swa += v.len() as i64;
                }
            }
            for &child in n.children.values() {
                if self.nodes[child as usize].value.is_some() {
                    stack.push(child);
                }
            }
        }
        (full, swa)
    }

    /// Live children of a node (debug/parity tooling).
    pub fn node_children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .get(id as usize)
            .map(|n| n.children.values().copied().collect())
            .unwrap_or_default()
    }

    pub fn node_tombstone(&self, id: NodeId) -> bool {
        self.nodes
            .get(id as usize)
            .map(|n| n.swa_tombstone)
            .unwrap_or(false)
    }

    pub fn node_full_lock_ref(&self, id: NodeId) -> u32 {
        self.nodes
            .get(id as usize)
            .map(|n| n.full_lock_ref)
            .unwrap_or(u32::MAX)
    }

    pub fn node_swa_lock_ref(&self, id: NodeId) -> u32 {
        self.nodes
            .get(id as usize)
            .map(|n| n.swa_lock_ref)
            .unwrap_or(u32::MAX)
    }

    /// Port of `match_prefix`.
    pub fn match_prefix(&mut self, key: &RadixKey) -> SWAMatchResult {
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
            return SWAMatchResult::default();
        }
        let flat = key.flatten_page_aligned(self.page_size);
        if flat.is_empty() {
            return SWAMatchResult::default();
        }

        let (runs, best_node, best_len) = self.match_prefix_helper(&flat, &key);

        // `_match_post_processor`: move the matched chain to the MRU end in
        // both lists (the SWA list skips tombstones).
        self.chain_mru(false, best_node, ROOT);
        self.chain_mru(true, best_node, ROOT);
        // The Python post-processor re-stamps `last_access_time` for every
        // node on the chain, deepest first; mirror it with the tick.
        let mut tick = self.tick();
        let mut cur = Some(best_node);
        while let Some(id) = cur {
            self.nodes[id as usize].last_access = tick;
            tick = tick.saturating_sub(1);
            cur = self.nodes[id as usize].parent;
        }

        // `value[:best_value_len]` in the Python post-processor slices the
        // per-NODE run list (not a token slice); concatenate what survives.
        let mut indices = Vec::with_capacity(best_len);
        for run in runs.iter().take(best_len) {
            indices.extend_from_slice(run);
        }
        SWAMatchResult {
            indices,
            last_node: best_node,
        }
    }

    /// Port of `_match_prefix_helper` over a flattened, page-aligned key.
    /// Returns (one value run per visited node, best_last_node,
    /// best_value_len as a RUN count, not tokens).
    fn match_prefix_helper(
        &mut self,
        flat: &[i64],
        key: &RadixKey,
    ) -> (Vec<Vec<i64>>, NodeId, usize) {
        // `None` = the Python `float("inf")`: the path so far is connected
        // to the root without a tombstone, so it is always a valid prefix.
        let mut since_tombstone: Option<usize> = None;
        let mut best_value_len = 0usize;
        let mut best_last_node = ROOT;
        let mut value: Vec<Vec<i64>> = Vec::new();

        let window = self.sliding_window_size;
        let window_ok = |n: &Option<usize>| n.map_or(true, |x| x >= window);
        let add_run = |state: &mut Option<usize>, run: &Vec<i64>| {
            if let Some(n) = state {
                *n += run.len();
            }
        };

        let mut node = ROOT;
        let mut remaining: &[i64] = flat;
        let mut child_key = self.child_key_for(key, remaining);
        while !remaining.is_empty() {
            let Some(&child) = self.nodes[node as usize].children.get(&child_key) else {
                break;
            };
            let c = &self.nodes[child as usize];
            if c.swa_tombstone {
                // `best_value_len` counts RUNS, not tokens: the Python tree
                // stores `len(value)` (the per-node tensor list) and the
                // post-processor slices it with `value[:best_value_len]`.
                if window_ok(&since_tombstone) {
                    best_value_len = value.len();
                    best_last_node = node;
                }
                since_tombstone = Some(0);
            }
            let m_flat = common_prefix_len(remaining, &c.key);
            let m = m_flat / self.unit_size;
            let child_logical = c.key.len() / self.unit_size;
            if m < child_logical {
                // Match ends inside the child: split, take the front half.
                let front = self.split_node(child, m * self.unit_size);
                let v = self.nodes[front as usize]
                    .value
                    .clone()
                    .unwrap_or_default();
                if !self.nodes[front as usize].swa_tombstone {
                    add_run(&mut since_tombstone, &v);
                }
                value.push(v);
                node = front;
                break;
            } else {
                let v = c.value.clone().unwrap_or_default();
                if !c.swa_tombstone {
                    add_run(&mut since_tombstone, &v);
                }
                value.push(v);
                node = child;
                remaining = &remaining[m * self.unit_size..];
                if !remaining.is_empty() {
                    child_key = self.child_key_for(key, remaining);
                }
            }
        }
        if window_ok(&since_tombstone) {
            best_value_len = value.len();
            best_last_node = node;
        }
        (value, best_last_node, best_value_len)
    }

    /// Port of `insert` + `_insert_helper`.
    pub fn insert(
        &mut self,
        key: &RadixKey,
        value: &[i64],
        update_kv_after_len: usize,
        swa_evicted_seqlen: usize,
    ) -> SWAInsertResult {
        let mut res = SWAInsertResult::default();
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

        let (prefix_len, last_node) = self.insert_helper(
            ns,
            key.is_bigram,
            &flat,
            value,
            update_kv_after_len,
            swa_evicted_seqlen,
            &mut res.free,
            &mut res.recover_locked_full,
        );
        res.prefix_len = prefix_len;
        res.last_node = last_node;
        res
    }

    /// Port of `_insert_helper`. Returns `(total_prefix_length, node)`
    /// where `node` is the deepest node the walk touched (the new leaf
    /// when one was created).
    fn insert_helper(
        &mut self,
        ns: u32,
        is_bigram: bool,
        flat: &[i64],
        value: &[i64],
        update_kv_after_len: usize,
        swa_evicted_seqlen: usize,
        free: &mut FreeOps,
        recover: &mut Vec<SWARecover>,
    ) -> (usize, NodeId) {
        self.nodes[ROOT as usize].last_access = self.tick();
        let mut remaining: &[i64] = flat;
        let mut val: &[i64] = value;
        if remaining.is_empty() {
            return (0, ROOT);
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
            if !self.nodes[node as usize].swa_tombstone {
                self.swa_lru.reset_mru(node);
            }
            let m_flat = common_prefix_len(remaining, &self.nodes[node as usize].key);
            let prefix_len = m_flat / self.unit_size;

            if prefix_len < self.nodes[node as usize].key.len() / self.unit_size {
                let front = self.split_node(node, prefix_len * self.unit_size);
                node = front;
            }

            // Tombstone recovery (and the plain "free the overlap" path).
            if update_kv_after_len < total_prefix_length + prefix_len {
                if self.nodes[node as usize].swa_tombstone {
                    assert!(
                        self.nodes[node as usize].swa_lock_ref == 0,
                        "tombstone swa_lock_ref should always be 0, full={}",
                        self.nodes[node as usize].full_lock_ref
                    );
                    assert!(
                        swa_evicted_seqlen % self.page_size == 0,
                        "swa_evicted_seqlen must be page aligned, swa_evicted_seqlen={swa_evicted_seqlen}"
                    );
                    if swa_evicted_seqlen <= total_prefix_length {
                        // Branch 1: the whole overlap still has live SWA
                        // peers on the incoming side — adopt it.
                        if self.nodes[node as usize].full_lock_ref > 0 {
                            recover.push(SWARecover {
                                tree_value: self
                                    .node_value_run(node, 0, prefix_len),
                                incoming: val[..prefix_len].to_vec(),
                            });
                            // Tree-side of the recover (the mapping ops
                            // travel back with the recover entry): the node
                            // is live-SWA again. The `free_full(incoming)`
                            // the Python tree makes is reported as a full
                            // run, in walk order.
                            self.nodes[node as usize].swa_tombstone = false;
                            self.swa_lru.insert_mru(node);
                            self.swa_evictable_size += prefix_len as i64;
                            free.full.push(val[..prefix_len].to_vec());
                        } else {
                            free.kv.push(self.node_value_run(node, 0, prefix_len));
                            self.nodes[node as usize].value =
                                Some(val[..prefix_len].to_vec());
                            self.nodes[node as usize].swa_tombstone = false;
                            self.swa_lru.insert_mru(node);
                            self.swa_evictable_size +=
                                prefix_len as i64;
                        }
                    } else if swa_evicted_seqlen < total_prefix_length + prefix_len {
                        // Branch 2: the eviction boundary cuts the overlap.
                        // After the split, the local `node` is the TAIL
                        // ([start_update_idx, prefix_len)); the front
                        // tombstone keeps its (possibly locked) full KV.
                        let start_update_idx =
                            swa_evicted_seqlen - total_prefix_length;
                        let recovered_len = prefix_len - start_update_idx;
                        if self.nodes[node as usize].full_lock_ref > 0 {
                            self.split_node(node, start_update_idx * self.unit_size);
                            // `node` is now the tail; its value is exactly
                            // the recovered range, held by the locked full
                            // slots.
                            let tail_value =
                                self.node_value_run(node, 0, recovered_len);
                            recover.push(SWARecover {
                                tree_value: tail_value,
                                incoming: val[start_update_idx..prefix_len].to_vec(),
                            });
                            self.nodes[node as usize].swa_tombstone = false;
                            self.swa_lru.insert_mru(node);
                            self.swa_evictable_size += recovered_len as i64;
                            free.full.push(val[start_update_idx..prefix_len].to_vec());
                            free.kv.push(val[..start_update_idx].to_vec());
                        } else {
                            let old_tail =
                                self.node_value_run(node, start_update_idx, recovered_len);
                            free.kv.push(old_tail);
                            self.split_node(node, start_update_idx * self.unit_size);
                            // `node` is the tail now; overwrite its value
                            // with the incoming KV and revive its SWA side.
                            self.nodes[node as usize].value =
                                Some(val[start_update_idx..prefix_len].to_vec());
                            self.nodes[node as usize].swa_tombstone = false;
                            self.swa_lru.insert_mru(node);
                            self.swa_evictable_size += recovered_len as i64;
                            free.kv.push(val[..start_update_idx].to_vec());
                        }
                    } else {
                        // Branch 3: the whole overlap is past the eviction
                        // boundary — nothing to recover.
                        free.kv.push(val[..prefix_len].to_vec());
                    }
                } else {
                    // Non-tombstone overlap: the tree already holds this KV.
                    free.kv.push(val[..prefix_len].to_vec());
                }
            }

            total_prefix_length += prefix_len;
            remaining = &remaining[prefix_len * self.unit_size..];
            val = &val[prefix_len..];
        }

        if !remaining.is_empty() {
            // Defensive guard (case 3 in the Python comment): the SWA pool
            // evicted the entire remainder — free it, create no node (leaf
            // nodes must not be tombstone).
            let rem_logical = remaining.len() / self.unit_size;
            if swa_evicted_seqlen == total_prefix_length + rem_logical {
                free.kv.push(val.to_vec());
                return (total_prefix_length, node);
            }

            if swa_evicted_seqlen > total_prefix_length
                && swa_evicted_seqlen < total_prefix_length + rem_logical
            {
                let swa_tombstone_len = swa_evicted_seqlen - total_prefix_length;
                node = self.add_new_node(
                    node,
                    ns,
                    is_bigram,
                    &remaining[..swa_tombstone_len * self.unit_size],
                    &val[..swa_tombstone_len],
                    true,
                );
                remaining = &remaining[swa_tombstone_len * self.unit_size..];
                val = &val[swa_tombstone_len..];
            }

            node = self.add_new_node(node, ns, is_bigram, remaining, val, false);
        }

        (total_prefix_length, node)
    }

    /// Port of `evict` (dual budgets, two phases).
    pub fn evict(
        &mut self,
        full_num_tokens: usize,
        swa_num_tokens: usize,
    ) -> SWAEvictResult {
        let mut res = SWAEvictResult::default();
        if self.disable {
            return res;
        }
        let mut full_num_evicted = 0usize;
        let mut swa_num_evicted = 0usize;

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
                assert!(
                    self.nodes[xid as usize].full_lock_ref == 0,
                    "node is in use, id={xid}"
                );

                // 1. free the node's KV (full+swa, or full-only when
                //    tombstoned) and count the evictions.
                let (fe, se) = self.free_node_value(xid, &mut res.free);
                full_num_evicted += fe;
                swa_num_evicted += se;

                // 2. next candidate, then detach the node.
                let x_next = self.next_leaf_no_lock(xid);
                self.full_lru.remove(xid);
                if !self.nodes[xid as usize].swa_tombstone {
                    self.swa_lru.remove(xid);
                }

                // 3. delete the leaf.
                self.delete_leaf(xid);

                // 4. iteratively delete tombstoned parents that lost their
                //    last child.
                let (x_after, leaf_full) =
                    self.iteratively_delete_tombstone_leaf(xid, &mut res.free);
                full_num_evicted += leaf_full;

                // 5. if the parent became a leaf, restart the LRU scan.
                let parent = self.nodes[x_after as usize].parent.unwrap_or(ROOT);
                if self.nodes[parent as usize].children.is_empty() {
                    x = self.full_get_leaf_lru_no_lock();
                } else {
                    x = x_next;
                }
            }
        }

        if swa_num_evicted < swa_num_tokens {
            let mut x = self.swa_get_lru_no_lock();
            while swa_num_evicted < swa_num_tokens {
                let Some(xid) = x else {
                    break;
                };
                assert!(
                    !self.nodes[xid as usize].swa_tombstone,
                    "duplicate swa tombstone node, id={xid}"
                );
                assert!(xid != ROOT, "root node is not evictable");
                assert!(
                    self.nodes[xid as usize].swa_lock_ref == 0,
                    "node is in use by swa kv indices, id={xid}"
                );

                if !self.nodes[xid as usize].children.is_empty() {
                    // 1. internal node: free SWA only.
                    let v = self
                        .nodes[xid as usize]
                        .value
                        .clone()
                        .unwrap_or_default();
                    res.free.swa.push(v.clone());
                    swa_num_evicted += v.len();
                    let x_next = self.next_swa_no_lock(xid);
                    self.swa_lru.remove(xid);
                    // 2. tombstone it.
                    self.tombstone_internal_node(xid);
                    x = x_next;
                } else if self.nodes[xid as usize].full_lock_ref > 0 {
                    // 3. leaf still full-locked: free SWA, tombstone.
                    let v = self
                        .nodes[xid as usize]
                        .value
                        .clone()
                        .unwrap_or_default();
                    res.free.swa.push(v.clone());
                    swa_num_evicted += v.len();
                    let x_next = self.next_swa_no_lock(xid);
                    self.swa_lru.remove(xid);
                    self.swa_evictable_size -= v.len() as i64;
                    self.nodes[xid as usize].swa_tombstone = true;
                    x = x_next;
                } else {
                    // 4. unlocked leaf: free full+swa, delete.
                    let (fe, se) = self.free_node_value(xid, &mut res.free);
                    full_num_evicted += fe;
                    swa_num_evicted += se;
                    let x_next = self.next_swa_no_lock(xid);
                    self.full_lru.remove(xid);
                    self.swa_lru.remove(xid);
                    self.delete_leaf(xid);
                    self.iteratively_delete_tombstone_leaf(xid, &mut res.free);
                    x = x_next;
                }
            }
        }

        res.full_num_evicted = full_num_evicted;
        res.swa_num_evicted = swa_num_evicted;
        res
    }

    /// Port of `inc_lock_ref`: full lock up to (exclusive) the root; the
    /// SWA lock up to the sliding-window boundary. Returns
    /// `(swa_uuid_for_lock, full-side tokens moved evictable -> protected
    /// (<= 0))`.
    pub fn inc_lock_ref(&mut self, mut node: NodeId) -> (Option<u64>, i64) {
        let mut full_delta = 0i64;
        if self.disable {
            return (None, 0);
        }
        let mut swa_lock_size = 0usize;
        let mut swa_uuid_for_lock: Option<u64> = None;
        while node != ROOT {
            {
                let n = &mut self.nodes[node as usize];
                if n.full_lock_ref == 0 {
                    let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0) as i64;
                    self.full_evictable_size -= l;
                    self.full_protected_size += l;
                    full_delta -= l;
                }
                n.full_lock_ref += 1;

                if swa_lock_size < self.sliding_window_size {
                    assert!(
                        !n.swa_tombstone,
                        "inc_lock_swa on swa_tombstone node {node}"
                    );
                    if n.swa_lock_ref == 0 {
                        let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0) as i64;
                        self.swa_evictable_size -= l;
                        self.swa_protected_size += l;
                    }
                    n.swa_lock_ref += 1;
                    let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0);
                    swa_lock_size += l;
                    if swa_lock_size >= self.sliding_window_size {
                        let id = node;
                        let uuid = match self.nodes[id as usize].swa_uuid {
                            Some(u) => u,
                            None => {
                                let u = self.next_swa_uuid();
                                self.nodes[id as usize].swa_uuid = Some(u);
                                u
                            }
                        };
                        swa_uuid_for_lock = Some(uuid);
                    }
                }
            }
            node = self.nodes[node as usize].parent.expect("walk reaches root");
        }
        (swa_uuid_for_lock, full_delta)
    }

    /// Port of `dec_lock_ref`. Returns the full-side tokens moved
    /// protected -> evictable (>= 0).
    pub fn dec_lock_ref(
        &mut self,
        mut node: NodeId,
        swa_uuid_for_lock: Option<u64>,
        skip_swa: bool,
    ) -> i64 {
        let mut full_delta = 0i64;
        if self.disable {
            return 0;
        }
        let mut dec_lock_swa = !skip_swa;
        while node != ROOT {
            {
                let n = &mut self.nodes[node as usize];
                assert!(
                    n.full_lock_ref > 0,
                    "dec_lock_ref on node {node} with full_lock_ref={}",
                    n.full_lock_ref
                );
                if n.full_lock_ref == 1 {
                    let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0) as i64;
                    self.full_evictable_size += l;
                    self.full_protected_size -= l;
                    full_delta += l;
                }
                n.full_lock_ref -= 1;

                if dec_lock_swa {
                    assert!(
                        !n.swa_tombstone,
                        "dec_lock_ref on swa_tombstone node {node}"
                    );
                    assert!(
                        n.swa_lock_ref > 0,
                        "dec_lock_ref on node {node} with swa_lock_ref=0"
                    );
                    if n.swa_lock_ref == 1 {
                        let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0) as i64;
                        self.swa_evictable_size += l;
                        self.swa_protected_size -= l;
                    }
                    n.swa_lock_ref -= 1;
                    if let Some(u) = swa_uuid_for_lock {
                        if self.nodes[node as usize].swa_uuid == Some(u) {
                            dec_lock_swa = false;
                        }
                    }
                }
            }
            node = self.nodes[node as usize].parent.expect("walk reaches root");
        }
        full_delta
    }

    /// Port of `dec_swa_lock_only`: release only the SWA lock along
    /// `[node, swa_uuid_for_lock]` (inclusive). A locked leaf is
    /// tombstoned and its SWA slots freed immediately; an internal node
    /// simply moves back to swa-evictable.
    pub fn dec_swa_lock_only(
        &mut self,
        mut node: NodeId,
        swa_uuid_for_lock: Option<u64>,
    ) -> SWADecResult {
        let mut res = SWADecResult::default();
        if self.disable {
            return res;
        }
        while node != ROOT {
            {
                let n = &mut self.nodes[node as usize];
                assert!(
                    !n.swa_tombstone,
                    "dec_swa_lock_only on swa_tombstone node {node}"
                );
                assert!(
                    n.swa_lock_ref > 0,
                    "dec_swa_lock_only on node {node} with swa_lock_ref=0"
                );

                if n.swa_lock_ref == 1 {
                    let l = n.value.as_ref().map(|v| v.len()).unwrap_or(0) as i64;
                    self.swa_protected_size -= l;
                    if n.children.is_empty() {
                        let v = n.value.clone().unwrap_or_default();
                        res.free_swa.push(v);
                        self.swa_lru.remove(node);
                        n.swa_tombstone = true;
                    } else {
                        self.swa_evictable_size += l;
                    }
                }
                n.swa_lock_ref -= 1;
            }
            if let Some(u) = swa_uuid_for_lock {
                if self.nodes[node as usize].swa_uuid == Some(u) {
                    break;
                }
            }
            node = self.nodes[node as usize].parent.expect("walk reaches root");
        }
        res
    }

    // ---- internals ----

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn next_swa_uuid(&mut self) -> u64 {
        // `gen_swa_uuid`: post-increment from 1 (first uuid is 2).
        self.swa_uuid_counter += 1;
        self.swa_uuid_counter
    }

    /// `value[start..start+len)` of a node's KV run.
    fn node_value_run(&self, node: NodeId, start: usize, len: usize) -> Vec<i64> {
        self.nodes[node as usize]
            .value
            .as_deref()
            .map(|v| v[start..start + len].to_vec())
            .unwrap_or_default()
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

    /// Port of `reset_node_and_parents_mru` on one of the two lists:
    /// move the `node -> root` chain to the MRU end, deepest node most
    /// recent; the SWA list skips tombstones (they are not in the list).
    fn chain_mru(&mut self, is_swa: bool, mut node: NodeId, root: NodeId) {
        let mut prev_node = Lru::Head;
        while node != root {
            let tombstone = self.nodes[node as usize].swa_tombstone;
            if !is_swa || !tombstone {
                let lru = if is_swa {
                    &mut self.swa_lru
                } else {
                    &mut self.full_lru
                };
                debug_assert!(
                    lru.in_list(node),
                    "chain_mru: node {node} not in the {} list",
                    if is_swa { "swa" } else { "full" }
                );
                lru.remove(node);
                lru.add_after(prev_node, node);
                prev_node = Lru::Node(node);
            }
            node = self.nodes[node as usize]
                .parent
                .expect("walk reaches the root");
        }
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

    /// Port of `get_lru_no_lock` on the SWA list.
    fn swa_get_lru_no_lock(&self) -> Option<NodeId> {
        let mut x = self.swa_lru.predecessor(Lru::Tail);
        while let Lru::Node(id) = x {
            if self.nodes[id as usize].swa_lock_ref == 0 {
                return Some(id);
            }
            x = self.swa_lru.predecessor(Lru::Node(id));
        }
        None
    }

    /// Port of `get_prev_leaf_no_lock(from)`: walk the full list from
    /// `from`'s predecessor (more recent) toward the head.
    fn next_leaf_no_lock(&self, from: NodeId) -> Option<NodeId> {
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

    /// Port of `get_prev_no_lock(from)` on the SWA list.
    fn next_swa_no_lock(&self, from: NodeId) -> Option<NodeId> {
        let mut x = self.swa_lru.predecessor(Lru::Node(from));
        while let Lru::Node(id) = x {
            if self.nodes[id as usize].swa_lock_ref == 0 {
                return Some(id);
            }
            x = self.swa_lru.predecessor(Lru::Node(id));
        }
        None
    }

    /// Port of `_free_node_value`: `(full_evicted, swa_evicted)`.
    fn free_node_value(&self, node: NodeId, free: &mut FreeOps) -> (usize, usize) {
        let v = self
            .nodes[node as usize]
            .value
            .clone()
            .unwrap_or_default();
        let num = v.len();
        if self.nodes[node as usize].swa_tombstone {
            free.full.push(v);
            (num, 0)
        } else {
            free.kv.push(v);
            (num, num)
        }
    }

    /// Port of `_add_new_node`.
    fn add_new_node(
        &mut self,
        parent: NodeId,
        ns: u32,
        is_bigram: bool,
        key: &[i64],
        value: &[i64],
        swa_tombstone: bool,
    ) -> NodeId {
        assert!(!key.is_empty(), "key should not be empty");
        let new_id = self.new_node_slot();
        let new_last_access = self.tick();
        {
            let n = &mut self.nodes[new_id as usize];
            n.parent = Some(parent);
            n.key = key.to_vec();
            n.value = Some(value.to_vec());
            n.swa_tombstone = swa_tombstone;
            n.last_access = new_last_access;
        }
        let ck = ChildKey {
            ns,
            head: self.head_from_flat(key, is_bigram),
        };
        self.nodes[parent as usize].children.insert(ck, new_id);
        self.full_lru.insert_mru(new_id);
        self.full_evictable_size += value.len() as i64;
        if !swa_tombstone {
            self.swa_lru.insert_mru(new_id);
            self.swa_evictable_size += value.len() as i64;
        }
        new_id
    }

    /// Port of `_split_node(key, child, split_len)`: split `child` after
    /// `split_flat` flattened elements. Returns the new front node; the
    /// `child` record is mutated in place into the tail (as in the Python
    /// tree, where the original node object becomes the tail).
    fn split_node(&mut self, child: NodeId, split_flat: usize) -> NodeId {
        let (child_key, gp, tombstone, fl, sl, uuid) = {
            let c = &self.nodes[child as usize];
            (
                c.key.clone(),
                c.parent.expect("split target must have a parent"),
                c.swa_tombstone,
                c.full_lock_ref,
                c.swa_lock_ref,
                c.swa_uuid,
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
        // grandchildren-map mutations below hold no live helper borrows.
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

        // Detach the child from both LRU lists (tombstoned children are
        // not in the SWA list).
        self.full_lru.remove(child);
        if !tombstone {
            self.swa_lru.remove(child);
        }

        let new_id = self.new_node_slot();
        let new_last_access = self.tick();
        // The tail's stamp must be fresher than the front's (the Python
        // split re-stamps `child.last_access_time` after construction).
        let tail_last_access = self.tick();
        {
            let mut children = HashMap::with_capacity(1);
            children.insert(tail_ck, child);
            let n = &mut self.nodes[new_id as usize];
            *n = SWANode {
                children,
                parent: Some(gp),
                key: front_key,
                value: Some(front_value),
                full_lock_ref: fl,
                swa_lock_ref: sl,
                swa_tombstone: tombstone,
                swa_uuid: uuid,
                last_access: new_last_access,
                id: new_id,
            };
        }
        // Mutate the tail node in place; it loses the SWA lock-boundary
        // marker (the front node keeps it) and gets a fresher stamp than
        // the front node.
        {
            let c = &mut self.nodes[child as usize];
            c.key = tail_key;
            c.value = Some(tail_value);
            c.parent = Some(new_id);
            c.swa_uuid = None;
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
        if !tombstone {
            self.swa_lru.insert_mru(new_id);
            self.swa_lru.insert_mru(child);
        }
        new_id
    }

    /// Port of `_delete_leaf`.
    fn delete_leaf(&mut self, node: NodeId) {
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
        let logical = key.len() as i64 / self.unit_size as i64;
        self.full_evictable_size -= logical;
        if !self.nodes[node as usize].swa_tombstone {
            self.swa_evictable_size -= logical;
        }
    }

    /// Port of `_tombstone_internal_node`.
    fn tombstone_internal_node(&mut self, node: NodeId) {
        assert!(
            !self.nodes[node as usize].children.is_empty(),
            "cannot tombstone a leaf node, id={node}"
        );
        self.nodes[node as usize].swa_tombstone = true;
        self.swa_evictable_size -= self.nodes[node as usize]
            .key
            .len() as i64
            / self.unit_size as i64;
    }

    /// Port of `_delete_tombstone_leaf`.
    fn delete_tombstone_leaf(&mut self, node: NodeId) {
        assert!(
            self.nodes[node as usize].swa_tombstone,
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
        free: &mut FreeOps,
    ) -> (NodeId, usize) {
        let mut full_num_evicted = 0usize;
        loop {
            let Some(parent) = self.nodes[node as usize].parent else {
                break;
            };
            let p = &self.nodes[parent as usize];
            if !p.swa_tombstone || !p.children.is_empty() {
                break;
            }
            if parent == ROOT {
                break;
            }
            if p.full_lock_ref > 0 {
                break;
            }
            assert!(
                p.swa_lock_ref == 0,
                "tombstone swa_lock_ref should always be 0, full={}",
                p.full_lock_ref
            );
            let (fe, _) = self.free_node_value(parent, free);
            full_num_evicted += fe;
            self.full_lru.remove(parent);
            self.delete_tombstone_leaf(parent);
            node = parent;
        }
        (node, full_num_evicted)
    }

    /// Push a fresh node record and grow the LRU arrays with it.
    fn new_node_slot(&mut self) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(SWANode {
            children: HashMap::new(),
            parent: None,
            key: vec![],
            value: Some(vec![]),
            full_lock_ref: 0,
            swa_lock_ref: 0,
            swa_tombstone: false,
            swa_uuid: None,
            last_access: 0,
            id,
        });
        self.full_lru.grow(self.nodes.len());
        self.swa_lru.grow(self.nodes.len());
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
        let mut t = SWARadixTree::new(1, false, 64);
        let ids: Vec<i64> = (0..100).collect();
        let r = t.insert(&RadixKey::new(&ids), &vals(100, 1000), 0, 0);
        assert_eq!(r.prefix_len, 0);
        assert_eq!(t.full_evictable_size(), 100);
        assert_eq!(t.swa_evictable_size(), 100);

        let m = t.match_prefix(&RadixKey::new(&ids));
        assert_eq!(m.indices.len(), 100);
        assert_eq!(m.last_node, r.last_node);
        assert_eq!(t.full_evictable_size(), 100); // match does not lock
        assert_eq!(t.full_protected_size(), 0);

        let (uuid, delta) = t.inc_lock_ref(r.last_node);
        assert_eq!(delta, -100);
        assert_eq!(t.full_protected_size(), 100);
        // window (64) < 100: the SWA lock boundary lands inside this node,
        // but the lock is per-NODE, so the whole node's SWA is protected.
        assert_eq!(uuid, Some(2)); // first issued uuid (counter starts at 1, post-increment)
        assert_eq!(t.swa_protected_size(), 100);
        assert_eq!(t.dec_lock_ref(r.last_node, uuid, false), 100);
        assert_eq!(t.swa_protected_size(), 0);
        assert_eq!(t.full_protected_size(), 0);
    }

    #[test]
    fn evict_dual_budget() {
        let mut t = SWARadixTree::new(1, false, 4);
        // Two disjoint 10-token leaves.
        let a: Vec<i64> = (0..10).collect();
        let b: Vec<i64> = (100..110).collect();
        t.insert(&RadixKey::new(&a), &vals(10, 1000), 0, 0);
        t.insert(&RadixKey::new(&b), &vals(10, 2000), 0, 0);

        // SWA-only evict: internal-node path is unreachable with leaves;
        // the unlocked-leaf path frees full+swa for 10 tokens.
        let r = t.evict(0, 10);
        assert_eq!(r.full_num_evicted, 10);
        assert_eq!(r.swa_num_evicted, 10);
        assert_eq!(t.full_evictable_size(), 10);
        assert_eq!(t.swa_evictable_size(), 10);
        assert_eq!(r.free.kv.len(), 1);
        assert_eq!(r.free.kv[0].len(), 10);
    }

    #[test]
    fn swa_evict_tombstones_internal_nodes() {
        let mut t = SWARadixTree::new(1, false, 4);
        // Chain: root -> A(10) -> B(10): insert A, then A+B.
        let a: Vec<i64> = (0..10).collect();
        t.insert(&RadixKey::new(&a), &vals(10, 1000), 0, 0);
        let ab: Vec<i64> = (0..20).collect();
        t.insert(&RadixKey::new(&ab), &vals(20, 1000), 0, 0);
        assert_eq!(t.full_evictable_size(), 20);
        assert_eq!(t.swa_evictable_size(), 20);

        // SWA-evict 5: the LRU non-leaf is A (internal, unlocked) -> free
        // swa only, tombstone it.
        let r = t.evict(0, 5);
        assert_eq!(r.full_num_evicted, 0);
        assert_eq!(r.swa_num_evicted, 10); // overfills to the node boundary
        assert_eq!(r.free.swa.len(), 1);
        assert_eq!(r.free.swa[0].len(), 10);
        assert_eq!(t.swa_evictable_size(), 10); // only B's SWA remains
        assert_eq!(t.full_evictable_size(), 20);
    }

    #[test]
    fn match_stops_at_tombstone_gap() {
        // root -> A(100) -> B(100).
        let mut t = SWARadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..100).collect();
        t.insert(&RadixKey::new(&a), &vals(100, 1000), 0, 0);
        let ab: Vec<i64> = (0..200).collect();
        t.insert(&RadixKey::new(&ab), &vals(200, 1000), 0, 0);

        // SWA-evict with a huge budget: the LRU node is A (internal,
        // unlocked) -> free SWA only, tombstone it (100). Then the scan
        // continues from A's predecessor — B (leaf, unlocked) -> free
        // full+swa, delete (100 + 100). Deleting B cascades: A lost its
        // last child while tombstoned and unlocked, so its full KV is
        // freed (into `free.full`) and it is deleted too. Faithful to
        // the Python quirk, the cascade's full tokens are NOT added to
        // `full_num_evicted` in phase 2 (the return value is discarded
        // in `SWARadixCache.evict`).
        let r = t.evict(0, 10_000);
        assert_eq!(r.full_num_evicted, 100);
        assert_eq!(r.swa_num_evicted, 200);
        // B went out via `kv`, A's full side via `full`.
        assert_eq!(
            r.free.kv.iter().map(|v| v.len()).collect::<Vec<_>>(),
            vec![100]
        );
        assert_eq!(
            r.free.full.iter().map(|v| v.len()).collect::<Vec<_>>(),
            vec![100]
        );
        assert!(t.node_children(ROOT).is_empty()); // both A and B detached
        assert_eq!(t.full_evictable_size(), 0);
        assert_eq!(t.swa_evictable_size(), 0);

        // The tree is empty now: a match of the full 200 tokens finds
        // nothing.
        let m = t.match_prefix(&RadixKey::new(&ab));
        assert_eq!(m.indices.len(), 0);
        assert_eq!(m.last_node, ROOT);
    }

    #[test]
    fn match_valid_across_tombstone_when_window_covered() {
        // root -> A(100, tombstone) -> B(100, live). Distance since the
        // tombstone at B is 100 >= window 64, so the full match is valid
        // and B's run is the reusable KV.
        let mut t = SWARadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..100).collect();
        t.insert(&RadixKey::new(&a), &vals(100, 1000), 0, 0);
        let ab: Vec<i64> = (0..200).collect();
        t.insert(&RadixKey::new(&ab), &vals(200, 2000), 0, 0);
        // Tombstone A only: A is the LRU node and internal, so its whole
        // 100-token SWA run is freed and it is tombstoned; B (MRU) is
        // untouched.
        let r = t.evict(0, 10);
        assert_eq!(r.full_num_evicted, 0);
        assert_eq!(r.swa_num_evicted, 100); // overfills to A's boundary
        assert!(t.node_tombstone(1));

        let m = t.match_prefix(&RadixKey::new(&ab));
        // Runs: [A(100), B(100)]; best run count = 2 (distance 100 >= 64
        // at the end), but the post-processor keeps runs up to best_len
        // inclusive... A is a tombstone run: it still holds full KV, so
        // its indices are reusable only as far as the window rule allows.
        // The window rule validates the LAST node (B, 100 since last
        // tombstone), so both runs survive the list slice.
        assert_eq!(m.indices.len(), 200);
        assert_eq!(m.last_node, 2); // B
    }

    #[test]
    fn dec_swa_lock_only_tombstones_leaf() {
        let mut t = SWARadixTree::new(1, false, 4);
        let ids: Vec<i64> = (0..10).collect();
        let leaf = t.insert(&RadixKey::new(&ids), &vals(10, 1000), 0, 0).last_node;
        let (uuid, _) = t.inc_lock_ref(leaf);
        assert_eq!(t.swa_protected_size(), 10); // window 4 <= 10, whole leaf

        let r = t.dec_swa_lock_only(leaf, uuid);
        assert_eq!(r.free_swa.len(), 1);
        assert_eq!(r.free_swa[0].len(), 10);
        assert!(t.node_tombstone(leaf));
        assert_eq!(t.node_full_lock_ref(leaf), 1); // full lock untouched
        assert_eq!(t.swa_protected_size(), 0);

        // Final release skips the SWA side (already released).
        let delta = t.dec_lock_ref(leaf, None, true);
        assert_eq!(delta, 10);
        assert_eq!(t.full_protected_size(), 0);
    }

    #[test]
    fn insert_recover_locked_full() {
        // root -> A(100) -> B(100); lock B; SWA-evict tombstones A while
        // its full side stays locked; re-inserting the prefix adopts the
        // incoming SWA for A (recover) and frees B's overlap.
        let mut t = SWARadixTree::new(1, false, 64);
        let a: Vec<i64> = (0..100).collect();
        t.insert(&RadixKey::new(&a), &vals(100, 1000), 0, 0);
        let ab: Vec<i64> = (0..200).collect();
        let b = t.insert(&RadixKey::new(&ab), &vals(200, 2000), 0, 0).last_node;
        let (uuid, _) = t.inc_lock_ref(b);
        assert!(uuid.is_some());

        let r = t.evict(0, 10);
        assert_eq!(r.swa_num_evicted, 100);
        assert_eq!(r.full_num_evicted, 0);
        assert!(t.node_tombstone(1)); // A
        assert_eq!(t.node_full_lock_ref(1), 1);

        let new_ab = vals(200, 9000);
        let r = t.insert(&RadixKey::new(&ab), &new_ab, 0, 0);
        assert_eq!(r.prefix_len, 200);
        assert_eq!(r.recover_locked_full.len(), 1);
        assert_eq!(r.recover_locked_full[0].tree_value, vals(100, 1000));
        assert_eq!(r.recover_locked_full[0].incoming, new_ab[..100].to_vec());
        // free_full(incoming) is reported in walk order; B's overlap via kv.
        assert_eq!(r.free.full, vec![new_ab[..100].to_vec()]);
        assert_eq!(r.free.kv, vec![new_ab[100..].to_vec()]);
        // A is live-SWA again, its value is the ADOPTED tree value (the
        // locked full slots stayed in place). B is non-tombstone, so the
        // tree keeps ITS value and the incoming overlap is freed.
        assert!(!t.node_tombstone(1));
        assert_eq!(t.swa_evictable_size(), 100);
        let m = t.match_prefix(&RadixKey::new(&ab));
        let mut expected = vals(100, 1000);
        expected.extend(vals(200, 2000)[100..].iter());
        assert_eq!(m.indices, expected);

        // Second lock (B keeps its uuid marker, so the same uuid applies);
        // release both locks, then drain to zero.
        let (uuid2, _) = t.inc_lock_ref(b);
        assert_eq!(uuid2, uuid);
        t.dec_lock_ref(b, uuid, false);
        t.dec_lock_ref(b, uuid2, false);
        let r = t.evict(10_000, 10_000);
        assert_eq!(r.full_num_evicted, 200);
        assert_eq!(r.swa_num_evicted, 200);
        assert_eq!(t.total_size(), (0, 0));
    }

    #[test]
    fn insert_with_swa_evicted_seqlen_creates_tombstone_leaf() {
        let mut t = SWARadixTree::new(1, false, 4);
        // 20-token insert where SWA has evicted [5, 12): the first 5 go
        // into a tombstoned node, [12, 20) into a live leaf.
        let ids: Vec<i64> = (0..20).collect();
        let r = t.insert(&RadixKey::new(&ids), &vals(20, 1000), 0, 12);
        assert_eq!(r.prefix_len, 0);
        // Full: both nodes hold full KV. SWA: only the live tail.
        assert_eq!(t.full_evictable_size(), 20);
        assert_eq!(t.swa_evictable_size(), 8);
        assert_eq!(t.total_size(), (20, 8));
    }
}
