//! The HiRadix host-tier radix tree — a faithful port of the tree
//! semantics in `python/sglang/srt/mem_cache/hiradix_cache.py`
//! (`HiRadixCache`), plus the host-tier pieces of its base `RadixCache`
//! that it inherits or overrides.
//!
//! Unlike the plain `RadixTree` (which removes evicted nodes from the
//! tree), this tree keeps device-evicted nodes in place: a node's
//! `value` (device KV) and `host_value` (host-tier copy) are
//! independent `Option`s, so an evicted-but-backuped node stays
//! reachable for host matches, `evict_host`, and `load_back`. Device
//! eviction DEMOTES (drops the value, keeps the node when backed up) or
//! deletes outright (regular leaves); host eviction DELETES the node
//! from the tree.
//!
//! Differences from the Python implementation, deliberately:
//! - `last_access_time` / `creation_time` are a monotonic u64 walk clock
//!   instead of `time.monotonic()`: the plan requires deterministic
//!   eviction order. Every walk (`match_prefix`, `insert`,
//!   `insert_host`) stamps ALL visited nodes with the SAME op-time tick
//!   (Python stamps each visited node with a fresh `time.monotonic()`
//!   value that may or may not advance within one walk); node creation
//!   (new leaf, split front) gets a fresh, strictly newer tick.
//! - The eviction heaps are rebuilt per evict call from
//!   `evictable_leaves` / `evictable_host_leaves` (exactly like Python's
//!   `heapq.heapify` snapshot), ordered by the `EvictionPolicy`
//!   priority with a node-id tie-break. Stale entries are filtered at
//!   pop time, faithful to Python (`lock_ref > 0` on device pops,
//!   `!evicted` and `host_ref_counter > 0` on host pops).
//! - No async: host DMA (`cache_controller.write` / `load` /
//!   `evict_device` / `evict_host`), storage prefetch/backup, KV events
//!   and metrics stay caller-side. The tree reports the runs to free
//!   (`free_device` / `free_host`) and the pending write-through state
//!   (`backup_pending`, `splits`), and exposes the two-phase
//!   `init_load_back` / `finish_load_back` + `begin_backup` /
//!   `end_backup` + `protect_host` / `release_host` operations the
//!   Python facade drives its controller through.
//! - `hash_value` (page hashes for L3 storage) stays caller-side: the
//!   facade recomputes it from node keys (`compute_node_hash_values`)
//!   when storage is enabled.
//! - The deprecated write_back eviction loop is decomposed into the
//!   tree primitives it is made of (`detach_backuped`,
//!   `drop_subtree_no_host`, `evictable_leaves_ordered`,
//!   `promote_parent`, `begin_backup`, `protect_host`) so the facade
//!   can orchestrate it without duplicating tree state; the active
//!   write-through loop runs fully in Rust.
//!
//! Invariants maintained by every mutation (all mirroring Python):
//! - a node in `evictable_leaves` is live (non-evicted), unlocked, and
//!   has no live children;
//! - a node in `evictable_host_leaves` is evicted, host-unlocked, and
//!   has no backuped children (in practice: no children at all, since
//!   an evicted node's children, when they exist, are backuped — the
//!   contiguous-prefix backup invariant);
//! - backed-up nodes form a contiguous prefix from the root in
//!   write-through mode (`begin_backup` is only called by the facade
//!   after `write_backup`'s parent check).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::key::{common_prefix_len, RadixKey};
use crate::policy::{EvictionPolicy, Prio};
use crate::tree::{ChildKey, Head, NodeId, ROOT};

/// Host write policy (Python `hicache_write_policy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiPolicy {
    /// `write_through` / `write_through_selective`: backups fire on the
    /// hit-count threshold; device eviction demotes backuped leaves and
    /// drops the rest.
    WriteThrough,
    /// `write_back`: eviction stages non-backuped leaves to host first.
    /// Deprecated in Python; orchestrated by the facade via the tree
    /// primitives.
    WriteBack,
}

impl HiPolicy {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "write_through" | "write_through_selective" => Ok(Self::WriteThrough),
            "write_back" => Ok(Self::WriteBack),
            other => Err(format!(
                "unknown hicache write policy {other:?} (expected
write_through | write_through_selective | write_back)"
            )),
        }
    }
}

/// One HiRadix node. `key` is flattened (one element per token, two per
/// bigram); `value` and `host_value` each have one index per logical
/// unit.
#[derive(Debug, Clone)]
pub struct HiRadixNode {
    pub children: HashMap<ChildKey, NodeId>,
    pub parent: Option<NodeId>,
    pub key: Vec<i64>,
    /// Device KV; `None` = evicted (demoted to host only). Root:
    /// `Some(vec![])`.
    pub value: Option<Vec<i64>>,
    /// Host-tier copy; `None` = not backed up. Root: `Some(vec![])`
    /// (Python `reset` sets `root_node.host_value = []`, so the root is
    /// "backuped" and anchors the `last_host_node` walk).
    pub host_value: Option<Vec<i64>>,
    /// Device lock (node->root exclusive walk in inc/dec_lock_ref).
    pub lock_ref: u32,
    /// Host reference counter (prefetch / load-back protection).
    pub host_ref: u32,
    /// Write-through hit counter (backup trigger at the threshold).
    pub hit_count: u64,
    /// Eviction priority (lower evicts earlier under `Priority`).
    /// Root is `i32::MIN` (Python `-sys.maxsize`).
    pub priority: i32,
    /// Walk-clock stamp; one op stamps every visited node with the same
    /// tick.
    pub last_access: u64,
    /// A write-through backup was enqueued for this node (its host_value
    /// is set but the DMA ack has not been processed yet). Split nodes
    /// inherit it on both halves (Python
    /// `_replace_pending_write_through_node`).
    pub backup_pending: bool,
    pub id: NodeId,
}

impl HiRadixNode {
    pub fn evicted(&self) -> bool {
        self.value.is_none()
    }

    pub fn backuped(&self) -> bool {
        self.host_value.is_some()
    }

    fn logical_len(&self, unit_size: usize) -> usize {
        self.key.len() / unit_size
    }
}

/// Port of `match_prefix` (with the host-hit / last-host-node post
/// processing).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HiMatchResult {
    /// Device KV of the longest live-device prefix
    /// (`device_indices`); empty when the match ran into evicted or
    /// unknown keys.
    pub indices: Vec<i64>,
    /// `last_device_node`: the terminal of the match walked up through
    /// the evicted suffix (root when the whole match is evicted).
    pub last_device_node: NodeId,
    /// `last_host_node` / Python `best_match_node`: the deepest matched
    /// node walked up to the nearest backuped ancestor. The
    /// `init_load_back` start node.
    pub last_host_node: NodeId,
    /// Host tokens of the evicted suffix of the matched path.
    pub host_hit_length: usize,
    /// `(front, tail)` for every node split by this match (the facade
    /// re-links its pending write-through chains on these).
    pub splits: Vec<(NodeId, NodeId)>,
}

/// Port of `insert`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HiInsertResult {
    /// Live-device tokens already in the tree (evicted re-attach runs
    /// are NOT counted, faithful to `HiRadixCache.insert`).
    pub prefix_len: usize,
    /// Deepest node the walk touched (new leaf when created).
    pub last_node: NodeId,
    /// Nodes that hit the write-through threshold without a backup yet
    /// (the hit-count path fired `write_backup`). The caller enqueues
    /// the host DMA and then calls
    /// `begin_backup(node, host_indices, lock=true)`.
    pub backup_needed: Vec<NodeId>,
    /// `(front, tail)` for every node split by this insert.
    pub splits: Vec<(NodeId, NodeId)>,
}

/// Port of `evict` (the write-through loop, the default policy).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HiEvictResult {
    pub num_tokens_evicted: usize,
    /// Device runs released, in eviction order: demoted backuped leaves
    /// and dropped regular leaves alike. The caller releases them
    /// through the device allocator / controller.
    pub free_device: Vec<Vec<i64>>,
}

/// Port of `evict_host`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HiHostEvictResult {
    pub num_tokens_evicted: usize,
    /// Host runs released, in eviction order.
    pub free_host: Vec<Vec<i64>>,
    /// Node ids deleted from the tree, in eviction order.
    pub deleted: Vec<NodeId>,
}

/// A load-back in flight: the evicted chain plus its host indices. The
/// caller owns the DMA between `init_load_back` and
/// `finish_load_back` / `abort_load_back` (the retry after a failed
/// `evict` happens caller-side too, exactly like Python's
/// `cache_controller.load` retry).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadBackPlan {
    /// The non-evicted ancestor the chain was temporarily locked
    /// through.
    pub ancestor: NodeId,
    /// The requested (deepest evicted) node.
    pub last_node: NodeId,
    /// Chain in ancestor -> last order, all evicted + backuped. Each is
    /// host-protected and device-locked while the plan is open.
    pub nodes: Vec<NodeId>,
    /// Concatenated host indices of the chain (in `nodes` order) — the
    /// DMA source.
    pub host_indices: Vec<i64>,
}

/// Port of `_drop_subtree_no_host`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DropSubtreeResult {
    /// Device tokens freed; 0 when the subtree was refused (a node held
    /// a host reference).
    pub freed_device: i64,
    pub free_device: Vec<Vec<i64>>,
    pub free_host: Vec<Vec<i64>>,
}

#[derive(Clone)]
pub struct HiRadixTree {
    nodes: Vec<HiRadixNode>,
    page_size: usize,
    /// Flattened elements per logical unit: 1 (plain) or 2 (bigram).
    unit_size: usize,
    is_eagle: bool,
    disable: bool,
    policy: HiPolicy,
    strategy: EvictionPolicy,
    /// `write_through_threshold` (Python: 1 for write_through, 2 for
    /// write_through_selective).
    write_through_threshold: u64,
    /// `load_back_threshold` (Python: 10).
    load_back_threshold: usize,
    evictable_leaves: HashSet<NodeId>,
    evictable_host_leaves: HashSet<NodeId>,
    /// Device tokens in evictable (unlocked, live) nodes.
    evictable_size: i64,
    /// Device tokens in locked (protected) nodes.
    protected_size: i64,
    clock: u64,
    ns_map: HashMap<(String, String), u32>,
    ns_list: Vec<(String, String)>,
}

impl HiRadixTree {
    pub fn new(
        page_size: usize,
        is_eagle: bool,
        policy: HiPolicy,
        strategy: EvictionPolicy,
        write_through_threshold: u64,
        load_back_threshold: usize,
    ) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        assert!(
            write_through_threshold >= 1,
            "write_through_threshold must be >= 1"
        );
        let mut t = Self {
            nodes: Vec::new(),
            page_size,
            unit_size: if is_eagle { 2 } else { 1 },
            is_eagle,
            disable: false,
            policy,
            strategy,
            write_through_threshold,
            load_back_threshold,
            evictable_leaves: HashSet::new(),
            evictable_host_leaves: HashSet::new(),
            evictable_size: 0,
            protected_size: 0,
            clock: 0,
            ns_map: HashMap::new(),
            ns_list: Vec::new(),
        };
        t.reset();
        t
    }

    /// `reset` — recreate the root and clear all bookkeeping (Python
    /// `HiRadixCache.reset` + base `reset`).
    pub fn reset(&mut self) {
        let root = HiRadixNode {
            children: HashMap::new(),
            parent: None,
            key: vec![],
            value: Some(vec![]),
            host_value: Some(vec![]),
            lock_ref: 1,
            host_ref: 0,
            hit_count: 0,
            priority: i32::MIN,
            last_access: 0,
            backup_pending: false,
            id: ROOT,
        };
        self.nodes = vec![root];
        self.evictable_leaves.clear();
        self.evictable_host_leaves.clear();
        self.evictable_size = 0;
        self.protected_size = 0;
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

    pub fn policy(&self) -> HiPolicy {
        self.policy
    }

    pub fn strategy(&self) -> EvictionPolicy {
        self.strategy
    }

    pub fn write_through_threshold(&self) -> u64 {
        self.write_through_threshold
    }

    pub fn load_back_threshold(&self) -> usize {
        self.load_back_threshold
    }

    pub fn evictable_size(&self) -> i64 {
        self.evictable_size
    }

    pub fn protected_size(&self) -> i64 {
        self.protected_size
    }

    /// Device tokens across live nodes (Python `total_size`).
    pub fn total_size(&self) -> i64 {
        self.total_over(0)
    }

    /// Host tokens across backuped nodes (debug/parity helper; the root's
    /// empty `host_value` contributes 0).
    pub fn total_host_size(&self) -> i64 {
        self.total_over(1)
    }

    fn total_over(&self, tier: u8) -> i64 {
        let mut total = 0i64;
        let mut stack = vec![ROOT];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id as usize];
            let run = match tier {
                0 => n.value.as_deref(),
                _ => n.host_value.as_deref(),
            };
            if let Some(v) = run {
                total += (v.len() / self.unit_size) as i64;
            }
            for &child in n.children.values() {
                stack.push(child);
            }
        }
        total
    }

    // ---- node accessors (debug / parity tooling) ----

    pub fn node_children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .get(id as usize)
            .map(|n| n.children.values().copied().collect())
            .unwrap_or_default()
    }

    pub fn node_evicted(&self, id: NodeId) -> bool {
        self.nodes
            .get(id as usize)
            .map(|n| n.evicted())
            .unwrap_or(true)
    }

    pub fn node_backuped(&self, id: NodeId) -> bool {
        self.nodes
            .get(id as usize)
            .map(|n| n.backuped())
            .unwrap_or(false)
    }

    pub fn node_value(&self, id: NodeId) -> Option<Vec<i64>> {
        self.nodes
            .get(id as usize)
            .and_then(|n| n.value.clone())
    }

    pub fn node_host_value(&self, id: NodeId) -> Option<Vec<i64>> {
        self.nodes
            .get(id as usize)
            .and_then(|n| n.host_value.clone())
    }

    pub fn node_lock_ref(&self, id: NodeId) -> u32 {
        self.nodes
            .get(id as usize)
            .map(|n| n.lock_ref)
            .unwrap_or(u32::MAX)
    }

    pub fn node_host_ref(&self, id: NodeId) -> u32 {
        self.nodes
            .get(id as usize)
            .map(|n| n.host_ref)
            .unwrap_or(u32::MAX)
    }

    pub fn node_hit_count(&self, id: NodeId) -> u64 {
        self.nodes
            .get(id as usize)
            .map(|n| n.hit_count)
            .unwrap_or(u64::MAX)
    }

    pub fn node_priority(&self, id: NodeId) -> i32 {
        self.nodes
            .get(id as usize)
            .map(|n| n.priority)
            .unwrap_or(i32::MIN)
    }

    pub fn node_last_access(&self, id: NodeId) -> u64 {
        self.nodes
            .get(id as usize)
            .map(|n| n.last_access)
            .unwrap_or(u64::MAX)
    }

    pub fn node_key(&self, id: NodeId) -> Option<Vec<i64>> {
        self.nodes.get(id as usize).map(|n| n.key.clone())
    }

    pub fn node_backup_pending(&self, id: NodeId) -> bool {
        self.nodes
            .get(id as usize)
            .map(|n| n.backup_pending)
            .unwrap_or(false)
    }

    /// `strategy.get_priority(node)` for the facade's own heaps (the
    /// write_back path).
    pub fn node_prio(&self, id: NodeId) -> Prio {
        self.prio_of(id)
    }

    /// Members of the evictable-device-leaf set (order undefined).
    pub fn evictable_leaves(&self) -> Vec<NodeId> {
        self.evictable_leaves.iter().copied().collect()
    }

    /// Members of the evictable-host-leaf set (order undefined).
    pub fn evictable_host_leaves(&self) -> Vec<NodeId> {
        self.evictable_host_leaves.iter().copied().collect()
    }

    /// `evictable_leaves` in eviction order (heap order, ties by id) —
    /// the write_back facade's starting heap.
    pub fn evictable_leaves_ordered(&self) -> Vec<NodeId> {
        let mut v: Vec<NodeId> = self.evictable_leaves.iter().copied().collect();
        v.sort_by_key(|&id| self.prio_of(id));
        v
    }

    fn prio_of(&self, id: NodeId) -> Prio {
        let n = &self.nodes[id as usize];
        self.strategy.prio_fields(id, n.last_access, n.hit_count, n.priority)
    }

    // ---- ported tree ops ----

    /// Port of `match_prefix` (+ `_match_prefix_helper` + the
    /// host-hit / last-host-node post processing).
    pub fn match_prefix(&mut self, key: &RadixKey) -> HiMatchResult {
        let mut res = HiMatchResult::default();
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
            return res;
        }
        let flat = key.flatten_page_aligned(self.page_size);
        if flat.is_empty() {
            return res;
        }

        // `_match_prefix_helper`: one access_time stamps every visited
        // node (Python computes one `time.monotonic()` at entry).
        let op_time = self.tick();
        self.nodes[ROOT as usize].last_access = op_time;

        let mut node = ROOT;
        let mut runs: Vec<Vec<i64>> = Vec::new();
        let mut remaining: &[i64] = &flat;
        let mut ck = self.child_key_for(&key, is_bigram, remaining);
        while !remaining.is_empty() {
            let Some(&child) = self.nodes[node as usize].children.get(&ck) else {
                break;
            };
            self.nodes[child as usize].last_access = op_time;
            let m = self.page_floor_match(remaining, &self.nodes[child as usize].key);
            let child_logical = self.nodes[child as usize].logical_len(self.unit_size);
            if m < child_logical {
                // Match ends inside the child: split, take the front half
                // (device KV only when it is live).
                let front = self.split_node(child, m, is_bigram, &mut res.splits);
                if !self.nodes[front as usize].evicted() {
                    runs.push(
                        self.nodes[front as usize]
                            .value
                            .clone()
                            .unwrap_or_default(),
                    );
                }
                node = front;
                break;
            }
            if !self.nodes[child as usize].evicted() {
                runs.push(
                    self.nodes[child as usize]
                        .value
                        .clone()
                        .unwrap_or_default(),
                );
            }
            node = child;
            remaining = &remaining[m * self.unit_size..];
            if !remaining.is_empty() {
                ck = self.child_key_for(&key, is_bigram, remaining);
            }
        }

        // Host-hit post-processing (verbatim walk): climb the evicted
        // suffix counting host tokens, then climb the deepest node to
        // its nearest backuped ancestor (the root is backuped, so both
        // walks terminate).
        let mut last_device_node = node;
        let mut host_hit_length = 0usize;
        while self.nodes[last_device_node as usize].evicted() {
            host_hit_length += self.nodes[last_device_node as usize]
                .host_value
                .as_ref()
                .map(|v| v.len())
                .unwrap_or(0);
            last_device_node = self
                .nodes[last_device_node as usize]
                .parent
                .unwrap_or(ROOT);
        }
        let mut last_host_node = node;
        while !self.nodes[last_host_node as usize].backuped() {
            last_host_node = self
                .nodes[last_host_node as usize]
                .parent
                .unwrap_or(ROOT);
        }

        for run in &runs {
            res.indices.extend_from_slice(run);
        }
        res.last_device_node = last_device_node;
        res.last_host_node = last_host_node;
        res.host_hit_length = host_hit_length;
        res
    }

    /// Port of `insert` (the HiRadix override: evicted re-attach,
    /// hit-count write-through trigger, priority propagation).
    pub fn insert(
        &mut self,
        key: &RadixKey,
        value: &[i64],
        priority: i32,
        chunked: bool,
    ) -> HiInsertResult {
        let mut res = HiInsertResult::default();
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
        // Python: `value = value[:len(key)]` — logical, page-aligned
        // length (one KV index per logical unit, bigrams included).
        let logical = flat.len() / self.unit_size;
        let value = &value[..logical.min(value.len())];
        debug_assert_eq!(
            value.len(),
            logical,
            "insert value must have one KV index per logical unit"
        );

        // HiRadix `insert` stamps neither the root's priority nor its
        // last_access_time (the walk starts from the first child); the
        // one op-time stamps every visited child, and each visited
        // child's priority is maxed with the insert's.
        let op_time = self.tick();

        let mut node = ROOT;
        let mut total_prefix_length = 0usize;
        let mut remaining: &[i64] = &flat;
        let mut val: &[i64] = value;
        let mut ck = self.child_key_for(&key, is_bigram, remaining);
        while !remaining.is_empty() {
            let Some(&child) = self.nodes[node as usize].children.get(&ck) else {
                break;
            };
            {
                let c = &mut self.nodes[child as usize];
                c.last_access = op_time;
                c.priority = c.priority.max(priority);
            }
            let m = self.page_floor_match(remaining, &self.nodes[child as usize].key);
            let child_logical = self.nodes[child as usize].logical_len(self.unit_size);

            if m == child_logical {
                if self.nodes[child as usize].evicted() {
                    // Re-attach the device value (KV recomputation); the
                    // re-attached prefix is NOT counted in
                    // total_prefix_length.
                    self.nodes[child as usize].value = Some(val[..m].to_vec());
                    self.evictable_size += m as i64;
                    self.update_leaf_status(child);
                    self.update_host_leaf_status(child);
                    let p = self.nodes[child as usize]
                        .parent
                        .unwrap_or(ROOT);
                    self.update_leaf_status(p);
                } else {
                    self.inc_hit_count(child, chunked, &mut res.backup_needed);
                    total_prefix_length += m;
                }
                // Python advances `node` at the top of the loop on every
                // matched path.
                node = child;
            } else {
                let front = self.split_node(child, m, is_bigram, &mut res.splits);
                {
                    let f = &mut self.nodes[front as usize];
                    f.priority = f.priority.max(priority);
                }
                if self.nodes[front as usize].evicted() {
                    self.nodes[front as usize].value = Some(val[..m].to_vec());
                    self.evictable_size += m as i64;
                    self.update_leaf_status(front);
                    self.update_host_leaf_status(front);
                    let p = self.nodes[front as usize]
                        .parent
                        .unwrap_or(ROOT);
                    self.update_leaf_status(p);
                } else {
                    self.inc_hit_count(front, chunked, &mut res.backup_needed);
                    total_prefix_length += m;
                }
                node = front;
            }

            remaining = &remaining[m * self.unit_size..];
            val = &val[m..];
            if !remaining.is_empty() {
                ck = self.child_key_for(&key, is_bigram, remaining);
            }
        }

        if !remaining.is_empty() {
            let new_node =
                self.add_new_device_node(node, ck, remaining, val, priority, op_time);
            self.evictable_size += (remaining.len() / self.unit_size) as i64;
            self.update_leaf_status(node);
            self.update_leaf_status(new_node);
            // New device leaf: the write-through hit count starts at 1
            // (Python `_inc_hit_count(new_node)`), except in write_back.
            if self.policy == HiPolicy::WriteThrough {
                self.inc_hit_count(new_node, chunked, &mut res.backup_needed);
            }
            node = new_node;
        }

        res.prefix_len = total_prefix_length;
        res.last_node = node;
        res
    }

    /// Port of `evict` (the write-through loop).
    pub fn evict(&mut self, num_tokens: usize) -> HiEvictResult {
        let mut res = HiEvictResult::default();
        if self.disable || num_tokens == 0 {
            return res;
        }
        let mut heap = self.device_heap();

        let mut num_evicted = 0usize;
        while num_evicted < num_tokens {
            let Some(item) = heap.pop() else {
                break;
            };
            let x = item.id;
            if self.nodes[x as usize].lock_ref > 0 {
                continue;
            }
            if self.nodes[x as usize].backuped() {
                // `_evict_backuped`: demote (keep the host copy), the
                // caller releases the device run.
                let device = self.detach_backuped(x);
                num_evicted += device.len();
                res.free_device.push(device);
            } else {
                // `_evict_regular`: drop the node entirely.
                let device = self.evict_regular(x);
                num_evicted += device.len();
                res.free_device.push(device);
            }
            if let Some(p) = self.promote_parent(x) {
                heap.push(HeapItem {
                    prio: self.prio_of(p),
                    id: p,
                });
            }
        }
        res.num_tokens_evicted = num_evicted;
        res
    }

    /// Port of `evict_host`.
    pub fn evict_host(&mut self, num_tokens: usize) -> HiHostEvictResult {
        let mut res = HiHostEvictResult::default();
        if num_tokens == 0 {
            return res;
        }
        let mut heap: BinaryHeap<HeapItem> = self
            .evictable_host_leaves
            .iter()
            .map(|&id| HeapItem {
                prio: self.prio_of(id),
                id,
            })
            .collect();

        let mut num_evicted = 0usize;
        while num_evicted < num_tokens {
            let Some(item) = heap.pop() else {
                break;
            };
            let x = item.id;
            if x == ROOT {
                break;
            }
            // only evict the host value of evicted nodes
            if !self.nodes[x as usize].evicted() {
                continue;
            }
            if self.nodes[x as usize].host_ref > 0 {
                continue;
            }

            let hv = self.nodes[x as usize]
                .host_value
                .take()
                .unwrap_or_default();
            num_evicted += hv.len();
            res.free_host.push(hv);

            let parent = self.nodes[x as usize]
                .parent
                .unwrap_or(ROOT);
            let ck = self.child_key_to(parent, x);
            let removed = self.nodes[parent as usize].children.remove(&ck);
            assert!(
                removed == Some(x),
                "evict_host: parent does not have child key for node {x}"
            );
            self.evictable_host_leaves.remove(&x);
            self.update_host_leaf_status(parent);

            if self.nodes[parent as usize].children.is_empty()
                && self.nodes[parent as usize].evicted()
            {
                heap.push(HeapItem {
                    prio: self.prio_of(parent),
                    id: parent,
                });
            }
            res.deleted.push(x);
        }
        res.num_tokens_evicted = num_evicted;
        res
    }

    /// Phase 1 of `load_back`: collect the evicted chain, lock the
    /// ancestor, check the threshold/quota, and host-protect the chain.
    /// Returns `None` when the load was skipped (the temporary lock is
    /// released in that case) or when the requested node is already
    /// live.
    pub fn init_load_back(
        &mut self,
        last_node: NodeId,
        mem_quota: Option<i64>,
    ) -> Option<LoadBackPlan> {
        if !self.nodes[last_node as usize].evicted() {
            // `while node.evicted` never runs: nothing to load.
            return None;
        }
        // collect the evicted chain in ancestor -> last order
        let mut nodes: Vec<NodeId> = Vec::new();
        let mut up = last_node;
        while self.nodes[up as usize].evicted() {
            assert!(
                self.nodes[up as usize].backuped(),
                "load_back: no backup on evicted node {up}"
            );
            nodes.push(up);
            up = self.nodes[up as usize].parent.unwrap_or(ROOT);
        }
        nodes.reverse();
        let ancestor = up;
        let delta = self.inc_lock_ref(ancestor);

        let host_indices: Vec<i64> = nodes
            .iter()
            .flat_map(|&id| self.nodes[id as usize].host_value.clone().unwrap_or_default())
            .collect();
        if host_indices.len() < self.load_back_threshold
            || (mem_quota.is_some_and(|q| (host_indices.len() as i64) > q + delta))
        {
            // skip loading back: too small or over quota
            self.dec_lock_ref(ancestor);
            return None;
        }
        for &id in &nodes {
            self.nodes[id as usize].host_ref += 1;
        }
        Some(LoadBackPlan {
            ancestor,
            last_node,
            nodes,
            host_indices,
        })
    }

    /// Phase 2 of `load_back` with the DMA result. `device_indices`
    /// carries the freshly materialized device run, or `None` when the
    /// controller failed even after the caller's evict+retry:
    ///
    /// - always releases the temporary ancestor lock (Python calls
    ///   `dec_lock_ref(ancestor)` before the failure check);
    /// - on `None`: releases the host protections and returns 0;
    /// - on success: releases the host protections, re-attaches the
    ///   values, grows `evictable_size`, and takes the PERMANENT lock
    ///   over the chain (`inc_lock_ref(last_node)`), returning its
    ///   delta (the caller must `dec_lock_ref(last_node)` when the
    ///   request is done).
    pub fn finish_load_back(
        &mut self,
        plan: &LoadBackPlan,
        device_indices: Option<&[i64]>,
    ) -> i64 {
        self.dec_lock_ref(plan.ancestor);
        let Some(device_indices) = device_indices else {
            for &id in &plan.nodes {
                self.nodes[id as usize].host_ref -= 1;
            }
            return 0;
        };
        for &id in &plan.nodes {
            self.nodes[id as usize].host_ref -= 1;
        }
        let mut offset = 0usize;
        for &id in &plan.nodes {
            let n = &mut self.nodes[id as usize];
            let host_len = n.host_value.as_ref().map(|h| h.len()).unwrap_or(0);
            n.value = Some(device_indices[offset..offset + host_len].to_vec());
            offset += host_len;
        }
        self.evictable_size += device_indices.len() as i64;
        // `self.ongoing_load_back[last_hit_node.id]` stays caller-side.
        self.inc_lock_ref(plan.last_node)
    }

    /// Abandon an open load-back without a DMA result (request aborted):
    /// release the host protections and the temporary lock.
    pub fn abort_load_back(&mut self, plan: &LoadBackPlan) {
        self.dec_lock_ref(plan.ancestor);
        for &id in &plan.nodes {
            self.release_host(id);
        }
    }

    /// Port of the successful `write_backup` tree-side effects: attach
    /// the host copy, mark the backup pending (Python
    /// `write_through_pending_id`), and take the protective device lock
    /// (write-through only; write_back staging uses `protect_host`
    /// instead and passes `lock=false`).
    ///
    /// The caller has already done the `cache_controller.write` DMA and
    /// the host-memory fallback (`evict_host` + retry); it passes the
    /// resulting host indices. Returns the lock delta (0 when
    /// `lock=false`).
    pub fn begin_backup(&mut self, node: NodeId, host_indices: &[i64], lock: bool) -> i64 {
        assert!(
            !self.nodes[node as usize].evicted(),
            "begin_backup on evicted node {node}"
        );
        assert!(!host_indices.is_empty(), "begin_backup with empty host run");
        self.nodes[node as usize].host_value = Some(host_indices.to_vec());
        self.nodes[node as usize].backup_pending = true;
        if lock {
            self.inc_lock_ref(node)
        } else {
            0
        }
    }

    /// DMA ack processed: clear the pending flag on one publish node.
    /// (Split nodes carry the flag on both halves; the facade walks its
    /// publish list and calls this for each pending node.)
    pub fn end_backup(&mut self, node: NodeId) {
        self.nodes[node as usize].backup_pending = false;
    }

    /// `node.protect_host()`.
    pub fn protect_host(&mut self, node: NodeId) {
        self.nodes[node as usize].host_ref += 1;
    }

    /// `node.release_host()` — Python raises `RuntimeError` at 0; a
    /// host-ref underflow is a caller bug, so this asserts.
    pub fn release_host(&mut self, node: NodeId) {
        let n = &mut self.nodes[node as usize];
        assert!(n.host_ref > 0, "release_host on node {node} with host_ref == 0");
        n.host_ref -= 1;
    }

    /// Port of `inc_lock_ref`: walk to the root (terminal node included,
    /// root excluded). Returns the delta of tokens moved
    /// evictable -> protected (<= 0).
    pub fn inc_lock_ref(&mut self, node: NodeId) -> i64 {
        if self.disable {
            return 0;
        }
        let mut delta = 0i64;
        let mut cur = Some(node);
        while let Some(id) = cur {
            if id == ROOT {
                break; // Python: `while node != self.root_node`
            }
            let parent = {
                let n = &mut self.nodes[id as usize];
                if n.lock_ref == 0 {
                    let l = n.logical_len(self.unit_size) as i64;
                    self.evictable_size -= l;
                    self.protected_size += l;
                    delta -= l;
                }
                n.lock_ref += 1;
                n.parent
            };
            self.update_leaf_status(id);
            self.update_host_leaf_status(id);
            cur = parent;
        }
        delta
    }

    /// Port of `dec_lock_ref`. Returns tokens moved protected ->
    /// evictable (>= 0).
    pub fn dec_lock_ref(&mut self, node: NodeId) -> i64 {
        if self.disable {
            return 0;
        }
        let mut delta = 0i64;
        let mut cur = Some(node);
        while let Some(id) = cur {
            if id == ROOT {
                break; // Python: `while node != self.root_node`
            }
            let parent = {
                let n = &mut self.nodes[id as usize];
                if n.lock_ref == 1 {
                    let l = n.logical_len(self.unit_size) as i64;
                    self.evictable_size += l;
                    self.protected_size -= l;
                    delta += l;
                }
                n.lock_ref -= 1;
                n.parent
            };
            self.update_leaf_status(id);
            self.update_host_leaf_status(id);
            cur = parent;
        }
        delta
    }

    // ---- write_back facade primitives ----

    /// Port of `_detach_backuped`: demote a node to host-only while the
    /// caller keeps the device slots for its staged DMA (write_back
    /// eviction). Returns the device run.
    pub fn detach_backuped(&mut self, node: NodeId) -> Vec<i64> {
        let device = self.nodes[node as usize]
            .value
            .take()
            .unwrap_or_default();
        assert!(
            !device.is_empty(),
            "detach_backuped on empty-value node {node}"
        );
        self.evictable_size -= device.len() as i64;
        self.update_leaf_status(node);
        self.update_host_leaf_status(node);
        let parent = self.nodes[node as usize]
            .parent
            .unwrap_or(ROOT);
        self.update_leaf_status(parent);
        device
    }

    /// Port of `_drop_subtree_no_host`: free host + device of an entire
    /// subtree and detach it. Refused (no-op, 0) when any node in the
    /// subtree holds a host reference.
    pub fn drop_subtree_no_host(&mut self, root: NodeId) -> DropSubtreeResult {
        let mut res = DropSubtreeResult::default();
        let mut nodes: Vec<NodeId> = Vec::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            nodes.push(id);
            for &child in self.nodes[id as usize].children.values() {
                stack.push(child);
            }
        }
        if nodes
            .iter()
            .any(|&id| self.nodes[id as usize].host_ref > 0)
        {
            return res;
        }

        for &id in &nodes {
            if self.nodes[id as usize].host_value.is_some() {
                res.free_host.push(
                    self.nodes[id as usize]
                        .host_value
                        .take()
                        .unwrap_or_default(),
                );
            }
            if self.nodes[id as usize].value.is_some() {
                let v = self.nodes[id as usize]
                    .value
                    .take()
                    .unwrap_or_default();
                res.freed_device += v.len() as i64;
                self.evictable_size -= v.len() as i64;
                res.free_device.push(v);
            }
            self.nodes[id as usize].backup_pending = false;
            self.evictable_leaves.remove(&id);
            self.evictable_host_leaves.remove(&id);
        }

        let parent = self.nodes[root as usize]
            .parent
            .expect("drop_subtree_no_host root must have a parent");
        let ck = self.child_key_to(parent, root);
        self.nodes[parent as usize].children.remove(&ck);
        self.update_leaf_status(parent);
        self.update_host_leaf_status(parent);
        res
    }

    /// Port of `_promote_parent`: the node's parent becomes a device
    /// leaf once all of its children are evicted (root excluded).
    /// Returns the parent id for the caller's heap.
    pub fn promote_parent(&mut self, node: NodeId) -> Option<NodeId> {
        let p = self.nodes[node as usize].parent?;
        if p == ROOT {
            return None;
        }
        let all_evicted = self.nodes[p as usize]
            .children
            .values()
            .all(|&c| self.nodes[c as usize].evicted());
        all_evicted.then_some(p)
    }

    /// Port of `_insert_helper_host` (the prefetch-result path creating
    /// host-ONLY nodes: `value = None`, `host_value` set).
    /// Returns the matched length (the caller frees the overlapping
    /// host indices of the prefetch).
    pub fn insert_host(
        &mut self,
        start_node: NodeId,
        key: &RadixKey,
        host_value: &[i64],
    ) -> usize {
        let mut matched_length = 0usize;
        if self.disable {
            return matched_length;
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
            return matched_length;
        }

        let op_time = self.tick();
        {
            let n = &mut self.nodes[start_node as usize];
            n.last_access = op_time;
        }

        let mut node = start_node;
        let mut remaining: &[i64] = &flat;
        let mut hv: &[i64] = host_value;
        let mut ck = self.child_key_for(&key, is_bigram, remaining);
        while !remaining.is_empty() {
            let Some(&child) = self.nodes[node as usize].children.get(&ck) else {
                break;
            };
            self.nodes[child as usize].last_access = op_time;
            let m = self.page_floor_match(remaining, &self.nodes[child as usize].key);
            matched_length += m;
            remaining = &remaining[m * self.unit_size..];
            hv = &hv[m..];
            if m < self.nodes[child as usize].logical_len(self.unit_size) {
                let front = self.split_node(child, m, is_bigram, &mut Vec::new());
                node = front;
            } else {
                node = child;
            }
            if !remaining.is_empty() {
                ck = self.child_key_for(&key, is_bigram, remaining);
            }
        }

        if !remaining.is_empty() {
            let new_node = self.add_new_host_node(node, ck, remaining, hv, op_time);
            self.update_host_leaf_status(new_node);
            self.update_leaf_status(node);
            self.update_host_leaf_status(node);
        }
        matched_length
    }

    // ---- internals ----

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn device_heap(&self) -> BinaryHeap<HeapItem> {
        self.evictable_leaves
            .iter()
            .map(|&id| HeapItem {
                prio: self.prio_of(id),
                id,
            })
            .collect()
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

    fn child_key_for(&mut self, key: &RadixKey, is_bigram: bool, remaining: &[i64]) -> ChildKey {
        let ns = self.intern_ns(key.extra_key, key.cache_salt);
        let head = self.head_from_flat(remaining, is_bigram);
        ChildKey { ns, head }
    }

    /// The child key naming `child` in `parent`'s children map.
    fn child_key_to(&self, parent: NodeId, child: NodeId) -> ChildKey {
        self.nodes[parent as usize]
            .children
            .iter()
            .find(|item| item.1 == &child)
            .map(|(k, _)| k.clone())
            .expect("child missing from parent's children map")
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

    /// Shared logical units between `remaining` and a stored key, floored
    /// to `page_size` — Python `RadixKey.match`.
    fn page_floor_match(&self, remaining: &[i64], child_key: &[i64]) -> usize {
        let m_flat = common_prefix_len(remaining, child_key);
        m_flat / self.unit_size / self.page_size * self.page_size
    }

    /// Port of `HiRadixCache._split_node`: split `child` after
    /// `split_logical` logical units. The front node inherits the
    /// child's priority / hit_count / lock_ref / pending flag; `value`
    /// and `host_value` are sliced only when present. Returns the front
    /// id; the tail keeps `child`'s id.
    fn split_node(
        &mut self,
        child: NodeId,
        split_logical: usize,
        is_bigram: bool,
        splits: &mut Vec<(NodeId, NodeId)>,
    ) -> NodeId {
        let cut = split_logical * self.unit_size;
        assert!(
            cut > 0 && cut < self.nodes[child as usize].key.len(),
            "split_node: bad split point {cut} for node {child}"
        );

        // Snapshot everything we need from the child before mutation.
        let (child_key, child_value, child_host, gp, priority, hit, lock, pending) = {
            let c = &self.nodes[child as usize];
            (
                c.key.clone(),
                c.value.clone(),
                c.host_value.clone(),
                c.parent.expect("split target must have a parent"),
                c.priority,
                c.hit_count,
                c.lock_ref,
                c.backup_pending,
            )
        };
        let ns = self.nodes[gp as usize]
            .children
            .iter()
            .find(|item| item.1 == &child)
            .map(|(k, _)| k.ns)
            .expect("split target missing from parent's children");

        let front_key = child_key[..cut].to_vec();
        let tail_key = child_key[cut..].to_vec();
        // Values are logical (one KV index per unit); sliced only when
        // the tier is present.
        let front_value = child_value.as_ref().map(|v| v[..split_logical].to_vec());
        let tail_value = child_value.map(|v| v[split_logical..].to_vec());
        let front_host = child_host.as_ref().map(|h| h[..split_logical].to_vec());
        let tail_host = child_host.map(|h| h[split_logical..].to_vec());

        let new_id = self.nodes.len() as NodeId;
        let front_ck = ChildKey {
            ns,
            head: self.head_from_flat(&front_key, is_bigram),
        };
        let tail_ck = ChildKey {
            ns,
            head: self.head_from_flat(&tail_key, is_bigram),
        };

        let mut children = HashMap::with_capacity(1);
        children.insert(tail_ck.clone(), child);
        let front = HiRadixNode {
            children,
            parent: Some(gp),
            key: front_key,
            value: front_value,
            host_value: front_host,
            lock_ref: lock,
            host_ref: 0,
            hit_count: hit,
            priority,
            last_access: self.tick(),
            backup_pending: pending,
            id: new_id,
        };
        self.nodes.push(front);

        // Mutate the tail node in place.
        {
            let c = &mut self.nodes[child as usize];
            c.key = tail_key;
            c.value = tail_value;
            c.host_value = tail_host;
            c.parent = Some(new_id);
        }
        // Re-parent at the grandparent: the old entry pointed at
        // `child`; it now points at the new front node, which owns the
        // tail.
        {
            let g = &mut self.nodes[gp as usize];
            g.children.remove(&tail_ck);
            g.children.insert(front_ck, new_id);
        }

        splits.push((new_id, child));
        new_id
    }

    /// New live leaf (device) under `parent` via `ck`; one walk-time
    /// stamp.
    fn add_new_device_node(
        &mut self,
        parent: NodeId,
        ck: ChildKey,
        key: &[i64],
        value: &[i64],
        priority: i32,
        last_access: u64,
    ) -> NodeId {
        let new_id = self.nodes.len() as NodeId;
        let new = HiRadixNode {
            children: HashMap::new(),
            parent: Some(parent),
            key: key.to_vec(),
            value: Some(value.to_vec()),
            host_value: None,
            lock_ref: 0,
            host_ref: 0,
            hit_count: 0,
            priority,
            last_access,
            backup_pending: false,
            id: new_id,
        };
        self.nodes.push(new);
        self.nodes[parent as usize].children.insert(ck, new_id);
        new_id
    }

    /// New host-ONLY leaf under `parent` via `ck` (prefetch result
    /// path): `value = None`, `host_value` set, priority inherited from
    /// the parent.
    fn add_new_host_node(
        &mut self,
        parent: NodeId,
        ck: ChildKey,
        key: &[i64],
        host_value: &[i64],
        last_access: u64,
    ) -> NodeId {
        let new_id = self.nodes.len() as NodeId;
        let priority = self.nodes[parent as usize].priority;
        let new = HiRadixNode {
            children: HashMap::new(),
            parent: Some(parent),
            key: key.to_vec(),
            value: None,
            host_value: Some(host_value.to_vec()),
            lock_ref: 0,
            host_ref: 0,
            hit_count: 0,
            priority,
            last_access,
            backup_pending: false,
            id: new_id,
        };
        self.nodes.push(new);
        self.nodes[parent as usize].children.insert(ck, new_id);
        new_id
    }

    /// Port of `_inc_hit_count` (HiRadix override): bump the hit count
    /// (write-through mode only, non-chunked) and request a backup when
    /// the threshold is crossed.
    fn inc_hit_count(&mut self, node: NodeId, chunked: bool, out: &mut Vec<NodeId>) {
        if self.policy == HiPolicy::WriteBack || chunked {
            return;
        }
        let n = &mut self.nodes[node as usize];
        n.hit_count += 1;
        if !n.backuped() && n.hit_count >= self.write_through_threshold {
            out.push(node);
        }
    }

    /// Port of `_evict_regular`: drop a non-backuped leaf entirely.
    fn evict_regular(&mut self, node: NodeId) -> Vec<i64> {
        assert!(
            self.nodes[node as usize].children.is_empty(),
            "evict_regular on non-leaf node {} (children present)",
            node
        );
        let device = self.nodes[node as usize]
            .value
            .take()
            .unwrap_or_default();
        self.delete_leaf(node);
        device
    }

    /// Port of `_delete_leaf`.
    fn delete_leaf(&mut self, node: NodeId) {
        let parent = self.nodes[node as usize]
            .parent
            .expect("evicted node has no parent?");
        let key_len = self.nodes[node as usize].logical_len(self.unit_size);
        let ck = self.child_key_to(parent, node);
        let removed = self.nodes[parent as usize].children.remove(&ck);
        assert!(removed == Some(node), "parent does not have child key");
        self.evictable_size -= key_len as i64;
        self.evictable_leaves.remove(&node);
        self.update_leaf_status(parent);
    }

    /// Port of the inherited `_update_leaf_status`.
    fn update_leaf_status(&mut self, node: NodeId) {
        let n = &self.nodes[node as usize];
        if n.evicted() || n.lock_ref > 0 {
            self.evictable_leaves.remove(&node);
            return;
        }
        for &child in n.children.values() {
            if !self.nodes[child as usize].evicted() {
                self.evictable_leaves.remove(&node);
                return;
            }
        }
        self.evictable_leaves.insert(node);
    }

    /// Port of `HiRadixCache._update_host_leaf_status`.
    fn update_host_leaf_status(&mut self, node: NodeId) {
        let n = &self.nodes[node as usize];
        if !n.evicted() || n.lock_ref > 0 {
            self.evictable_host_leaves.remove(&node);
            return;
        }
        for &child in n.children.values() {
            if self.nodes[child as usize].backuped() {
                self.evictable_host_leaves.remove(&node);
                return;
            }
        }
        self.evictable_host_leaves.insert(node);
    }
}

impl Default for HiRadixTree {
    fn default() -> Self {
        Self::new(
            1,
            false,
            HiPolicy::WriteThrough,
            EvictionPolicy::Lru,
            1,
            10,
        )
    }
}

#[derive(Clone, Copy)]
struct HeapItem {
    prio: Prio,
    id: NodeId,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.prio == other.prio && self.id == other.id
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    // BinaryHeap is a max-heap; invert so the SMALLEST prio pops first
    // (the id tie-break rides on Prio.c).
    fn cmp(&self, other: &Self) -> Ordering {
        other.prio.key().cmp(&self.prio.key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// write-through / LRU / threshold 1 / load_back_threshold 10.
    fn wt() -> HiRadixTree {
        HiRadixTree::new(1, false, HiPolicy::WriteThrough, EvictionPolicy::Lru, 1, 10)
    }

    fn key<'a>(tokens: &'a [i64]) -> RadixKey<'a> {
        RadixKey::new(tokens)
    }

    /// Find the root's child whose value is `value` (child-map order is
    /// arbitrary).
    fn child_with_value(t: &HiRadixTree, value: &[i64]) -> u32 {
        t.node_children(ROOT)
            .into_iter()
            .find(|&id| t.node_value(id).as_deref() == Some(value))
            .expect("child with that value not found")
    }

    fn insert(t: &mut HiRadixTree, tokens: &[i64], vals: &[i64]) -> HiInsertResult {
        t.insert(&key(tokens), vals, 0, false)
    }

    /// Insert `tokens` with fresh device values `100..` and return the
    /// new leaf id.
    fn insert_vals(t: &mut HiRadixTree, tokens: &[i64], base: i64) -> (HiInsertResult, i64) {
        let vals: Vec<i64> = (0..tokens.len()).map(|i| base + i as i64).collect();
        let r = insert(t, tokens, &vals);
        (r, base)
    }

    #[test]
    fn match_host_hit_length_and_last_host_node() {
        let mut t = wt();
        let key_a: Vec<i64> = (0..4).collect();
        let (r1, _) = insert_vals(&mut t, &key_a, 0);
        let a = r1.last_node;
        // extend: child B over [4,8)
        let key_ab: Vec<i64> = (0..8).collect();
        let (r2, _) = insert_vals(&mut t, &key_ab, 100);
        let b = r2.last_node;
        assert_ne!(a, b);

        // back up both (write-through: lock then "ack")
        let ha: Vec<i64> = (1000..1004).collect();
        let hb: Vec<i64> = (2000..2004).collect();
        t.begin_backup(a, &ha, true);
        t.dec_lock_ref(a);
        t.begin_backup(b, &hb, true);
        t.dec_lock_ref(b);

        // evict the whole chain from the device (both demoted)
        let ev = t.evict(8);
        assert_eq!(ev.num_tokens_evicted, 8);
        assert!(t.node_evicted(a) && t.node_evicted(b));
        assert!(t.node_children(b).is_empty());

        let m = t.match_prefix(&key(&key_ab));
        assert!(m.indices.is_empty(), "no device KV left");
        assert_eq!(m.host_hit_length, 8);
        assert_eq!(m.last_device_node, ROOT);
        assert_eq!(m.last_host_node, b);
    }

    #[test]
    fn insert_reattach_not_counted() {
        let mut t = wt();
        let tokens: Vec<i64> = (0..4).collect();
        let (r1, _) = insert_vals(&mut t, &tokens, 0);
        let a = r1.last_node;
        assert_eq!(r1.prefix_len, 0);

        let ha: Vec<i64> = (1000..1004).collect();
        t.begin_backup(a, &ha, true);
        t.dec_lock_ref(a);
        let ev = t.evict(4);
        assert_eq!(ev.num_tokens_evicted, 4);
        assert!(t.node_evicted(a));
        assert_eq!(t.evictable_size(), 0);

        // re-insert the same prefix: the node is re-attached, NOT
        // counted as a live prefix.
        let vals: Vec<i64> = (50..54).collect();
        let r2 = t.insert(&key(&tokens), &vals, 0, false);
        assert_eq!(r2.prefix_len, 0, "re-attach must not count toward the prefix");
        assert_eq!(r2.last_node, a);
        assert!(!t.node_evicted(a));
        assert_eq!(t.node_value(a), Some(vals.clone()));
        assert_eq!(t.node_host_value(a), Some(ha.clone()));
        assert_eq!(t.evictable_size(), 4);
        assert_eq!(t.total_size(), 4);
    }

    #[test]
    fn hit_count_threshold_triggers_backup_needed() {
        // threshold 2: new leaf hit=1 (no backup), second insert hit=2.
        let mut t = HiRadixTree::new(1, false, HiPolicy::WriteThrough, EvictionPolicy::Lru, 2, 10);
        let tokens: Vec<i64> = (0..4).collect();
        let vals: Vec<i64> = (0..4).collect();
        let r1 = t.insert(&key(&tokens), &vals, 0, false);
        let a = r1.last_node;
        assert!(
            r1.backup_needed.is_empty(),
            "hit_count 1 must not trigger at threshold 2"
        );
        assert_eq!(t.node_hit_count(a), 1);

        let r2 = t.insert(&key(&tokens), &vals, 0, false);
        assert_eq!(r2.prefix_len, 4);
        assert_eq!(r2.backup_needed, vec![a]);
        assert_eq!(t.node_hit_count(a), 2);
        assert!(!t.node_backuped(a), "backup is the caller's job");

        // the caller enqueues the DMA and attaches the host copy
        let ha: Vec<i64> = (9000..9004).collect();
        let delta = t.begin_backup(a, &ha, true);
        assert_eq!(delta, -4);
        assert!(t.node_backuped(a));
        assert!(t.node_backup_pending(a));
        assert_eq!(t.node_lock_ref(a), 1);
        t.end_backup(a);
        assert!(!t.node_backup_pending(a));
        // ack the protective lock
        assert_eq!(t.dec_lock_ref(a), 4);
    }

    #[test]
    fn evict_demotes_backuped_and_drops_regular_in_lru_order() {
        let mut t = wt();
        let ka: Vec<i64> = (0..4).collect();
        let kb: Vec<i64> = (8..12).collect();
        let va: Vec<i64> = (0..4).collect();
        let vb: Vec<i64> = (100..104).collect();
        let r1 = t.insert(&key(&ka), &va, 0, false);
        let a = r1.last_node;
        let r2 = t.insert(&key(&kb), &vb, 0, false);
        let b = r2.last_node;

        // back up A only (older under LRU)
        let ha: Vec<i64> = (1000..1004).collect();
        t.begin_backup(a, &ha, true);
        t.dec_lock_ref(a);

        let ev = t.evict(8);
        assert_eq!(ev.num_tokens_evicted, 8);
        assert_eq!(ev.free_device, vec![va.clone(), vb.clone()]);
        // A demoted (kept in tree with its host copy), B deleted
        assert!(t.node_evicted(a));
        assert_eq!(t.node_host_value(a), Some(ha));
        assert!(t.node_value(b).is_none());
        assert_eq!(t.node_children(ROOT), vec![a]);
        assert_eq!(t.evictable_size(), 0);
    }

    #[test]
    fn evict_respects_locks_and_promotes_parent() {
        let mut t = wt();
        let tokens: Vec<i64> = (0..8).collect();
        // two-node chain: A=[0,4), B=[4,8)
        let half: Vec<i64> = (0..4).collect();
        insert_vals(&mut t, &half, 0);
        let vals8: Vec<i64> = (0..8).collect();
        let r2 = insert(&mut t, &tokens, &vals8);
        let b = r2.last_node;
        let a = t.node_children(ROOT)[0];
        assert_ne!(a, b);

        // lock A (device protection); B stays unlocked
        let delta = t.inc_lock_ref(a);
        assert_eq!(delta, -4);

        let ev = t.evict(8);
        // B is dropped (4 tokens); A is promoted but skipped: locked.
        assert_eq!(ev.num_tokens_evicted, 4);
        assert_eq!(ev.free_device.len(), 1);
        assert!(!t.node_evicted(a), "locked node must survive eviction");
        // A is a live leaf again (its only child is gone) but still
        // locked, so it must stay OUT of the evictable set
        assert!(!t.evictable_leaves().contains(&a));

        // unlock: A re-enters the set
        assert_eq!(t.dec_lock_ref(a), 4);
        assert!(t.evictable_leaves().contains(&a));
    }

    #[test]
    fn evict_host_skip_rules_and_parent_promotion() {
        let mut t = wt();
        let tokens: Vec<i64> = (0..8).collect();
        let half: Vec<i64> = (0..4).collect();
        insert_vals(&mut t, &half, 0);
        let vals8: Vec<i64> = (0..8).collect();
        let r2 = insert(&mut t, &tokens, &vals8);
        let b = r2.last_node;
        let a = t.node_children(ROOT)[0];

        let ha: Vec<i64> = (1000..1004).collect();
        let hb: Vec<i64> = (2000..2004).collect();
        t.begin_backup(a, &ha, true);
        t.dec_lock_ref(a);
        t.begin_backup(b, &hb, true);
        t.dec_lock_ref(b);

        // demote the whole chain from the device
        let ev = t.evict(8);
        assert_eq!(ev.num_tokens_evicted, 8);
        assert!(t.node_evicted(a) && t.node_evicted(b));
        // host-leaf set: only B (A has a backuped child)
        assert_eq!(t.evictable_host_leaves(), vec![b]);

        // host-protect B: eviction must skip it
        t.protect_host(b);
        let eh = t.evict_host(8);
        assert_eq!(eh.num_tokens_evicted, 0);
        assert!(t.node_children(a).contains(&b), "protected node survives");
        t.release_host(b);

        // now evict the whole host chain: B first, then promoted A
        let eh = t.evict_host(8);
        assert_eq!(eh.num_tokens_evicted, 8);
        assert_eq!(eh.deleted, vec![b, a]);
        assert_eq!(eh.free_host, vec![hb.clone(), ha.clone()]);
        assert_eq!(t.node_children(ROOT), vec![]);
        assert!(t.evictable_host_leaves().is_empty());
    }

    #[test]
    fn load_back_skips_below_threshold_and_releases_lock() {
        let mut t = wt(); // threshold 10
        let tokens: Vec<i64> = (0..4).collect();
        let (_, _) = insert_vals(&mut t, &tokens, 0);
        let a = t.node_children(ROOT)[0];
        let ha: Vec<i64> = (1000..1004).collect();
        t.begin_backup(a, &ha, true);
        t.dec_lock_ref(a);
        assert_eq!(t.evict(4).num_tokens_evicted, 4);
        assert!(t.node_evicted(a));

        // 4 < 10: skipped, temporary lock must be released
        let plan = t.init_load_back(a, None);
        assert!(plan.is_none());
        assert_eq!(t.node_lock_ref(a), 0);
        assert_eq!(t.node_host_ref(a), 0);
        assert!(t.node_evicted(a));
        assert_eq!(t.evictable_size(), 0);
    }

    #[test]
    fn load_back_roundtrip_reattaches_and_locks_chain() {
        let mut t = wt();
        let tokens12: Vec<i64> = (0..12).collect();
        let first6: Vec<i64> = (0..6).collect();
        insert_vals(&mut t, &first6, 0);
        let vals12: Vec<i64> = (0..12).collect();
        insert(&mut t, &tokens12, &vals12);
        let b = t.node_children(t.node_children(ROOT)[0])[0];
        let a = t.node_children(ROOT)[0];
        let ha: Vec<i64> = (1000..1006).collect();
        let hb: Vec<i64> = (2000..2006).collect();
        t.begin_backup(a, &ha, true);
        t.dec_lock_ref(a);
        t.begin_backup(b, &hb, true);
        t.dec_lock_ref(b);
        assert_eq!(t.evict(12).num_tokens_evicted, 12);

        let plan = t.init_load_back(b, None).expect("12 >= threshold 10");
        assert_eq!(plan.ancestor, ROOT);
        assert_eq!(plan.nodes, vec![a, b]);
        assert_eq!(plan.host_indices, [1000, 1001, 1002, 1003, 1004, 1005, 2000, 2001, 2002, 2003, 2004, 2005]);
        assert_eq!(t.node_host_ref(a), 1);
        assert_eq!(t.node_host_ref(b), 1);

        let device: Vec<i64> = (500..512).collect();
        let lock_delta = t.finish_load_back(&plan, Some(&device));
        assert_eq!(lock_delta, -12, "permanent chain lock moved 12 tokens to protected");
        assert_eq!(t.node_value(a), Some(device[..6].to_vec()));
        assert_eq!(t.node_value(b), Some(device[6..12].to_vec()));
        // the re-attached tokens are immediately locked by the chain
        // lock, so all 12 are protected and none evictable
        assert_eq!(t.evictable_size(), 0);
        assert_eq!(t.protected_size(), 12);
        assert_eq!(t.node_lock_ref(a), 1);
        assert_eq!(t.node_lock_ref(b), 1);
        // host copies survive the load-back
        assert_eq!(t.node_host_value(a), Some(ha));
        assert_eq!(t.node_host_value(b), Some(hb));
        // caller releases the permanent lock when the request ends
        assert_eq!(t.dec_lock_ref(b), 12);
        assert_eq!(t.evictable_size(), 12);
        assert_eq!(t.protected_size(), 0);
    }

    #[test]
    fn load_back_quota_and_failure_paths() {
        // A live (10 tokens), B evicted+backuped (10 tokens)
        let mut t = wt();
        let tokens20: Vec<i64> = (0..20).collect();
        let first10: Vec<i64> = (0..10).collect();
        insert_vals(&mut t, &first10, 0);
        let vals20: Vec<i64> = (0..20).collect();
        insert(&mut t, &tokens20, &vals20);
        let b = t.node_children(t.node_children(ROOT)[0])[0];
        let a = t.node_children(ROOT)[0];
        let hb: Vec<i64> = (2000..2010).collect();
        t.begin_backup(b, &hb, true);
        t.dec_lock_ref(b);
        assert_eq!(t.evict(10).num_tokens_evicted, 10);
        assert!(t.node_evicted(b));

        // quota: 10 tokens > quota(15) + delta(-10) = 5 -> skip
        let plan = t.init_load_back(b, Some(15));
        assert!(plan.is_none());
        assert_eq!(t.node_lock_ref(a), 0, "temporary lock released on skip");

        let plan = t.init_load_back(b, None).expect("no quota: load proceeds");
        assert_eq!(plan.ancestor, a);
        assert_eq!(plan.nodes, vec![b]);
        assert_eq!(t.node_lock_ref(a), 1, "temporary ancestor lock held");

        // DMA failure (even after the caller's evict+retry)
        let delta = t.finish_load_back(&plan, None);
        assert_eq!(delta, 0);
        assert!(t.node_evicted(b), "no value attached on failure");
        assert_eq!(t.node_lock_ref(a), 0, "temporary lock released");
        assert_eq!(t.node_host_ref(b), 0, "host protection released");
        // host copy is intact for a later retry
        assert_eq!(t.node_host_value(b), Some(hb));
    }

    #[test]
    fn split_slices_host_value_and_inherits_state() {
        let mut t = wt();
        let tokens: Vec<i64> = (0..10).collect();
        let vals: Vec<i64> = (0..10).collect();
        let r = t.insert(&key(&tokens), &vals, 5, false);
        let c = r.last_node;
        let hc: Vec<i64> = (9000..9010).collect();
        t.begin_backup(c, &hc, false); // no lock, pending=true

        // match 4 of the 10: split c into front(4) + tail(6)
        let four: Vec<i64> = (0..4).collect();
        let m = t.match_prefix(&key(&four));
        assert_eq!(m.indices, vals[..4].to_vec());
        assert_eq!(m.splits.len(), 1);
        let (front, tail) = m.splits[0];
        assert_eq!(tail, c, "the tail keeps the original id");
        assert_eq!(front, t.node_children(ROOT)[0]);
        assert_eq!(t.node_children(front), vec![c]);

        assert_eq!(t.node_value(front), Some(vals[..4].to_vec()));
        assert_eq!(t.node_value(c), Some(vals[4..].to_vec()));
        assert_eq!(t.node_host_value(front), Some(hc[..4].to_vec()));
        assert_eq!(t.node_host_value(c), Some(hc[4..].to_vec()));
        // inherited state on the front
        assert_eq!(t.node_hit_count(front), 1);
        assert_eq!(t.node_priority(front), 5);
        assert_eq!(t.node_lock_ref(front), 0);
        // pending flag rides the split onto both halves
        assert!(t.node_backup_pending(front));
        assert!(t.node_backup_pending(c));
        // fresh creation tick on the front
        assert!(t.node_last_access(front) > t.node_last_access(c));

        // match post-processing on the split path
        assert_eq!(m.last_device_node, front);
        assert_eq!(m.last_host_node, front, "the front is backuped");
        assert_eq!(m.host_hit_length, 0);

        t.end_backup(front);
        t.end_backup(c);
        assert!(!t.node_backup_pending(front) && !t.node_backup_pending(c));
    }

    #[test]
    fn lock_walk_updates_leaf_sets() {
        let mut t = wt();
        let ka: Vec<i64> = (0..4).collect();
        let kb: Vec<i64> = (8..12).collect();
        insert_vals(&mut t, &ka, 0);
        insert_vals(&mut t, &kb, 100);
        let a = child_with_value(&t, &[0, 1, 2, 3]);
        let b = child_with_value(&t, &[100, 101, 102, 103]);

        let mut leaves = t.evictable_leaves();
        leaves.sort();
        assert_eq!(leaves, vec![a, b]);

        // lock A: it leaves the set; B is untouched
        assert_eq!(t.inc_lock_ref(a), -4);
        assert_eq!(t.evictable_leaves(), vec![b]);

        // evict drops only B
        let ev = t.evict(4);
        assert_eq!(ev.num_tokens_evicted, 4);
        assert!(!t.node_evicted(a));
        assert!(t.node_children(ROOT).contains(&a));

        // unlock: A re-enters the set (live leaf, no children)
        assert_eq!(t.dec_lock_ref(a), 4);
        assert_eq!(t.evictable_leaves(), vec![a]);
    }

    #[test]
    fn drop_subtree_refused_when_host_protected_then_frees_all() {
        let mut t = wt();
        let tokens: Vec<i64> = (0..8).collect();
        let first4: Vec<i64> = (0..4).collect();
        insert_vals(&mut t, &first4, 0);
        let vals8: Vec<i64> = (0..8).collect();
        let rp = insert(&mut t, &tokens, &vals8);
        let q = rp.last_node;
        let p = t.node_children(ROOT)[0];
        let qh: Vec<i64> = (2000..2004).collect();
        t.begin_backup(q, &qh, false);

        // refused while Q holds a host reference
        t.protect_host(q);
        let refused = t.drop_subtree_no_host(p);
        assert_eq!(refused.freed_device, 0);
        assert!(refused.free_device.is_empty() && refused.free_host.is_empty());
        assert!(!t.node_evicted(p), "refusal must not mutate the tree");
        assert_eq!(t.node_children(p), vec![q]);
        t.release_host(q);

        let res = t.drop_subtree_no_host(p);
        assert_eq!(res.freed_device, 8);
        assert_eq!(res.free_device.len(), 2);
        assert_eq!(res.free_host, vec![qh]);
        assert_eq!(t.node_children(ROOT), vec![]);
        assert_eq!(t.evictable_size(), 0);
    }

    #[test]
    fn insert_host_creates_evicted_only_node() {
        let mut t = wt();
        let key4: Vec<i64> = (0..4).collect();
        let key8: Vec<i64> = (0..8).collect();
        insert_vals(&mut t, &key4, 0);
        let a = t.node_children(ROOT)[0];

        // prefetch result: 8 host tokens, the first 4 already on host?
        // No — A is not backuped yet, so the whole 8-token host run is
        // fresh; 4 overlap the device node A.
        let hv: Vec<i64> = (7000..7008).collect();
        let matched = t.insert_host(ROOT, &key(&key8), &hv);
        assert_eq!(matched, 4, "overlap with the live node A");
        let h = t.node_children(a)[0];
        assert!(t.node_evicted(h), "prefetch nodes are host-only");
        assert_eq!(t.node_host_value(h), Some(hv[4..].to_vec()));
        assert_eq!(t.node_priority(h), t.node_priority(a));
        assert!(t.evictable_host_leaves().contains(&h));

        // match sees the device prefix + the host suffix
        let m = t.match_prefix(&key(&key8));
        assert_eq!(m.indices, (0..4).collect::<Vec<i64>>());
        assert_eq!(m.host_hit_length, 4);
        assert_eq!(m.last_device_node, a);
        assert_eq!(m.last_host_node, h);

        // host-evict the prefetch node
        let eh = t.evict_host(4);
        assert_eq!(eh.deleted, vec![h]);
        assert_eq!(eh.free_host, vec![hv[4..].to_vec()]);
        assert!(t.node_children(a).is_empty());
    }

    #[test]
    fn bigram_eagle_tree_works() {
        let mut t = HiRadixTree::new(1, true, HiPolicy::WriteThrough, EvictionPolicy::Lru, 1, 10);
        // 5 raw tokens -> 4 bigrams
        let toks: Vec<i64> = (1..6).collect();
        let vals: Vec<i64> = (0..4).collect();
        let r = t.insert(&RadixKey { is_bigram: true, ..key(&toks) }, &vals, 0, false);
        assert_eq!(r.prefix_len, 0);
        let a = r.last_node;
        assert_eq!(t.node_key(a).unwrap().len(), 8, "flattened bigram key");

        let ha: Vec<i64> = (1000..1004).collect();
        t.begin_backup(a, &ha, true);
        t.dec_lock_ref(a);
        let ev = t.evict(4);
        assert_eq!(ev.num_tokens_evicted, 4, "4 bigram units");

        let m = t.match_prefix(&RadixKey {
            is_bigram: true,
            ..key(&toks)
        });
        assert!(m.indices.is_empty());
        assert_eq!(m.host_hit_length, 4);
        assert_eq!(m.last_host_node, a);
    }

    #[test]
    fn reset_clears_everything() {
        let mut t = wt();
        let toks: Vec<i64> = (0..8).collect();
        insert_vals(&mut t, &toks, 0);
        t.reset();
        assert_eq!(t.total_size(), 0);
        assert_eq!(t.total_host_size(), 0);
        assert_eq!(t.evictable_size(), 0);
        assert_eq!(t.protected_size(), 0);
        assert!(t.evictable_leaves().is_empty());
        assert!(t.evictable_host_leaves().is_empty());
        // root invariants
        assert_eq!(t.node_value(ROOT), Some(vec![]));
        assert_eq!(t.node_host_value(ROOT), Some(vec![]));
        assert_eq!(t.node_lock_ref(ROOT), 1);
        assert_eq!(t.node_priority(ROOT), i32::MIN);
        let m = t.match_prefix(&key(&toks));
        assert!(m.indices.is_empty());
        assert_eq!(m.last_device_node, ROOT);
        assert_eq!(m.last_host_node, ROOT);
    }
}
