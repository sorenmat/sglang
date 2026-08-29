//! The radix tree — a faithful port of the base `RadixCache` from
//! `python/sglang/srt/mem_cache/radix_cache.py` (single-tier, device-only,
//! MHA semantics).
//!
//! Differences from the Python implementation, deliberately:
//! - `last_access_time` is a per-tree u64 walk clock instead of a
//!   `time.monotonic()` float. Each match/insert walk and each node
//!   allocation gets its own strictly-increasing tick, so LRU order is
//!   deterministic across runs (Python floats can tie inside one
//!   iteration; the experimental C++ tree used the same clock approach).
//! - Nodes live in a `Vec` and are never freed (the parent's child map is
//!   the only reference path). Node ids stay stable for the whole tree
//!   lifetime, which request bookkeeping (`last_node` handles) relies on.
//! - `evict` returns the evicted value runs for the caller to release
//!   through the allocator; the Python class owns its allocator inline.
//! - Namespaces (`extra_key`, `cache_salt`) are interned to `u32` ids; the
//!   children map key is `(ns, head)` exactly like the Python
//!   `child_key(page_size)` namespacing.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::key::{common_prefix_len, RadixKey};
use crate::policy::{EvictionPolicy, Prio};

pub type NodeId = u32;

/// The tree root. It is never evicted: `lock_ref == 1` from `reset()`.
pub const ROOT: NodeId = 0;

/// The hashable children-map head: the first `page_size` logical units of a
/// child's key. Mirrors the `plain` part of `RadixKey.child_key(page_size)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Head {
    /// page_size == 1, plain key.
    Token(i64),
    /// page_size == 1, bigram key: first pair.
    Bigram(i64, i64),
    /// page_size > 1: the first `page_size * unit_size` flattened elements.
    Tokens(Vec<i64>),
}

/// `(namespace, head)` — the full children-map key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChildKey {
    pub ns: u32,
    pub head: Head,
}

/// One tree node. `key` and `value` are parallel: `value` has one KV index
/// per logical unit; `key` holds the flattened units (bigram keys: 2
/// elements per unit).
#[derive(Debug, Clone)]
pub struct RawNode {
    pub children: HashMap<ChildKey, NodeId>,
    pub parent: Option<NodeId>,
    pub key: Vec<i64>,
    /// KV cache indices, one per logical unit. `None` = evicted.
    /// Root: `Some(vec![])` (never "evicted", matching Python's `[]`).
    pub value: Option<Vec<i64>>,
    pub lock_ref: u32,
    /// Walk clock of the last match/insert that visited this node.
    pub last_access: u64,
    pub hit_count: u64,
    /// Eviction priority (lower evicts earlier under `Priority` policy).
    /// Root is `i32::MIN` so any real priority overrides it.
    pub priority: i32,
    pub id: NodeId,
}

impl RawNode {
    #[cfg(test)]
    pub fn test_node(id: NodeId, last_access: u64, hit_count: u64, priority: i32) -> Self {
        Self {
            children: HashMap::new(),
            parent: None,
            key: vec![],
            value: Some(vec![]),
            lock_ref: 0,
            last_access,
            hit_count,
            priority,
            id,
        }
    }

    fn logical_len(&self, unit_size: usize) -> usize {
        self.key.len() / unit_size
    }
}

/// Result of `match_prefix`: the concatenated KV indices of the longest
/// cached prefix (page-aligned), and the terminal node handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    pub indices: Vec<i64>,
    pub last_node: NodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertResult {
    /// Logical tokens already present in the tree before this insert
    /// (excludes the new leaf's tokens) — the range the caller may free
    /// from its own pool copy.
    pub prefix_len: usize,
    pub last_node: NodeId,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvictResult {
    /// Evicted value runs, in eviction order. The caller releases them
    /// through the allocator (`free_segment(run, start_pos=0)`).
    pub evicted_values: Vec<Vec<i64>>,
    pub num_tokens_evicted: usize,
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
    // BinaryHeap is a max-heap; invert so the SMALLEST prio pops first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .prio
            .key()
            .cmp(&self.prio.key())
            .then(self.id.cmp(&other.id))
    }
}

#[derive(Clone)]
pub struct RadixTree {
    nodes: Vec<RawNode>,
    page_size: usize,
    /// Flattened elements per logical unit: 1 (plain) or 2 (bigram).
    unit_size: usize,
    is_eagle: bool,
    disable: bool,
    policy: EvictionPolicy,
    /// Logical tokens currently in evictable (unlocked, live) nodes.
    evictable_size: i64,
    /// Logical tokens in locked (protected) nodes.
    protected_size: i64,
    /// Nodes that are live, unlocked, and have no live children.
    evictable_leaves: HashSet<NodeId>,
    /// Monotonic walk clock (deterministic stand-in for monotonic time).
    clock: u64,
    ns_map: HashMap<(String, String), u32>,
    ns_list: Vec<(String, String)>,
}

impl RadixTree {
    pub fn new(page_size: usize, is_eagle: bool, policy: EvictionPolicy) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        let mut t = Self {
            nodes: Vec::new(),
            page_size,
            unit_size: if is_eagle { 2 } else { 1 },
            is_eagle,
            disable: false,
            policy,
            evictable_size: 0,
            protected_size: 0,
            evictable_leaves: HashSet::new(),
            clock: 0,
            ns_map: HashMap::new(),
            ns_list: Vec::new(),
        };
        t.reset();
        t
    }

    /// `reset` — recreate the root and clear all bookkeeping.
    pub fn reset(&mut self) {
        let root = RawNode {
            children: HashMap::new(),
            parent: None,
            key: vec![],
            value: Some(vec![]),
            lock_ref: 1,
            last_access: 0,
            hit_count: 0,
            priority: i32::MIN,
            id: ROOT,
        };
        self.nodes = vec![root];
        self.evictable_size = 0;
        self.protected_size = 0;
        self.evictable_leaves.clear();
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

    pub fn is_eagle(&self) -> bool {
        self.is_eagle
    }

    pub fn root_node(&self) -> NodeId {
        ROOT
    }

    pub fn evictable_size(&self) -> i64 {
        self.evictable_size
    }

    pub fn protected_size(&self) -> i64 {
        self.protected_size
    }

    /// O(nodes) sum of live values, matching `total_size()`.
    pub fn total_size(&self) -> i64 {
        let mut total = 0i64;
        let mut stack = vec![ROOT];
        while let Some(id) = stack.pop() {
            if let Some(v) = &self.nodes[id as usize].value {
                total += v.len() as i64;
            }
            for &child in self.nodes[id as usize].children.values() {
                if self.nodes[child as usize].value.is_some() {
                    stack.push(child);
                }
            }
        }
        total
    }

    /// Whether `id` names a live (non-evicted) node of this tree.
    pub fn is_live_node(&self, id: NodeId) -> bool {
        self.nodes
            .get(id as usize)
            .is_some_and(|n| n.value.is_some())
    }

    pub fn node_value(&self, id: NodeId) -> Option<&[i64]> {
        self.nodes.get(id as usize).and_then(|n| n.value.as_deref())
    }

    pub fn node_lock_ref(&self, id: NodeId) -> u32 {
        self.nodes.get(id as usize).map(|n| n.lock_ref).unwrap_or(u32::MAX)
    }

    /// Sorted snapshot of the evictable-leaf set (debug/parity tooling).
    pub fn debug_evictable_leaves(&self) -> Vec<NodeId> {
        let mut v: Vec<NodeId> = self.evictable_leaves.iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Live children of a node (evicted children are already detached).
    pub fn node_children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .get(id as usize)
            .map(|n| n.children.values().copied().collect())
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

    fn child_key_for(
        &mut self,
        key: &RadixKey,
        is_bigram: bool,
        remaining: &[i64],
    ) -> ChildKey {
        let ns = self.intern_ns(key.extra_key, key.cache_salt);
        let head = self.head_from_flat(remaining, is_bigram);
        ChildKey { ns, head }
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

    /// Logical units in a flattened key.
    fn logical_len(&self, flat_key_len: usize) -> usize {
        flat_key_len / self.unit_size
    }

    /// Shared logical units between `remaining` and a stored key, floored
    /// to `page_size` — Python `RadixKey.match`:
    /// `(matched // page_size) * page_size`. (The child lookup already
    /// guarantees at least one full page of agreement.)
    fn page_floor_match(&self, remaining: &[i64], child_key: &[i64]) -> usize {
        let m_flat = common_prefix_len(remaining, child_key);
        self.logical_len(m_flat) / self.page_size * self.page_size
    }

    /// Port of `match_prefix` + `_match_prefix_helper`.
    pub fn match_prefix(&mut self, key: &RadixKey) -> MatchResult {
        // `maybe_to_bigram_view`: the tree forces bigram when is_eagle.
        let is_bigram = if self.is_eagle { true } else { key.is_bigram };
        let key = RadixKey { is_bigram, ..*key };
        if self.disable || key.logical_len() == 0 {
            return MatchResult {
                indices: vec![],
                last_node: ROOT,
            };
        }
        let flat = key.flatten_page_aligned(self.page_size);
        if flat.is_empty() {
            return MatchResult {
                indices: vec![],
                last_node: ROOT,
            };
        }
        let access = self.tick();
        let mut node = ROOT;
        self.nodes[node as usize].last_access = access;
        let mut out = Vec::with_capacity(flat.len());
        let mut remaining: &[i64] = &flat;
        let mut child_key = self.child_key_for(&key, is_bigram, remaining);
        while !remaining.is_empty() {
            let Some(&child) = self.nodes[node as usize].children.get(&child_key) else {
                break;
            };
            self.nodes[child as usize].last_access = access;
            let m = self.page_floor_match(remaining, &self.nodes[child as usize].key);
            let child_logical = self.logical_len(self.nodes[child as usize].key.len());
            if m < child_logical {
                // Match ends inside the child: split, take the front half.
                let new_id = self.split_node(child, m, is_bigram);
                out.extend_from_slice(
                    &self.nodes[new_id as usize].value.clone().unwrap_or_default(),
                );
                return MatchResult {
                    indices: out,
                    last_node: new_id,
                };
            }
            out.extend_from_slice(
                &self.nodes[child as usize].value.clone().unwrap_or_default(),
            );
            node = child;
            remaining = &remaining[m * self.unit_size..];
            if !remaining.is_empty() {
                child_key = self.child_key_for(&key, is_bigram, remaining);
            }
        }
        MatchResult {
            indices: out,
            last_node: node,
        }
    }

    /// Port of `insert` + `_insert_helper`.
    pub fn insert(
        &mut self,
        key: &RadixKey,
        value: &[i64],
        priority: i32,
        chunked: bool,
    ) -> InsertResult {
        if self.disable {
            return InsertResult {
                prefix_len: 0,
                last_node: ROOT,
            };
        }
        // `maybe_to_bigram_view`: the tree forces bigram when is_eagle.
        let is_bigram = if self.is_eagle { true } else { key.is_bigram };
        let key = RadixKey { is_bigram, ..*key };
        let flat = key.flatten_page_aligned(self.page_size);
        if flat.is_empty() {
            return InsertResult {
                prefix_len: 0,
                last_node: ROOT,
            };
        }
        // Python: `value = value[:len(key)]` — logical, page-aligned length
        // (one KV index per logical unit, bigrams included).
        let logical = self.logical_len(flat.len());
        let value = &value[..logical.min(value.len())];
        debug_assert_eq!(
            value.len(),
            logical,
            "insert value must have one KV index per logical unit"
        );

        let access = self.tick();
        let mut node = ROOT;
        self.nodes[node as usize].last_access = access;
        self.nodes[node as usize].priority = self.nodes[node as usize].priority.max(priority);

        let mut total_prefix = 0usize;
        let mut remaining: &[i64] = &flat;
        let mut val: &[i64] = value;
        let mut child_key = self.child_key_for(&key, is_bigram, remaining);
        while !remaining.is_empty() {
            let Some(&child) = self.nodes[node as usize].children.get(&child_key) else {
                break;
            };
            let m = self.page_floor_match(remaining, &self.nodes[child as usize].key);
            let child_logical = self.logical_len(self.nodes[child as usize].key.len());
            total_prefix += m;
            remaining = &remaining[m * self.unit_size..];
            val = &val[m..];
            self.nodes[child as usize].last_access = access;
            if m < child_logical {
                let new_id = self.split_node(child, m, is_bigram);
                self.nodes[new_id as usize].priority =
                    self.nodes[new_id as usize].priority.max(priority);
                self.inc_hit(new_id, chunked);
                node = new_id;
            } else {
                let n = &mut self.nodes[child as usize];
                n.priority = n.priority.max(priority);
                self.inc_hit(child, chunked);
                node = child;
            }
            if !remaining.is_empty() {
                child_key = self.child_key_for(&key, is_bigram, remaining);
            }
        }

        if !remaining.is_empty() {
            let new_id = self.nodes.len() as NodeId;
            let parent = node;
            let new = RawNode {
                children: HashMap::new(),
                parent: Some(parent),
                key: remaining.to_vec(),
                value: Some(val.to_vec()),
                lock_ref: 0,
                last_access: self.tick(),
                hit_count: 0,
                priority,
                id: new_id,
            };
            self.nodes.push(new);
            self.nodes[parent as usize].children.insert(child_key, new_id);
            self.inc_hit(new_id, chunked);
            self.evictable_size += self.logical_len(remaining.len()) as i64;
            self.update_leaf_status(parent);
            self.update_leaf_status(new_id);
            node = new_id;
        }
        InsertResult {
            prefix_len: total_prefix,
            last_node: node,
        }
    }

    /// Port of `evict`: rebuild the heap from `evictable_leaves` and pop
    /// until the budget is met. Node values are returned for the caller to
    /// free; the tree itself only mutates structure.
    pub fn evict(&mut self, num_tokens: usize) -> EvictResult {
        if self.disable || num_tokens == 0 {
            return EvictResult::default();
        }
        let mut heap = BinaryHeap::with_capacity(self.evictable_leaves.len());
        for &id in &self.evictable_leaves {
            let prio = self.policy.prio(&self.nodes[id as usize]);
            heap.push(HeapItem { prio, id });
        }

        let mut evicted_values = Vec::new();
        let mut num_evicted = 0usize;
        while num_evicted < num_tokens {
            let Some(item) = heap.pop() else {
                break;
            };
            let x = item.id;
            debug_assert_eq!(
                self.nodes[x as usize].lock_ref,
                0,
                "evict popped locked node {x}"
            );
            debug_assert!(
                self.nodes[x as usize].children.is_empty(),
                "evict popped node {x} with live children"
            );
            debug_assert!(
                self.nodes[x as usize].value.is_some(),
                "evict popped already-evicted node {x}"
            );
            let values = self.nodes[x as usize].value.take().unwrap_or_default();
            num_evicted += values.len();
            evicted_values.push(values);
            self.delete_leaf(x);
            let parent = self.nodes[x as usize].parent.unwrap_or(ROOT);
            if self.nodes[parent as usize].children.is_empty()
                && self.nodes[parent as usize].lock_ref == 0
            {
                let prio = self.policy.prio(&self.nodes[parent as usize]);
                heap.push(HeapItem { prio, id: parent });
            }
        }
        EvictResult {
            num_tokens_evicted: num_evicted,
            evicted_values,
        }
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
            let n = &mut self.nodes[id as usize];
            if n.lock_ref == 0 {
                let l = n.logical_len(self.unit_size) as i64;
                self.evictable_size -= l;
                self.protected_size += l;
                delta -= l;
            }
            n.lock_ref += 1;
            let parent = n.parent;
            self.update_leaf_status_id(id);
            cur = parent;
        }
        delta
    }

    /// Port of `dec_lock_ref`. Returns tokens moved protected -> evictable
    /// (>= 0).
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
            let n = &mut self.nodes[id as usize];
            if n.lock_ref == 1 {
                let l = n.logical_len(self.unit_size) as i64;
                self.evictable_size += l;
                self.protected_size -= l;
                delta += l;
            }
            n.lock_ref -= 1;
            let parent = n.parent;
            self.update_leaf_status_id(id);
            cur = parent;
        }
        delta
    }

    /// Human-readable dump, same shape as `pretty_print`.
    pub fn pretty_print(&self) -> String {
        let mut out = String::new();
        let mut stack = vec![(ROOT, 0)];
        while let Some((id, indent)) = stack.pop() {
            let n = &self.nodes[id as usize];
            let preview: Vec<i64> = n.key.iter().take(10).copied().collect();
            out.push_str(&format!(
                "{}{} {:?} r={}\n",
                " ".repeat(indent * 2),
                self.logical_len(n.key.len()),
                preview,
                n.lock_ref
            ));
            for &child in n.children.values() {
                stack.push((child, indent + 1));
            }
        }
        out.push_str(&format!("#tokens: {}\n", self.total_size()));
        out
    }

    // ---- internals ----

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn inc_hit(&mut self, id: NodeId, chunked: bool) {
        if chunked {
            return;
        }
        self.nodes[id as usize].hit_count += 1;
    }

    /// Port of `_split_node`: split `child` after `split_len` logical
    /// units. `0 < split_len < key length`, page-aligned (the match floor
    /// guarantees `split_len >= page_size`). The new front node inherits
    /// the child's priority/hit_count/lock_ref; both halves stay
    /// consistently locked, so size bookkeeping needs no adjustment.
    fn split_node(&mut self, child: NodeId, split_len: usize, is_bigram: bool) -> NodeId {
        let cut = split_len * self.unit_size;

        // Snapshot everything we need from the child before mutation.
        let (child_key, child_value, gp, priority, hit, lock) = {
            let c = &self.nodes[child as usize];
            (
                c.key.clone(),
                c.value.clone().unwrap_or_default(),
                c.parent.expect("split target must have a parent"),
                c.priority,
                c.hit_count,
                c.lock_ref,
            )
        };
        // The grandparent's entry naming `child` carries the namespace.
        let gp_entry = self
            .nodes[gp as usize]
            .children
            .iter()
            .find(|item| item.1 == &child)
            .map(|(k, _)| k.clone())
            .expect("split target missing from parent's children");
        let ns = gp_entry.ns;

        let front_key = child_key[..cut].to_vec();
        let tail_key = child_key[cut..].to_vec();
        // Values are logical (one KV index per unit); keys are flattened.
        let front_value = child_value[..split_len].to_vec();
        let tail_value = child_value[split_len..].to_vec();

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
        children.insert(tail_ck, child);
        let new_node = RawNode {
            children,
            parent: Some(gp),
            key: front_key,
            value: Some(front_value),
            lock_ref: lock,
            last_access: self.tick(),
            hit_count: hit,
            priority,
            id: new_id,
        };
        self.nodes.push(new_node);

        // Mutate the tail node in place.
        {
            let c = &mut self.nodes[child as usize];
            c.key = tail_key;
            c.value = Some(tail_value);
            c.parent = Some(new_id);
        }
        // Re-parent at the grandparent: the old entry pointed at `child`;
        // it now points at the new front node, which owns the tail.
        {
            let g = &mut self.nodes[gp as usize];
            g.children.remove(&gp_entry);
            g.children.insert(front_ck, new_id);
        }

        new_id
    }

    /// Port of `_delete_leaf`.
    fn delete_leaf(&mut self, node: NodeId) {
        let parent = self
            .nodes[node as usize]
            .parent
            .expect("evicted node has no parent?");
        let key = self.nodes[node as usize].key.clone();
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
        assert!(removed == Some(node), "parent does not have child key");
        self.evictable_size -= self.logical_len(key.len()) as i64;
        self.evictable_leaves.remove(&node);
        self.update_leaf_status(parent);
    }

    /// Port of `_update_leaf_status`.
    fn update_leaf_status(&mut self, node: NodeId) {
        let evicted = self.nodes[node as usize].value.is_none();
        let locked = self.nodes[node as usize].lock_ref > 0;
        if evicted || locked {
            self.evictable_leaves.remove(&node);
            return;
        }
        for &child in self.nodes[node as usize].children.values() {
            if self.nodes[child as usize].value.is_some() {
                self.evictable_leaves.remove(&node);
                return;
            }
        }
        self.evictable_leaves.insert(node);
    }

    /// Same as `update_leaf_status`, split out to satisfy the borrow
    /// checker inside the lock walks (which hold a `&mut` on the node's
    /// `parent` field).
    fn update_leaf_status_id(&mut self, node: NodeId) {
        self.update_leaf_status(node);
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new(1, false, EvictionPolicy::default())
    }
}
