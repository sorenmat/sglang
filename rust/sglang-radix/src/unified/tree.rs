//! The unified tree core: arena, LRUs, match, node splitting, leaf-set and
//! duplicate tracking. Ported method-for-method from
//! `unified_tree_core.py` (`UnifiedTreeCore`, `UnifiedTreeNode`,
//! `UnifiedLRUList`).

use std::collections::HashMap;

use crate::unified::evict::{FullEvictWalk, LruEvictWalk};
use crate::unified::insert::InsertState;
use crate::unified::{
    UCacheAction, UConfig, UHead, UNodeDump, UrkvEvent, CT_BASE, CT_FULL, CT_MAMBA, CT_SWA,
    NUM_CT,
};

pub(crate) const PARENT_NONE: u32 = u32::MAX;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct UChildKey {
    pub ns: u32,
    pub head: UHead,
}

/// One tree node (arena-resident). Sentinel LRU nodes live in the same arena
/// with `sentinel = true`.
#[derive(Clone, Debug)]
pub struct UNode {
    pub children: HashMap<UChildKey, u32>,
    /// Insertion-ordered child keys (deterministic walks/dumps; kept in sync
    /// with `children`).
    pub child_order: Vec<UChildKey>,
    pub parent: u32,
    /// Namespace id for this key (stable per path; root = 0, unused).
    pub ns: u32,
    /// Raw token ids; bigram mode holds N+1 raw tokens for N logical units.
    pub key: Vec<i64>,
    /// Component device values (pool index lists), logical-token domain.
    pub value: [Option<Vec<i64>>; NUM_CT],
    pub host_value: [Option<Vec<i64>>; NUM_CT],
    pub lock_ref: [i32; NUM_CT],
    pub host_lock_ref: [i32; NUM_CT],
    /// Session machinery is out of scope: always 0.
    pub session_ref: [i32; NUM_CT],
    pub last_access: f64,
    pub creation: f64,
    pub hit_count: i64,
    pub priority: i64,
    pub hash_value: Option<Vec<String>>,
    pub event_hash_value: Option<Vec<String>>,
    pub wt_pending: Option<i64>,
    pub lb_pending: Option<u32>,
    /// SWA lock-boundary uuids ("uuid" / "host_uuid" metadata).
    pub swa_uuid: Option<i64>,
    pub swa_host_uuid: Option<i64>,
    /// Intrusive LRU links; slot = ct*2 + layer (0 device, 1 host).
    pub lru_prev: [u32; NUM_CT * 2],
    pub lru_next: [u32; NUM_CT * 2],
    /// Bit per slot: node is currently in that LRU list.
    pub in_lru: u16,
    pub sentinel: bool,
    pub deleted: bool,
}

impl UNode {
    fn new_sentinel() -> Self {
        UNode {
            children: HashMap::new(),
            child_order: Vec::new(),
            parent: PARENT_NONE,
            ns: 0,
            key: vec![],
            value: [const { None }; NUM_CT],
            host_value: [const { None }; NUM_CT],
            lock_ref: [0; NUM_CT],
            host_lock_ref: [0; NUM_CT],
            session_ref: [0; NUM_CT],
            last_access: 0.0,
            creation: 0.0,
            hit_count: 0,
            priority: 0,
            hash_value: None,
            event_hash_value: None,
            wt_pending: None,
            lb_pending: None,
            swa_uuid: None,
            swa_host_uuid: None,
            lru_prev: [0; NUM_CT * 2],
            lru_next: [0; NUM_CT * 2],
            in_lru: 0,
            sentinel: true,
            deleted: false,
        }
    }
}

/// `MatchResult` with the node fields exposed as NodeIds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UMatchResult {
    pub device_indices: Vec<i64>,
    pub last_device_node: u32,
    pub last_host_node: u32,
    pub best_match_node: u32,
    pub host_hit_length: i64,
    pub swa_host_hit_length: i64,
    pub mamba_host_hit_length: i64,
    pub mamba_branching_seqlen: Option<i64>,
    pub full_kv_hit_length: i64,
    /// Python `MatchResult.cache_actions` (the match-walk split action, if any).
    pub actions: Vec<UCacheAction>,
}

/// `InsertResult`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UIntsertResult {
    pub prefix_len: i64,
    pub total_len: i64,
    pub last_device_node: u32,
    pub mamba_exist: bool,
    pub inserted_host_node: Option<u32>,
    pub host_insert_dropped: bool,
    /// Nodes created by `_add_new_node` during this insert (the caller must
    /// backfill `hash_value` for these when storage is enabled).
    pub created_nodes: Vec<u32>,
    /// `InsertResult.cache_actions` — set by insert_host (split actions).
    pub cache_actions: Vec<UCacheAction>,
}

/// `InsertStepResult`: `actions` flushed at this barrier, `result` when the
/// insert completed.
#[derive(Debug, Clone, Default)]
pub struct UStepResult {
    pub actions: Vec<UCacheAction>,
    pub result: Option<UIntsertResult>,
}

/// `RadixCacheWalkResult` (kv-canary rows).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UWalkResult {
    pub slot_indices: Vec<i64>,
    pub positions: Vec<i64>,
    pub prev_slot_indices: Vec<i64>,
}

#[derive(Clone)]
pub struct UnifiedRadixTree {
    pub cfg: UConfig,
    pub has_swa: bool,
    pub has_mamba: bool,
    pub nodes: Vec<UNode>,
    pub root: u32,
    /// Time counter (Python `get_and_increase_time_counter`), 1.0-based.
    pub time_counter: f64,
    /// Component uuid counter (Python `next_component_uuid`), inc-then-return.
    pub uuid_counter: i64,
    pub enable_hicache: bool,
    pub enable_storage: bool,
    pub evictable_size: [i64; NUM_CT],
    pub protected_size: [i64; NUM_CT],
    pub d_leaves: std::collections::HashSet<u32>,
    pub h_leaves: std::collections::HashSet<u32>,
    /// Insertion-ordered full-host-duplicate ids (+ live set for O(1) checks).
    pub dup_ids: Vec<u32>,
    pub dup_set: std::collections::HashSet<u32>,
    pub reclaim_digest: u64,
    /// LRU sentinels: slot -> head / tail node ids.
    pub(crate) lru_head: [u32; NUM_CT * 2],
    pub(crate) lru_tail: [u32; NUM_CT * 2],
    pub(crate) ongoing: Option<InsertState>,
    pub(crate) full_evict: Option<FullEvictWalk>,
    pub(crate) swa_evict: Option<LruEvictWalk>,
    pub(crate) mamba_evict: Option<LruEvictWalk>,
    pub kv_log: Vec<UrkvEvent>,
}

impl UnifiedRadixTree {
    pub fn new(cfg: UConfig) -> Self {
        assert!(
            !cfg.enable_session_radix_cache,
            "UnifiedRadixTree (Rust): --enable-session-radix-cache is not supported by the Rust core"
        );
        assert!(
            !cfg.is_eagle || !cfg.has_mamba(),
            "UnifiedRadixTree (Rust): is_eagle is off when Mamba is present"
        );
        let mut t = UnifiedRadixTree {
            has_swa: cfg.has_swa(),
            has_mamba: cfg.has_mamba(),
            nodes: Vec::new(),
            root: 0,
            time_counter: 1.0,
            uuid_counter: 1,
            enable_hicache: false,
            enable_storage: false,
            evictable_size: [0; NUM_CT],
            protected_size: [0; NUM_CT],
            d_leaves: std::collections::HashSet::new(),
            h_leaves: std::collections::HashSet::new(),
            dup_ids: Vec::new(),
            dup_set: std::collections::HashSet::new(),
            reclaim_digest: 0,
            lru_head: [0; NUM_CT * 2],
            lru_tail: [0; NUM_CT * 2],
            ongoing: None,
            full_evict: None,
            swa_evict: None,
            mamba_evict: None,
            kv_log: Vec::new(),
            cfg,
        };
        t.reset();
        t
    }

    /// `reset()` — rebuild root, LRUs, sizes, leaf sets.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.time_counter = 1.0;
        self.uuid_counter = 1;
        self.enable_hicache = false;
        self.enable_storage = false;
        self.evictable_size = [0; NUM_CT];
        self.protected_size = [0; NUM_CT];
        self.d_leaves.clear();
        self.h_leaves.clear();
        self.dup_ids.clear();
        self.dup_set.clear();
        self.reclaim_digest = 0;
        self.ongoing = None;
        self.full_evict = None;
        self.swa_evict = None;
        self.mamba_evict = None;
        self.kv_log.clear();

        let root = self.new_node(0);
        self.root = root;
        let r = &mut self.nodes[root as usize];
        r.priority = i64::MIN; // -sys.maxsize
        r.key.clear();
        r.value[CT_BASE as usize] = Some(Vec::new());
        r.hash_value = Some(Vec::new());
        for ct in 0..NUM_CT {
            r.lock_ref[ct] = 1;
        }
        for slot in 0..NUM_CT * 2 {
            let head = self.new_sentinel();
            let tail = self.new_sentinel();
            self.lru_head[slot] = head;
            self.lru_tail[slot] = tail;
            let h = &mut self.nodes[head as usize];
            h.lru_next[slot] = tail;
            let t = &mut self.nodes[tail as usize];
            t.lru_prev[slot] = head;
        }
    }

    pub(crate) fn new_node(&mut self, priority: i64) -> u32 {
        let id = self.nodes.len() as u32;
        let la = self.tick();
        let cr = self.tick();
        self.nodes.push(UNode {
            children: HashMap::new(),
            child_order: Vec::new(),
            parent: PARENT_NONE,
            ns: 0,
            key: Vec::new(),
            value: [const { None }; NUM_CT],
            host_value: [const { None }; NUM_CT],
            lock_ref: [0; NUM_CT],
            host_lock_ref: [0; NUM_CT],
            session_ref: [0; NUM_CT],
            last_access: la,
            creation: cr,
            hit_count: 0,
            priority,
            hash_value: None,
            event_hash_value: None,
            wt_pending: None,
            lb_pending: None,
            swa_uuid: None,
            swa_host_uuid: None,
            lru_prev: [0; NUM_CT * 2],
            lru_next: [0; NUM_CT * 2],
            in_lru: 0,
            sentinel: false,
            deleted: false,
        });
        id
    }

    fn new_sentinel(&mut self) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(UNode::new_sentinel());
        id
    }

    pub(crate) fn tick(&mut self) -> f64 {
        let v = self.time_counter;
        self.time_counter += 1.0;
        v
    }

    /// Python `next_component_uuid`: increment first, then return.
    pub(crate) fn next_uuid(&mut self) -> i64 {
        self.uuid_counter += 1;
        self.uuid_counter
    }

    /// Active component ids in facade order (FULL, SWA, MAMBA).
    pub fn active_cts(&self) -> Vec<u8> {
        let mut v = vec![CT_FULL];
        if self.has_swa {
            v.push(CT_SWA);
        }
        if self.has_mamba {
            v.push(CT_MAMBA);
        }
        v
    }

    // ==== key helpers ====

    /// Logical length of a raw key (bigram mode: N+1 raw -> N).
    pub fn key_len(&self, raw: &[i64]) -> i64 {
        if raw.is_empty() {
            return 0;
        }
        if self.cfg.is_eagle {
            raw.len() as i64 - 1
        } else {
            raw.len() as i64
        }
    }

    /// Page-align a raw key from the left (Python `page_aligned`); returns the
    /// raw slice to keep.
    pub(crate) fn page_align_raw<'a>(&self, raw: &'a [i64]) -> &'a [i64] {
        let page = i64::from(self.cfg.page_size);
        if page == 1 {
            return raw;
        }
        let aligned = (self.key_len(raw) / page) * page;
        // Logical [0, aligned) -> raw [0, aligned + (1 if bigram)); an empty
        // aligned slice keeps zero raw tokens.
        let keep = if aligned == 0 {
            0
        } else {
            aligned as usize + self.cfg.is_eagle as usize
        };
        &raw[..keep.min(raw.len())]
    }

    /// `child_key(page_size)` over the first page of logical units.
    pub(crate) fn child_key_of(&self, raw: &[i64], ns: u32) -> UChildKey {
        let page = self.cfg.page_size as usize;
        let head = if self.cfg.is_eagle {
            if page == 1 {
                UHead::Bigram(raw[0], raw[1])
            } else {
                UHead::Tokens(raw[..2 * page].to_vec())
            }
        } else if page == 1 {
            UHead::Token(raw[0])
        } else {
            UHead::Tokens(raw[..page].to_vec())
        };
        UChildKey { ns, head }
    }

    fn child_key_of_node(&self, node: u32) -> UChildKey {
        let n = &self.nodes[node as usize];
        self.child_key_of(&n.key, n.ns)
    }

    /// `RadixKey.match` — page-aligned logical prefix length.
    pub(crate) fn match_len(&self, a: &[i64], b: &[i64]) -> i64 {
        let n = a.len().min(b.len());
        let mut m = 0;
        while m < n && a[m] == b[m] {
            m += 1;
        }
        let page = i64::from(self.cfg.page_size);
        if self.cfg.is_eagle {
            let matched = (m as i64 - 1).max(0).min(self.key_len(a)).min(self.key_len(b));
            if page > 1 {
                matched / page * page
            } else {
                matched
            }
        } else {
            let matched = m as i64;
            if page > 1 {
                matched / page * page
            } else {
                matched
            }
        }
    }

    /// Split a raw node key at logical position `split` into (front, back).
    pub(crate) fn split_raw_key(&self, raw: &[i64], split: usize) -> (Vec<i64>, Vec<i64>) {
        if self.cfg.is_eagle {
            (raw[..split + 1].to_vec(), raw[split..].to_vec())
        } else {
            (raw[..split].to_vec(), raw[split..].to_vec())
        }
    }

    // ==== node predicates ====

    pub fn is_backuped(&self, node: u32) -> bool {
        self.nodes[node as usize].host_value[CT_BASE as usize].is_some()
    }

    pub fn is_root(&self, node: u32) -> bool {
        node == self.root
    }

    pub fn is_evicted(&self, node: u32) -> bool {
        self.nodes[node as usize].parent != PARENT_NONE
            && self.nodes[node as usize].value[CT_BASE as usize].is_none()
    }

    // ==== LRU (slot = ct*2 + layer) ====

    fn lru_slot(ct: u8, layer: u8) -> usize {
        ct as usize * 2 + layer as usize
    }

    fn lru_insert_after(&mut self, slot: usize, prev: u32, node: u32) {
        let bit = 1u16 << slot;
        let nxt = self.nodes[prev as usize].lru_next[slot];
        self.nodes[node as usize].lru_prev[slot] = prev;
        self.nodes[node as usize].lru_next[slot] = nxt;
        self.nodes[nxt as usize].lru_prev[slot] = node;
        self.nodes[prev as usize].lru_next[slot] = node;
        self.nodes[node as usize].in_lru |= bit;
    }

    pub(crate) fn lru_remove(&mut self, slot: usize, node: u32) {
        let bit = 1u16 << slot;
        let (p, q) = {
            let n = &self.nodes[node as usize];
            (n.lru_prev[slot], n.lru_next[slot])
        };
        self.nodes[q as usize].lru_prev[slot] = p;
        self.nodes[p as usize].lru_next[slot] = q;
        let n = &mut self.nodes[node as usize];
        n.lru_prev[slot] = 0;
        n.lru_next[slot] = 0;
        n.in_lru &= !bit;
    }

    /// `in_list`.
    pub(crate) fn lru_in(&self, slot: usize, node: u32) -> bool {
        self.nodes[node as usize].in_lru & (1u16 << slot) != 0
    }

    /// `insert_mru` (asserts not in list).
    pub(crate) fn lru_insert_mru(&mut self, slot: usize, node: u32) {
        debug_assert!(!self.lru_in(slot, node), "insert_mru: node in list");
        self.lru_insert_after(slot, self.lru_head[slot], node);
    }

    /// `reset_node_mru`.
    pub(crate) fn lru_reset_mru(&mut self, slot: usize, node: u32) {
        if self.lru_in(slot, node) {
            self.lru_remove(slot, node);
        }
        self.lru_insert_mru(slot, node);
    }

    /// `reset_node_and_window_ancestors_mru`.
    pub(crate) fn lru_reset_window_ancestors_mru(&mut self, ct: u8, node: u32, window: i64) {
        let slot = Self::lru_slot(ct, 0);
        let mut prev = self.lru_head[slot];
        let mut cur = node;
        let mut accumulated = 0i64;
        while cur != self.root && accumulated < window {
            if self.nodes[cur as usize].value[ct as usize].is_some() {
                self.lru_remove(slot, cur);
                self.lru_insert_after(slot, prev, cur);
                prev = cur;
            }
            accumulated += self.key_len(&self.nodes[cur as usize].key);
            cur = self.nodes[cur as usize].parent;
        }
    }

    /// `get_prev_no_lock` walking toward head from `start`; None at head.
    pub(crate) fn lru_prev_no_lock(&self, slot: usize, start: u32, ct: u8) -> Option<u32> {
        let mut x = self.nodes[start as usize].lru_prev[slot];
        while self.nodes[x as usize].lock_ref[ct as usize] > 0 {
            x = self.nodes[x as usize].lru_prev[slot];
        }
        (x != self.lru_head[slot]).then_some(x)
    }

    /// `get_prev_no_host_lock`.
    pub(crate) fn lru_prev_no_host_lock(&self, slot: usize, start: u32, ct: u8) -> Option<u32> {
        let mut x = self.nodes[start as usize].lru_prev[slot];
        while self.nodes[x as usize].host_lock_ref[ct as usize] > 0 {
            x = self.nodes[x as usize].lru_prev[slot];
        }
        (x != self.lru_head[slot]).then_some(x)
    }

    /// `get_lru_no_lock` — from the tail sentinel.
    pub(crate) fn lru_lru_no_lock(&self, slot: usize, ct: u8) -> Option<u32> {
        self.lru_prev_no_lock(slot, self.lru_tail[slot], ct)
    }

    pub(crate) fn lru_lru_no_host_lock(&self, slot: usize, ct: u8) -> Option<u32> {
        self.lru_prev_no_host_lock(slot, self.lru_tail[slot], ct)
    }

    // ==== leaf-set / duplicate tracking ====

    pub fn is_device_leaf(&self, node: u32) -> bool {
        if node == self.root || self.is_evicted(node) {
            return false;
        }
        let n = &self.nodes[node as usize];
        if n.lock_ref.iter().any(|&l| l > 0) {
            return false;
        }
        let child_full = n
            .children
            .values()
            .any(|&c| self.nodes[c as usize].value[CT_BASE as usize].is_some());
        !child_full
    }

    pub fn is_host_leaf(&self, node: u32) -> bool {
        if node == self.root || !self.is_evicted(node) {
            return false;
        }
        let n = &self.nodes[node as usize];
        if n.host_value[CT_BASE as usize].is_none() {
            return false;
        }
        if n.host_lock_ref.iter().any(|&l| l > 0) {
            return false;
        }
        n.children.is_empty()
    }

    pub(crate) fn update_leaf_sets(&mut self, node: u32) {
        if self.is_device_leaf(node) {
            self.d_leaves.insert(node);
        } else {
            self.d_leaves.remove(&node);
        }
        if self.is_host_leaf(node) {
            self.h_leaves.insert(node);
        } else {
            self.h_leaves.remove(&node);
        }
    }

    pub(crate) fn is_settled_duplicate(&self, node: u32) -> bool {
        let n = &self.nodes[node as usize];
        node != self.root
            && n.value[CT_BASE as usize].is_some()
            && n.host_value[CT_BASE as usize].is_some()
            && n.wt_pending.is_none()
            && n.lb_pending.is_none()
    }

    pub(crate) fn update_duplicate_tracking(&mut self, node: u32) {
        if self.is_settled_duplicate(node) {
            // setdefault: keep the first insertion position.
            if !self.dup_set.contains(&node) {
                self.dup_set.insert(node);
                self.dup_ids.push(node);
            }
        } else {
            self.dup_set.remove(&node);
            // Lazy: dup_ids is swept at reclaim time.
        }
    }

    // ==== node split (Python `_split_node`) ====

    /// Split `child` at logical position `split_len`; returns the new front
    /// fragment and the optional ReplaceWriteThroughOnNodeSplit action.
    pub fn split_node(&mut self, child: u32, split_len: i64) -> (u32, Option<UCacheAction>) {
        let split_len = split_len as usize;
        let new_node = self.new_node(self.nodes[child as usize].priority);
        let ckey = self.nodes[child as usize].key.clone();
        let ns = self.nodes[child as usize].ns;
        let (front_key, back_key) = self.split_raw_key(&ckey, split_len);
        let back_child_key = self.child_key_of(&back_key, ns);
        let cparent = self.nodes[child as usize].parent;

        {
            let chit = self.nodes[child as usize].hit_count;
            let ccreation = self.nodes[child as usize].creation;
            let clb = self.nodes[child as usize].lb_pending;
            let n = &mut self.nodes[new_node as usize];
            n.children.insert(back_child_key.clone(), child);
            n.child_order.push(back_child_key);
            n.parent = cparent;
            n.ns = ns;
            n.key = front_key;
            n.hit_count = chit;
            n.creation = ccreation;
            n.lb_pending = clb;
        }

        // Remove child from every aux device LRU it is in.
        for ct in self.active_cts() {
            if ct == CT_FULL {
                continue;
            }
            let slot = Self::lru_slot(ct, 0);
            if self.lru_in(slot, child) {
                self.lru_remove(slot, child);
            }
        }

        let action = {
            let (front_hash, child_hash) = Self::split_hash_list(
                self.nodes[child as usize].hash_value.take(),
                split_len,
                self.cfg.page_size,
            );
            let (front_evt, child_evt) = Self::split_hash_list(
                self.nodes[child as usize].event_hash_value.take(),
                split_len,
                self.cfg.page_size,
            );
            let c = &mut self.nodes[child as usize];
            c.parent = new_node;
            c.key = back_key;
            // hash_value / event_hash_value split (Python `split_node_hash_value`).
            c.hash_value = child_hash;
            c.event_hash_value = child_evt;
            self.nodes[new_node as usize].hash_value = front_hash;
            self.nodes[new_node as usize].event_hash_value = front_evt;

            // Component redistribution.
            self.redistribute_full(new_node, child);
            if self.has_swa {
                self.redistribute_swa(new_node, child);
            }
            if self.has_mamba {
                self.redistribute_mamba(new_node, child);
            }

            // The new node replaces the child under the ORIGINAL parent; its
            // head equals the child's, so the old entry is overwritten.
            let np = cparent;
            let front_ck = self.child_key_of_node(new_node);
            self.nodes[np as usize].children.insert(front_ck.clone(), new_node);
            {
                let order = &mut self.nodes[np as usize].child_order;
                match order.iter().position(|k| *k == front_ck) {
                    Some(i) => order[i] = front_ck,
                    None => order.push(front_ck),
                }
            }

            if let Some(ack) = self.nodes[child as usize].wt_pending {
                self.nodes[new_node as usize].wt_pending = Some(ack);
                Some(UCacheAction::ReplaceWT {
                    ack_id: ack,
                    old_node: child,
                    new_node,
                    new_child_node: child,
                })
            } else {
                None
            }
        };

        // Both fragments back into the aux device LRUs (skip_existing).
        for ct in self.active_cts() {
            if ct == CT_FULL {
                continue;
            }
            let slot = Self::lru_slot(ct, 0);
            if self.nodes[new_node as usize].value[ct as usize].is_some()
                && !self.lru_in(slot, new_node)
            {
                self.lru_insert_mru(slot, new_node);
            }
            if self.nodes[child as usize].value[ct as usize].is_some()
                && !self.lru_in(slot, child)
            {
                self.lru_insert_mru(slot, child);
            }
        }

        self.nodes[child as usize].last_access = self.tick();
        self.update_leaf_sets(new_node);
        self.update_leaf_sets(child);
        self.update_duplicate_tracking(new_node);
        (new_node, action)
    }

    /// Python `split_node_hash_value`: (front, child) halves.
    fn split_hash_list(
        v: Option<Vec<String>>,
        split_len: usize,
        page_size: u32,
    ) -> (Option<Vec<String>>, Option<Vec<String>>) {
        let mut v = match v {
            Some(v) => v,
            None => return (None, None),
        };
        let pages = if page_size == 1 {
            split_len
        } else {
            split_len / page_size as usize
        };
        let front: Vec<String> = v.drain(..pages).collect();
        (Some(front), Some(v))
    }

    // Python `redistribute_on_node_split`, FULL.
    fn redistribute_full(&mut self, new_parent: u32, child: u32) {
        let split_len = self.key_len(&self.nodes[new_parent as usize].key) as usize;
        {
            let lr = self.nodes[child as usize].lock_ref[CT_BASE as usize];
            let sr = self.nodes[child as usize].session_ref[CT_BASE as usize];
            self.nodes[new_parent as usize].lock_ref[CT_BASE as usize] = lr;
            self.nodes[new_parent as usize].session_ref[CT_BASE as usize] = sr;
        }
        let v = self.nodes[child as usize].value[CT_BASE as usize].take();
        if let Some(mut val) = v {
            let front: Vec<i64> = val.drain(..split_len).collect();
            self.nodes[new_parent as usize].value[CT_BASE as usize] = Some(front);
            self.nodes[child as usize].value[CT_BASE as usize] = Some(val);
        }
        let h = self.nodes[child as usize].host_value[CT_BASE as usize].take();
        if let Some(mut val) = h {
            let front: Vec<i64> = val.drain(..split_len).collect();
            self.nodes[new_parent as usize].host_value[CT_BASE as usize] = Some(front);
            self.nodes[child as usize].host_value[CT_BASE as usize] = Some(val);
        }
    }

    // Python `redistribute_on_node_split`, SWA.
    fn redistribute_swa(&mut self, new_parent: u32, child: u32) {
        let split_len = self.key_len(&self.nodes[new_parent as usize].key) as usize;
        {
            let lr = self.nodes[child as usize].lock_ref[CT_SWA as usize];
            let sr = self.nodes[child as usize].session_ref[CT_SWA as usize];
            self.nodes[new_parent as usize].lock_ref[CT_SWA as usize] = lr;
            self.nodes[new_parent as usize].session_ref[CT_SWA as usize] = sr;
        }
        let v = self.nodes[child as usize].value[CT_SWA as usize].take();
        if let Some(mut val) = v {
            let front: Vec<i64> = val.drain(..split_len).collect();
            self.nodes[new_parent as usize].value[CT_SWA as usize] = Some(front);
            self.nodes[child as usize].value[CT_SWA as usize] = Some(val);
        } else {
            self.nodes[new_parent as usize].value[CT_SWA as usize] = None;
        }
        let h = self.nodes[child as usize].host_value[CT_SWA as usize].take();
        if let Some(mut val) = h {
            let front: Vec<i64> = val.drain(..split_len).collect();
            self.nodes[new_parent as usize].host_value[CT_SWA as usize] = Some(front);
            self.nodes[child as usize].host_value[CT_SWA as usize] = Some(val);
            let slot = Self::lru_slot(CT_SWA, 1);
            if self.nodes[new_parent as usize].value[CT_SWA as usize].is_none() {
                self.lru_insert_mru(slot, new_parent);
            }
            if self.nodes[child as usize].value[CT_SWA as usize].is_none()
                && !self.lru_in(slot, child)
            {
                self.lru_insert_mru(slot, child);
            }
        }
        // Parent inherits the swa uuid; the child drops it.
        {
            let u = self.nodes[child as usize].swa_uuid;
            self.nodes[new_parent as usize].swa_uuid = u;
            self.nodes[child as usize].swa_uuid = None;
        }
    }

    // Python `redistribute_on_node_split`, MAMBA.
    fn redistribute_mamba(&mut self, new_parent: u32, _child: u32) {
        let n = &mut self.nodes[new_parent as usize];
        n.value[CT_MAMBA as usize] = None;
        n.lock_ref[CT_MAMBA as usize] = 0;
        n.session_ref[CT_MAMBA as usize] = 0;
        n.host_value[CT_MAMBA as usize] = None;
        n.host_lock_ref[CT_MAMBA as usize] = 0;
    }

    // ==== match (Python `match_prefix` + helpers + post-processor) ====

    pub fn match_prefix(&mut self, ns: u32, raw_tokens: &[i64]) -> UMatchResult {
        let key = self.page_align_raw(raw_tokens);
        if self.key_len(key) == 0 {
            return self.empty_match_result();
        }

        let mut chunks: Vec<Vec<i64>> = Vec::new();
        let mut best_match = self.root;
        let mut best_dev = self.root;
        let mut best_dev_len = 0usize;
        let mut full_kv_hit = 0i64;
        let mut action: Option<UCacheAction> = None;
        let separate = self.enable_hicache;

        // SWA match-state (two independent counters when separate). None = +inf.
        let mut swa_len: Option<i64> = None;
        let mut swa_dev_len: Option<i64> = None;
        let swa_device_only_hicache = !self.cfg.has_swa_host_pool && self.enable_hicache;
        let window = self.cfg.sliding_window_size;

        let mut node = self.root;
        let mut key_rem: &[i64] = key;
        let mut child_key = self.child_key_of(key_rem, ns);
        while !key_rem.is_empty() {
            let child = match self.nodes[node as usize].children.get(&child_key) {
                Some(&c) => c,
                None => break,
            };
            // HiCache: dead node (evicted + not backuped) — stop traversal.
            if self.is_evicted(child) && !self.is_backuped(child) {
                break;
            }
            let ckey = self.nodes[child as usize].key.clone();
            let prefix_len = self.match_len(&ckey, key_rem);
            full_kv_hit += prefix_len;
            let partial = prefix_len < self.key_len(&ckey);
            let target = if partial {
                let (front, split_action) = self.split_node(child, prefix_len);
                action = split_action;
                front
            } else {
                child
            };
            if !self.is_evicted(target) {
                chunks.push(
                    self.nodes[target as usize]
                        .value[CT_BASE as usize]
                        .clone()
                        .unwrap_or_default(),
                );
            }
            // `_update_best_if_valid`: Python builds the validator list fully
            // (every validator runs and mutates its state) before the all().
            let matched = if separate {
                self.all_valid_full(&mut swa_len, window, swa_device_only_hicache, target)
            } else {
                self.all_valid_dev(&mut swa_len, window, swa_device_only_hicache, target)
            };
            if matched {
                best_match = target;
            }
            if !separate {
                if matched {
                    best_dev_len = chunks.len();
                    best_dev = target;
                }
            } else {
                let dev = self.all_valid_dev(&mut swa_dev_len, window, swa_device_only_hicache, target);
                if dev {
                    best_dev_len = chunks.len();
                    best_dev = target;
                }
            }
            if partial {
                break;
            }
            node = target;
            let adv = prefix_len as usize;
            key_rem = &key_rem[adv..];
            if !key_rem.is_empty() {
                child_key = self.child_key_of(key_rem, ns);
            }
        }

        let mut result =
            self.match_post_process(chunks, best_match, best_dev, best_dev_len, full_kv_hit);
        if let Some(a) = action {
            result.actions.push(a);
        }
        result
    }

    /// All "full" validators pass at `node` (HiCache: value or backuped).
    /// Every validator runs (state mutation) regardless of earlier results.
    fn all_valid_full(
        &self,
        swa_len: &mut Option<i64>,
        window: i64,
        device_only_hicache: bool,
        node: u32,
    ) -> bool {
        let f = self.full_validator(node);
        let s = self.swa_validator(swa_len, window, device_only_hicache, node, false);
        let m = self.mamba_validator(node, false);
        f && s && m
    }

    /// All device-only validators pass at `node`.
    fn all_valid_dev(
        &self,
        swa_dev_len: &mut Option<i64>,
        window: i64,
        device_only_hicache: bool,
        node: u32,
    ) -> bool {
        let f = self.full_validator_dev(node);
        let s = self.swa_validator(swa_dev_len, window, device_only_hicache, node, true);
        let m = self.mamba_validator(node, true);
        f && s && m
    }

    fn full_validator(&self, node: u32) -> bool {
        self.nodes[node as usize].value[CT_BASE as usize].is_some()
            || self.is_backuped(node)
    }

    fn full_validator_dev(&self, node: u32) -> bool {
        self.nodes[node as usize].value[CT_BASE as usize].is_some()
    }

    /// Python SWA `create_match_validator`; `state` is the running window len
    /// (None = +inf, which always satisfies the window check).
    fn swa_validator(
        &self,
        state: &mut Option<i64>,
        window: i64,
        device_only_hicache: bool,
        node: u32,
        device_only: bool,
    ) -> bool {
        if !self.has_swa {
            return true;
        }
        let n = &self.nodes[node as usize];
        let has_val = n.value[CT_SWA as usize].is_some();
        let has_host = n.host_value[CT_SWA as usize].is_some();
        if !has_val && (device_only || !has_host) {
            *state = Some(0);
            if device_only_hicache && (self.is_backuped(node) || !self.is_evicted(node)) {
                return true;
            }
            return false;
        }
        match state {
            Some(cur) => {
                let nv = *cur + self.key_len(&n.key);
                *state = Some(nv);
                nv >= window
            }
            None => true, // inf
        }
    }

    fn mamba_validator(&self, node: u32, device_only: bool) -> bool {
        if !self.has_mamba {
            return true;
        }
        let n = &self.nodes[node as usize];
        if device_only {
            n.value[CT_MAMBA as usize].is_some()
        } else {
            n.value[CT_MAMBA as usize].is_some() || n.host_value[CT_MAMBA as usize].is_some()
        }
    }

    fn match_post_process(
        &mut self,
        chunks: Vec<Vec<i64>>,
        best_match: u32,
        best_dev: u32,
        best_dev_len: usize,
        full_kv_hit: i64,
    ) -> UMatchResult {
        // MATCH_END LRU refresh (aux only).
        if self.has_swa {
            let window = self.cfg.sliding_window_size + i64::from(self.cfg.page_size);
            self.lru_reset_window_ancestors_mru(CT_SWA, best_match, window);
        }
        if self.has_mamba {
            let slot = Self::lru_slot(CT_MAMBA, 0);
            if self.nodes[best_match as usize].value[CT_MAMBA as usize].is_some() {
                self.lru_reset_mru(slot, best_match);
            }
        }

        // Timestamp walk (fresh tick at best_match, -0.00001 per ancestor up,
        // root included).
        let mut cur_time = self.tick();
        let mut node_update = best_match;
        loop {
            self.nodes[node_update as usize].last_access = cur_time;
            cur_time -= 0.00001;
            if node_update == self.root {
                break;
            }
            node_update = self.nodes[node_update as usize].parent;
        }

        let last_host = if self.enable_hicache { best_match } else { best_dev };

        let device_indices: Vec<i64> = if best_dev_len > 0 {
            chunks[..best_dev_len].concat()
        } else {
            Vec::new()
        };

        let mut host_hit = 0i64;
        let mut swa_host_hit = 0i64;
        let mut mamba_host_hit = 0i64;
        let mut mamba_branching: Option<i64> = None;

        // FULL finalize (first component in facade order).
        {
            let mut kv_host_hit = 0i64;
            let mut node = best_match;
            while node != best_dev && node != self.root {
                if let Some(hv) = &self.nodes[node as usize].host_value[CT_BASE as usize] {
                    kv_host_hit += hv.len() as i64;
                }
                node = self.nodes[node as usize].parent;
            }
            if kv_host_hit > 0 {
                host_hit = host_hit.max(kv_host_hit);
            }
        }

        // SWA finalize.
        if self.has_swa {
            let mut n_swa = 0i64;
            let mut node = best_match;
            let window = self.cfg.sliding_window_size;
            while node != self.root && n_swa < window {
                let n = &self.nodes[node as usize];
                if let Some(v) = &n.value[CT_SWA as usize] {
                    n_swa += v.len() as i64;
                } else if let Some(hv) = &n.host_value[CT_SWA as usize] {
                    swa_host_hit += hv.len() as i64;
                    n_swa += hv.len() as i64;
                } else {
                    break;
                }
                node = n.parent;
            }
            if swa_host_hit > 0 {
                swa_host_hit = swa_host_hit.max(0);
            }
        }

        // Mamba finalize.
        if self.has_mamba {
            let boundary = device_indices.len() as i64 + host_hit;
            let grid = self.cfg.mamba_checkpoint_grid;
            let aligned = (full_kv_hit / grid) * grid;
            mamba_branching = (aligned > boundary).then_some(aligned);
            let n = &self.nodes[best_match as usize];
            if n.value[CT_MAMBA as usize].is_none()
                && n.host_value[CT_MAMBA as usize].is_some()
            {
                mamba_host_hit = mamba_host_hit.max(1);
            }
        }

        UMatchResult {
            device_indices,
            last_device_node: best_dev,
            last_host_node: last_host,
            best_match_node: best_match,
            host_hit_length: host_hit,
            swa_host_hit_length: swa_host_hit,
            mamba_host_hit_length: mamba_host_hit,
            mamba_branching_seqlen: mamba_branching,
            full_kv_hit_length: full_kv_hit,
            actions: Vec::new(),
        }
    }

    /// `empty_match_result` equivalent.
    pub fn empty_match_result(&self) -> UMatchResult {
        UMatchResult {
            device_indices: Vec::new(),
            last_device_node: self.root,
            last_host_node: self.root,
            best_match_node: self.root,
            ..Default::default()
        }
    }

    /// `is_full_device_evicted`.
    pub fn is_full_device_evicted(&self, node: u32) -> bool {
        self.is_evicted(node)
    }

    /// `collect_full_device_indices` (root order, from_node up to excl. until).
    pub fn collect_full_device_indices(&self, from_node: u32, until_node: u32) -> Vec<i64> {
        let mut chunks: Vec<Vec<i64>> = Vec::new();
        let mut node = from_node;
        while node != until_node {
            let v = self.nodes[node as usize].value[CT_BASE as usize].clone();
            debug_assert!(v.is_some(), "collect_full_device_indices: evicted ancestor");
            chunks.push(v.unwrap_or_default());
            node = self.nodes[node as usize].parent;
        }
        chunks.reverse();
        chunks.concat()
    }

    /// `walk_for_kv_canary` (child order = insertion order).
    pub fn walk_for_kv_canary(&self, unlocked_only: bool, swa_resident_only: bool) -> UWalkResult {
        let mut res = UWalkResult::default();
        let swa_filter = swa_resident_only && self.has_swa;
        let mut stack: Vec<(u32, i64, i64)> = vec![(self.root, 0, -1)];
        while let Some((node, depth, parent_last_slot)) = stack.pop() {
            let slots: Vec<i64> = self.nodes[node as usize]
                .value[CT_BASE as usize]
                .clone()
                .unwrap_or_default();
            let mut emit = node != self.root;
            if unlocked_only {
                let lock_ct = if swa_filter { CT_SWA } else { CT_BASE };
                emit = emit && self.nodes[node as usize].lock_ref[lock_ct as usize] == 0;
            }
            if swa_filter {
                emit = emit && self.nodes[node as usize].value[CT_SWA as usize].is_some();
            }
            let mut chain_last = parent_last_slot;
            for (j, &slot) in slots.iter().enumerate() {
                if emit {
                    res.slot_indices.push(slot);
                    res.positions.push(depth + j as i64);
                    res.prev_slot_indices.push(if j == 0 {
                        parent_last_slot
                    } else {
                        slots[j - 1]
                    });
                }
                chain_last = slot;
            }
            let child_depth = if node == self.root || self.nodes[node as usize].key.is_empty() {
                depth + slots.len() as i64
            } else {
                depth + self.key_len(&self.nodes[node as usize].key)
            };
            let children: Vec<u32> = self.nodes[node as usize].child_order.iter().map(|k| {
                *self.nodes[node as usize].children.get(k).unwrap_or(&0)
            }).filter(|&c| c != 0).collect();
            for c in children.into_iter().rev() {
                stack.push((c, child_depth, chain_last));
            }
        }
        res
    }

    /// `all_values_flatten` (preorder, child insertion order).
    pub fn all_values_flatten(&self) -> Vec<i64> {
        let mut out = Vec::new();
        self.flatten_component(CT_BASE, self.root, &mut out);
        out
    }

    /// `all_mamba_values_flatten`.
    pub fn all_mamba_values_flatten(&self) -> Vec<i64> {
        let mut out = Vec::new();
        if self.has_mamba {
            self.flatten_component(CT_MAMBA, self.root, &mut out);
        }
        out
    }

    fn flatten_component(&self, ct: u8, node: u32, out: &mut Vec<i64>) {
        for k in self.nodes[node as usize].child_order.clone() {
            if let Some(&c) = self.nodes[node as usize].children.get(&k) {
                if let Some(v) = &self.nodes[c as usize].value[ct as usize] {
                    out.extend_from_slice(v);
                }
                self.flatten_component(ct, c, out);
            }
        }
    }

    // ==== sizes ====

    pub fn total_size(&self) -> (i64, i64) {
        let mut total = 0i64;
        let mut total_aux = 0i64;
        let mut stack = vec![self.root];
        while let Some(node) = stack.pop() {
            if let Some(v) = &self.nodes[node as usize].value[CT_BASE as usize] {
                total += v.len() as i64;
            }
            for &ct in self.active_cts().iter() {
                if ct == CT_FULL {
                    continue;
                }
                if let Some(v) = &self.nodes[node as usize].value[ct as usize] {
                    total_aux += v.len() as i64;
                }
            }
            stack.extend(self.nodes[node as usize].children.values());
        }
        (total, total_aux)
    }

    pub fn evictable_size(&self) -> i64 {
        self.evictable_size[CT_FULL as usize]
    }

    pub fn protected_size(&self) -> i64 {
        self.protected_size[CT_FULL as usize]
    }

    pub fn component_evictable_size(&self, ct: u8) -> i64 {
        self.evictable_size[ct as usize]
    }

    pub fn component_protected_size(&self, ct: u8) -> i64 {
        self.protected_size[ct as usize]
    }

    // ==== hash accessors (wrapper-managed) ====

    pub fn get_hash_values_opt(&self, node: u32) -> Option<Vec<String>> {
        self.nodes[node as usize].hash_value.clone()
    }

    pub fn get_hash_values(&self, node: u32) -> Vec<String> {
        self.nodes[node as usize].hash_value.clone().unwrap_or_default()
    }

    pub fn set_hash_values(&mut self, node: u32, values: Vec<String>) {
        self.nodes[node as usize].hash_value = Some(values);
    }

    pub fn get_last_hash_value(&self, node: u32) -> Option<String> {
        self.nodes[node as usize].hash_value.as_ref().and_then(|v| v.last().cloned())
    }

    /// `get_prefix_hash_values`: ancestor hashes, root-to-parent.
    pub fn get_prefix_hash_values(&self, node: u32) -> Vec<String> {
        let mut chain: Vec<u32> = Vec::new();
        let mut cur = self.nodes[node as usize].parent;
        while cur != PARENT_NONE {
            chain.push(cur);
            if cur == self.root {
                break;
            }
            cur = self.nodes[cur as usize].parent;
        }
        chain.reverse();
        let mut out = Vec::new();
        for c in chain {
            if !self.nodes[c as usize].key.is_empty() {
                out.extend(self.nodes[c as usize].hash_value.clone().unwrap_or_default());
            }
        }
        out
    }

    pub fn get_event_hash_values_opt(&self, node: u32) -> Option<Vec<String>> {
        self.nodes[node as usize].event_hash_value.clone()
    }

    pub fn get_event_hash_values(&self, node: u32) -> Vec<String> {
        self.nodes[node as usize]
            .event_hash_value
            .clone()
            .unwrap_or_default()
    }

    pub fn set_event_hash_values(&mut self, node: u32, values: Vec<String>) {
        self.nodes[node as usize].event_hash_value = Some(values);
    }

    /// Raw KV-event log drain (caller rebuilds BlockStored/BlockRemoved).
    pub fn take_kv_events(&mut self) -> Vec<UrkvEvent> {
        std::mem::take(&mut self.kv_log)
    }

    // ==== dump ====

    /// Full deterministic state dump (preorder).
    pub fn dump_nodes(&self) -> Vec<UNodeDump> {
        let mut out = Vec::new();
        let mut stack = vec![self.root];
        while let Some(node) = stack.pop() {
            let n = &self.nodes[node as usize];
            out.push(UNodeDump {
                id: node,
                key: n.key.clone(),
                last_access: n.last_access,
                creation: n.creation,
                hit_count: n.hit_count,
                priority: n.priority,
                full_value: n.value[CT_BASE as usize].clone(),
                full_host_value: n.host_value[CT_BASE as usize].clone(),
                swa_value: n.value[CT_SWA as usize].clone(),
                swa_host_value: n.host_value[CT_SWA as usize].clone(),
                mamba_value: n.value[CT_MAMBA as usize].clone(),
                mamba_host_value: n.host_value[CT_MAMBA as usize].clone(),
                lock_refs: n.lock_ref,
                host_lock_refs: n.host_lock_ref,
                swa_uuid: n.swa_uuid,
                swa_host_uuid: n.swa_host_uuid,
                write_through_pending: n.wt_pending,
                load_back_pending: n.lb_pending,
                in_device_leaves: self.d_leaves.contains(&node),
                in_host_leaves: self.h_leaves.contains(&node),
                is_duplicate_tracked: self.dup_set.contains(&node),
            });
            let children: Vec<u32> = n.child_order.iter().filter_map(|k| n.children.get(k).copied()).collect();
            for c in children.into_iter().rev() {
                stack.push(c);
            }
        }
        out
    }

    pub fn lru_order(&self, ct: u8, layer: u8) -> Vec<u32> {
        let slot = Self::lru_slot(ct, layer);
        let mut out = Vec::new();
        let mut x = self.nodes[self.lru_head[slot] as usize].lru_next[slot];
        while x != self.lru_tail[slot] {
            out.push(x);
            x = self.nodes[x as usize].lru_next[slot];
        }
        out
    }

    pub fn node_key(&self, node: u32) -> Vec<i64> {
        self.nodes[node as usize].key.clone()
    }

    pub fn node_parent(&self, node: u32) -> Option<u32> {
        let p = self.nodes[node as usize].parent;
        (p != PARENT_NONE).then_some(p)
    }

    /// Live duplicate ids in insertion order.
    pub fn duplicate_ids(&self) -> Vec<u32> {
        self.dup_ids
            .iter()
            .copied()
            .filter(|id| self.dup_set.contains(id))
            .collect()
    }

    pub fn reclaim_digest(&self) -> u64 {
        self.reclaim_digest
    }

    /// Sanity-check result: empty when invariants hold (Python raises).
    pub fn sanity_check(
        &self,
        ongoing_write_through: &[(i64, u32)],
        ongoing_load_back: &[(i64, u32)],
    ) -> Vec<String> {
        crate::unified::sanity::sanity_check(self, ongoing_write_through, ongoing_load_back)
    }

    pub fn set_hicache_enabled(&mut self) {
        self.enable_hicache = true;
    }

    pub fn set_storage_enabled(&mut self) {
        self.enable_storage = true;
    }

    pub fn set_write_back(&mut self) {
        self.cfg.is_write_back = true;
    }

    pub fn set_swa_host_pool(&mut self, has: bool) {
        self.cfg.has_swa_host_pool = has;
    }

    pub fn write_through_threshold(&self) -> i64 {
        self.cfg.write_through_threshold
    }

    pub fn set_write_through_threshold(&mut self, v: i64) {
        self.cfg.write_through_threshold = v;
    }

    pub fn is_write_back(&self) -> bool {
        self.cfg.is_write_back
    }

    pub fn root_id(&self) -> u32 {
        self.root
    }

    // Internal helpers used by the insert/locks/evict/hicache modules.

    pub(crate) fn touch_node(&mut self, node: u32) {
        self.nodes[node as usize].last_access = self.tick();
        // WALKDOWN LRU refresh: no-op for SWA and Mamba; the default (C128)
        // refresh is unreachable in this port.
    }

    pub(crate) fn add_new_node(
        &mut self,
        ns: u32,
        parent: u32,
        raw_key: Vec<i64>,
        value: Vec<i64>,
        priority: i64,
    ) -> u32 {
        let new_node = self.new_node(priority);
        let ck = self.child_key_of(&raw_key, ns);
        {
            let n = &mut self.nodes[new_node as usize];
            n.parent = parent;
            n.ns = ns;
            n.key = raw_key;
            n.value[CT_BASE as usize] = Some(value);
        }
        self.nodes[parent as usize].children.insert(ck.clone(), new_node);
        self.nodes[parent as usize].child_order.push(ck);
        self.evictable_size[CT_BASE as usize] += self.key_len(&self.nodes[new_node as usize].key);
        self.update_leaf_sets(new_node);
        self.update_leaf_sets(parent);
        self.kv_log.push(UrkvEvent {
            op: 1, // store
            node: new_node,
            medium: 1, // GPU
        });
        new_node
    }

    pub(crate) fn unevict_node_on_insert(&mut self, node: u32, fresh_value: Vec<i64>) {
        let key_len = self.key_len(&self.nodes[node as usize].key);
        let n = &mut self.nodes[node as usize];
        debug_assert!(n.value[CT_BASE as usize].is_none());
        n.value[CT_BASE as usize] = Some(fresh_value);
        self.evictable_size[CT_BASE as usize] += key_len;
        self.update_leaf_sets(node);
        self.update_duplicate_tracking(node);
        if self.nodes[node as usize].parent != PARENT_NONE {
            self.update_leaf_sets(self.nodes[node as usize].parent);
        }
        self.kv_log.push(UrkvEvent {
            op: 1,
            node,
            medium: 1,
        });
    }

    pub(crate) fn record_store(&mut self, node: u32, medium: u8) {
        self.kv_log.push(UrkvEvent { op: 1, node, medium });
    }

    pub(crate) fn record_remove(&mut self, node: u32, medium: u8) {
        self.kv_log.push(UrkvEvent { op: 0, node, medium });
    }

    pub(crate) fn remove_leaf_from_parent(&mut self, node: u32) {
        let parent = self.nodes[node as usize].parent;
        let ck = self.child_key_of_node(node);
        self.nodes[parent as usize].children.remove(&ck);
        self.nodes[parent as usize].child_order.retain(|k| k != &ck);
        self.dup_set.remove(&node);
        // Lazy: dup_ids is swept at reclaim time.
        self.nodes[node as usize].deleted = true;
    }

    /// `set_component_device_value` (aux only).
    pub fn set_component_device_value(&mut self, node: u32, ct: u8, value: Vec<i64>) {
        assert!(ct != CT_BASE, "Full stores go through the insert paths");
        let len = value.len() as i64;
        let host_slot = Self::lru_slot(ct, 1);
        let dev_slot = Self::lru_slot(ct, 0);
        self.nodes[node as usize].value[ct as usize] = Some(value);
        if self.lru_in(host_slot, node) {
            self.lru_remove(host_slot, node);
        }
        self.lru_insert_mru(dev_slot, node);
        self.evictable_size[ct as usize] += len;
    }

    pub fn get_component_device_value(&self, node: u32, ct: u8) -> Option<Vec<i64>> {
        self.nodes[node as usize].value[ct as usize].clone()
    }

    pub fn component_has_host_value_only(&self, node: u32, ct: u8) -> bool {
        let n = &self.nodes[node as usize];
        n.value[ct as usize].is_none() && n.host_value[ct as usize].is_some()
    }

    pub fn node_by_id(&self, node_id: u32) -> &UNode {
        &self.nodes[node_id as usize]
    }

    pub(crate) fn lru_slot_public(ct: u8, layer: u8) -> usize {
        Self::lru_slot(ct, layer)
    }
}

#[cfg(test)]
mod tests {
    use super::UnifiedRadixTree;
    use crate::policy::EvictionPolicy;
    use crate::unified::{
        UCacheAction, UConfig, UDecLockParams, UInsertParams, UStepResult, UIntsertResult,
        CT_FULL, CT_MAMBA, CT_SWA,
    };

    fn base_cfg() -> UConfig {
        UConfig {
            page_size: 1,
            is_eagle: false,
            sliding_window_size: 0,
            mamba_checkpoint_grid: 0,
            mamba_max_states_per_path: -1,
            eviction_policy: EvictionPolicy::Lru,
            write_through_threshold: 2,
            is_write_back: false,
            has_swa_host_pool: false,
            enable_session_radix_cache: false,
        }
    }

    fn params() -> UInsertParams {
        UInsertParams {
            prev_prefix_len: 0,
            chunked: false,
            priority: 0,
            swa_evicted_seqlen: 0,
            mamba_value: None,
        }
    }

    /// Insert `value = tokens + 1000`, pumping through any barriers and
    /// applying deferred MambaEvictExcess actions like the facade would.
    fn insert(tree: &mut UnifiedRadixTree, ns: u32, tokens: &[i64]) -> UIntsertResult {
        let value: Vec<i64> = tokens.iter().map(|t| t + 1000).collect();
        insert_with_value(tree, ns, tokens, &value, &params())
    }

    fn insert_with_value(
        tree: &mut UnifiedRadixTree,
        ns: u32,
        tokens: &[i64],
        value: &[i64],
        p: &UInsertParams,
    ) -> UIntsertResult {
        let step = tree.begin_insert(ns, tokens, Some(value.to_vec()), p);
        pump(tree, step)
    }

    fn pump(tree: &mut UnifiedRadixTree, mut step: UStepResult) -> UIntsertResult {
        while step.result.is_none() {
            for a in &step.actions {
                if let UCacheAction::MambaEvictExcess { tail_node } = a {
                    tree.evict_excess_path_states(*tail_node);
                }
            }
            step = tree.resume_insert();
        }
        step.result.unwrap()
    }

    fn tracker_full(t: &[(u8, i64)], ct: u8) -> i64 {
        t.iter().find(|(c, _)| *c == ct).map(|(_, n)| *n).unwrap_or(0)
    }

    #[test]
    fn full_insert_split_and_match() {
        let mut t = UnifiedRadixTree::new(base_cfg());
        let r1 = insert(&mut t, 1, &[1, 2, 3, 4]);
        assert_eq!(r1.prefix_len, 0);
        let l1 = r1.last_device_node;
        assert_eq!(t.node_key(l1), vec![1, 2, 3, 4]);

        let r2 = insert(&mut t, 1, &[1, 2, 5, 6]);
        assert_eq!(r2.prefix_len, 2);
        let l2 = r2.last_device_node;
        // The first node was split into "12" (new) + "34" (l1 kept its id).
        assert_eq!(t.node_key(l1), vec![3, 4]);
        let mid = t.node_parent(l1).expect("l1 has a parent");
        assert_eq!(t.node_key(mid), vec![1, 2]);
        assert_eq!(t.node_parent(l2), Some(mid));
        assert_eq!(t.node_key(l2), vec![5, 6]);

        let m = t.match_prefix(1, &[1, 2, 3, 4]);
        assert_eq!(m.device_indices, vec![1001, 1002, 1003, 1004]);
        assert_eq!(m.full_kv_hit_length, 4);
        assert_eq!(m.last_device_node, l1);
        let m2 = t.match_prefix(1, &[1, 2, 5, 7]);
        // "12" (2) + partial "5" (1) — the partial match still counts, and
        // the match walk split "56" into "5" (new device fragment) + "6".
        assert_eq!(m2.full_kv_hit_length, 3);
        assert_eq!(t.node_key(m2.last_device_node), vec![5]);
        assert!(t.sanity_check(&[], &[]).is_empty());
        assert_eq!(t.dump_nodes().len(), 5); // root + mid + l1 + "5" + "6"
    }

    #[test]
    fn page_alignment_truncates() {
        let mut cfg = base_cfg();
        cfg.page_size = 4;
        let mut t = UnifiedRadixTree::new(cfg);
        let r = insert(&mut t, 1, &[1, 2, 3, 4, 5, 6]);
        // Key truncated to whole pages.
        assert_eq!(t.node_key(r.last_device_node), vec![1, 2, 3, 4]);
        assert_eq!(t.match_prefix(1, &[1, 2]).full_kv_hit_length, 0);
        assert_eq!(
            t.match_prefix(1, &[1, 2, 3, 4, 9, 9]).full_kv_hit_length,
            4
        );
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn namespaces_are_isolated() {
        let mut t = UnifiedRadixTree::new(base_cfg());
        let r1 = insert(&mut t, 7, &[1, 2, 3, 4]);
        let r2 = insert(&mut t, 9, &[1, 2, 3, 4]);
        assert_ne!(r1.last_device_node, r2.last_device_node);
        assert_eq!(t.match_prefix(7, &[1, 2, 3, 4]).full_kv_hit_length, 4);
        assert_eq!(t.match_prefix(9, &[1, 2, 3, 4]).full_kv_hit_length, 4);
        assert_eq!(t.match_prefix(11, &[1, 2, 3, 4]).full_kv_hit_length, 0);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn full_lock_protects_from_eviction() {
        let mut t = UnifiedRadixTree::new(base_cfg());
        let la = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        let lb = insert(&mut t, 1, &[5, 6, 7, 8]).last_device_node;

        let inc = t.inc_lock_ref(la, &[]);
        assert_eq!(inc.delta, 4);
        assert_eq!(t.component_protected_size(CT_FULL), 4);
        assert!(!t.d_leaves.contains(&la));

        t.evict_device_start(CT_FULL, 4);
        let step = t.evict_device_next_node(CT_FULL, 0);
        assert_eq!(step.node_id, Some(lb)); // la locked -> lb picked
        let out = t.evict_device_leaf(lb, false);
        assert_eq!(tracker_full(&out.tracker, CT_FULL), 4);
        t.evict_device_end(CT_FULL);
        // Write-through, unbacked leaf is deleted outright.
        assert_eq!(t.match_prefix(1, &[5, 6, 7, 8]).full_kv_hit_length, 0);

        t.dec_lock_ref(la, &UDecLockParams::default(), false);
        assert_eq!(t.component_evictable_size(CT_FULL), 4);
        assert_eq!(t.component_protected_size(CT_FULL), 0);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn write_back_evict_backup_demote_reinsert() {
        let mut cfg = base_cfg();
        cfg.is_write_back = true;
        let mut t = UnifiedRadixTree::new(cfg);
        t.set_hicache_enabled(); // host hits are only surfaced in HiCache mode
        let la = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;

        // Evict under write-back: the leaf asks for a D->H backup first.
        let out = t.evict_device_leaf(la, true);
        let backup = out.backup_kv.expect("write-back defers the demote");
        assert_eq!(backup, vec![la]);
        assert!(!t.is_evicted(la), "node stays device-resident until backup");

        // Facade commits the backup, then demotes.
        t.commit_backup(la, &[9001, 9002, 9003, 9004], &[]);
        assert!(t.is_backuped(la));
        let out = t.demote_node(la);
        assert_eq!(tracker_full(&out.tracker, CT_FULL), 4);
        assert!(t.is_evicted(la));
        assert!(t.component_has_host_value_only(la, CT_FULL));

        // Host hit only, no device KV.
        let m = t.match_prefix(1, &[1, 2, 3, 4]);
        assert_eq!(m.full_kv_hit_length, 4); // tree match incl. host copy
        assert!(m.device_indices.is_empty());
        assert_eq!(m.last_device_node, t.root_id());
        assert_eq!(m.host_hit_length, 4);
        assert_eq!(m.last_host_node, la);

        // Re-insert over the evicted node: it is un-evicted in place.
        let r = insert(&mut t, 1, &[1, 2, 3, 4]);
        assert_eq!(r.last_device_node, la);
        assert!(r.created_nodes.is_empty());
        let m = t.match_prefix(1, &[1, 2, 3, 4]);
        assert_eq!(m.device_indices, vec![1001, 1002, 1003, 1004]);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn swa_window_lock_and_release() {
        let mut cfg = base_cfg();
        cfg.sliding_window_size = 4;
        let mut t = UnifiedRadixTree::new(cfg);
        let l = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        t.set_component_device_value(l, CT_SWA, vec![2001, 2002, 2003, 2004]);
        assert_eq!(t.component_evictable_size(CT_SWA), 4);

        // Lock only the SWA layer (FULL skipped -> the leaf stays a D-leaf).
        let inc = t.inc_lock_ref(l, &[CT_FULL]);
        assert!(inc.skip_ids(CT_FULL).contains(&l));
        let uuid = inc.swa_uuid_for_lock.expect("window filled -> uuid");
        assert_eq!(t.component_evictable_size(CT_SWA), 0);
        assert_eq!(t.component_protected_size(CT_SWA), 4);

        // dec_swa_lock_only releases the window; at ref 0 the leaf is a
        // device leaf, so the SWA value is evicted inline.
        t.dec_swa_lock_only(l, Some(uuid), &[]);
        assert!(t.get_component_device_value(l, CT_SWA).is_none());
        assert_eq!(t.component_evictable_size(CT_SWA), 0);
        assert_eq!(t.component_protected_size(CT_SWA), 0);
        // FULL was never locked (skipped at acquire); release with skip ids.
        let dec = UDecLockParams {
            skip_lock_node_ids: inc.skip_lock_node_ids.clone(),
            ..Default::default()
        };
        t.dec_lock_ref(l, &dec, true /* skip SWA */);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn swa_lru_cursor_eviction() {
        let mut cfg = base_cfg();
        cfg.sliding_window_size = 4;
        let mut t = UnifiedRadixTree::new(cfg);
        let l1 = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        let l2 = insert(&mut t, 1, &[5, 6, 7, 8]).last_device_node;
        t.set_component_device_value(l1, CT_SWA, vec![3001, 3002, 3003, 3004]);
        t.set_component_device_value(l2, CT_SWA, vec![3005, 3006, 3007, 3008]);
        // MRU order: l2 was inserted last.
        assert_eq!(t.lru_order(CT_SWA, 0), vec![l2, l1]);

        t.evict_device_start(CT_SWA, 4);
        let step = t.evict_device_next_node(CT_SWA, 0);
        assert_eq!(step.node_id, Some(l1)); // LRU first
        let out = t.evict_device_leaf(l1, false);
        assert_eq!(tracker_full(&out.tracker, CT_FULL), 4);
        t.evict_device_end(CT_SWA);
        assert_eq!(t.lru_order(CT_SWA, 0), vec![l2]);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn mamba_excess_path_states() {
        let mut cfg = base_cfg();
        cfg.mamba_checkpoint_grid = 4;
        cfg.mamba_max_states_per_path = 1;
        let mut t = UnifiedRadixTree::new(cfg);
        let p1 = UInsertParams {
            mamba_value: Some(vec![5001]),
            ..params()
        };
        let full1 = vec![2001, 2002, 2003, 2004];
        let l1 = insert_with_value(&mut t, 1, &[1, 2, 3, 4], &full1, &p1).last_device_node;
        assert_eq!(
            t.get_component_device_value(l1, CT_MAMBA),
            Some(vec![5001])
        );
        // Extend the path: a second checkpoint beyond the cap of 1.
        let p2 = UInsertParams {
            mamba_value: Some(vec![5002]),
            ..params()
        };
        let full2: Vec<i64> = (5..13).collect();
        let l2 = insert_with_value(&mut t, 1, &[1, 2, 3, 4, 5, 6, 7, 8], &full2, &p2)
            .last_device_node;
        assert_eq!(t.get_component_device_value(l2, CT_MAMBA), Some(vec![5002]));
        // The shallow checkpoint (l1) was evicted; the tail one stays.
        assert!(t.get_component_device_value(l1, CT_MAMBA).is_none());
        // FULL data survived on both.
        assert_eq!(t.node_by_id(l1).value[CT_FULL as usize].as_ref().unwrap().len(), 4);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn write_through_backup_on_hit() {
        let mut cfg = base_cfg();
        cfg.write_through_threshold = 1;
        let mut t = UnifiedRadixTree::new(cfg);
        t.set_hicache_enabled();
        let value: Vec<i64> = [1, 2, 3, 4].iter().map(|x| x + 1000).collect();
        let p = params();
        let step = t.begin_insert(1, &[1, 2, 3, 4], Some(value), &p);
        // A single step completes a plain insert; the terminal step flushes
        // the BackupKV emitted by the hit check.
        let r = step.result.expect("insert completes in one step");
        let l = r.last_device_node;
        assert!(
            step
                .actions
                .iter()
                .any(|a| matches!(a, UCacheAction::BackupKV { node_ids } if node_ids == &[l])),
            "hit at threshold 1 must emit BackupKV: {:?}",
            step.actions
        );
        // Facade replay: spec + commit + ack.
        let (host, comp) = t.build_backup_spec(l);
        assert_eq!(host, vec![1001, 1002, 1003, 1004]);
        t.commit_backup(l, &host, &comp);
        assert!(t.is_backuped(l));
        t.mark_write_through_pending(l);
        t.finish_write_through(&[l], 7);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn host_eviction_and_load_back_round_trip() {
        let cfg = base_cfg();
        let mut t = UnifiedRadixTree::new(cfg);
        t.set_hicache_enabled();
        let l = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        let (host, comp) = t.build_backup_spec(l);
        t.commit_backup(l, &host, &comp);

        // Demote: device freed, host copy kept (host-only node).
        let out = t.evict_device_leaf(l, false);
        assert!(out.backup_kv.is_none());
        assert!(t.is_evicted(l));
        assert!(t.is_backuped(l));

        // Load back from host: device value restored from the new indices.
        let spec = t.build_load_back_spec(l, None).expect("host copy to load");
        assert_eq!(spec.kv.host_indices, Some(vec![1001, 1002, 1003, 1004]));
        let actions = t.commit_load_back(
            l,
            Some(&[7001, 7002, 7003, 7004]),
            &spec.kv.nodes_to_load,
            &[],
        );
        assert!(actions.is_empty());
        t.finish_load_back(l);
        assert!(!t.is_evicted(l));
        let m = t.match_prefix(1, &[1, 2, 3, 4]);
        assert_eq!(m.device_indices, vec![7001, 7002, 7003, 7004]);

        // Demote again, then host-evict the now host-only node.
        t.evict_device_leaf(l, false);
        assert!(t.is_evicted(l) && t.is_backuped(l));
        let out = t.drive_host_eviction(CT_FULL, 4);
        assert_eq!(tracker_full(&out.tracker, CT_FULL), 4);
        assert!(!t.is_backuped(l));
        assert!(t.is_evicted(l));
        // Dead node (evicted, unbacked): the match stops at the root.
        assert_eq!(t.match_prefix(1, &[1, 2, 3, 4]).full_kv_hit_length, 0);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn split_preserves_hash_lists() {
        let mut t = UnifiedRadixTree::new(base_cfg());
        let l = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        t.set_hash_values(l, vec!["h0".into(), "h1".into(), "h2".into(), "h3".into()]);
        let (new_id, action) = t.split_node(l, 2);
        assert!(action.is_none());
        assert_eq!(t.node_key(new_id), vec![1, 2]);
        assert_eq!(t.node_key(l), vec![3, 4]);
        assert_eq!(t.node_parent(l), Some(new_id));
        assert_eq!(t.get_hash_values(new_id), vec!["h0".to_string(), "h1".to_string()]);
        assert_eq!(t.get_hash_values(l), vec!["h2".to_string(), "h3".to_string()]);
        assert_eq!(
            t.get_prefix_hash_values(l),
            vec!["h0".to_string(), "h1".to_string()]
        );
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn insert_host_builds_backuped_path() {
        let cfg = base_cfg();
        let mut t = UnifiedRadixTree::new(cfg);
        t.set_hicache_enabled();
        let r = t.insert_host(t.root_id(), 1, &[1, 2, 3, 4], vec![9001, 9002, 9003, 9004], None);
        let l = r
            .inserted_host_node
            .or(Some(r.last_device_node))
            .expect("host node");
        assert!(t.is_backuped(l));
        assert!(t.is_evicted(l), "host-only node is device-evicted");
        assert!(t.component_has_host_value_only(l, CT_FULL));
        let m = t.match_prefix(1, &[1, 2, 3, 4]);
        assert_eq!(m.host_hit_length, 4);
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn drop_subtree_no_host_removes_host_only_branch() {
        let mut cfg = base_cfg();
        cfg.is_write_back = true; // host refills are kept even when unbacked
        let mut t = UnifiedRadixTree::new(cfg);
        t.set_hicache_enabled();
        // Device leaf "1234" (unbacked) with a host-only child "56".
        let l = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        let host_all = vec![8000, 8001, 8002, 8003, 8004, 8005];
        let r = t.insert_host(t.root_id(), 1, &[1, 2, 3, 4, 5, 6], host_all, None);
        let h = r.inserted_host_node.expect("host-only child created");
        assert!(!r.host_insert_dropped);
        assert_eq!(t.node_by_id(h).host_value[CT_FULL as usize], Some(vec![8004, 8005]));
        // l is still a D-leaf (its only child holds no device FULL value).
        assert!(t.is_device_leaf(l));
        let out = t.drop_subtree_no_host(l);
        assert!(out.is_dropped);
        assert_eq!(t.match_prefix(1, &[1, 2, 3, 4]).full_kv_hit_length, 0);
        assert_eq!(t.match_prefix(1, &[1, 2, 3, 4, 5, 6]).host_hit_length, 0);
        let _ = h;
        assert!(t.sanity_check(&[], &[]).is_empty());
    }

    #[test]
    fn kv_event_log_tracks_stores_and_removals() {
        let mut t = UnifiedRadixTree::new(base_cfg());
        let l = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        let ev = t.take_kv_events();
        assert!(ev.iter().any(|e| e.op == 1 && e.node == l && e.medium == 1));
        t.evict_device_start(CT_FULL, 4);
        let step = t.evict_device_next_node(CT_FULL, 0);
        assert_eq!(step.node_id, Some(l));
        t.evict_device_leaf(l, false);
        t.evict_device_end(CT_FULL);
        let ev = t.take_kv_events();
        assert!(ev.iter().any(|e| e.op == 0 && e.node == l && e.medium == 1));
        assert!(t.take_kv_events().is_empty(), "log drains");
    }

    #[test]
    fn sanity_check_catches_corruption() {
        let mut t = UnifiedRadixTree::new(base_cfg());
        let _ = insert(&mut t, 1, &[1, 2, 3, 4]).last_device_node;
        // Corrupt: give the root a FULL device value but drop it from no set —
        // instead clear a live node's value while keeping it a D-leaf.
        let l = t.match_prefix(1, &[1, 2, 3, 4]).last_device_node;
        t.nodes[l as usize].value[CT_FULL as usize] = None;
        assert!(!t.sanity_check(&[], &[]).is_empty());
    }
}
