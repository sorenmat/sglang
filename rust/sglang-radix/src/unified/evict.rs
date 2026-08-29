//! Device/host eviction — port of the tree-level eviction methods in
//! `unified_tree_core.py` plus the component drivers' `evict_component` /
//! `drive_host_eviction` hooks.
//!
//! FULL device eviction walks a min-heap over D-leaves (Python
//! `FullComponent._evict_device_*`); SWA/Mamba device eviction walks the
//! component LRU cursor. Host eviction drives are per-component; under
//! write-back, FULL pressure first reclaims redundant Full host copies.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::unified::tree::PARENT_NONE;
use crate::unified::tree::UnifiedRadixTree;
use crate::unified::{
    UCacheAction, Layer, UEvictOutcome, UEvictStep, CT_BASE, CT_FULL, CT_MAMBA, CT_SWA,
};

/// Python `_RECLAIM_DIGEST_MASK` = (1 << 42) - 1.
const RECLAIM_DIGEST_MASK: u64 = (1 << 42) - 1;

/// One eviction-strategy key: the Python `get_priority` tuple flattened.
/// The trailing f64 is last_access or creation (policy-dependent) — unique
/// per node, which matches Python's `TreeNode.__lt__` tie-break and makes
/// the heap order total and deterministic.
#[derive(Clone, Copy)]
pub(crate) struct StratKey {
    a: i64,
    b: i64,
    t: f64,
}

fn strat_key_cmp(a: &StratKey, b: &StratKey) -> Ordering {
    a.a.cmp(&b.a).then_with(|| a.b.cmp(&b.b)).then_with(|| a.t.total_cmp(&b.t))
}

impl PartialEq for StratKey {
    fn eq(&self, other: &Self) -> bool {
        strat_key_cmp(self, other) == Ordering::Equal
    }
}
impl Eq for StratKey {}
impl PartialOrd for StratKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for StratKey {
    fn cmp(&self, other: &Self) -> Ordering {
        strat_key_cmp(self, other)
    }
}

/// FULL device-eviction walk (heap of D-leaves; stale pops discarded).
#[derive(Clone)]
pub(crate) struct FullEvictWalk {
    pub request_cnt: i64,
    pub last_node: Option<u32>,
    pub heap: BinaryHeap<std::cmp::Reverse<(StratKey, u32)>>,
}

/// SWA/Mamba device-eviction walk (LRU cursor).
#[derive(Clone)]
pub(crate) struct LruEvictWalk {
    pub request_cnt: i64,
    pub cursor: Option<u32>,
}

fn tracker_add(tracker: &mut Vec<(u8, i64)>, ct: u8, n: i64) {
    if n == 0 {
        return;
    }
    if let Some(entry) = tracker.iter_mut().find(|(c, _)| *c == ct) {
        entry.1 += n;
    } else {
        tracker.push((ct, n));
    }
}

fn tracker_get(tracker: &[(u8, i64)], ct: u8) -> i64 {
    tracker.iter().find(|(c, _)| *c == ct).map(|(_, n)| *n).unwrap_or(0)
}

impl UnifiedRadixTree {
    /// `eviction_strategy.get_priority(node)` — `evict_policy.py` tuples.
    pub(crate) fn strategy_key(&self, node: u32) -> StratKey {
        let n = &self.nodes[node as usize];
        match &self.cfg.eviction_policy {
            crate::policy::EvictionPolicy::Lru => StratKey { a: 0, b: 0, t: n.last_access },
            crate::policy::EvictionPolicy::Lfu => {
                StratKey { a: n.hit_count, b: 0, t: n.last_access }
            }
            crate::policy::EvictionPolicy::Fifo => StratKey { a: 0, b: 0, t: n.creation },
            crate::policy::EvictionPolicy::Mru => StratKey { a: 0, b: 0, t: -n.last_access },
            crate::policy::EvictionPolicy::Filo => StratKey { a: 0, b: 0, t: -n.creation },
            crate::policy::EvictionPolicy::Priority => StratKey {
                a: n.priority,
                b: 0,
                t: n.last_access,
            },
            crate::policy::EvictionPolicy::Slru { threshold } => StratKey {
                a: i64::from(n.hit_count >= *threshold as i64),
                b: 0,
                t: n.last_access,
            },
        }
    }

    /// `node_has_component_data(node, target)`.
    fn node_has_data(&self, node: u32, ct: u8, target: Layer) -> bool {
        let n = &self.nodes[node as usize];
        match target {
            Layer::Device => n.value[ct as usize].is_some(),
            Layer::Host => n.host_value[ct as usize].is_some(),
            Layer::All => unreachable!(),
        }
    }

    /// `eviction_priority(ct, is_leaf)`: leaf all 0; internal FULL=2 > SWA=1.
    fn evict_priority(ct: u8, is_leaf: bool) -> i64 {
        if is_leaf {
            return 0;
        }
        match ct {
            CT_BASE => 2,
            CT_SWA => 1,
            _ => 0,
        }
    }

    // ==== per-component evict (Python `evict_component`) ====

    /// Returns (device_freed, host_freed).
    pub(crate) fn evict_component(
        &mut self,
        node: u32,
        ct: u8,
        target: Layer,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) -> (i64, i64) {
        let mut freed = 0i64;
        let mut host_freed = 0i64;

        if target.device() && self.nodes[node as usize].value[ct as usize].is_some() {
            match ct {
                CT_BASE => {
                    // FULL: keep the value on the node (deferred None in the
                    // cascade) — SWA's free still needs to read it.
                    let v = self.nodes[node as usize].value[CT_BASE as usize]
                        .clone()
                        .expect("full value present");
                    freed = v.len() as i64;
                    Self::push_free(device_frees, CT_BASE, v);
                    self.evictable_size[CT_BASE as usize] -= freed;
                }
                CT_SWA => {
                    // Pass the FULL indices (free_swa skips slots without a
                    // SWA pair; freeing the SWA value would double-free).
                    let full = self.nodes[node as usize].value[CT_BASE as usize]
                        .clone()
                        .expect("SWA device evict requires Full value");
                    freed = self.nodes[node as usize].value[CT_SWA as usize]
                        .as_ref()
                        .unwrap()
                        .len() as i64;
                    self.evictable_size[CT_SWA as usize] -= freed;
                    self.nodes[node as usize].value[CT_SWA as usize] = None;
                    Self::push_free(device_frees, CT_SWA, full);
                }
                _ => {
                    let v = self.nodes[node as usize]
                        .value[CT_MAMBA as usize]
                        .take()
                        .expect("mamba value present");
                    freed = v.len() as i64;
                    self.evictable_size[CT_MAMBA as usize] -= freed;
                    Self::push_free(device_frees, CT_MAMBA, v);
                }
            }
        }

        if target.host() && self.nodes[node as usize].host_value[ct as usize].is_some() {
            let h = self.nodes[node as usize]
                .host_value[ct as usize]
                .take()
                .expect("host value present");
            host_freed = h.len() as i64;
            Self::push_free(host_frees, ct, h);
            if ct != CT_BASE {
                let slot = Self::lru_slot_public(ct, 1);
                if self.lru_in(slot, node) {
                    self.lru_remove(slot, node);
                }
            }
        }

        // After a device tombstone: if host value remains, move into host LRU.
        if target == Layer::Device && ct != CT_BASE {
            let n = &self.nodes[node as usize];
            if n.value[ct as usize].is_none() && n.host_value[ct as usize].is_some() {
                let slot = Self::lru_slot_public(ct, 1);
                if !self.lru_in(slot, node) {
                    self.lru_insert_mru(slot, node);
                }
            }
        }

        (freed, host_freed)
    }

    fn push_free(v: &mut Vec<(u8, Vec<Vec<i64>>)>, ct: u8, chunk: Vec<i64>) {
        if let Some(entry) = v.iter_mut().find(|(c, _)| *c == ct) {
            entry.1.push(chunk);
        } else {
            v.push((ct, vec![chunk]));
        }
    }

    /// `evict_component_and_detach_lru` — evict + detach from the LRU lists;
    /// `None` tracker = no credit (Python `tracker=None`).
    pub(crate) fn evict_component_and_detach_lru(
        &mut self,
        node: u32,
        ct: u8,
        target: Layer,
        tracker: Option<&mut Vec<(u8, i64)>>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) -> (i64, i64) {
        let (df, hf) = self.evict_component(node, ct, target, device_frees, host_frees);
        if let Some(tracker) = tracker {
            if target.device() {
                tracker_add(tracker, ct, df);
            } else if target.host() {
                tracker_add(tracker, ct, hf);
            }
        }
        for layer in [0u8, 1u8] {
            if (layer == 0 && target.device()) || (layer == 1 && target.host()) {
                let slot = Self::lru_slot_public(ct, layer);
                if self.lru_in(slot, node) {
                    self.lru_remove(slot, node);
                }
            }
        }
        (df, hf)
    }

    // ==== cascade (Python `_cascade_evict` / `_should_cascade_evict_component`) ====

    fn should_cascade_evict(
        &self,
        node: u32,
        trigger: u8,
        comp: u8,
        target: Layer,
        is_leaf: bool,
    ) -> bool {
        let trigger_priority = Self::evict_priority(trigger, is_leaf);
        if Self::evict_priority(comp, is_leaf) > trigger_priority {
            return false;
        }
        if comp == trigger || !self.node_has_data(node, comp, target) {
            return false;
        }
        let n = &self.nodes[node as usize];
        // Internal-priority equal-or-higher tier: a lock is a legitimate pin.
        if Self::evict_priority(comp, false) >= Self::evict_priority(trigger, false) {
            if target.device() && n.lock_ref[comp as usize] != 0 {
                return false;
            }
            if target.host() && n.host_lock_ref[comp as usize] != 0 {
                return false;
            }
            // session_ref pinning is out of scope (always 0 here).
        }
        debug_assert!(
            !target.device() || n.lock_ref[comp as usize] == 0,
            "cascade evict: locked component on evicted layer"
        );
        debug_assert!(
            !target.host() || n.host_lock_ref[comp as usize] == 0,
            "cascade evict: host-locked component on evicted layer"
        );
        true
    }

    pub(crate) fn cascade_evict(
        &mut self,
        node: u32,
        trigger: u8,
        tracker: &mut Vec<(u8, i64)>,
        target: Layer,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        let is_leaf = if target == Layer::Device {
            self.d_leaves.contains(&node)
        } else if target == Layer::Host {
            self.h_leaves.contains(&node)
        } else {
            false
        };
        let mut base_evicted = false;
        for &ct in self.active_cts().iter() {
            if self.should_cascade_evict(node, trigger, ct, target, is_leaf) {
                self.evict_component_and_detach_lru(
                    node,
                    ct,
                    target,
                    Some(tracker),
                    device_frees,
                    host_frees,
                );
                if ct == CT_BASE {
                    base_evicted = true;
                }
            }
        }
        // Deferred FULL tombstone: all lower-priority components (SWA) have
        // read the FULL value by now.
        if target.device()
            && ((trigger == CT_BASE) || base_evicted)
        {
            self.nodes[node as usize].value[CT_BASE as usize] = None;
        }
        self.update_leaf_sets(node);
    }

    // ==== demote / leaf delete (Python `_demote`, `evict_device_leaf`,
    // `_delete_unbacked_device_leaf`, `_release_all_component_layers`) ====

    pub(crate) fn demote(
        &mut self,
        node: u32,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        debug_assert!(!self.is_evicted(node) && self.is_backuped(node), "demote precondition");
        self.evict_component_and_detach_lru(
            node,
            CT_BASE,
            Layer::Device,
            Some(tracker),
            device_frees,
            host_frees,
        );
        self.cascade_evict(
            node,
            CT_BASE,
            tracker,
            Layer::Device,
            device_frees,
            host_frees,
        );
        self.record_remove(node, 1 /* GPU */);

        // After device eviction, insert aux components into host LRU.
        for &ct in self.active_cts().iter() {
            if ct == CT_FULL {
                continue;
            }
            if self.nodes[node as usize].host_value[ct as usize].is_some() {
                let slot = Self::lru_slot_public(ct, 1);
                if !self.lru_in(slot, node) {
                    self.lru_insert_mru(slot, node);
                }
            }
        }
        self.update_leaf_sets(self.nodes[node as usize].parent);
    }

    /// `release_all_component_layers` (FULL evict HOST leaves the FULL host
    /// layer freed; aux host slices follow their own pools).
    pub(crate) fn release_all_component_layers(
        &mut self,
        node: u32,
        medium: u8,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        self.record_remove(node, medium);
        for &ct in self.active_cts().iter() {
            self.evict_component_and_detach_lru(
                node,
                ct,
                Layer::All,
                Some(tracker),
                device_frees,
                host_frees,
            );
        }
        self.d_leaves.remove(&node);
        self.h_leaves.remove(&node);
    }

    pub(crate) fn delete_unbacked_device_leaf(
        &mut self,
        node: u32,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        self.release_all_component_layers(node, 1, tracker, device_frees, host_frees);
        let parent = self.nodes[node as usize].parent;
        self.remove_leaf_from_parent(node);
        self.update_leaf_sets(parent);
        self.iteratively_delete_tombstone_leaf(node, tracker, device_frees, host_frees);
    }

    /// `iteratively_delete_tombstone_leaf` / `_ancestors`.
    pub(crate) fn iteratively_delete_tombstone_leaf(
        &mut self,
        deleted_node: u32,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        let parent = self.nodes[deleted_node as usize].parent;
        self.iteratively_delete_tombstone_ancestors(parent, tracker, device_frees, host_frees);
    }

    pub(crate) fn iteratively_delete_tombstone_ancestors(
        &mut self,
        mut cur: u32,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        while cur != self.root && self.nodes[cur as usize].children.is_empty() {
            let n = &self.nodes[cur as usize];
            if n
                .lock_ref
                .iter()
                .chain(n.host_lock_ref.iter())
                .any(|&l| l > 0)
            {
                break;
            }
            let has_device = n.value[CT_BASE as usize].is_some();
            let has_host = n.host_value[CT_BASE as usize].is_some();

            if has_device {
                self.update_leaf_sets(cur);
                break;
            }
            // Full device absent — clean up orphaned aux device data.
            for &ct in self.active_cts().iter() {
                if self.node_has_data(cur, ct, Layer::Device) {
                    self.evict_component_and_detach_lru(
                        cur,
                        ct,
                        Layer::Device,
                        Some(tracker),
                        device_frees,
                        host_frees,
                    );
                }
            }
            if has_host {
                self.update_leaf_sets(cur);
                break;
            }
            // Full absent on both layers — evict remaining host data, delete.
            for &ct in self.active_cts().iter() {
                if self.node_has_data(cur, ct, Layer::Host) {
                    self.evict_component_and_detach_lru(
                        cur,
                        ct,
                        Layer::Host,
                        Some(tracker),
                        device_frees,
                        host_frees,
                    );
                }
            }
            self.h_leaves.remove(&cur);
            self.remove_leaf_from_parent(cur);
            let parent = self.nodes[cur as usize].parent;
            self.update_leaf_sets(parent);
            cur = parent;
        }
    }

    // ==== host leaves (Python `_evict_host_leaf`) ====

    pub(crate) fn evict_host_leaf(
        &mut self,
        node: u32,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        debug_assert!(self.is_host_leaf(node), "not an H-leaf");
        self.record_remove(node, 2 /* CPU */);
        for &ct in self.active_cts().iter() {
            let (_df, hf) = self.evict_component_and_detach_lru(
                node,
                ct,
                Layer::All,
                None,
                device_frees,
                host_frees,
            );
            tracker_add(tracker, ct, hf);
        }
        self.h_leaves.remove(&node);
        self.remove_leaf_from_parent(node);
        self.iteratively_delete_tombstone_leaf(node, tracker, device_frees, host_frees);
    }

    // ==== public device-leaf eviction (Python `evict_device_leaf`) ====

    pub fn evict_device_leaf(&mut self, node: u32, is_write_back: bool) -> UEvictOutcome {
        let mut result = UEvictOutcome::default();
        debug_assert!(self.is_device_leaf(node), "node is not a D-leaf");
        if !self.is_backuped(node) {
            if is_write_back {
                // `build_backup_kv` always returns a BackupKV action.
                match self.build_backup_kv(node, true) {
                    UCacheAction::BackupKV { node_ids } => {
                        result.backup_kv = Some(node_ids);
                    }
                    _ => unreachable!("backup kv action"),
                }
                return result;
            }
            self.delete_unbacked_device_leaf(
                node,
                &mut result.tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
            return result;
        }
        self.demote(
            node,
            &mut result.tracker,
            &mut result.device_frees,
            &mut result.host_frees,
        );
        result
    }

    // ==== drop_subtree_no_host (Python `drop_subtree_no_host`) ====

    pub fn drop_subtree_no_host(&mut self, node: u32) -> UEvictOutcome {
        let mut result = UEvictOutcome::default();
        debug_assert!(self.is_device_leaf(node), "node is not a D-leaf");
        debug_assert!(
            !self.is_backuped(node)
                && self.nodes[node as usize].wt_pending.is_none(),
            "drop_subtree_no_host precondition"
        );
        let n = &self.nodes[node as usize];
        if n.host_lock_ref.iter().any(|&l| l > 0) {
            return result;
        }
        let mut descendants: Vec<u32> = Vec::new();
        let mut stack: Vec<u32> = self.nodes[node as usize].children.values().copied().collect();
        while let Some(cur) = stack.pop() {
            let c = &self.nodes[cur as usize];
            if c
                .lock_ref
                .iter()
                .chain(c.host_lock_ref.iter())
                .any(|&l| l > 0)
            {
                return result;
            }
            descendants.push(cur);
            stack.extend(c.children.values());
        }
        for &desc in descendants.iter().rev() {
            debug_assert!(
                self.is_evicted(desc) && self.is_backuped(desc),
                "descendant not host-only"
            );
            debug_assert!(self.nodes[desc as usize].wt_pending.is_none());
            self.release_all_component_layers(
                desc,
                2 /* CPU */,
                &mut result.tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
            self.remove_leaf_from_parent(desc);
        }
        self.delete_unbacked_device_leaf(
            node,
            &mut result.tracker,
            &mut result.device_frees,
            &mut result.host_frees,
        );
        result.is_dropped = true;
        result
    }

    // ==== demote (public) ====

    pub fn demote_node(&mut self, node: u32) -> UEvictOutcome {
        let mut result = UEvictOutcome::default();
        self.demote(
            node,
            &mut result.tracker,
            &mut result.device_frees,
            &mut result.host_frees,
        );
        result
    }

    // ==== device-eviction walks (start/next/end) ====

    pub fn evict_device_start(&mut self, ct: u8, request_cnt: i64) {
        match ct {
            CT_FULL => {
                let heap = self
                    .d_leaves
                    .iter()
                    .map(|&n| std::cmp::Reverse((self.strategy_key(n), n)))
                    .collect();
                self.full_evict = Some(FullEvictWalk {
                    request_cnt,
                    last_node: None,
                    heap,
                });
            }
            CT_SWA | CT_MAMBA => {
                let cursor = self.lru_lru_no_lock(Self::lru_slot_public(ct, 0), ct);
                *self.walk_mut(ct) = Some(LruEvictWalk { request_cnt, cursor });
            }
            _ => {}
        }
    }

    fn walk_mut(&mut self, ct: u8) -> &mut Option<LruEvictWalk> {
        match ct {
            CT_SWA => &mut self.swa_evict,
            _ => &mut self.mamba_evict,
        }
    }

    /// `evict_device_next_node`; `running` is the caller's running tracker
    /// total for this component (Python seeds `defaultdict(int, tracker)`).
    /// Returns this step's deltas only.
    pub fn evict_device_next_node(&mut self, ct: u8, running: i64) -> UEvictStep {
        match ct {
            CT_FULL => self.full_evict_next(running),
            CT_SWA | CT_MAMBA => self.lru_evict_next(ct, running),
            _ => UEvictStep::default(),
        }
    }

    fn full_evict_next(&mut self, running: i64) -> UEvictStep {
        let mut step = UEvictStep::default();
        let mut walk = match self.full_evict.take() {
            Some(w) => w,
            None => return step,
        };
        if let Some(lv) = walk.last_node {
            let parent = self.nodes[lv as usize].parent;
            if parent != PARENT_NONE && self.d_leaves.contains(&parent) {
                walk.heap
                    .push(std::cmp::Reverse((self.strategy_key(parent), parent)));
            }
            walk.last_node = None;
        }
        while running < walk.request_cnt {
            let Some(std::cmp::Reverse((_, x))) = walk.heap.pop() else {
                break;
            };
            if !self.d_leaves.contains(&x) {
                continue;
            }
            walk.last_node = Some(x);
            step.node_id = Some(x);
            step.made_progress = true;
            break;
        }
        self.full_evict = Some(walk);
        step
    }

    fn lru_evict_next(&mut self, ct: u8, running: i64) -> UEvictStep {
        let mut step = UEvictStep::default();
        let mut walk = match self.walk_mut(ct).take() {
            Some(w) => w,
            None => return step,
        };
        let slot = Self::lru_slot_public(ct, 0);
        if let Some(c) = walk.cursor && !self.lru_in(slot, c) {
            walk.cursor = self.lru_lru_no_lock(slot, ct);
        }
        if running >= walk.request_cnt {
            *self.walk_mut(ct) = Some(walk);
            return step;
        }
        let Some(x) = walk.cursor else {
            *self.walk_mut(ct) = Some(walk);
            return step;
        };
        if !self.lru_in(slot, x) {
            *self.walk_mut(ct) = Some(walk);
            return step;
        }
        debug_assert!(
            self.nodes[x as usize].value[ct as usize].is_some(),
            "evict cursor node without component value"
        );
        if self.d_leaves.contains(&x) {
            walk.cursor = self.lru_prev_no_lock(slot, x, ct);
            step.node_id = Some(x);
            step.made_progress = true;
            *self.walk_mut(ct) = Some(walk);
            return step;
        }
        let x_next = self.lru_prev_no_lock(slot, x, ct);
        self.evict_component_and_detach_lru(
            x,
            ct,
            Layer::Device,
            Some(&mut step.tracker),
            &mut step.device_frees,
            &mut step.host_frees,
        );
        self.cascade_evict(
            x,
            ct,
            &mut step.tracker,
            Layer::Device,
            &mut step.device_frees,
            &mut step.host_frees,
        );
        walk.cursor = x_next;
        step.made_progress = true;
        *self.walk_mut(ct) = Some(walk);
        step
    }

    pub fn evict_device_end(&mut self, ct: u8) {
        match ct {
            CT_FULL => self.full_evict = None,
            _ => *self.walk_mut(ct) = None,
        }
    }

    // ==== host-eviction drives (Python `drive_host_eviction`) ====

    pub fn drive_host_eviction(&mut self, ct: u8, num_tokens: i64) -> UEvictOutcome {
        let mut result = UEvictOutcome::default();
        if self.is_write_back() && ct == CT_FULL {
            self.reclaim_full_host_duplicates(
                num_tokens,
                &mut result.tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
        }
        match ct {
            CT_FULL => self.drive_host_eviction_full(&mut result, num_tokens),
            CT_SWA => self.drive_host_eviction_aux(CT_SWA, &mut result, num_tokens),
            CT_MAMBA => self.drive_host_eviction_aux(CT_MAMBA, &mut result, num_tokens),
            _ => {}
        }
        result
    }

    fn drive_host_eviction_full(&mut self, result: &mut UEvictOutcome, num_tokens: i64) {
        let mut heap: BinaryHeap<std::cmp::Reverse<(StratKey, u32)>> = self
            .h_leaves
            .iter()
            .map(|&n| std::cmp::Reverse((self.strategy_key(n), n)))
            .collect();
        while tracker_get(&result.tracker, CT_FULL) < num_tokens {
            let Some(std::cmp::Reverse((_, x))) = heap.pop() else {
                break;
            };
            if !self.h_leaves.contains(&x) {
                continue;
            }
            self.evict_host_leaf(
                x,
                &mut result.tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
            let parent = self.nodes[x as usize].parent;
            if parent != PARENT_NONE && self.h_leaves.contains(&parent) {
                heap.push(std::cmp::Reverse((self.strategy_key(parent), parent)));
            }
        }
    }

    /// SWA/Mamba host drive; `extra_leaf_sets` mirrors Python's per-component
    /// difference (Mamba refreshes the leaf sets after an internal step).
    fn drive_host_eviction_aux(
        &mut self,
        ct: u8,
        result: &mut UEvictOutcome,
        num_tokens: i64,
    ) {
        let slot = Self::lru_slot_public(ct, 1);
        let mut x = self.lru_lru_no_host_lock(slot, ct);
        while tracker_get(&result.tracker, ct) < num_tokens {
            let Some(cur) = x else {
                break;
            };
            if !self.lru_in(slot, cur) {
                break;
            }
            let x_next = self.lru_prev_no_host_lock(slot, cur, ct);
            if self.h_leaves.contains(&cur) {
                self.evict_host_leaf(
                    cur,
                    &mut result.tracker,
                    &mut result.device_frees,
                    &mut result.host_frees,
                );
            } else {
                debug_assert!(
                    self.nodes[cur as usize].host_value[ct as usize].is_some(),
                    "internal host-evict without host value"
                );
                self.evict_component_and_detach_lru(
                    cur,
                    ct,
                    Layer::Host,
                    Some(&mut result.tracker),
                    &mut result.device_frees,
                    &mut result.host_frees,
                );
                self.cascade_evict(
                    cur,
                    ct,
                    &mut result.tracker,
                    Layer::Host,
                    &mut result.device_frees,
                    &mut result.host_frees,
                );
                if ct == CT_MAMBA {
                    // Python mamba_component line 895: extra leaf-set refresh.
                    self.update_leaf_sets(cur);
                }
            }
            x = x_next;
        }
    }

    // ==== write-back duplicate reclaim (Python `_reclaim_full_host_duplicates`) ====

    fn reclaim_full_host_duplicates(
        &mut self,
        num_tokens: i64,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        let mut swept_ids: Vec<u32> = Vec::new();
        // No new duplicates join during the walk (tracking is not updated
        // here), so a snapshot mirrors Python's live-dict iteration.
        let ids = self.dup_ids.clone();
        for spare_imminent_demotes in [true, false] {
            if tracker_get(tracker, CT_BASE) >= num_tokens {
                break;
            }
            for &node in ids.iter() {
                if tracker_get(tracker, CT_BASE) >= num_tokens {
                    break;
                }
                let n = &self.nodes[node as usize];
                if n.value[CT_BASE as usize].is_none()
                    || n.host_value[CT_BASE as usize].is_none()
                {
                    swept_ids.push(node); // stale entry
                    continue;
                }
                if spare_imminent_demotes && self.d_leaves.contains(&node) {
                    continue;
                }
                if !self.can_reclaim_full_host_duplicate(node) {
                    continue;
                }
                self.release_full_host_duplicate(
                    node,
                    tracker,
                    device_frees,
                    host_frees,
                );
                swept_ids.push(node);
            }
        }
        // Sweep after the walk: tracking must not be mutated mid-iteration.
        for nid in swept_ids {
            self.dup_set.remove(&nid);
        }
    }

    fn can_reclaim_full_host_duplicate(&self, node: u32) -> bool {
        let n = &self.nodes[node as usize];
        if node == self.root
            || n.value[CT_BASE as usize].is_none()
            || n.host_value[CT_BASE as usize].is_none()
        {
            return false;
        }
        if n.wt_pending.is_some() || n.lb_pending.is_some() {
            return false;
        }
        n.host_lock_ref[CT_BASE as usize] == 0
    }

    fn release_full_host_duplicate(
        &mut self,
        node: u32,
        tracker: &mut Vec<(u8, i64)>,
        device_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
        host_frees: &mut Vec<(u8, Vec<Vec<i64>>)>,
    ) {
        debug_assert!(self.can_reclaim_full_host_duplicate(node));
        self.record_remove(node, 2 /* CPU */);
        self.evict_component_and_detach_lru(
            node,
            CT_BASE,
            Layer::Host,
            Some(tracker),
            device_frees,
            host_frees,
        );
        self.reclaim_digest =
            (self.reclaim_digest.wrapping_mul(1000003) + node as u64 + 1) & RECLAIM_DIGEST_MASK;
    }

    // ==== Mamba excess path states (Python `_evict_excess_path_states`) ====

    pub fn evict_excess_path_states(&mut self, tail: u32) -> UEvictOutcome {
        let mut result = UEvictOutcome::default();
        let cap = self.cfg.mamba_max_states_per_path;
        if cap < 0 {
            return result;
        }
        let mut holders: Vec<u32> = Vec::new();
        let mut node = tail;
        while node != self.root {
            if self.nodes[node as usize].value[CT_MAMBA as usize].is_some() {
                holders.push(node);
            }
            node = self.nodes[node as usize].parent;
        }
        let mut excess = holders.len() as i64 - cap;
        if excess <= 0 {
            return result;
        }
        for node in holders.iter().rev() {
            if excess <= 0 || *node == tail {
                break;
            }
            let n = &self.nodes[*node as usize];
            if n.lock_ref[CT_MAMBA as usize] > 0 || n.children.len() != 1 {
                continue;
            }
            if self.d_leaves.contains(node) {
                continue;
            }
            self.evict_component_and_detach_lru(
                *node,
                CT_MAMBA,
                Layer::Device,
                Some(&mut result.tracker),
                &mut result.device_frees,
                &mut result.host_frees,
            );
            self.cascade_evict(
                *node,
                CT_MAMBA,
                &mut result.tracker,
                Layer::Device,
                &mut result.device_frees,
                &mut result.host_frees,
            );
            excess -= 1;
        }
        result
    }
}
