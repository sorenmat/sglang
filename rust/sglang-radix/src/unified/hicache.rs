//! HiCache tree-level machinery — port of `unified_tree_core.py`'s HiCache
//! section (`insert_host`, `build_backup_spec`, `build_storage_backup_spec`,
//! `build_load_back_spec`, `commit_backup`, `commit_load_back`,
//! `finish_load_back`, write-through marks) plus the tree-state parts of the
//! SWA/Mamba `build_hicache_transfers` / `commit_hicache_transfer` hooks
//! (allocator DMAs and mapping rebuilds stay facade-side via
//! [`UCacheAction`]).
//!
//! Pool ids: 0 = KV, 1 = SWA, 2 = MAMBA (Python `PoolName`).
//! Phases: 0 = BACKUP_HOST, 1 = LOAD_BACK, 2 = BACKUP_STORAGE, 3 = PREFETCH.
//! `UTransfer.hit_policy`: 0 = exact, 1 = TRAILING_PAGES.

use crate::unified::tree::PARENT_NONE;
use crate::unified::tree::UnifiedRadixTree;
use crate::unified::{
    UCacheAction, UTransfer, UIntsertResult, CT_BASE, CT_FULL, CT_MAMBA, CT_SWA,
};

pub const PHASE_BACKUP_HOST: u8 = 0;
pub const PHASE_LOAD_BACK: u8 = 1;
pub const PHASE_BACKUP_STORAGE: u8 = 2;
pub const PHASE_PREFETCH: u8 = 3;

const POOL_KV: u8 = 0;
const POOL_SWA: u8 = 1;
const POOL_MAMBA: u8 = 2;

/// `StorageBackupSpec` (tree-visible fields).
#[derive(Debug, Clone, Default)]
pub struct UStorageBackupSpec {
    pub host_value: Option<Vec<i64>>,
    /// Raw node key tokens.
    pub token_ids: Vec<i64>,
    pub hash_value: Option<Vec<String>>,
    pub prefix_keys: Option<Vec<String>>,
    pub comp_xfers: Vec<(u8, Vec<UTransfer>)>,
}

/// `build_load_back_spec` result; `None` = conflict (a node is pinned by
/// another load-back anchor) — the facade emits the empty spec instead.
#[derive(Debug, Clone, Default)]
pub struct ULoadBackSpec {
    pub kv: UTransfer,
    pub comp_xfers: Vec<(u8, Vec<UTransfer>)>,
}

impl UnifiedRadixTree {
    /// `build_backup_spec` — FULL device value (empty when already backuped)
    /// plus the aux components' BACKUP_HOST transfers.
    pub fn build_backup_spec(&self, node: u32) -> (Vec<i64>, Vec<(u8, Vec<UTransfer>)>) {
        let device_value = if self.is_backuped(node) {
            Vec::new()
        } else {
            self.nodes[node as usize]
                .value[CT_BASE as usize]
                .clone()
                .expect("backup spec on evicted node")
        };
        let mut comp_xfers: Vec<(u8, Vec<UTransfer>)> = Vec::new();
        for &ct in self.active_cts().iter() {
            if ct == CT_FULL {
                continue;
            }
            if self.nodes[node as usize].host_value[ct as usize].is_some() {
                continue;
            }
            if let Some(xfer) = self.build_comp_transfer(ct, node, PHASE_BACKUP_HOST, None, None)
            {
                comp_xfers.push((ct, vec![xfer]));
            }
        }
        (device_value, comp_xfers)
    }

    /// `build_storage_backup_spec`; None when the node is not backuped.
    pub fn build_storage_backup_spec(&self, node: u32, pass_prefix_keys: bool) -> Option<UStorageBackupSpec> {
        if !self.is_backuped(node) {
            return None;
        }
        let n = &self.nodes[node as usize];
        let mut spec = UStorageBackupSpec {
            host_value: n.host_value[CT_BASE as usize].clone(),
            token_ids: n.key.clone(),
            hash_value: n.hash_value.clone(),
            prefix_keys: pass_prefix_keys.then(|| self.get_prefix_hash_values(node)),
            ..Default::default()
        };
        for &ct in self.active_cts().iter() {
            if ct == CT_FULL {
                continue;
            }
            if let Some(xfer) =
                self.build_comp_transfer(ct, node, PHASE_BACKUP_STORAGE, None, None)
            {
                spec.comp_xfers.push((ct, vec![xfer]));
            }
        }
        Some(spec)
    }

    /// SWA/Mamba `build_hicache_transfers` (tree-state parts).
    pub fn build_comp_transfer(
        &self,
        ct: u8,
        node: u32,
        phase: u8,
        host_indices: Option<&[i64]>,
        mamba_pool_idx: Option<&[i64]>,
    ) -> Option<UTransfer> {
        let page = i64::from(self.cfg.page_size);
        let n = &self.nodes[node as usize];
        match ct {
            CT_SWA => {
                // unified_kv keeps SWA as a device-only ring.
                if !self.cfg.has_swa_host_pool && self.enable_hicache {
                    return None;
                }
                match phase {
                    PHASE_BACKUP_HOST => {
                        let v = n.value[CT_SWA as usize].clone()?;
                        Some(UTransfer {
                            pool: POOL_SWA,
                            device_indices: Some(v),
                            ..Default::default()
                        })
                    }
                    PHASE_LOAD_BACK => {
                        let window = self.cfg.sliding_window_size;
                        let mut n_swa = 0i64;
                        let mut backed_up: Vec<Vec<i64>> = Vec::new();
                        let mut nodes: Vec<u32> = Vec::new();
                        let mut cur = node;
                        while cur != self.root && n_swa < window {
                            let cd = &self.nodes[cur as usize];
                            let has_val = cd.value[CT_SWA as usize].is_some();
                            let has_host = cd.host_value[CT_SWA as usize].is_some();
                            debug_assert!(has_val || has_host, "SWA load-back gap");
                            if has_val {
                                n_swa += cd
                                    .value[CT_SWA as usize]
                                    .as_ref()
                                    .unwrap()
                                    .len() as i64;
                            } else {
                                backed_up.push(cd.host_value[CT_SWA as usize].clone().unwrap());
                                nodes.push(cur);
                                n_swa += cd
                                    .host_value[CT_SWA as usize]
                                    .as_ref()
                                    .unwrap()
                                    .len() as i64;
                            }
                            cur = cd.parent;
                        }
                        if backed_up.is_empty() {
                            return None;
                        }
                        backed_up.reverse();
                        nodes.reverse();
                        Some(UTransfer {
                            pool: POOL_SWA,
                            host_indices: Some(backed_up.concat()),
                            device_indices: None,
                            nodes_to_load: nodes,
                            ..Default::default()
                        })
                    }
                    PHASE_BACKUP_STORAGE => {
                        let hv = n.host_value[CT_SWA as usize].clone()?;
                        let hashes = n.hash_value.as_ref()?;
                        let num_pages = hv.len() as i64 / page;
                        if num_pages == 0 {
                            return None;
                        }
                        let start = hv.len() - (num_pages as usize) * page as usize;
                        Some(UTransfer {
                            pool: POOL_SWA,
                            host_indices: Some(hv[start..].to_vec()),
                            keys: Some(hashes[hashes.len() - num_pages as usize..].to_vec()),
                            hit_policy: 1,
                            ..Default::default()
                        })
                    }
                    PHASE_PREFETCH => {
                        let hi = host_indices?;
                        let num_pages = hi.len() as i64 / page;
                        Some(UTransfer {
                            pool: POOL_SWA,
                            host_indices: Some(hi.to_vec()),
                            keys: Some(vec!["__placeholder__".to_string(); num_pages as usize]),
                            hit_policy: 1,
                            ..Default::default()
                        })
                    }
                    _ => None,
                }
            }
            CT_MAMBA => match phase {
                PHASE_BACKUP_HOST => {
                    let v = n.value[CT_MAMBA as usize].clone()?;
                    Some(UTransfer {
                        pool: POOL_MAMBA,
                        device_indices: Some(v),
                        ..Default::default()
                    })
                }
                PHASE_LOAD_BACK => {
                    if n.value[CT_MAMBA as usize].is_some() {
                        return None;
                    }
                    let mut transfers: Vec<UTransfer> = Vec::new();
                    let hv = n.host_value[CT_MAMBA as usize].clone();
                    if let Some(hv) = hv.clone() {
                        transfers.push(UTransfer {
                            pool: POOL_MAMBA,
                            host_indices: Some(hv),
                            nodes_to_load: vec![node],
                            ..Default::default()
                        });
                    }
                    if let (Some(pool_idx), Some(hv)) = (mamba_pool_idx, hv) {
                        transfers.push(UTransfer {
                            pool: POOL_MAMBA,
                            host_indices: Some(hv),
                            device_indices: Some(pool_idx.to_vec()),
                            ..Default::default()
                        });
                    }
                    transfers.into_iter().next()
                }
                PHASE_BACKUP_STORAGE => {
                    let hv = n.host_value[CT_MAMBA as usize].clone()?;
                    let hashes = n.hash_value.as_ref()?;
                    Some(UTransfer {
                        pool: POOL_MAMBA,
                        host_indices: Some(hv),
                        keys: Some(vec![hashes.last().cloned().unwrap()]),
                        hit_policy: 1,
                        ..Default::default()
                    })
                }
                PHASE_PREFETCH => {
                    let hi = host_indices?;
                    Some(UTransfer {
                        pool: POOL_MAMBA,
                        host_indices: Some(hi.to_vec()),
                        keys: Some(vec!["__placeholder__".to_string()]),
                        hit_policy: 1,
                        ..Default::default()
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// `build_load_back_spec` — FULL evicted-chain KV transfer + aux transfers.
    /// `None` when a would-be loaded node is pinned by a different anchor.
    pub fn build_load_back_spec(
        &self,
        node: u32,
        mamba_pool_idx: Option<&[i64]>,
    ) -> Option<ULoadBackSpec> {
        // FULL LOAD_BACK: walk the evicted chain (leaves up to a device node).
        let mut backed_up: Vec<Vec<i64>> = Vec::new();
        let mut nodes: Vec<u32> = Vec::new();
        let mut cur = node;
        while self.is_evicted(cur) {
            let hv = self.nodes[cur as usize]
                .host_value[CT_BASE as usize]
                .clone()
                .expect("load-back on node without host backup");
            backed_up.push(hv);
            nodes.push(cur);
            cur = self.nodes[cur as usize].parent;
        }
        backed_up.reverse();
        nodes.reverse();
        let kv = UTransfer {
            pool: POOL_KV,
            host_indices: Some(backed_up.concat()),
            device_indices: None,
            nodes_to_load: nodes,
            ..Default::default()
        };

        let mut comp_xfers: Vec<(u8, Vec<UTransfer>)> = Vec::new();
        for &ct in self.active_cts().iter() {
            if ct == CT_FULL {
                continue;
            }
            if let Some(xfer) =
                self.build_comp_transfer(ct, node, PHASE_LOAD_BACK, None, mamba_pool_idx)
            {
                comp_xfers.push((ct, vec![xfer]));
            }
        }

        // Conflict check: reject any node pinned by another load-back anchor.
        let mut candidates: Vec<u32> = kv.nodes_to_load.clone();
        for (_ct, xfers) in &comp_xfers {
            for x in xfers {
                candidates.extend_from_slice(&x.nodes_to_load);
            }
        }
        if candidates.iter().any(|&nid| {
            matches!(self.nodes[nid as usize].lb_pending, Some(p) if p != node)
        }) {
            return None;
        }
        Some(ULoadBackSpec { kv, comp_xfers })
    }

    /// `commit_backup` — applies a successful D->H backup. No actions are
    /// emitted (Python `assert not cache_actions`).
    pub fn commit_backup(
        &mut self,
        node: u32,
        host_indices: &[i64],
        comp_xfers: &[(u8, Vec<UTransfer>)],
    ) {
        if !host_indices.is_empty() {
            self.nodes[node as usize].host_value[CT_BASE as usize] = Some(host_indices.to_vec());
        }
        for &(ct, ref xfers) in comp_xfers {
            let Some(xfer) = xfers.first() else { continue };
            if let Some(hi) = &xfer.host_indices
                && self.nodes[node as usize].host_value[ct as usize].is_none()
            {
                self.nodes[node as usize].host_value[ct as usize] = Some(hi.clone());
            }
        }
    }

    /// `commit_load_back` — applies a successful H->D load-back; returns the
    /// deferred actions (SWA full->swa mapping rebuilds).
    pub fn commit_load_back(
        &mut self,
        node: u32,
        device_indices: Option<&[i64]>,
        kv_nodes_to_load: &[u32],
        comp_xfers: &[(u8, Vec<UTransfer>)],
    ) -> Vec<UCacheAction> {
        let mut actions: Vec<UCacheAction> = Vec::new();

        if self.is_write_back() {
            for &nid in kv_nodes_to_load {
                let pinned = &self.nodes[nid as usize];
                debug_assert!(
                    pinned.lb_pending.is_none() || pinned.lb_pending == Some(node),
                    "node pinned by a different load-back anchor"
                );
                self.nodes[nid as usize].lb_pending = Some(node);
            }
        }

        // FULL LOAD_BACK commit.
        if let Some(dev) = device_indices {
            let mut offset = 0usize;
            for &nid in kv_nodes_to_load {
                let n_len = self.nodes[nid as usize]
                    .host_value[CT_BASE as usize]
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0);
                let chunk = dev[offset..offset + n_len].to_vec();
                self.nodes[nid as usize].value[CT_BASE as usize] = Some(chunk);
                offset += n_len;
                self.evictable_size[CT_BASE as usize] += n_len as i64;
                self.update_leaf_sets(nid);
                self.record_store(nid, 1 /* GPU */);
            }
            self.update_leaf_sets(node);
        } else {
            self.update_leaf_sets(node);
        }

        // Aux component commits.
        for &(ct, ref xfers) in comp_xfers {
            let Some(xfer) = xfers.first() else { continue };
            match ct {
                CT_SWA => {
                    let dev = xfer
                        .device_indices
                        .as_ref()
                        .expect("SWA load-back without device indices");
                    let mut full_chunks: Vec<Vec<i64>> = Vec::new();
                    let mut swa_chunks: Vec<Vec<i64>> = Vec::new();
                    let mut offset = 0usize;
                    for &nid in &xfer.nodes_to_load {
                        let n_tokens = self.nodes[nid as usize]
                            .host_value[CT_SWA as usize]
                            .as_ref()
                            .map(|v| v.len())
                            .unwrap_or(0);
                        let swa_chunk = dev[offset..offset + n_tokens].to_vec();
                        self.set_component_device_value(nid, CT_SWA, swa_chunk.clone());
                        let full = self.nodes[nid as usize]
                            .value[CT_BASE as usize]
                            .as_ref()
                            .expect("SWA load-back requires Full device value");
                        debug_assert_eq!(full.len(), n_tokens);
                        full_chunks.push(full.clone());
                        swa_chunks.push(swa_chunk.clone());
                        offset += n_tokens;
                    }
                    if !full_chunks.is_empty() {
                        actions.push(UCacheAction::RebuildFullToSWAMapping {
                            full_indices: full_chunks,
                            swa_indices: swa_chunks,
                        });
                    }
                }
                CT_MAMBA => {
                    if let Some(dev) = &xfer.device_indices {
                        let count = dev.len();
                        self.nodes[node as usize].value[CT_MAMBA as usize] = Some(dev.clone());
                        let host_slot = Self::lru_slot_public(CT_MAMBA, 1);
                        if self.lru_in(host_slot, node) {
                            self.lru_remove(host_slot, node);
                        }
                        self.lru_insert_mru(Self::lru_slot_public(CT_MAMBA, 0), node);
                        self.evictable_size[CT_MAMBA as usize] += count as i64;
                    }
                }
                _ => {}
            }
        }

        self.update_leaf_sets(node);
        actions
    }

    /// `finish_load_back` — finalize load-back state along the anchor path.
    pub fn finish_load_back(&mut self, anchor: u32) {
        let mut node = anchor;
        while node != PARENT_NONE && node != self.root {
            if self.is_write_back() {
                if self.nodes[node as usize].lb_pending != Some(anchor) {
                    node = self.nodes[node as usize].parent;
                    continue;
                }
                self.nodes[node as usize].lb_pending = None;
            }
            self.update_duplicate_tracking(node);
            node = self.nodes[node as usize].parent;
        }
    }

    /// `mark_write_through_pending` (the pending id is the node itself).
    pub fn mark_write_through_pending(&mut self, node: u32) {
        self.nodes[node as usize].wt_pending = Some(node as i64);
    }

    /// `finish_write_through`.
    pub fn finish_write_through(&mut self, node_ids: &[u32], ack_id: i64) {
        for &node_id in node_ids {
            let cleared = {
                let n = &mut self.nodes[node_id as usize];
                let matched = n.wt_pending == Some(ack_id);
                if matched {
                    n.wt_pending = None;
                }
                matched
            };
            if cleared {
                // The backed-up copy becomes a tracked duplicate only now.
                self.update_duplicate_tracking(node_id);
            }
            self.record_store(node_id, 2 /* CPU */);
        }
    }

    /// `prefetch_anchor_info` — the anchor's namespace id (the facade maps
    /// it back to (extra_key, cache_salt)).
    pub fn prefetch_anchor_ns(&self, node: u32) -> u32 {
        self.nodes[node as usize].ns
    }

    // ==== insert_host (Python `insert_host`) ====

    /// `insert_host` — insert a host-side (backuped) path below `anchor`.
    /// `ns` is the key's namespace (the Python key carries it inside
    /// extra_key/cache_salt); it must agree with the anchor's when the
    /// anchor is a path node. `raw_tokens` is the (already page-aligned)
    /// raw key; `hash_value` the node's page hashes.
    pub fn insert_host(
        &mut self,
        anchor: u32,
        ns: u32,
        raw_tokens: &[i64],
        host_value: Vec<i64>,
        hash_value: Option<Vec<String>>,
    ) -> UIntsertResult {
        let total_len = self.key_len(raw_tokens);
        self.touch_node(anchor);
        if anchor != self.root {
            debug_assert_eq!(self.nodes[anchor as usize].ns, ns, "insert_host ns mismatch");
        }
        if total_len == 0 {
            return UIntsertResult {
                prefix_len: 0,
                total_len: 0,
                mamba_exist: true,
                ..Default::default()
            };
        }

        let mut node = anchor;
        let mut key: Vec<i64> = raw_tokens.to_vec();
        let mut host_value = host_value;
        let mut hash_value = hash_value;
        let mut matched_length = 0i64;
        let mut cache_actions: Vec<UCacheAction> = Vec::new();

        while !key.is_empty() {
            let ck = self.child_key_of(&key, ns);
            let next = match self.nodes[node as usize].children.get(&ck) {
                Some(&c) => c,
                None => break,
            };
            node = next;
            self.touch_node(node);
            let ckey = self.nodes[node as usize].key.clone();
            let prefix_len = self.match_len(&ckey, &key);
            // Advance the remaining key/value/hashes.
            key = self.slice_key_back(&key, prefix_len as usize);
            host_value = host_value[prefix_len as usize..].to_vec();
            hash_value = hash_value.map(|h| {
                let skip = ((prefix_len / i64::from(self.cfg.page_size)) as usize)
                .min(h.len());
                h[skip..].to_vec()
            });
            matched_length += prefix_len;

            if prefix_len < self.key_len(&ckey) {
                let (front, action) = self.split_node(node, prefix_len);
                if let Some(a) = action {
                    cache_actions.push(a);
                }
                node = front;
            }
        }

        let mut result = UIntsertResult {
            prefix_len: matched_length,
            total_len,
            cache_actions,
            ..Default::default()
        };

        if key.is_empty() {
            if node != self.root
                && self.nodes[node as usize].host_value[CT_BASE as usize].is_some()
            {
                result.inserted_host_node = Some(node);
            }
            return result;
        }

        // Drop the refill only under write-through.
        if node != self.root
            && !self.is_backuped(node)
            && !self.is_write_back()
        {
            result.host_insert_dropped = true;
            return result;
        }

        let priority = self.nodes[node as usize].priority;
        let new_node = self.new_node(priority);
        {
            let n = &mut self.nodes[new_node as usize];
            n.parent = node;
            n.ns = ns;
            n.key = key;
            n.hash_value = hash_value;
            n.host_value[CT_BASE as usize] = Some(host_value);
        }
        let ck = self.child_key_of(&self.nodes[new_node as usize].key, ns);
        self.nodes[node as usize].children.insert(ck.clone(), new_node);
        self.nodes[node as usize].child_order.push(ck);
        self.update_leaf_sets(new_node);
        self.update_leaf_sets(node);
        result.inserted_host_node = Some(new_node);
        result
    }

    /// Logical back-slice of a raw key (bigram keeps the boundary token).
    fn slice_key_back(&self, raw: &[i64], split: usize) -> Vec<i64> {
        self.split_raw_key(raw, split).1
    }

    // ==== SWA prefetch commit (Python `_commit_prefetch`) ====

    /// `SWAComponent._commit_prefetch` (tree-state parts; host releases are
    /// `FreeComponentHostSlot` actions).
    pub fn commit_swa_prefetch(
        &mut self,
        anchor: u32,
        host_indices: Vec<i64>,
        loaded_pages: i64,
        insert_result: &UIntsertResult,
    ) -> Vec<UCacheAction> {
        let mut actions: Vec<UCacheAction> = Vec::new();
        let page = i64::from(self.cfg.page_size);
        let window_require_pages = host_indices.len() as i64 / page;
        let full_window_pages = self.cfg.swa_tail_size() / page;
        let target = insert_result.inserted_host_node;

        if anchor != self.root && window_require_pages < full_window_pages {
            // Cache-mode graft commit with a hit-shrunk window: drop it.
            if !host_indices.is_empty() {
                actions.push(UCacheAction::FreeComponentHostSlot {
                    ct: CT_SWA,
                    chunks: vec![host_indices],
                });
            }
            return actions;
        }
        if target.is_none()
            || window_require_pages == 0
            || loaded_pages < window_require_pages
        {
            if !host_indices.is_empty() {
                actions.push(UCacheAction::FreeComponentHostSlot {
                    ct: CT_SWA,
                    chunks: vec![host_indices],
                });
            }
            return actions;
        }
        let target = target.expect("checked above");

        let loaded_start = insert_result.total_len - window_require_pages * page;
        let mut pos = insert_result.total_len;
        let mut cur = target;
        while cur != anchor && pos > loaded_start {
            let node_len = self.key_len(&self.nodes[cur as usize].key);
            let node_start = pos - node_len;
            let fill_start = node_start.max(loaded_start);
            let fill_len = pos - fill_start;
            let buf_off = fill_start - loaded_start;
            let host_slice = host_indices[buf_off as usize..(buf_off + fill_len) as usize].to_vec();

            if self.nodes[cur as usize].host_value[CT_SWA as usize].is_none() && fill_len > 0 {
                if fill_start > node_start {
                    let (_, action) =
                        self.split_node(cur, fill_start - node_start);
                    if let Some(a) = action {
                        actions.push(a);
                    }
                }
                self.attach_swa_host_value(cur, host_slice);
            } else if fill_len > 0 {
                // Already has SWA: drop this slice.
                actions.push(UCacheAction::FreeComponentHostSlot {
                    ct: CT_SWA,
                    chunks: vec![host_slice],
                });
            }
            pos = node_start;
            cur = self.nodes[cur as usize].parent;
        }

        if pos > loaded_start {
            actions.push(UCacheAction::FreeComponentHostSlot {
                ct: CT_SWA,
                chunks: vec![host_indices[..(pos - loaded_start) as usize].to_vec()],
            });
        }
        actions
    }

    /// `SWAComponent._attach_swa_host_value`.
    fn attach_swa_host_value(&mut self, node: u32, host_indices: Vec<i64>) {
        let in_list = self.lru_in(Self::lru_slot_public(CT_SWA, 1), node);
        self.nodes[node as usize].host_value[CT_SWA as usize] = Some(host_indices);
        let value_none = self.nodes[node as usize].value[CT_SWA as usize].is_none();
        if value_none && !in_list {
            self.lru_insert_mru(Self::lru_slot_public(CT_SWA, 1), node);
        }
        self.update_leaf_sets(node);
        let parent = self.nodes[node as usize].parent;
        if parent != PARENT_NONE {
            self.update_leaf_sets(parent);
        }
    }

    // ==== Mamba prefetch commit (Python MambaComponent PREFETCH commit) ====

    pub fn commit_mamba_prefetch(
        &mut self,
        host_indices: Option<Vec<i64>>,
        loaded: bool,
        insert_result: &mut UIntsertResult,
    ) -> Vec<UCacheAction> {
        let target = insert_result.inserted_host_node;
        let has_host = target.map(|t| self.nodes[t as usize].host_value[CT_MAMBA as usize].is_some());
        if host_indices.is_none()
            || target.is_none()
            || !loaded
            || has_host.unwrap_or(false)
        {
            if insert_result.inserted_host_node.is_some() {
                insert_result.mamba_exist = true;
            }
            return vec![UCacheAction::FreeComponentHostSlot {
                ct: CT_MAMBA,
                chunks: vec![host_indices.unwrap_or_default()],
            }];
        }
        let target = target.expect("checked above");
        let host_indices = host_indices.expect("checked above");
        self.nodes[target as usize].host_value[CT_MAMBA as usize] = Some(host_indices);
        if self.nodes[target as usize].value[CT_MAMBA as usize].is_none()
            && !self.lru_in(Self::lru_slot_public(CT_MAMBA, 1), target)
        {
            self.lru_insert_mru(Self::lru_slot_public(CT_MAMBA, 1), target);
        }
        insert_result.mamba_exist = false;
        Vec::new()
    }
}
