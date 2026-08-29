//! The stepped (resumable) insert — port of `UnifiedTreeCore.begin_insert` /
//! `_advance_insert` / `_insert_walk_step` / `_insert_commit_step` /
//! `_insert_tail_step` plus the SWA and Mamba tree-level hooks invoked from
//! the walk/commit/tail phases.

use crate::unified::tree::UnifiedRadixTree;
use crate::unified::{UCacheAction, UIntsertResult, UStepResult, CT_BASE, CT_FULL, CT_MAMBA, CT_SWA};

/// `InsertParams` (tree-relevant fields).
#[derive(Debug, Clone, Default)]
pub struct UInsertParams {
    pub prev_prefix_len: i64,
    pub chunked: bool,
    pub priority: i64,
    pub swa_evicted_seqlen: i64,
    pub mamba_value: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertPhase {
    Walk,
    Commit,
    Tail,
}

/// In-flight resumable-insert state (Python `_InsertWalkState`).
#[derive(Debug, Clone)]
pub struct InsertState {
    pub phase: InsertPhase,
    pub node: u32,
    pub ns: u32,
    /// Remaining raw key tokens.
    pub key: Vec<i64>,
    /// Remaining value (logical domain, aligned to `key`).
    pub value: Vec<i64>,
    pub params: UInsertParams,
    pub priority: i64,
    pub total_prefix_length: i64,
    pub is_new_leaf: bool,
    pub target_node: u32,
    pub result: Option<UIntsertResult>,
    /// Actions awaiting the next barrier flush (or the final step).
    pub pending: Vec<UCacheAction>,
}

impl UnifiedRadixTree {
    /// `begin_insert`.
    pub fn begin_insert(
        &mut self,
        ns: u32,
        raw_tokens: &[i64],
        value: Option<Vec<i64>>,
        params: &UInsertParams,
    ) -> UStepResult {
        debug_assert!(self.ongoing.is_none(), "concurrent insert walks");
        let priority = params.priority;
        let key = self.page_align_raw(raw_tokens);
        self.touch_node(self.root);
        {
            let r = &mut self.nodes[self.root as usize];
            r.priority = r.priority.max(priority);
        }
        if self.key_len(key) == 0 {
            return UStepResult {
                actions: Vec::new(),
                result: Some(UIntsertResult {
                    prefix_len: 0,
                    total_len: 0,
                    last_device_node: self.root,
                    mamba_exist: true,
                    ..Default::default()
                }),
            };
        }
        let logical = self.key_len(key) as usize;
        let value = match value {
            Some(mut v) => {
                v.truncate(logical);
                v
            }
            None => key[..logical].to_vec(),
        };
        self.ongoing = Some(InsertState {
            phase: InsertPhase::Walk,
            node: self.root,
            ns,
            key: key.to_vec(),
            value,
            params: params.clone(),
            priority,
            total_prefix_length: 0,
            is_new_leaf: false,
            target_node: 0,
            result: None,
            pending: Vec::new(),
        });
        Self::advance_insert(self)
    }

    /// `resume_insert`.
    pub fn resume_insert(&mut self) -> UStepResult {
        Self::advance_insert(self)
    }

    pub fn has_ongoing_insert(&self) -> bool {
        self.ongoing.is_some()
    }

    /// `end_insert` (idempotent): drains still-pending actions.
    pub fn end_insert(&mut self) -> Vec<UCacheAction> {
        self.ongoing.take().map(|s| s.pending).unwrap_or_default()
    }

    fn advance_insert(t: &mut UnifiedRadixTree) -> UStepResult {
        loop {
            let mut state = t.ongoing.take().expect("no in-flight insert");
            let flushed_len = state.pending.len();
            match state.phase {
                InsertPhase::Walk => Self::walk_step(t, &mut state),
                InsertPhase::Commit => Self::commit_step(t, &mut state),
                InsertPhase::Tail => {
                    Self::tail_step(t, &mut state);
                    t.ongoing = None;
                    return UStepResult {
                        actions: state.pending,
                        result: state.result,
                    };
                }
            }
            let new_actions = &state.pending[flushed_len..];
            // Suspend only when a step emitted a non-deferrable action; the
            // walk state stays in flight (Python flushes and clears
            // pending_actions but keeps the walk state).
            if !new_actions.is_empty() && new_actions.iter().any(|a| !a.is_deferrable()) {
                let flushed = std::mem::take(&mut state.pending);
                t.ongoing = Some(state);
                return UStepResult {
                    actions: flushed,
                    result: None,
                };
            }
            t.ongoing = Some(state);
        }
    }

    /// Python `_inc_hit_count_and_check`.
    pub(crate) fn inc_hit_count_and_check(&mut self, node: u32, chunked: bool) -> bool {
        if self.is_evicted(node) || chunked {
            return false;
        }
        if self.cfg.is_write_back {
            return false;
        }
        self.nodes[node as usize].hit_count += 1;
        self.enable_hicache
            && !self.is_backuped(node)
            && self.nodes[node as usize].hit_count >= self.cfg.write_through_threshold
    }

    /// Python `_build_backup_kv_action` (write_back == false chain form).
    pub(crate) fn build_backup_kv(&self, node: u32, write_back: bool) -> UCacheAction {
        let mut chain = vec![node];
        if !write_back {
            let mut ancestor = self.nodes[node as usize].parent;
            while ancestor != crate::unified::tree::PARENT_NONE
                && ancestor != self.root
                && !self.is_backuped(ancestor)
            {
                chain.push(ancestor);
                ancestor = self.nodes[ancestor as usize].parent;
            }
            chain.reverse();
        }
        UCacheAction::BackupKV { node_ids: chain }
    }

    /// `needs_incremental_component_backup` (any aux component). SWA: always
    /// False. Mamba: device value present, host value absent.
    fn needs_incremental_component_backup(&self, node: u32) -> bool {
        if self.has_mamba {
            let n = &self.nodes[node as usize];
            if n.value[CT_MAMBA as usize].is_some()
                && n.host_value[CT_MAMBA as usize].is_none()
            {
                return true;
            }
        }
        false
    }

    fn should_backup_existing(&self, node: u32) -> bool {
        self.enable_hicache
            && !self.cfg.is_write_back
            && self.is_backuped(node)
            && self.nodes[node as usize].wt_pending.is_none()
            && self.needs_incremental_component_backup(node)
    }
}

impl UnifiedRadixTree {
    fn walk_step(t: &mut UnifiedRadixTree, state: &mut InsertState) {
        if t.key_len(&state.key) == 0 {
            state.phase = InsertPhase::Commit;
            return;
        }
        let ck = t.child_key_of(&state.key, state.ns);
        let next = match t.nodes[state.node as usize].children.get(&ck) {
            Some(&c) => c,
            None => {
                state.phase = InsertPhase::Commit;
                return;
            }
        };
        t.touch_node(next);
        let ckey = t.nodes[next as usize].key.clone();
        let prefix_len = t.match_len(&ckey, &state.key);
        let mut node = next;
        if prefix_len < t.key_len(&ckey) {
            let (front, action) = t.split_node(next, prefix_len);
            if let Some(a) = action {
                state.pending.push(a);
            }
            node = front;
        }
        {
            let n = &mut t.nodes[node as usize];
            n.priority = n.priority.max(state.priority);
        }

        if t.is_evicted(node) {
            let fresh = state.value[..prefix_len as usize].to_vec();
            t.unevict_node_on_insert(node, fresh);
            if t.has_swa {
                swa_recover_after_unevict(
                    t,
                    node,
                    prefix_len,
                    state.total_prefix_length,
                    &state.params,
                    &mut state.pending,
                );
            }
            // Mamba recover_after_unevict: no-op.
        } else {
            let value_slice: &[i64] = &state.value[..prefix_len as usize];
            let mut consumed_from = prefix_len;
            // FULL: default (no override) -> prefix_len.
            if t.has_swa {
                let c = swa_insert_overlap(
                    t,
                    node,
                    prefix_len,
                    state.total_prefix_length,
                    value_slice,
                    &state.params,
                    &mut state.pending,
                );
                consumed_from = consumed_from.min(c);
            }
            // Mamba: default -> prefix_len.
            let dup_start = 0i64.max(state.params.prev_prefix_len - state.total_prefix_length);
            if dup_start < consumed_from {
                let chunk = value_slice[dup_start as usize..consumed_from as usize].to_vec();
                state.pending.push(UCacheAction::FreeDeviceKV {
                    chunks: vec![chunk],
                });
            }
        }

        if t.inc_hit_count_and_check(node, state.params.chunked) {
            state.pending.push(t.build_backup_kv(node, false));
        }

        state.node = node;
        state.total_prefix_length += prefix_len;
        state.key = state.key[prefix_len as usize..].to_vec();
        state.value = state.value[prefix_len as usize..].to_vec();
    }

    fn commit_step(t: &mut UnifiedRadixTree, state: &mut InsertState) {
        let ns = state.ns;
        let target = if t.key_len(&state.key) > 0 {
            let new_leaf = t.add_new_node(
                ns,
                state.node,
                state.key.clone(),
                state.value.clone(),
                state.priority,
            );
            state.is_new_leaf = true;
            new_leaf
        } else {
            state.node
        };
        state.target_node = target;
        state.result = Some(UIntsertResult {
            prefix_len: state.total_prefix_length,
            total_len: 0,
            last_device_node: target,
            mamba_exist: false,
            inserted_host_node: None,
            host_insert_dropped: false,
            created_nodes: if state.is_new_leaf { vec![target] } else { Vec::new() },
            ..Default::default()
        });

        // Component commit hooks in facade order (FULL is a no-op).
        if t.has_swa {
            swa_commit_insert(
                t,
                target,
                state.is_new_leaf,
                state.total_prefix_length,
                &state.params,
                &mut state.pending,
            );
        }
        if t.has_mamba {
            mamba_commit_insert(
                t,
                target,
                state.is_new_leaf,
                &state.params,
                state.result.as_mut().unwrap(),
                &mut state.pending,
            );
        }
        state.phase = InsertPhase::Tail;
    }

    fn tail_step(t: &mut UnifiedRadixTree, state: &mut InsertState) {
        let target = state.target_node;
        if target != t.root && t.has_swa {
            let window = t.cfg.sliding_window_size + i64::from(t.cfg.page_size);
            t.lru_reset_window_ancestors_mru(CT_SWA, target, window);
            // Mamba INSERT_END: no-op.
        }
        // `_should_backup_after_insert`: new leaf -> hit-count check (mutates);
        // existing target -> incremental aux backup check.
        let backup = if state.is_new_leaf {
            t.inc_hit_count_and_check(target, state.params.chunked)
        } else {
            t.should_backup_existing(target)
        };
        if backup {
            state.pending.push(t.build_backup_kv(target, false));
        }
        // `total_len` stays 0 for device inserts (only insert_host sets it).
    }
}

// SWA tree-level hooks ------------------------------------------------------

/// Python `SWAComponent.update_component_on_insert_overlap`.
fn swa_insert_overlap(
    t: &mut UnifiedRadixTree,
    node: u32,
    prefix_len: i64,
    total_prefix_len: i64,
    value_slice: &[i64],
    params: &UInsertParams,
    pending: &mut Vec<UCacheAction>,
) -> i64 {
    if params.prev_prefix_len >= total_prefix_len + prefix_len {
        return prefix_len;
    }
    if t.nodes[node as usize].value[CT_SWA as usize].is_some() {
        return prefix_len; // not a tombstone
    }
    let swa_evicted_seqlen = params.swa_evicted_seqlen;
    debug_assert_eq!(
        t.nodes[node as usize].lock_ref[CT_SWA as usize],
        0,
        "tombstone SWA lock_ref should be 0"
    );
    let page = i64::from(t.cfg.page_size);
    debug_assert_eq!(swa_evicted_seqlen % page, 0, "swa_evicted_seqlen must be page-aligned");

    if swa_evicted_seqlen <= total_prefix_len {
        // Branch 1: entire value_slice within the SWA window.
        let full_cd = &t.nodes[node as usize];
        if full_cd.lock_ref[CT_FULL as usize] > 0 {
            let old_full = full_cd.value[CT_BASE as usize].clone().unwrap_or_default();
            pending.push(UCacheAction::RecoverSWAWithLockedFull {
                node,
                kept_full: old_full,
                incoming_full: value_slice.to_vec(),
            });
            return 0;
        }
        let old_full = t.nodes[node as usize].value[CT_BASE as usize].take().unwrap_or_default();
        t.nodes[node as usize].value[CT_BASE as usize] = Some(value_slice.to_vec());
        pending.push(UCacheAction::FreeDeviceKVFullOnly {
            chunks: vec![old_full],
        });
        pending.push(UCacheAction::SWARebuild {
            node,
            source_value: value_slice.to_vec(),
        });
        0
    } else if swa_evicted_seqlen < total_prefix_len + prefix_len {
        // Branch 2: partial recover; split at the boundary.
        let start_idx = swa_evicted_seqlen - total_prefix_len;
        let is_locked = t.nodes[node as usize].lock_ref[CT_FULL as usize] > 0;
        let old_full = t.nodes[node as usize]
            .value[CT_BASE as usize]
            .as_ref()
            .unwrap()
            .iter()
            .skip(start_idx as usize)
            .copied()
            .collect::<Vec<i64>>();
        let (_, action) = t.split_node(node, start_idx);
        if let Some(a) = action {
            pending.push(a);
        }
        let new_full = value_slice[start_idx as usize..].to_vec();
        if is_locked {
            pending.push(UCacheAction::RecoverSWAWithLockedFull {
                node,
                kept_full: old_full,
                incoming_full: new_full,
            });
            return start_idx;
        }
        t.nodes[node as usize].value[CT_BASE as usize] = Some(new_full.clone());
        pending.push(UCacheAction::FreeDeviceKVFullOnly {
            chunks: vec![old_full],
        });
        pending.push(UCacheAction::SWARebuild {
            node,
            source_value: new_full,
        });
        start_idx
    } else {
        // Branch 3: entirely outside the window.
        prefix_len
    }
}

/// Python `SWAComponent.recover_after_unevict`.
fn swa_recover_after_unevict(
    t: &mut UnifiedRadixTree,
    node: u32,
    prefix_len: i64,
    total_prefix_len: i64,
    params: &UInsertParams,
    pending: &mut Vec<UCacheAction>,
) {
    if t.nodes[node as usize].value[CT_SWA as usize].is_some() {
        return;
    }
    debug_assert_eq!(
        t.nodes[node as usize].lock_ref[CT_SWA as usize],
        0,
        "tombstone SWA lock_ref should be 0 on unevict"
    );
    let swa_evicted_seqlen = params.swa_evicted_seqlen;
    let page = i64::from(t.cfg.page_size);
    debug_assert_eq!(swa_evicted_seqlen % page, 0);

    if swa_evicted_seqlen <= total_prefix_len {
        // entire node within the window
    } else if swa_evicted_seqlen < total_prefix_len + prefix_len {
        let start_idx = swa_evicted_seqlen - total_prefix_len;
        let (_, action) = t.split_node(node, start_idx);
        if let Some(a) = action {
            pending.push(a);
        }
    } else {
        return;
    }
    let src = t.nodes[node as usize]
        .value[CT_BASE as usize]
        .clone()
        .unwrap_or_default();
    pending.push(UCacheAction::SWARebuild { node, source_value: src });
}

/// Python `SWAComponent.commit_insert_component_data`.
fn swa_commit_insert(
    t: &mut UnifiedRadixTree,
    node: u32,
    is_new_leaf: bool,
    result_prefix_len: i64,
    params: &UInsertParams,
    pending: &mut Vec<UCacheAction>,
) {
    if !is_new_leaf {
        return;
    }
    let node_start = result_prefix_len;
    let split_pos = params.swa_evicted_seqlen - node_start;
    if split_pos >= t.key_len(&t.nodes[node as usize].key) {
        // Entire leaf outside the window: leave the tombstone.
        return;
    }
    if split_pos > 0 {
        let (_, action) = t.split_node(node, split_pos);
        debug_assert!(action.is_none(), "new leaf cannot be write-through-pending");
    }
    // Cap the in-window leaf at one page-aligned window (lock granularity).
    if let Some(capped_parent) = t.maybe_split_leaf_for_swa_lock(node) {
        let src = t.nodes[capped_parent as usize]
            .value[CT_BASE as usize]
            .clone()
            .unwrap_or_default();
        pending.push(UCacheAction::SWARebuild {
            node: capped_parent,
            source_value: src,
        });
    }
    let src = t.nodes[node as usize].value[CT_BASE as usize].clone().unwrap_or_default();
    pending.push(UCacheAction::SWARebuild {
        node,
        source_value: src,
    });
}

impl UnifiedRadixTree {
    /// Python `SWAComponent._maybe_split_leaf_for_swa_lock`.
    pub(crate) fn maybe_split_leaf_for_swa_lock(&mut self, leaf: u32) -> Option<u32> {
        if leaf == self.root || self.nodes[leaf as usize].lock_ref[CT_SWA as usize] > 0 {
            return None;
        }
        let tail_size = self.cfg.swa_tail_size();
        let leaf_len = self.key_len(&self.nodes[leaf as usize].key);
        if leaf_len <= tail_size {
            return None;
        }
        let split_at = leaf_len - tail_size;
        let page = i64::from(self.cfg.page_size);
        if page > 1 && (split_at % page != 0 || leaf_len % page != 0) {
            return None;
        }
        let (new_parent, action) = self.split_node(leaf, split_at);
        debug_assert!(action.is_none(), "fresh SWA leaf cannot be write-through-pending");
        Some(new_parent)
    }
}

// Mamba tree-level hooks ----------------------------------------------------

/// Python `MambaComponent.commit_insert_component_data`.
fn mamba_commit_insert(
    t: &mut UnifiedRadixTree,
    node: u32,
    is_new_leaf: bool,
    params: &UInsertParams,
    result: &mut UIntsertResult,
    pending: &mut Vec<UCacheAction>,
) {
    let mamba_value = params
        .mamba_value
        .clone()
        .expect("mamba_value is required when Mamba is present");
    let dev_slot = UnifiedRadixTree::lru_slot_public(CT_MAMBA, 0);
    let host_slot = UnifiedRadixTree::lru_slot_public(CT_MAMBA, 1);
    if is_new_leaf {
        t.nodes[node as usize].value[CT_MAMBA as usize] = Some(mamba_value.clone());
        t.lru_insert_mru(dev_slot, node);
        t.evictable_size[CT_MAMBA as usize] += mamba_value.len() as i64;
        if t.cfg.mamba_max_states_per_path >= 0 {
            pending.push(UCacheAction::MambaEvictExcess { tail_node: node });
        }
        return;
    }
    if t.nodes[node as usize].value[CT_MAMBA as usize].is_none() {
        t.nodes[node as usize].value[CT_MAMBA as usize] = Some(mamba_value.clone());
        if t.lru_in(host_slot, node) {
            t.lru_remove(host_slot, node);
        }
        t.lru_insert_mru(dev_slot, node);
        t.evictable_size[CT_MAMBA as usize] += mamba_value.len() as i64;
        t.nodes[node as usize].last_access = t.tick();
        if t.cfg.mamba_max_states_per_path >= 0 {
            pending.push(UCacheAction::MambaEvictExcess { tail_node: node });
        }
        return;
    }
    t.lru_reset_mru(dev_slot, node);
    t.nodes[node as usize].last_access = t.tick();
    result.mamba_exist = true;
}
