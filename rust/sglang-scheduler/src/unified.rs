//! PyO3 bindings for the unified multi-pool radix tree core
//! (`sglang_radix::unified::UnifiedRadixTree`), exposed as
//! `UnifiedRadixTree` in the `_scheduler` module.
//!
//! The tree is pure CPU and torch-free: component "values" are plain `i64`
//! pool-index lists, and the Python wrapper
//! (`python/sglang/srt/mem_cache/unified_cache/tree_core_rust.py`) applies
//! the emitted actions against the real allocators.
//!
//! Tuple shapes (field order is part of the module ABI — keep in sync with
//! `tree_core_rust.py`):
//!
//! - `match_prefix` / `empty_match_result` →
//!   `(device_indices, last_device_node, last_host_node, best_match_node,
//!   host_hit_length, swa_host_hit_length, mamba_host_hit_length,
//!   mamba_branching_seqlen, full_kv_hit_length, actions)`
//! - insert results →
//!   `(prefix_len, total_len, last_device_node, mamba_exist,
//!   inserted_host_node, host_insert_dropped, created_nodes, cache_actions)`
//! - `begin_insert` / `resume_insert` → `(actions, result_or_None)`
//! - evict step (`evict_device_next_node`) →
//!   `(node_id, made_progress, tracker, device_frees, host_frees)`
//! - evict outcome (`evict_device_leaf` / `demote` / `drop_subtree_no_host`
//!   / `drive_host_eviction` / `evict_excess_path_states` /
//!   `dec_swa_lock_only`) →
//!   `(tracker, device_frees, host_frees, is_dropped, backup_kv)`
//!   with `tracker`/`*_frees` as lists of `(component_type, items)`.
//! - lock results (`inc_lock_ref` / `inc_host_lock_ref`) →
//!   `(delta, swa_uuid_for_lock, swa_uuid_for_host_lock, skip_lock_node_ids)`
//!   with `skip_lock_node_ids` a list of `(component_type, node_ids)`.
//! - transfers →
//!   `(pool, device_indices, host_indices, keys, nodes_to_load, hit_policy)`,
//!   `None` for absent fields; `comp_xfers` = list of
//!   `(component_type, [transfer, ...])`.
//! - `build_storage_backup_spec` →
//!   `(host_value, token_ids, hash_value, prefix_keys, comp_xfers)`
//! - `build_load_back_spec` → `(kv_transfer, comp_xfers)` or `None`
//!   (conflict: a would-be loaded node is pinned by another anchor).
//! - node dumps (`dump_nodes`) →
//!   `(id, key, last_access, creation, hit_count, priority, full_value,
//!   full_host_value, swa_value, swa_host_value, mamba_value,
//!   mamba_host_value, lock_refs, host_lock_refs, swa_uuid, swa_host_uuid,
//!   write_through_pending, load_back_pending, in_device_leaves,
//!   in_host_leaves, is_duplicate_tracked)`
//! - actions:
//!   `("ReplaceWT", ack_id, old_node, new_node, new_child_node)`,
//!   `("FreeDeviceKV", chunks)`, `("FreeDeviceKVFullOnly", chunks)`,
//!   `("BackupKV", node_ids)`,
//!   `("FreeComponentDeviceSlot", ct, chunks)`,
//!   `("FreeComponentHostSlot", ct, chunks)`,
//!   `("MambaEvictExcess", tail_node)`,
//!   `("RebuildFullToSWAMapping", full_indices, swa_indices)`,
//!   `("RecoverSWAWithLockedFull", node, kept_full, incoming_full)`,
//!   `("SWARebuild", node, source_value)`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use sglang_radix::{
    EvictionPolicy, UCacheAction, UConfig, UDecLockParams, UInsertParams, UIncLockResult,
    ULoadBackSpec, UMatchResult, UNodeDump, UStepResult, UStorageBackupSpec, UTransfer,
    UWalkResult, UnifiedRadixTree as UTree, UIntsertResult, UEvictOutcome, UEvictStep,
};

// ---------------------------------------------------------------- converters

fn action_to_py(py: Python, a: &UCacheAction) -> PyResult<Py<PyAny>> {
    match a {
        UCacheAction::ReplaceWT {
            ack_id,
            old_node,
            new_node,
            new_child_node,
        } => PyTuple::new(
            py,
            [
                "ReplaceWT".into_py_any(py)?,
                ack_id.into_py_any(py)?,
                old_node.into_py_any(py)?,
                new_node.into_py_any(py)?,
                new_child_node.into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
        UCacheAction::FreeDeviceKV { chunks } => PyTuple::new(
            py,
            ["FreeDeviceKV".into_py_any(py)?, chunks.clone().into_py_any(py)?],
        )?
        .into_py_any(py),
        UCacheAction::FreeDeviceKVFullOnly { chunks } => PyTuple::new(
            py,
            [
                "FreeDeviceKVFullOnly".into_py_any(py)?,
                chunks.clone().into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
        UCacheAction::BackupKV { node_ids } => PyTuple::new(
            py,
            ["BackupKV".into_py_any(py)?, node_ids.clone().into_py_any(py)?],
        )?
        .into_py_any(py),
        UCacheAction::FreeComponentDeviceSlot { ct, chunks } => PyTuple::new(
            py,
            [
                "FreeComponentDeviceSlot".into_py_any(py)?,
                u64::from(*ct).into_py_any(py)?,
                chunks.clone().into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
        UCacheAction::FreeComponentHostSlot { ct, chunks } => PyTuple::new(
            py,
            [
                "FreeComponentHostSlot".into_py_any(py)?,
                u64::from(*ct).into_py_any(py)?,
                chunks.clone().into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
        UCacheAction::MambaEvictExcess { tail_node } => PyTuple::new(
            py,
            [
                "MambaEvictExcess".into_py_any(py)?,
                tail_node.into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
        UCacheAction::RebuildFullToSWAMapping {
            full_indices,
            swa_indices,
        } => PyTuple::new(
            py,
            [
                "RebuildFullToSWAMapping".into_py_any(py)?,
                full_indices.clone().into_py_any(py)?,
                swa_indices.clone().into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
        UCacheAction::RecoverSWAWithLockedFull {
            node,
            kept_full,
            incoming_full,
        } => PyTuple::new(
            py,
            [
                "RecoverSWAWithLockedFull".into_py_any(py)?,
                node.into_py_any(py)?,
                kept_full.clone().into_py_any(py)?,
                incoming_full.clone().into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
        UCacheAction::SWARebuild { node, source_value } => PyTuple::new(
            py,
            [
                "SWARebuild".into_py_any(py)?,
                node.into_py_any(py)?,
                source_value.clone().into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
    }
}

fn actions_to_py(py: Python, actions: &[UCacheAction]) -> PyResult<Py<PyAny>> {
    let mut out = Vec::with_capacity(actions.len());
    for a in actions {
        out.push(action_to_py(py, a)?);
    }
    PyList::new(py, out)?.into_py_any(py)
}

fn transfer_to_py(py: Python, x: &UTransfer) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            u64::from(x.pool).into_py_any(py)?,
            x.device_indices.clone().into_py_any(py)?,
            x.host_indices.clone().into_py_any(py)?,
            x.keys.clone().into_py_any(py)?,
            x.nodes_to_load.clone().into_py_any(py)?,
            u64::from(x.hit_policy).into_py_any(py)?,
        ],
    )?
    .into_py_any(py)
}

fn xfers_to_py(py: Python, xfers: &[(u8, Vec<UTransfer>)]) -> PyResult<Py<PyAny>> {
    let mut out = Vec::with_capacity(xfers.len());
    for (ct, xfs) in xfers {
        let list: Vec<Py<PyAny>> = xfs
            .iter()
            .map(|x| transfer_to_py(py, x))
            .collect::<PyResult<_>>()?;
        out.push(
            PyTuple::new(
                py,
                [u64::from(*ct).into_py_any(py)?, list.into_py_any(py)?],
            )?
            .into_py_any(py)?,
        );
    }
    PyList::new(py, out)?.into_py_any(py)
}

fn inc_lock_to_py(py: Python, r: &UIncLockResult) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            r.delta.into_py_any(py)?,
            r.swa_uuid_for_lock.into_py_any(py)?,
            r.swa_uuid_for_host_lock.into_py_any(py)?,
            r.skip_lock_node_ids.clone().into_py_any(py)?,
        ],
    )?
    .into_py_any(py)
}

fn step_to_py(py: Python, s: &UStepResult) -> PyResult<Py<PyAny>> {
    let result = match s.result.as_ref() {
        Some(r) => insert_result_to_py(py, r)?,
        None => py.None(),
    };
    PyTuple::new(py, [actions_to_py(py, &s.actions)?, result])?.into_py_any(py)
}

/// Wire form of one `PoolTransfer` crossing the Python boundary.
type TransferPyTuple = (
    u8,
    Option<Vec<i64>>,
    Option<Vec<i64>>,
    Option<Vec<String>>,
    Vec<u32>,
    u8,
);

/// Parse `(ct, [transfer, ...])` lists back into tree form. A transfer is the
/// 6-tuple `(pool, device_indices, host_indices, keys, nodes_to_load,
/// hit_policy)` with `None` for absent fields.
fn xfers_from_py(xfers: &Bound<'_, PyAny>) -> PyResult<Vec<(u8, Vec<UTransfer>)>> {
    let mut out = Vec::new();
    for entry in xfers.try_iter()? {
        let entry = entry?;
        let ct: u8 = entry.get_item(0)?.extract()?;
        let mut list = Vec::new();
        for t in entry.get_item(1)?.try_iter()? {
            let t = t?;
            let (
                pool,
                device_indices,
                host_indices,
                keys,
                nodes_to_load,
                hit_policy,
            ): TransferPyTuple = t.extract()?;
            list.push(UTransfer {
                pool,
                device_indices,
                host_indices,
                keys,
                nodes_to_load,
                hit_policy,
            });
        }
        out.push((ct, list));
    }
    Ok(out)
}

fn insert_result_to_py(py: Python, r: &UIntsertResult) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            r.prefix_len.into_py_any(py)?,
            r.total_len.into_py_any(py)?,
            r.last_device_node.into_py_any(py)?,
            r.mamba_exist.into_py_any(py)?,
            r.inserted_host_node.into_py_any(py)?,
            r.host_insert_dropped.into_py_any(py)?,
            r.created_nodes.clone().into_py_any(py)?,
            actions_to_py(py, &r.cache_actions)?,
        ],
    )?
    .into_py_any(py)
}

/// Rebuild the tree-side insert result from the 8-tuple Python hands back
/// (the caller-side `cache_actions` are dropped; the tree does not need
/// them on the commit path).
fn insert_result_from_tuple(t: &Bound<'_, PyAny>) -> PyResult<UIntsertResult> {
    let (
        prefix_len,
        total_len,
        last_device_node,
        mamba_exist,
        inserted_host_node,
        host_insert_dropped,
        created_nodes,
    ): (i64, i64, u32, bool, Option<u32>, bool, Vec<u32>) = t.extract()?;
    Ok(UIntsertResult {
        prefix_len,
        total_len,
        last_device_node,
        mamba_exist,
        inserted_host_node,
        host_insert_dropped,
        created_nodes,
        cache_actions: Vec::new(),
    })
}

fn match_result_to_py(py: Python, m: &UMatchResult) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            m.device_indices.clone().into_py_any(py)?,
            m.last_device_node.into_py_any(py)?,
            m.last_host_node.into_py_any(py)?,
            m.best_match_node.into_py_any(py)?,
            m.host_hit_length.into_py_any(py)?,
            m.swa_host_hit_length.into_py_any(py)?,
            m.mamba_host_hit_length.into_py_any(py)?,
            m.mamba_branching_seqlen.into_py_any(py)?,
            m.full_kv_hit_length.into_py_any(py)?,
            actions_to_py(py, &m.actions)?,
        ],
    )?
    .into_py_any(py)
}

fn evict_step_to_py(py: Python, s: &UEvictStep) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            s.node_id.into_py_any(py)?,
            s.made_progress.into_py_any(py)?,
            s.tracker.clone().into_py_any(py)?,
            s.device_frees.clone().into_py_any(py)?,
            s.host_frees.clone().into_py_any(py)?,
        ],
    )?
    .into_py_any(py)
}

fn outcome_to_py(py: Python, o: &UEvictOutcome) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            o.tracker.clone().into_py_any(py)?,
            o.device_frees.clone().into_py_any(py)?,
            o.host_frees.clone().into_py_any(py)?,
            o.is_dropped.into_py_any(py)?,
            o.backup_kv.clone().into_py_any(py)?,
        ],
    )?
    .into_py_any(py)
}

fn walk_result_to_py(py: Python, w: &UWalkResult) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            w.slot_indices.clone().into_py_any(py)?,
            w.positions.clone().into_py_any(py)?,
            w.prev_slot_indices.clone().into_py_any(py)?,
        ],
    )?
    .into_py_any(py)
}

fn storage_spec_to_py(py: Python, s: &UStorageBackupSpec) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            s.host_value.clone().into_py_any(py)?,
            s.token_ids.clone().into_py_any(py)?,
            s.hash_value.clone().into_py_any(py)?,
            s.prefix_keys.clone().into_py_any(py)?,
            xfers_to_py(py, &s.comp_xfers)?,
        ],
    )?
    .into_py_any(py)
}

fn load_back_spec_to_py(py: Python, s: &ULoadBackSpec) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [transfer_to_py(py, &s.kv)?, xfers_to_py(py, &s.comp_xfers)?],
    )?
    .into_py_any(py)
}

fn node_dump_to_py(py: Python, d: &UNodeDump) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            d.id.into_py_any(py)?,
            d.key.clone().into_py_any(py)?,
            d.last_access.into_py_any(py)?,
            d.creation.into_py_any(py)?,
            d.hit_count.into_py_any(py)?,
            d.priority.into_py_any(py)?,
            d.full_value.clone().into_py_any(py)?,
            d.full_host_value.clone().into_py_any(py)?,
            d.swa_value.clone().into_py_any(py)?,
            d.swa_host_value.clone().into_py_any(py)?,
            d.mamba_value.clone().into_py_any(py)?,
            d.mamba_host_value.clone().into_py_any(py)?,
            d.lock_refs.to_vec().into_py_any(py)?,
            d.host_lock_refs.to_vec().into_py_any(py)?,
            d.swa_uuid.into_py_any(py)?,
            d.swa_host_uuid.into_py_any(py)?,
            d.write_through_pending.into_py_any(py)?,
            d.load_back_pending.into_py_any(py)?,
            d.in_device_leaves.into_py_any(py)?,
            d.in_host_leaves.into_py_any(py)?,
            d.is_duplicate_tracked.into_py_any(py)?,
        ],
    )?
    .into_py_any(py)
}

// ------------------------------------------------------------------ class

#[pyclass(name = "UnifiedRadixTree")]
pub struct UnifiedRadixTreePy {
    tree: UTree,
}

#[pymethods]
impl UnifiedRadixTreePy {
    #[new]
    #[pyo3(signature = (
        page_size,
        is_eagle,
        sliding_window_size,
        mamba_checkpoint_grid,
        mamba_max_states_per_path,
        eviction_policy,
        write_through_threshold,
        is_write_back,
        has_swa_host_pool,
        enable_session_radix_cache
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        page_size: u32,
        is_eagle: bool,
        sliding_window_size: i64,
        mamba_checkpoint_grid: i64,
        mamba_max_states_per_path: i64,
        eviction_policy: &str,
        write_through_threshold: i64,
        is_write_back: bool,
        has_swa_host_pool: bool,
        enable_session_radix_cache: bool,
    ) -> PyResult<Self> {
        let policy = EvictionPolicy::parse(eviction_policy).map_err(PyValueError::new_err)?;
        if enable_session_radix_cache {
            return Err(PyValueError::new_err(
                "the Rust unified tree core does not support --enable-session-radix-cache",
            ));
        }
        let cfg = UConfig {
            page_size,
            is_eagle,
            sliding_window_size,
            mamba_checkpoint_grid,
            mamba_max_states_per_path,
            eviction_policy: policy,
            write_through_threshold,
            is_write_back,
            has_swa_host_pool,
            enable_session_radix_cache,
        };
        if is_eagle && mamba_checkpoint_grid > 0 {
            return Err(PyValueError::new_err(
                "the Rust unified tree core does not support EAGLE bigram keys with Mamba",
            ));
        }
        Ok(Self {
            tree: UTree::new(cfg),
        })
    }

    // ------------------------------------------------ config / node info

    fn root_id(&self) -> u32 {
        self.tree.root_id()
    }

    fn active_cts(&self) -> Vec<u8> {
        self.tree.active_cts()
    }

    fn has_swa(&self) -> bool {
        self.tree.cfg.has_swa()
    }

    fn has_mamba(&self) -> bool {
        self.tree.cfg.has_mamba()
    }

    fn node_ns(&self, node: u32) -> u32 {
        self.tree.node_by_id(node).ns
    }

    fn node_key(&self, node: u32) -> Vec<i64> {
        self.tree.node_key(node)
    }

    fn node_parent(&self, node: u32) -> Option<u32> {
        self.tree.node_parent(node)
    }

    fn write_through_threshold(&self) -> i64 {
        self.tree.write_through_threshold()
    }

    fn is_write_back(&self) -> bool {
        self.tree.is_write_back()
    }

    fn reset(&mut self) {
        self.tree.reset()
    }

    // ------------------------------------------------------------- match

    fn match_prefix(&mut self, py: Python, ns: u32, raw_tokens: Vec<i64>) -> PyResult<Py<PyAny>> {
        let r = self.tree.match_prefix(ns, &raw_tokens);
        match_result_to_py(py, &r)
    }

    fn empty_match_result(&self, py: Python) -> PyResult<Py<PyAny>> {
        let r = self.tree.empty_match_result();
        match_result_to_py(py, &r)
    }

    fn is_full_device_evicted(&self, node: u32) -> bool {
        self.tree.is_full_device_evicted(node)
    }

    fn collect_full_device_indices(&self, from_node: u32, until_node: u32) -> Vec<i64> {
        self.tree.collect_full_device_indices(from_node, until_node)
    }

    // -------------------------------------------------------------- sizes

    fn total_size(&self) -> (i64, i64) {
        self.tree.total_size()
    }

    fn evictable_size(&self) -> i64 {
        self.tree.evictable_size()
    }

    fn protected_size(&self) -> i64 {
        self.tree.protected_size()
    }

    fn component_evictable_size(&self, ct: u8) -> i64 {
        self.tree.component_evictable_size(ct)
    }

    fn component_protected_size(&self, ct: u8) -> i64 {
        self.tree.component_protected_size(ct)
    }

    // -------------------------------------------------------- values / walk

    fn all_values_flatten(&self) -> Vec<i64> {
        self.tree.all_values_flatten()
    }

    fn all_mamba_values_flatten(&self) -> Vec<i64> {
        self.tree.all_mamba_values_flatten()
    }

    fn walk_for_kv_canary(
        &self,
        py: Python,
        unlocked_only: bool,
        swa_resident_only: bool,
    ) -> PyResult<Py<PyAny>> {
        let r = self.tree.walk_for_kv_canary(unlocked_only, swa_resident_only);
        walk_result_to_py(py, &r)
    }

    // ------------------------------------------------------------- hashes

    fn get_hash_values_opt(&self, node: u32) -> Option<Vec<String>> {
        self.tree.get_hash_values_opt(node)
    }

    fn set_hash_values(&mut self, node: u32, values: Vec<String>) {
        self.tree.set_hash_values(node, values);
    }

    fn get_last_hash_value(&self, node: u32) -> Option<String> {
        self.tree.get_last_hash_value(node)
    }

    fn get_prefix_hash_values(&self, node: u32) -> Vec<String> {
        self.tree.get_prefix_hash_values(node)
    }

    fn get_event_hash_values_opt(&self, node: u32) -> Option<Vec<String>> {
        self.tree.get_event_hash_values_opt(node)
    }

    fn set_event_hash_values(&mut self, node: u32, values: Vec<String>) {
        self.tree.set_event_hash_values(node, values);
    }

    // --------------------------------------------------------- kv events

    fn take_kv_events(&mut self) -> Vec<(u8, u32, u8)> {
        self.tree
            .take_kv_events()
            .into_iter()
            .map(|e| (e.op, e.node, e.medium))
            .collect()
    }

    // ------------------------------------------------------ stepped insert

    #[pyo3(signature = (
        ns,
        raw_tokens,
        value,
        prev_prefix_len,
        chunked,
        priority,
        swa_evicted_seqlen,
        mamba_value
    ))]
    #[allow(clippy::too_many_arguments)]
    fn begin_insert(
        &mut self,
        py: Python,
        ns: u32,
        raw_tokens: Vec<i64>,
        value: Option<Vec<i64>>,
        prev_prefix_len: i64,
        chunked: bool,
        priority: i64,
        swa_evicted_seqlen: i64,
        mamba_value: Option<Vec<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let params = UInsertParams {
            prev_prefix_len,
            chunked,
            priority,
            swa_evicted_seqlen,
            mamba_value,
        };
        let step = self.tree.begin_insert(ns, &raw_tokens, value, &params);
        step_to_py(py, &step)
    }

    fn resume_insert(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        let step = self.tree.resume_insert();
        step_to_py(py, &step)
    }

    fn has_ongoing_insert(&self) -> bool {
        self.tree.has_ongoing_insert()
    }

    fn end_insert(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        let actions = self.tree.end_insert();
        actions_to_py(py, &actions)
    }

    // -------------------------------------------------------------- locks

    fn inc_lock_ref(
        &mut self,
        py: Python,
        node: u32,
        skip_lock_components: Vec<u8>,
    ) -> PyResult<Py<PyAny>> {
        let r = self.tree.inc_lock_ref(node, &skip_lock_components);
        inc_lock_to_py(py, &r)
    }

    #[pyo3(signature = (
        node,
        swa_uuid_for_lock,
        swa_uuid_for_host_lock,
        skip_lock_node_ids,
        skip_swa
    ))]
    fn dec_lock_ref(
        &mut self,
        node: u32,
        swa_uuid_for_lock: Option<i64>,
        swa_uuid_for_host_lock: Option<i64>,
        skip_lock_node_ids: Vec<(u8, Vec<u32>)>,
        skip_swa: bool,
    ) {
        let params = UDecLockParams {
            swa_uuid_for_lock,
            swa_uuid_for_host_lock,
            skip_lock_node_ids,
        };
        self.tree.dec_lock_ref(node, &params, skip_swa);
    }

    fn dec_swa_lock_only(
        &mut self,
        py: Python,
        node: u32,
        swa_uuid_for_lock: Option<i64>,
        skip_lock_node_ids: Vec<(u8, Vec<u32>)>,
    ) -> PyResult<Py<PyAny>> {
        let r = self.tree.dec_swa_lock_only(node, swa_uuid_for_lock, &skip_lock_node_ids);
        outcome_to_py(py, &r)
    }

    fn inc_host_lock_ref(&mut self, py: Python, node: u32) -> PyResult<Py<PyAny>> {
        let r = self.tree.inc_host_lock_ref(node);
        inc_lock_to_py(py, &r)
    }

    fn dec_host_lock_ref(
        &mut self,
        node: u32,
        swa_uuid_for_lock: Option<i64>,
        swa_uuid_for_host_lock: Option<i64>,
        skip_lock_node_ids: Vec<(u8, Vec<u32>)>,
    ) {
        let params = UDecLockParams {
            swa_uuid_for_lock,
            swa_uuid_for_host_lock,
            skip_lock_node_ids,
        };
        self.tree.dec_host_lock_ref(node, &params);
    }

    // ------------------------------------------------------------ eviction

    fn evict_device_start(&mut self, ct: u8, request_cnt: i64) {
        self.tree.evict_device_start(ct, request_cnt);
    }

    fn evict_device_next_node(
        &mut self,
        py: Python,
        ct: u8,
        running: i64,
    ) -> PyResult<Py<PyAny>> {
        let r = self.tree.evict_device_next_node(ct, running);
        evict_step_to_py(py, &r)
    }

    fn evict_device_leaf(
        &mut self,
        py: Python,
        node: u32,
        is_write_back: bool,
    ) -> PyResult<Py<PyAny>> {
        let r = self.tree.evict_device_leaf(node, is_write_back);
        outcome_to_py(py, &r)
    }

    fn evict_device_end(&mut self, ct: u8) {
        self.tree.evict_device_end(ct);
    }

    fn drop_subtree_no_host(&mut self, py: Python, node: u32) -> PyResult<Py<PyAny>> {
        let r = self.tree.drop_subtree_no_host(node);
        outcome_to_py(py, &r)
    }

    fn demote(&mut self, py: Python, node: u32) -> PyResult<Py<PyAny>> {
        let r = self.tree.demote_node(node);
        outcome_to_py(py, &r)
    }

    fn drive_host_eviction(&mut self, py: Python, ct: u8, num_tokens: i64) -> PyResult<Py<PyAny>> {
        let r = self.tree.drive_host_eviction(ct, num_tokens);
        outcome_to_py(py, &r)
    }

    fn evict_excess_path_states(&mut self, py: Python, tail: u32) -> PyResult<Py<PyAny>> {
        let r = self.tree.evict_excess_path_states(tail);
        outcome_to_py(py, &r)
    }

    // ------------------------------------------------------------- hicache

    fn set_hicache_enabled(&mut self) {
        self.tree.set_hicache_enabled()
    }

    fn set_storage_enabled(&mut self) {
        self.tree.set_storage_enabled()
    }

    fn set_write_back(&mut self) {
        self.tree.set_write_back()
    }

    fn set_swa_host_pool(&mut self, has: bool) {
        self.tree.set_swa_host_pool(has)
    }

    fn set_write_through_threshold(&mut self, v: i64) {
        self.tree.set_write_through_threshold(v)
    }

    fn insert_host(
        &mut self,
        py: Python,
        anchor: u32,
        ns: u32,
        raw_tokens: Vec<i64>,
        host_value: Vec<i64>,
        hash_value: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let r = self.tree.insert_host(anchor, ns, &raw_tokens, host_value, hash_value);
        insert_result_to_py(py, &r)
    }

    fn build_backup_spec(
        &self,
        py: Python,
        node: u32,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (kv, xfers) = self.tree.build_backup_spec(node);
        Ok((kv.into_py_any(py)?, xfers_to_py(py, &xfers)?))
    }

    fn build_storage_backup_spec(
        &self,
        py: Python,
        node: u32,
        pass_prefix_keys: bool,
    ) -> PyResult<Py<PyAny>> {
        match self.tree.build_storage_backup_spec(node, pass_prefix_keys) {
            Some(spec) => storage_spec_to_py(py, &spec),
            None => Ok(py.None()),
        }
    }

    fn build_comp_transfer(
        &self,
        py: Python,
        ct: u8,
        node: u32,
        phase: u8,
        host_indices: Option<Vec<i64>>,
        mamba_pool_idx: Option<Vec<i64>>,
    ) -> PyResult<Py<PyAny>> {
        match self.tree.build_comp_transfer(
            ct,
            node,
            phase,
            host_indices.as_deref(),
            mamba_pool_idx.as_deref(),
        ) {
            Some(xfer) => transfer_to_py(py, &xfer),
            None => Ok(py.None()),
        }
    }

    fn build_load_back_spec(
        &self,
        py: Python,
        node: u32,
        mamba_pool_idx: Option<Vec<i64>>,
    ) -> PyResult<Py<PyAny>> {
        match self.tree.build_load_back_spec(node, mamba_pool_idx.as_deref()) {
            Some(spec) => load_back_spec_to_py(py, &spec),
            None => Ok(py.None()),
        }
    }

    fn commit_backup(
        &mut self,
        node: u32,
        host_indices: Vec<i64>,
        comp_xfers: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let xfers = xfers_from_py(comp_xfers)?;
        self.tree.commit_backup(node, &host_indices, &xfers);
        Ok(())
    }

    fn commit_load_back(
        &mut self,
        py: Python,
        node: u32,
        device_indices: Option<Vec<i64>>,
        kv_nodes_to_load: Vec<u32>,
        comp_xfers: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let xfers = xfers_from_py(comp_xfers)?;
        let actions = self
            .tree
            .commit_load_back(node, device_indices.as_deref(), &kv_nodes_to_load, &xfers);
        actions_to_py(py, &actions)
    }

    fn finish_load_back(&mut self, anchor: u32) {
        self.tree.finish_load_back(anchor)
    }

    fn mark_write_through_pending(&mut self, node: u32) {
        self.tree.mark_write_through_pending(node)
    }

    fn finish_write_through(&mut self, node_ids: Vec<u32>, ack_id: i64) {
        self.tree.finish_write_through(&node_ids, ack_id)
    }

    fn prefetch_anchor_ns(&self, node: u32) -> u32 {
        self.tree.prefetch_anchor_ns(node)
    }

    fn commit_swa_prefetch(
        &mut self,
        py: Python,
        anchor: u32,
        host_indices: Vec<i64>,
        loaded_pages: i64,
        insert_result: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let ir = insert_result_from_tuple(insert_result)?;
        let actions = self.tree.commit_swa_prefetch(anchor, host_indices, loaded_pages, &ir);
        actions_to_py(py, &actions)
    }

    fn commit_mamba_prefetch(
        &mut self,
        py: Python,
        host_indices: Option<Vec<i64>>,
        loaded: bool,
        insert_result: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let mut ir = insert_result_from_tuple(insert_result)?;
        let actions = self.tree.commit_mamba_prefetch(host_indices, loaded, &mut ir);
        PyTuple::new(
            py,
            [
                actions_to_py(py, &actions)?,
                ir.mamba_exist.into_py_any(py)?,
                ir.inserted_host_node.into_py_any(py)?,
            ],
        )?
        .into_py_any(py)
    }

    // -------------------------------------------------------- component values

    fn set_component_device_value(&mut self, node: u32, ct: u8, value: Vec<i64>) {
        self.tree.set_component_device_value(node, ct, value);
    }

    fn get_component_device_value(&self, node: u32, ct: u8) -> Option<Vec<i64>> {
        self.tree.get_component_device_value(node, ct)
    }

    fn component_has_host_value_only(&self, node: u32, ct: u8) -> bool {
        self.tree.component_has_host_value_only(node, ct)
    }

    // ------------------------------------------------------------- state

    fn is_backuped(&self, node: u32) -> bool {
        self.tree.is_backuped(node)
    }

    fn is_root(&self, node: u32) -> bool {
        self.tree.is_root(node)
    }

    fn is_evicted(&self, node: u32) -> bool {
        self.tree.is_evicted(node)
    }

    fn is_device_leaf(&self, node: u32) -> bool {
        self.tree.is_device_leaf(node)
    }

    fn is_host_leaf(&self, node: u32) -> bool {
        self.tree.is_host_leaf(node)
    }

    fn sanity_check(
        &self,
        ongoing_write_through: Vec<(i64, u32)>,
        ongoing_load_back: Vec<(i64, u32)>,
    ) -> Vec<String> {
        self.tree.sanity_check(&ongoing_write_through, &ongoing_load_back)
    }

    fn dump_nodes(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dumps = self.tree.dump_nodes();
        let mut out = Vec::with_capacity(dumps.len());
        for d in &dumps {
            out.push(node_dump_to_py(py, d)?);
        }
        PyList::new(py, out)?.into_py_any(py)
    }

    fn lru_order(&self, ct: u8, layer: u8) -> Vec<u32> {
        self.tree.lru_order(ct, layer)
    }

    fn reclaim_digest(&self) -> u64 {
        self.tree.reclaim_digest()
    }
}
