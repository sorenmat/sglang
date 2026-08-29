//! PyO3 bindings (feature `python`) — the
//! `sglang.srt.rust_extensions._scheduler` module.
//!
//! Three entry points over the same engine:
//!
//! - [`plan_next_batch`] — stateless **shadow planner**
//!   (`SGLANG_RUST_SCHEDULER=planner`): Python owns the queues, tree and
//!   allocator and passes compact snapshots each iteration; the returned
//!   plan is diffed against Python's own decision (trace capture) before
//!   (eventually) being applied.
//! - [`SchedulerCore`] — persistent **core**
//!   (`SGLANG_RUST_SCHEDULER=core`): the engine owns the queues, the radix
//!   tree and the new-token-ratio tracker; Python feeds ingress + results
//!   and executes each plan + event list.
//! - [`RadixTree`] — raw tree ops backing the Python `RadixCacheRust`
//!   facade (`SGLANG_RUST_SCHEDULER=radix`).
//!
//! Inputs are plain dicts/lists (no torch on the boundary); outputs are
//! compact nested tuples. Heavy entry points drop the GIL around the engine
//! work so planning can overlap Python-side CUDA launches.
//!
//! Tuple shapes (field order is part of the module ABI — keep in sync with
//! `python/sglang/srt/managers/rust_scheduler.py`):
//!
//! - `plan` → `(mode, batch_is_full, prefill, decode)` where
//!   - `prefill` = `(admitted, chunked_or_-1, mixed, extend_tokens,
//!     alloc_extend_pages)`, `admitted` a list of
//!     `(waiting_idx, prefix_len, extend_start, extend_end)`;
//!   - `decode` = `(decode, finished_removed, retract, abort, evict_tokens,
//!     alloc_decode_pages, ntr)`; the lists hold running-batch positions.
//! - events → a list of
//!   - `("evict", [run, ...])`
//!   - `("free_segments", pool_idx, [[start, end], ...])`
//!   - `("stash_row_write", pool_idx, start, [indices, ...])`
//!   - `("finished", core_idx, reason, out_len)`

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use sglang_radix::{EvictionPolicy, RadixKey, ROOT};

use crate::core::{Event, IngressReq, KvRow, ResultRow, SchedulerCore as EngineCore};
use crate::ntr::Ntr;
use crate::planner::plan_next_batch as engine_plan;
use crate::types::{
    AdmitReq, BatchPlan, Config, PlanReq, Policy, StepEnv, CHUNKED_IDX,
};

// ---------------------------------------------------------------- inputs

/// `Config` from a dict; **every key must be present** (None when unset)
/// — `#[pyo3(item)]` extraction is strict, and the Python driver builds the
/// full dict once per process.
#[derive(FromPyObject)]
struct InCfg {
    #[pyo3(item)]
    policy: Option<String>,
    #[pyo3(item)]
    page_size: Option<u32>,
    #[pyo3(item)]
    max_prefill_tokens: Option<u32>,
    #[pyo3(item)]
    chunked_prefill_size: Option<u32>,
    #[pyo3(item)]
    mixed_chunk: Option<bool>,
    #[pyo3(item)]
    priority_scheduling: Option<bool>,
    #[pyo3(item)]
    low_priority_values_first: Option<bool>,
    #[pyo3(item)]
    clip_max_new_tokens: Option<u32>,
    #[pyo3(item)]
    in_batch_check_threshold: Option<u32>,
    #[pyo3(item)]
    in_batch_deprioritize_threshold: Option<u32>,
    #[pyo3(item)]
    prefill_max_requests: Option<u32>,
    #[pyo3(item)]
    truncation_align_size: Option<u32>,
    #[pyo3(item)]
    lpm_queue_degrade_at: Option<u32>,
    #[pyo3(item)]
    random_seed: Option<u64>,
    #[pyo3(item)]
    disable_tree: Option<bool>,
    #[pyo3(item)]
    ntr_init_raw: Option<f64>,
    #[pyo3(item)]
    schedule_conservativeness: Option<f64>,
    #[pyo3(item)]
    ntr_min_factor: Option<f64>,
    #[pyo3(item)]
    ntr_decay_steps: Option<u32>,
    #[pyo3(item)]
    retract_decode_steps: Option<u32>,
}

impl InCfg {
    fn config(&self) -> PyResult<Config> {
        let policy = Policy::parse(self.policy.as_deref().unwrap_or("fcfs"))
            .map_err(PyValueError::new_err)?;
        let d = Config::default();
        Ok(Config {
            policy,
            page_size: self.page_size.unwrap_or(d.page_size),
            max_prefill_tokens: self.max_prefill_tokens.unwrap_or(d.max_prefill_tokens),
            chunked_prefill_size: self.chunked_prefill_size,
            mixed_chunk: self.mixed_chunk.unwrap_or(false),
            priority_scheduling: self.priority_scheduling.unwrap_or(false),
            low_priority_values_first: self.low_priority_values_first.unwrap_or(false),
            clip_max_new_tokens: self
                .clip_max_new_tokens
                .unwrap_or(d.clip_max_new_tokens),
            in_batch_check_threshold: self
                .in_batch_check_threshold
                .unwrap_or(d.in_batch_check_threshold),
            in_batch_deprioritize_threshold: self
                .in_batch_deprioritize_threshold
                .unwrap_or(d.in_batch_deprioritize_threshold),
            prefill_max_requests: self.prefill_max_requests,
            truncation_align_size: self.truncation_align_size,
            lpm_queue_degrade_at: self.lpm_queue_degrade_at.unwrap_or(d.lpm_queue_degrade_at),
            random_seed: self.random_seed.unwrap_or(0),
            disable_tree: self.disable_tree.unwrap_or(false),
            ntr_init_raw: self.ntr_init_raw.unwrap_or(d.ntr_init_raw),
            schedule_conservativeness: self
                .schedule_conservativeness
                .unwrap_or(d.schedule_conservativeness),
            ntr_min_factor: self.ntr_min_factor.unwrap_or(d.ntr_min_factor),
            ntr_decay_steps: self.ntr_decay_steps.unwrap_or(d.ntr_decay_steps),
            retract_decode_steps: self
                .retract_decode_steps
                .unwrap_or(d.retract_decode_steps),
        })
    }
}

/// One `PlanReq` snapshot from a dict. **Every key must be present**
/// (the `#[pyo3(item)]` extraction is strict); the Python driver fills the
/// full set from the `Req` snapshot.
#[derive(FromPyObject)]
struct InPlanReq {
    #[pyo3(item)]
    pool_idx: u32,
    #[pyo3(item)]
    origin_len: u32,
    #[pyo3(item)]
    out_len: u32,
    #[pyo3(item)]
    committed_len: u32,
    #[pyo3(item)]
    prefix_len: u32,
    #[pyo3(item)]
    last_node: Option<u32>,
    #[pyo3(item)]
    priority: Option<i32>,
    #[pyo3(item)]
    arrival_seq: Option<u64>,
    #[pyo3(item)]
    max_new_tokens: Option<u32>,
    #[pyo3(item)]
    routing_key: Option<u64>,
    #[pyo3(item)]
    ignore_eos: Option<bool>,
    #[pyo3(item)]
    finished: Option<bool>,
    #[pyo3(item)]
    retracted_stain: Option<bool>,
    #[pyo3(item)]
    host_hit_length: Option<u32>,
}

impl InPlanReq {
    fn plan_req(&self) -> PlanReq {
        PlanReq {
            pool_idx: self.pool_idx,
            origin_len: self.origin_len,
            out_len: self.out_len,
            committed_len: self.committed_len,
            prefix_len: self.prefix_len,
            last_node: self.last_node.unwrap_or(0),
            priority: self.priority.unwrap_or(0),
            arrival_seq: self.arrival_seq.unwrap_or(0),
            max_new_tokens: self.max_new_tokens.unwrap_or(0),
            routing_key: self.routing_key.unwrap_or(0),
            ignore_eos: self.ignore_eos.unwrap_or(false),
            finished: self.finished.unwrap_or(false),
            retracted_stain: self.retracted_stain.unwrap_or(false),
            host_hit_length: self.host_hit_length.unwrap_or(0),
        }
    }
}

#[derive(FromPyObject)]
struct InStepEnv {
    #[pyo3(item)]
    allocator_avail_tokens: u32,
    #[pyo3(item)]
    tree_evictable_tokens: u32,
    #[pyo3(item)]
    num_allocatable_reqs: u32,
    #[pyo3(item)]
    batch_is_full: bool,
    #[pyo3(item)]
    mixed_chunk_allowed: bool,
}

impl InStepEnv {
    fn step_env(&self) -> StepEnv {
        StepEnv {
            allocator_avail_tokens: self.allocator_avail_tokens,
            tree_evictable_tokens: self.tree_evictable_tokens,
            num_allocatable_reqs: self.num_allocatable_reqs,
            batch_is_full: self.batch_is_full,
            mixed_chunk_allowed: self.mixed_chunk_allowed,
        }
    }
}

#[derive(FromPyObject)]
struct InIngress {
    #[pyo3(item)]
    rid: u64,
    #[pyo3(item)]
    pool_idx: u32,
    #[pyo3(item)]
    origin: Vec<i64>,
    #[pyo3(item)]
    max_new_tokens: u32,
    #[pyo3(item)]
    priority: Option<i32>,
    #[pyo3(item)]
    arrival_seq: Option<u64>,
    #[pyo3(item)]
    routing_key: Option<u64>,
    #[pyo3(item)]
    ignore_eos: Option<bool>,
}

#[derive(FromPyObject)]
struct InResultRow {
    #[pyo3(item)]
    accepted: Vec<i64>,
    #[pyo3(item)]
    finished: bool,
    #[pyo3(item)]
    finish_reason: Option<u32>,
}

#[derive(FromPyObject)]
struct InKvRow {
    #[pyo3(item)]
    core_idx: u32,
    #[pyo3(item)]
    row: Vec<i64>,
}

// --------------------------------------------------------------- outputs

fn admit_tuple(py: Python<'_>, a: &AdmitReq) -> PyResult<Py<PyAny>> {
    PyTuple::new(
        py,
        [
            i64::from(a.waiting_idx),
            i64::from(a.prefix_len),
            i64::from(a.extend_start),
            i64::from(a.extend_end),
        ],
    )?
    .into_py_any(py)
}

fn u32_list(py: Python<'_>, v: &[u32]) -> PyResult<Py<PyAny>> {
    PyList::new(py, v.iter().map(|x| i64::from(*x)))?.into_py_any(py)
}

fn int_list(py: Python<'_>, v: &[i64]) -> PyResult<Py<PyAny>> {
    PyList::new(py, v.iter().copied())?.into_py_any(py)
}

/// `BatchPlan` → `(mode, batch_is_full, prefill, decode)`.
fn batch_plan_to_py(plan: &BatchPlan, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let prefill = match &plan.prefill {
        None => py.None().into_py_any(py)?,
        Some(p) => {
            let admitted: Vec<Py<PyAny>> = p
                .admitted
                .iter()
                .map(|a| admit_tuple(py, a))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(
                py,
                vec![
                    admitted.into_py_any(py)?,
                    p.chunked.map(i64::from).unwrap_or(-1).into_py_any(py)?,
                    p.mixed.into_py_any(py)?,
                    i64::from(p.extend_tokens).into_py_any(py)?,
                    i64::from(p.alloc_extend_pages).into_py_any(py)?,
                ],
            )?
            .into_py_any(py)?
        }
    };
    let decode = match &plan.decode {
        None => py.None().into_py_any(py)?,
        Some(d) => PyTuple::new(
            py,
            vec![
                u32_list(py, &d.decode)?,
                u32_list(py, &d.finished_removed)?,
                u32_list(py, &d.retract)?,
                u32_list(py, &d.abort)?,
                i64::from(d.evict_tokens).into_py_any(py)?,
                i64::from(d.alloc_decode_pages).into_py_any(py)?,
                d.ntr.into_py_any(py)?,
            ],
        )?
        .into_py_any(py)?,
    };
    PyTuple::new(
        py,
        vec![
            i64::from(plan.mode).into_py_any(py)?,
            plan.batch_is_full.into_py_any(py)?,
            prefill,
            decode,
        ],
    )?
    .into_py_any(py)
}

fn event_to_py(ev: &Event, py: Python<'_>) -> PyResult<Py<PyAny>> {
    match ev {
        Event::Evict { values } => {
            let runs: Vec<Py<PyAny>> = values
                .iter()
                .map(|run| int_list(py, run))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(
                py,
                vec![
                    "evict".to_string().into_py_any(py)?,
                    runs.into_py_any(py)?,
                ],
            )?
            .into_py_any(py)
        }
        Event::FreeSegments { pool_idx, ranges } => {
            let rs: Vec<Py<PyAny>> = ranges
                .iter()
                .map(|(s, e)| PyTuple::new(py, [i64::from(*s), i64::from(*e)]))
                .collect::<PyResult<Vec<Bound<'_, PyTuple>>>>()?
                .into_iter()
                .map(|t| t.into_py_any(py))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(
                py,
                vec![
                    "free_segments".to_string().into_py_any(py)?,
                    i64::from(*pool_idx).into_py_any(py)?,
                    rs.into_py_any(py)?,
                ],
            )?
            .into_py_any(py)
        }
        Event::StashRowWrite {
            pool_idx,
            start,
            new_indices,
        } => PyTuple::new(
            py,
            vec![
                "stash_row_write".to_string().into_py_any(py)?,
                i64::from(*pool_idx).into_py_any(py)?,
                i64::from(*start).into_py_any(py)?,
                int_list(py, new_indices)?,
            ],
        )?
        .into_py_any(py),
        Event::Finished {
            core_idx,
            reason,
            out_len,
        } => PyTuple::new(
            py,
            vec![
                "finished".to_string().into_py_any(py)?,
                i64::from(*core_idx).into_py_any(py)?,
                i64::from(*reason).into_py_any(py)?,
                i64::from(*out_len).into_py_any(py)?,
            ],
        )?
        .into_py_any(py),
    }
}

fn events_to_py(events: &[Event], py: Python<'_>) -> PyResult<Py<PyAny>> {
    let out: Vec<Py<PyAny>> = events
        .iter()
        .map(|ev| event_to_py(ev, py))
        .collect::<PyResult<Vec<_>>>()?;
    out.into_py_any(py)
}

// ------------------------------------------------------- shadow planner

/// One shadow-planner iteration.
///
/// ```python
/// mode, batch_is_full, prefill, decode = plan_next_batch(
///     cfg, ntr_current, waiting, running, chunked,
///     scores, deprioritized, env, iter,
/// )
/// ```
///
/// - `waiting` / `running` / `chunked`: dicts (or None) with **all** keys
///   `pool_idx, origin_len, out_len, committed_len, prefix_len, last_node,
///   priority, arrival_seq, max_new_tokens, routing_key, ignore_eos,
///   finished, retracted_stain, host_hit_length`.
/// - `env`: dict with `allocator_avail_tokens, tree_evictable_tokens,
///   num_allocatable_reqs, batch_is_full, mixed_chunk_allowed`.
/// - `ntr_current`: the Python tracker's current value; when the returned
///   decode is non-empty, update the tracker with `decode[6]` **only if**
///   `decode[0]` is non-empty (Python skips the update on the
///   `batch.is_empty()` return).
#[pyfunction]
#[pyo3(name = "plan_next_batch")]
#[allow(clippy::too_many_arguments)]
fn plan_next_batch_py(
    py: Python<'_>,
    cfg: InCfg,
    ntr_current: f64,
    waiting: Vec<InPlanReq>,
    running: Vec<InPlanReq>,
    chunked: Option<InPlanReq>,
    scores: Vec<u32>,
    deprioritized: Vec<bool>,
    env: InStepEnv,
    iter: u64,
) -> PyResult<Py<PyAny>> {
    let cfg = cfg.config()?;
    let mut ntr = Ntr::from_config(&cfg);
    ntr.set_current(ntr_current);
    let waiting: Vec<PlanReq> = waiting.iter().map(|r| r.plan_req()).collect();
    let running: Vec<PlanReq> = running.iter().map(|r| r.plan_req()).collect();
    let chunked = chunked.as_ref().map(|r| r.plan_req());
    let env = env.step_env();
    let plan = py.detach(|| {
        engine_plan(
            &cfg,
            &ntr,
            &waiting,
            &running,
            chunked.as_ref(),
            &scores,
            &deprioritized,
            &env,
            iter,
        )
    });
    batch_plan_to_py(&plan, py)
}

// --------------------------------------------------------- NTR helpers

/// `NewTokenRatioTracker.decay_step()` as a pure function: the value the
/// tracker would hold after one decay from `current`.
#[pyfunction]
fn ntr_next_after_decay(
    current: f64,
    init_raw: f64,
    conservativeness: f64,
    min_factor: f64,
    decay_steps: u32,
) -> f64 {
    let cfg = Config {
        ntr_init_raw: init_raw,
        schedule_conservativeness: conservativeness,
        ntr_min_factor: min_factor,
        ntr_decay_steps: decay_steps,
        ..Config::default()
    };
    let mut n = Ntr::from_config(&cfg);
    n.set_current(current);
    n.next_after_decay()
}

/// `estimate_new_token_ratio_after_retract(reqs)`.
#[pyfunction]
fn ntr_estimate_after_retract(out_lens: Vec<u32>, max_news: Vec<u32>, retract_steps: u32) -> f64 {
    Ntr::estimate_after_retract(&out_lens, &max_news, retract_steps)
}

// ------------------------------------------------------ radix tree facade

/// Raw `sglang-radix` tree ops (Python `RadixCacheRust` facade backing).
#[pyclass]
struct RadixTree {
    tree: sglang_radix::RadixTree,
}

#[pymethods]
impl RadixTree {
    #[new]
    fn new(page_size: u32, is_eagle: bool, policy: &str) -> PyResult<Self> {
        let policy = EvictionPolicy::parse(policy).map_err(PyValueError::new_err)?;
        Ok(Self {
            tree: sglang_radix::RadixTree::new(page_size as usize, is_eagle, policy),
        })
    }

    /// `match_prefix(keys)` → `(indices, last_node)` (page-floored prefix).
    fn match_prefix(&mut self, py: Python<'_>, keys: Vec<i64>) -> PyResult<(Vec<i64>, u32)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.match_prefix(&key)
        });
        Ok((r.indices, r.last_node))
    }

    /// Shadow-check fast path: `(matched_len, last_node)` without building
    /// the Python list of KV indices (the list conversion dominates cost
    /// for long prompts; the dual-write facade only needs length + handle).
    fn match_prefix_meta(&mut self, py: Python<'_>, keys: Vec<i64>) -> PyResult<(usize, u32)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.match_prefix(&key)
        });
        Ok((r.indices.len(), r.last_node))
    }

    /// `insert(keys, values, priority, chunked)` → `(prefix_len, last_node)`.
    fn insert(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
        values: Vec<i64>,
        priority: i32,
        chunked: bool,
    ) -> PyResult<(u32, u32)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.insert(&key, &values, priority, chunked)
        });
        Ok((u32::try_from(r.prefix_len).unwrap_or(u32::MAX), r.last_node))
    }

    /// `evict(tokens)` → `(evicted value runs, num_tokens_evicted)`.
    fn evict(&mut self, py: Python<'_>, tokens: u32) -> PyResult<(Vec<Vec<i64>>, u32)> {
        let r = py.detach(|| self.tree.evict(tokens as usize));
        Ok((
            r.evicted_values,
            u32::try_from(r.num_tokens_evicted).unwrap_or(u32::MAX),
        ))
    }

    fn inc_lock_ref(&mut self, node: u32) -> i64 {
        self.tree.inc_lock_ref(node)
    }

    fn dec_lock_ref(&mut self, node: u32) -> i64 {
        self.tree.dec_lock_ref(node)
    }

    fn evictable_size(&self) -> i64 {
        self.tree.evictable_size()
    }

    fn protected_size(&self) -> i64 {
        self.tree.protected_size()
    }

    fn total_size(&self) -> i64 {
        self.tree.total_size()
    }

    fn node_children(&self, node: u32) -> Vec<u32> {
        self.tree.node_children(node)
    }
}

/// Raw `sglang-radix` SWA (dual-counter) tree ops — Python `SWARadixCacheRust`
/// facade backing (plan.md M2/1b). The allocator is *not* behind this
/// boundary: every `free` / `free_full` / `free_swa` the Python tree would
/// make comes back as a value-run list instead.
#[pyclass]
struct SWARadixTree {
    tree: sglang_radix::SWARadixTree,
}

#[pymethods]
impl SWARadixTree {
    #[new]
    fn new(page_size: u32, is_eagle: bool, sliding_window_size: u32) -> Self {
        Self {
            tree: sglang_radix::SWARadixTree::new(
                page_size as usize,
                is_eagle,
                sliding_window_size as usize,
            ),
        }
    }

    /// `match_prefix(keys)` → `(indices, last_node)`.
    fn match_prefix(&mut self, py: Python<'_>, keys: Vec<i64>) -> PyResult<(Vec<i64>, u32)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.match_prefix(&key)
        });
        Ok((r.indices, r.last_node))
    }

    /// `insert(keys, values, update_kv_after_len, swa_evicted_seqlen)` →
    /// `(prefix_len, last_node, free_kv, free_full, free_swa, recover)`
    /// where `recover` is a list of `[tree_value, incoming]` runs the
    /// caller re-points (`set_full_to_swa_mapping` + `free_full`).
    fn insert(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
        values: Vec<i64>,
        update_kv_after_len: u32,
        swa_evicted_seqlen: u32,
    ) -> PyResult<
        (
            u32,
            u32,
            Vec<Vec<i64>>,
            Vec<Vec<i64>>,
            Vec<Vec<i64>>,
            Vec<(Vec<i64>, Vec<i64>)>,
        ),
    > {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.insert(
                &key,
                &values,
                update_kv_after_len as usize,
                swa_evicted_seqlen as usize,
            )
        });
        let recover = r
            .recover_locked_full
            .into_iter()
            .map(|e| (e.tree_value, e.incoming))
            .collect();
        Ok((
            u32::try_from(r.prefix_len).unwrap_or(u32::MAX),
            r.last_node,
            r.free.kv,
            r.free.full,
            r.free.swa,
            recover,
        ))
    }

    /// `evict(full_tokens, swa_tokens)` →
    /// `(full_evicted, swa_evicted, free_kv, free_full, free_swa)`.
    fn evict(
        &mut self,
        py: Python<'_>,
        full_tokens: u32,
        swa_tokens: u32,
    ) -> PyResult<(u32, u32, Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
        let r =
            py.detach(|| self.tree.evict(full_tokens as usize, swa_tokens as usize));
        Ok((
            u32::try_from(r.full_num_evicted).unwrap_or(u32::MAX),
            u32::try_from(r.swa_num_evicted).unwrap_or(u32::MAX),
            r.free.kv,
            r.free.full,
            r.free.swa,
        ))
    }

    /// `inc_lock_ref(node)` → `(swa_uuid_for_lock | None, full-side delta)`.
    fn inc_lock_ref(&mut self, node: u32) -> (Option<u64>, i64) {
        self.tree.inc_lock_ref(node)
    }

    /// `dec_lock_ref(node, swa_uuid_or_none, skip_swa)` → full-side delta.
    fn dec_lock_ref(
        &mut self,
        node: u32,
        swa_uuid: Option<u64>,
        skip_swa: bool,
    ) -> i64 {
        self.tree.dec_lock_ref(node, swa_uuid, skip_swa)
    }

    /// `dec_swa_lock_only(node, swa_uuid_or_none)` → `free_swa` runs.
    fn dec_swa_lock_only(&mut self, node: u32, swa_uuid: Option<u64>) -> Vec<Vec<i64>> {
        self.tree.dec_swa_lock_only(node, swa_uuid).free_swa
    }

    fn full_evictable_size(&self) -> i64 {
        self.tree.full_evictable_size()
    }

    fn swa_evictable_size(&self) -> i64 {
        self.tree.swa_evictable_size()
    }

    fn full_protected_size(&self) -> i64 {
        self.tree.full_protected_size()
    }

    fn swa_protected_size(&self) -> i64 {
        self.tree.swa_protected_size()
    }

    /// `total_size()` → `(full, swa)`.
    fn total_size(&self) -> (i64, i64) {
        self.tree.total_size()
    }

    fn node_children(&self, node: u32) -> Vec<u32> {
        self.tree.node_children(node)
    }

    fn node_tombstone(&self, node: u32) -> bool {
        self.tree.node_tombstone(node)
    }

    fn node_full_lock_ref(&self, node: u32) -> u32 {
        self.tree.node_full_lock_ref(node)
    }

    fn node_swa_lock_ref(&self, node: u32) -> u32 {
        self.tree.node_swa_lock_ref(node)
    }
}

/// Raw `sglang-radix` Mamba (hybrid full + SSM) tree ops — Python
/// `MambaRadixCacheRust` facade backing (plan.md M2/1c). The allocator
/// is *not* behind this boundary: every `free_segment(run, start_pos)`
/// and `free_mamba(run)` the Python tree would make comes back as a
/// run list instead.
#[pyclass]
struct MambaRadixTree {
    tree: sglang_radix::MambaRadixTree,
}

#[pymethods]
impl MambaRadixTree {
    #[new]
    fn new(page_size: u32, is_eagle: bool, mamba_cache_chunk_size: u32) -> Self {
        Self {
            tree: sglang_radix::MambaRadixTree::new(
                page_size as usize,
                is_eagle,
                mamba_cache_chunk_size as usize,
            ),
        }
    }

    /// `match_prefix(keys)` →
    /// `(indices, last_node, mamba_branching_seqlen | None)`.
    fn match_prefix(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
    ) -> PyResult<(Vec<i64>, u32, Option<u32>)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.match_prefix(&key)
        });
        Ok((
            r.indices,
            r.last_node,
            r.mamba_branching_seqlen
                .map(|v| u32::try_from(v).unwrap_or(u32::MAX)),
        ))
    }

    /// `insert(keys, values, mamba_values, prev_prefix_len)` →
    /// `(prefix_len, last_node, mamba_exist, free_kv_runs,
    /// free_kv_start_pos, free_mamba)` where `free_kv_runs[i]` is freed
    /// via `free_segment(run, start_pos=free_kv_start_pos[i])`.
    fn insert(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
        values: Vec<i64>,
        mamba_values: Vec<i64>,
        prev_prefix_len: u32,
    ) -> PyResult<(
        u32,
        u32,
        bool,
        Vec<Vec<i64>>,
        Vec<u32>,
        Vec<Vec<i64>>,
    )> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.insert(
                &key,
                &values,
                &mamba_values,
                prev_prefix_len as usize,
            )
        });
        let (kv_runs, kv_start_pos) = r
            .free
            .kv
            .into_iter()
            .map(|(run, pos)| (run, u32::try_from(pos).unwrap_or(u32::MAX)))
            .unzip();
        Ok((
            u32::try_from(r.prefix_len).unwrap_or(u32::MAX),
            r.last_node,
            r.mamba_exist,
            kv_runs,
            kv_start_pos,
            r.free.mamba,
        ))
    }

    /// `evict(full_tokens, mamba_num)` →
    /// `(full_evicted, mamba_evicted, free_kv_runs, free_kv_start_pos,
    /// free_mamba)`.
    fn evict(
        &mut self,
        py: Python<'_>,
        full_tokens: u32,
        mamba_num: u32,
    ) -> PyResult<(u32, u32, Vec<Vec<i64>>, Vec<u32>, Vec<Vec<i64>>)> {
        let r = py.detach(|| self.tree.evict(full_tokens as usize, mamba_num as usize));
        let (kv_runs, kv_start_pos) = r
            .free
            .kv
            .into_iter()
            .map(|(run, pos)| (run, u32::try_from(pos).unwrap_or(u32::MAX)))
            .unzip();
        Ok((
            u32::try_from(r.full_num_evicted).unwrap_or(u32::MAX),
            u32::try_from(r.mamba_num_evicted).unwrap_or(u32::MAX),
            kv_runs,
            kv_start_pos,
            r.free.mamba,
        ))
    }

    /// `inc_lock_ref(node)` → `(full_delta, mamba_delta)` (units moved
    /// evictable -> protected, <= 0).
    fn inc_lock_ref(&mut self, node: u32) -> (i64, i64) {
        self.tree.inc_lock_ref(node)
    }

    /// `dec_lock_ref(node)` → `(full_delta, mamba_delta)` (>= 0).
    fn dec_lock_ref(&mut self, node: u32) -> (i64, i64) {
        self.tree.dec_lock_ref(node)
    }

    fn full_evictable_size(&self) -> i64 {
        self.tree.full_evictable_size()
    }

    fn mamba_evictable_size(&self) -> i64 {
        self.tree.mamba_evictable_size()
    }

    fn full_protected_size(&self) -> i64 {
        self.tree.full_protected_size()
    }

    fn mamba_protected_size(&self) -> i64 {
        self.tree.mamba_protected_size()
    }

    /// `total_size()` → `(full, mamba)`.
    fn total_size(&self) -> (i64, i64) {
        self.tree.total_size()
    }

    fn node_children(&self, node: u32) -> Vec<u32> {
        self.tree.node_children(node)
    }

    fn node_mamba_tombstone(&self, node: u32) -> bool {
        self.tree.node_mamba_tombstone(node)
    }

    fn node_mamba_value(&self, node: u32) -> Option<Vec<i64>> {
        self.tree.node_mamba_value(node)
    }

    fn node_full_lock_ref(&self, node: u32) -> u32 {
        self.tree.node_full_lock_ref(node)
    }

    fn node_mamba_lock_ref(&self, node: u32) -> u32 {
        self.tree.node_mamba_lock_ref(node)
    }
}

// ------------------------------------------------------------ scheduler core

/// Persistent scheduler core: owns the queues, the radix tree and the NTR
/// tracker. See `sglang.srt.managers.rust_scheduler` for the driver.
#[pyclass]
struct SchedulerCore {
    core: EngineCore,
}

#[pymethods]
impl SchedulerCore {
    #[new]
    fn new(cfg: InCfg, tree_policy: &str) -> PyResult<Self> {
        let cfg = cfg.config()?;
        let tree = EvictionPolicy::parse(tree_policy).map_err(PyValueError::new_err)?;
        Ok(Self {
            core: EngineCore::new(cfg, tree),
        })
    }

    /// Feed incoming requests (before any planning). Returns the core
    /// indices.
    fn ingest(&mut self, py: Python<'_>, reqs: Vec<InIngress>) -> PyResult<Vec<u32>> {
        let reqs: Vec<IngressReq> = reqs
            .into_iter()
            .map(|r| IngressReq {
                rid: r.rid,
                pool_idx: r.pool_idx,
                origin: r.origin,
                max_new_tokens: r.max_new_tokens,
                priority: r.priority.unwrap_or(0),
                arrival_seq: r.arrival_seq.unwrap_or(0),
                routing_key: r.routing_key.unwrap_or(0),
                ignore_eos: r.ignore_eos.unwrap_or(false),
            })
            .collect();
        Ok(py.detach(|| self.core.ingest(reqs)))
    }

    /// Fold one last-batch result into the core (output append, finish
    /// detection, cache ops). Returns the event list to execute in Python.
    fn apply_result(
        &mut self,
        py: Python<'_>,
        rows: Vec<InResultRow>,
        kv_rows: Vec<InKvRow>,
    ) -> PyResult<Py<PyAny>> {
        let rows: Vec<ResultRow> = rows
            .into_iter()
            .map(|r| ResultRow {
                accepted: r.accepted,
                finished: r.finished,
                finish_reason: r.finish_reason.unwrap_or(0),
            })
            .collect();
        let kv_rows: Vec<KvRow> = kv_rows
            .into_iter()
            .map(|r| KvRow {
                core_idx: r.core_idx,
                row: r.row,
            })
            .collect();
        let events = py.detach(|| self.core.apply_result(&rows, &kv_rows));
        events_to_py(&events, py)
    }

    /// User-initiated abort of a queued request (Python `abort_request`).
    /// Returns the events to execute (KV frees + abort notice).
    fn drop(&mut self, py: Python<'_>, core_idx: u32) -> PyResult<Py<PyAny>> {
        let events = py.detach(|| self.core.drop_request(core_idx));
        events_to_py(&events, py)
    }

    /// Plan the next batch. Returns `(plan, events)`; `events` carries the
    /// tree evictions and KV frees to run before the batch's allocations.
    fn plan(&mut self, py: Python<'_>, env: InStepEnv) -> PyResult<Py<PyAny>> {
        let env = env.step_env();
        let out = py.detach(|| self.core.plan(&env));
        let plan = batch_plan_to_py(&out.plan, py)?;
        let events = events_to_py(&out.events, py)?;
        PyTuple::new(py, [plan, events])?.into_py_any(py)
    }

    // Observability / driver accessors.

    fn waiting(&self) -> Vec<u32> {
        self.core.waiting().to_vec()
    }

    fn running(&self) -> Vec<u32> {
        self.core.running().to_vec()
    }

    /// -1 when no chunked request is in flight.
    fn chunked_idx(&self) -> i64 {
        match self.core.chunked_idx() {
            None => -1,
            Some(i) => i as i64,
        }
    }

    fn new_token_ratio(&self) -> f64 {
        self.core.new_token_ratio()
    }

    fn batch_is_full(&self) -> bool {
        self.core.batch_is_full()
    }

    fn last_batch(&self) -> Vec<u32> {
        self.core.last_batch().to_vec()
    }

    fn req_pool_idx(&self, core_idx: u32) -> u32 {
        self.core.req_pool_idx(core_idx)
    }

    fn req_rid(&self, core_idx: u32) -> u64 {
        self.core.req_rid(core_idx)
    }

    fn req_out_len(&self, core_idx: u32) -> u32 {
        self.core.req_out_len(core_idx)
    }

    fn req_retracted_stain(&self, core_idx: u32) -> bool {
        self.core.req_retracted_stain(core_idx)
    }

    /// `(total_size, evictable_size, protected_size)`.
    fn tree_stats(&self) -> (i64, i64, i64) {
        let t = self.core.tree();
        (t.total_size(), t.evictable_size(), t.protected_size())
    }

    fn tree_node_children(&self, node: u32) -> Vec<u32> {
        self.core.tree().node_children(node)
    }
}

// ------------------------------------------------------------------ module

/// The `_scheduler` PyO3 extension module.
#[pymodule]
fn _scheduler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ROOT", ROOT)?;
    m.add("CHUNKED_IDX", CHUNKED_IDX)?;
    m.add("MODE_NONE", 0i32)?;
    m.add("MODE_PREFILL", 1i32)?;
    m.add("MODE_DECODE", 2i32)?;
    m.add_class::<SchedulerCore>()?;
    m.add_class::<RadixTree>()?;
    m.add_class::<SWARadixTree>()?;
    m.add_class::<MambaRadixTree>()?;
    m.add_function(wrap_pyfunction!(plan_next_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(ntr_next_after_decay, m)?)?;
    m.add_function(wrap_pyfunction!(ntr_estimate_after_retract, m)?)?;
    Ok(())
}
