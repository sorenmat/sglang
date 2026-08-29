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

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use sglang_radix::{EvictionPolicy, ROOT, RadixKey};

use crate::core::{Event, IngressReq, KvRow, ResultRow, SchedulerCore as EngineCore};
use crate::ntr::Ntr;
use crate::planner::plan_next_batch as engine_plan;
use crate::types::{AdmitReq, BatchPlan, CHUNKED_IDX, Config, PlanReq, Policy, StepEnv};
use crate::unified::UnifiedRadixTreePy;

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
            clip_max_new_tokens: self.clip_max_new_tokens.unwrap_or(d.clip_max_new_tokens),
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
            retract_decode_steps: self.retract_decode_steps.unwrap_or(d.retract_decode_steps),
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

/// Spec-v2 row metadata (plan §9). Python settles the counters with the
/// pre-step state; `settled` captures that gate.
#[derive(FromPyObject)]
struct InResultSpec {
    #[pyo3(item)]
    accept_len: u32,
    #[pyo3(item)]
    settled: bool,
    #[pyo3(item)]
    block_accept_len: Option<u32>,
    #[pyo3(item)]
    cap_len: Option<u32>,
}

#[derive(FromPyObject)]
struct InResultRow {
    #[pyo3(item)]
    accepted: Vec<i64>,
    #[pyo3(item)]
    finished: bool,
    #[pyo3(item)]
    finish_reason: Option<u32>,
    #[pyo3(item)]
    spec: Option<InResultSpec>,
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

fn u64_list(py: Python<'_>, v: &[u64]) -> PyResult<Py<PyAny>> {
    PyList::new(py, v.iter().map(|x| *x as i64))?.into_py_any(py)
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
                vec!["evict".to_string().into_py_any(py)?, runs.into_py_any(py)?],
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
    #[allow(clippy::type_complexity)]
    fn insert(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
        values: Vec<i64>,
        update_kv_after_len: u32,
        swa_evicted_seqlen: u32,
    ) -> PyResult<(
        u32,
        u32,
        Vec<Vec<i64>>,
        Vec<Vec<i64>>,
        Vec<Vec<i64>>,
        Vec<(Vec<i64>, Vec<i64>)>,
    )> {
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
    #[allow(clippy::type_complexity)]
    fn evict(
        &mut self,
        py: Python<'_>,
        full_tokens: u32,
        swa_tokens: u32,
    ) -> PyResult<(u32, u32, Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
        let r = py.detach(|| self.tree.evict(full_tokens as usize, swa_tokens as usize));
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
    fn dec_lock_ref(&mut self, node: u32, swa_uuid: Option<u64>, skip_swa: bool) -> i64 {
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
    #[allow(clippy::type_complexity)]
    fn insert(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
        values: Vec<i64>,
        mamba_values: Vec<i64>,
        prev_prefix_len: u32,
    ) -> PyResult<(u32, u32, bool, Vec<Vec<i64>>, Vec<u32>, Vec<Vec<i64>>)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree
                .insert(&key, &values, &mamba_values, prev_prefix_len as usize)
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
    #[allow(clippy::type_complexity)]
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

// ------------------------------------------------------------ HiRadix tree

/// Python facade backing for `HiRadixCache` (host-tier cache). The DMA /
/// controller side stays in Python; the tree reports the runs to free and
/// the pending write-through state, and exposes the two-phase
/// load-back + backup primitives the facade drives.
#[pyclass]
struct HiRadixTree {
    tree: sglang_radix::HiRadixTree,
}

#[pymethods]
impl HiRadixTree {
    #[new]
    fn new(
        page_size: u32,
        is_eagle: bool,
        write_policy: &str,
        eviction_policy: &str,
        write_through_threshold: u32,
        load_back_threshold: u32,
    ) -> PyResult<Self> {
        let policy = sglang_radix::HiPolicy::parse(write_policy).map_err(PyValueError::new_err)?;
        let strategy = EvictionPolicy::parse(eviction_policy).map_err(PyValueError::new_err)?;
        Ok(Self {
            tree: sglang_radix::HiRadixTree::new(
                page_size as usize,
                is_eagle,
                policy,
                strategy,
                write_through_threshold as u64,
                load_back_threshold as usize,
            ),
        })
    }

    fn reset(&mut self) {
        self.tree.reset();
    }

    /// `match_prefix(keys)` →
    /// `(indices, last_device_node, last_host_node, host_hit_length,
    /// splits)`; each split is `(front, tail)`.
    #[allow(clippy::type_complexity)]
    fn match_prefix(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
    ) -> PyResult<(Vec<i64>, u32, u32, u32, Vec<(u32, u32)>)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.match_prefix(&key)
        });
        Ok((
            r.indices,
            r.last_device_node,
            r.last_host_node,
            u32::try_from(r.host_hit_length).unwrap_or(u32::MAX),
            r.splits,
        ))
    }

    /// `insert(keys, values, priority, chunked)` →
    /// `(prefix_len, last_node, backup_needed, splits)`.
    /// `backup_needed` lists the nodes that crossed the write-through
    /// threshold without a backup: run the host DMA, then
    /// `begin_backup(node, host_indices, lock=True)`.
    #[allow(clippy::type_complexity)]
    fn insert(
        &mut self,
        py: Python<'_>,
        keys: Vec<i64>,
        values: Vec<i64>,
        priority: i32,
        chunked: bool,
    ) -> PyResult<(u32, u32, Vec<u32>, Vec<(u32, u32)>)> {
        let r = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.insert(&key, &values, priority, chunked)
        });
        Ok((
            u32::try_from(r.prefix_len).unwrap_or(u32::MAX),
            r.last_node,
            r.backup_needed,
            r.splits,
        ))
    }

    /// `insert_host(start_node, keys, host_value)` → matched length
    /// (the caller frees the overlapping host indices of the prefetch).
    fn insert_host(
        &mut self,
        py: Python<'_>,
        start_node: u32,
        keys: Vec<i64>,
        host_value: Vec<i64>,
    ) -> PyResult<u32> {
        let n = py.detach(|| {
            let key = RadixKey::new(&keys);
            self.tree.insert_host(start_node, &key, &host_value)
        });
        Ok(u32::try_from(n).unwrap_or(u32::MAX))
    }

    /// `evict(num_tokens)` → `(num_evicted, free_device)` (write-through
    /// loop; the caller releases each device run).
    fn evict(&mut self, py: Python<'_>, num_tokens: u32) -> PyResult<(u32, Vec<Vec<i64>>)> {
        let r = py.detach(|| self.tree.evict(num_tokens as usize));
        Ok((
            u32::try_from(r.num_tokens_evicted).unwrap_or(u32::MAX),
            r.free_device,
        ))
    }

    /// `evict_host(num_tokens)` →
    /// `(num_evicted, free_host, deleted_node_ids)`.
    fn evict_host(
        &mut self,
        py: Python<'_>,
        num_tokens: u32,
    ) -> PyResult<(u32, Vec<Vec<i64>>, Vec<u32>)> {
        let r = py.detach(|| self.tree.evict_host(num_tokens as usize));
        Ok((
            u32::try_from(r.num_tokens_evicted).unwrap_or(u32::MAX),
            r.free_host,
            r.deleted,
        ))
    }

    /// Phase 1 of `load_back`: `init_load_back(last_node, mem_quota)` →
    /// `(ancestor, last_node, nodes, host_indices)` or `None` when the
    /// load was skipped (threshold / quota / already live).
    fn init_load_back(
        &mut self,
        last_node: u32,
        mem_quota: Option<i64>,
    ) -> Option<(u32, u32, Vec<u32>, Vec<i64>)> {
        self.tree
            .init_load_back(last_node, mem_quota)
            .map(|p| (p.ancestor, p.last_node, p.nodes.clone(), p.host_indices))
    }

    /// Phase 2 of `load_back` with the DMA result.
    /// `device_indices=None` (controller failed after the evict+retry)
    /// releases the host protections and the temporary lock; otherwise
    /// it re-attaches the values and returns the PERMANENT chain-lock
    /// delta (release later with `dec_lock_ref(last_node)`).
    fn finish_load_back(
        &mut self,
        ancestor: u32,
        last_node: u32,
        nodes: Vec<u32>,
        device_indices: Option<Vec<i64>>,
    ) -> i64 {
        let plan = sglang_radix::LoadBackPlan {
            ancestor,
            last_node,
            nodes,
            host_indices: vec![],
        };
        self.tree.finish_load_back(&plan, device_indices.as_deref())
    }

    /// Abandon an open load-back without a DMA result (request aborted).
    fn abort_load_back(&mut self, ancestor: u32, nodes: Vec<u32>) {
        let plan = sglang_radix::LoadBackPlan {
            ancestor,
            last_node: 0,
            nodes,
            host_indices: vec![],
        };
        self.tree.abort_load_back(&plan);
    }

    /// Successful `write_backup` tree-side effects: attach the host copy,
    /// mark the backup pending, and take the protective device lock
    /// (`lock=True`, write-through). Returns the lock delta.
    fn begin_backup(&mut self, node: u32, host_indices: Vec<i64>, lock: bool) -> i64 {
        self.tree.begin_backup(node, &host_indices, lock)
    }

    /// DMA ack processed: clear the pending flag on one publish node.
    fn end_backup(&mut self, node: u32) {
        self.tree.end_backup(node);
    }

    fn protect_host(&mut self, node: u32) {
        self.tree.protect_host(node);
    }

    /// Panics (→ Python exception) when `host_ref == 0`, like Python's
    /// `RuntimeError`.
    fn release_host(&mut self, node: u32) {
        self.tree.release_host(node);
    }

    /// `inc_lock_ref(node)` → delta of tokens moved evictable ->
    /// protected (<= 0).
    fn inc_lock_ref(&mut self, node: u32) -> i64 {
        self.tree.inc_lock_ref(node)
    }

    /// `dec_lock_ref(node)` → tokens moved protected -> evictable (>= 0).
    fn dec_lock_ref(&mut self, node: u32) -> i64 {
        self.tree.dec_lock_ref(node)
    }

    /// write_back facade primitives. `_detach_backuped`: demote to host-only,
    /// returning the device run (the caller keeps it for the staged DMA).
    fn detach_backuped(&mut self, node: u32) -> Vec<i64> {
        self.tree.detach_backuped(node)
    }

    /// `_drop_subtree_no_host` →
    /// `(freed_device, free_device_runs, free_host_runs)`; all zeros when
    /// refused (a subtree node holds a host reference).
    fn drop_subtree_no_host(&mut self, root: u32) -> (i64, Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let r = self.tree.drop_subtree_no_host(root);
        (r.freed_device, r.free_device, r.free_host)
    }

    /// `_promote_parent`: the parent becomes a device leaf once all of
    /// its children are evicted; re-insert it into the caller's heap.
    fn promote_parent(&mut self, node: u32) -> Option<u32> {
        self.tree.promote_parent(node)
    }

    // ---- sizes / set membership / node accessors ----

    fn evictable_size(&self) -> i64 {
        self.tree.evictable_size()
    }

    fn protected_size(&self) -> i64 {
        self.tree.protected_size()
    }

    fn total_size(&self) -> i64 {
        self.tree.total_size()
    }

    fn total_host_size(&self) -> i64 {
        self.tree.total_host_size()
    }

    fn evictable_leaves(&self) -> Vec<u32> {
        self.tree.evictable_leaves()
    }

    fn evictable_host_leaves(&self) -> Vec<u32> {
        self.tree.evictable_host_leaves()
    }

    fn evictable_leaves_ordered(&self) -> Vec<u32> {
        self.tree.evictable_leaves_ordered()
    }

    fn node_children(&self, node: u32) -> Vec<u32> {
        self.tree.node_children(node)
    }

    fn node_value(&self, node: u32) -> Option<Vec<i64>> {
        self.tree.node_value(node)
    }

    fn node_host_value(&self, node: u32) -> Option<Vec<i64>> {
        self.tree.node_host_value(node)
    }

    fn node_evicted(&self, node: u32) -> bool {
        self.tree.node_evicted(node)
    }

    fn node_backuped(&self, node: u32) -> bool {
        self.tree.node_backuped(node)
    }

    fn node_lock_ref(&self, node: u32) -> u32 {
        self.tree.node_lock_ref(node)
    }

    fn node_host_ref(&self, node: u32) -> u32 {
        self.tree.node_host_ref(node)
    }

    fn node_hit_count(&self, node: u32) -> u32 {
        u32::try_from(self.tree.node_hit_count(node)).unwrap_or(u32::MAX)
    }

    fn node_priority(&self, node: u32) -> i32 {
        self.tree.node_priority(node)
    }

    fn node_last_access(&self, node: u32) -> u32 {
        u32::try_from(self.tree.node_last_access(node)).unwrap_or(u32::MAX)
    }

    fn node_key(&self, node: u32) -> Option<Vec<i64>> {
        self.tree.node_key(node)
    }

    fn node_backup_pending(&self, node: u32) -> bool {
        self.tree.node_backup_pending(node)
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
                spec: r.spec.map(|s| crate::core::ResultSpec {
                    accept_len: s.accept_len,
                    settled: s.settled,
                    block_accept_len: s.block_accept_len,
                    cap_len: s.cap_len,
                }),
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

    /// Spec-v2 counters for a live request (plan §9): a dict with
    /// `spec_verify_ct`, `spec_num_correct_drafts`,
    /// `spec_num_block_accept_tokens`, `spec_num_cap_tokens`,
    /// `correct_drafts_histogram`, `cap_lens_histogram` — or None for an
    /// out-of-range index.
    fn spec_counters(&self, py: Python<'_>, core_idx: u32) -> PyResult<Py<PyAny>> {
        match self.core.spec_counters(core_idx) {
            None => Ok(py.None()),
            Some(c) => {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("spec_verify_ct", c.spec_verify_ct)?;
                d.set_item("spec_num_correct_drafts", c.spec_num_correct_drafts)?;
                d.set_item(
                    "spec_num_block_accept_tokens",
                    c.spec_num_block_accept_tokens,
                )?;
                d.set_item("spec_num_cap_tokens", c.spec_num_cap_tokens)?;
                d.set_item(
                    "correct_drafts_histogram",
                    u64_list(py, &c.correct_drafts_histogram)?,
                )?;
                d.set_item("cap_lens_histogram", u64_list(py, &c.cap_lens_histogram)?)?;
                d.into_py_any(py)
            }
        }
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

// ------------------------------------------------- spec-v2 bookkeeping (M6)

/// Per-req spec counters (plan §9) — the `Req` spec fields + the two
/// growable histograms, driven from Python one settled step at a time.
#[pyclass]
struct SpecCounters(crate::spec::SpecCounters);

#[pymethods]
impl SpecCounters {
    #[new]
    fn new() -> Self {
        Self(crate::spec::SpecCounters::default())
    }

    /// One settled spec step: `correct_drafts` is the pre-grammar
    /// `accept_len - 1`; `block` / `cap` are the per-req lens
    /// (None when the batch column is absent).
    fn update(&mut self, correct_drafts: u32, block: Option<u32>, cap: Option<u32>) {
        self.0.update(correct_drafts, block, cap);
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let c = &self.0;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("spec_verify_ct", c.spec_verify_ct)?;
        d.set_item("spec_num_correct_drafts", c.spec_num_correct_drafts)?;
        d.set_item(
            "spec_num_block_accept_tokens",
            c.spec_num_block_accept_tokens,
        )?;
        d.set_item("spec_num_cap_tokens", c.spec_num_cap_tokens)?;
        d.set_item(
            "correct_drafts_histogram",
            u64_list(py, &c.correct_drafts_histogram)?,
        )?;
        d.set_item("cap_lens_histogram", u64_list(py, &c.cap_lens_histogram)?)?;
        d.into_py_any(py)
    }
}

/// Spec-v2 accept-run resolution (plan §9) — the CPU core of
/// `batch_result_processor._resolve_spec_v2_tokens`.
///
/// ```python
/// runs, num_correct_drafts, per_req, block_total, cap_total = \
///     resolve_spec_runs(
///         next_token_ids, stride, accept_lens,
///         retracted, finished,
///         grammar_retained,   # list[None | list[int]]
///         block_accept_lens,  # list[int] | None
///         cap_lens,           # list[int] | None
///     )
/// ```
///
/// `next_token_ids` is the flat stride-padded buffer (req `i`'s draft
/// slots are `[i * stride, i * stride + stride)`); `runs[i]` is the
/// committed run (grammar-truncated when a retained run is supplied;
/// empty for unsettled rows — retracted or pre-finished reqs).
#[pyfunction]
fn resolve_spec_runs(
    py: Python<'_>,
    next_token_ids: Vec<i64>,
    stride: u32,
    accept_lens: Vec<u32>,
    retracted: Vec<bool>,
    finished: Vec<bool>,
    grammar_retained: Option<Vec<Option<Vec<i64>>>>,
    block_accept_lens: Option<Vec<u32>>,
    cap_lens: Option<Vec<u32>>,
) -> PyResult<Py<PyAny>> {
    let n = accept_lens.len();
    if retracted.len() != n || finished.len() != n {
        return Err(PyValueError::new_err(
            "resolve_spec_runs: retracted/finished length != accept_lens",
        ));
    }
    if let Some(g) = &grammar_retained {
        if g.len() != n {
            return Err(PyValueError::new_err(
                "resolve_spec_runs: grammar_retained length != accept_lens",
            ));
        }
    }
    if let Some(b) = &block_accept_lens {
        if b.len() != n {
            return Err(PyValueError::new_err(
                "resolve_spec_runs: block_accept_lens length != accept_lens",
            ));
        }
    }
    if let Some(c) = &cap_lens {
        if c.len() != n {
            return Err(PyValueError::new_err(
                "resolve_spec_runs: cap_lens length != accept_lens",
            ));
        }
    }

    let rows: Vec<crate::spec::SpecRow> = (0..n)
        .map(|i| crate::spec::SpecRow {
            accept_len: accept_lens[i],
            retracted: retracted[i],
            finished: finished[i],
            grammar_retained: grammar_retained.as_ref().and_then(|g| g[i].clone()),
            block_accept_len: block_accept_lens.as_ref().map(|b| b[i]),
            cap_len: cap_lens.as_ref().map(|c| c[i]),
        })
        .collect();

    let res = py
        .detach(|| crate::spec::resolve_spec_runs(&next_token_ids, stride, &rows))
        .map_err(|e: crate::spec::SpecError| PyValueError::new_err(e.to_string()))?;

    let runs: Vec<Py<PyAny>> = res
        .runs
        .iter()
        .map(|r| int_list(py, &r.tokens))
        .collect::<PyResult<Vec<_>>>()?;
    let out = PyTuple::new(
        py,
        [
            runs.into_py_any(py)?,
            i64::from(res.num_correct_drafts).into_py_any(py)?,
            u32_list(py, &res.num_correct_drafts_per_req)?,
            i64::from(res.num_block_accept_tokens).into_py_any(py)?,
            i64::from(res.num_cap_tokens).into_py_any(py)?,
        ],
    )?;
    out.into_py_any(py)
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
    m.add("CT_FULL", sglang_radix::CT_FULL)?;
    m.add("CT_SWA", sglang_radix::CT_SWA)?;
    m.add("CT_MAMBA", sglang_radix::CT_MAMBA)?;
    m.add("CT_C128", sglang_radix::CT_C128)?;
    m.add("PHASE_BACKUP_HOST", sglang_radix::PHASE_BACKUP_HOST)?;
    m.add("PHASE_BACKUP_STORAGE", sglang_radix::PHASE_BACKUP_STORAGE)?;
    m.add("PHASE_LOAD_BACK", sglang_radix::PHASE_LOAD_BACK)?;
    m.add("PHASE_PREFETCH", sglang_radix::PHASE_PREFETCH)?;
    m.add_class::<SchedulerCore>()?;
    m.add_class::<SpecCounters>()?;
    m.add_class::<RadixTree>()?;
    m.add_class::<SWARadixTree>()?;
    m.add_class::<MambaRadixTree>()?;
    m.add_class::<HiRadixTree>()?;
    m.add_class::<UnifiedRadixTreePy>()?;
    m.add_function(wrap_pyfunction!(plan_next_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(ntr_next_after_decay, m)?)?;
    m.add_function(wrap_pyfunction!(ntr_estimate_after_retract, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_spec_runs, m)?)?;
    Ok(())
}
