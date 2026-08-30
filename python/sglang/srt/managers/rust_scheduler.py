"""Rust CPU control-plane driver (``SGLANG_RUST_SCHEDULER``).

Stages (additive; each implies the previous — see plan.md "Flag design"):

- ``radix``   — tree ops dual-run on the Rust tree (``mem_cache.rust_radix``);
- ``planner`` — the next-batch decision is shadowed in Rust and diffed
  against Python's actual decision (trace capture);
- ``core``    — queues / budgets / result bookkeeping run in a persistent
  ``SchedulerCore`` kept in lock-step with the Python scheduler (A/B);
- ``stream``  — reserved for Phase 4 (per-iteration payload building).

The default (``off``) makes everything here a no-op beyond one cheap
stage check per call. The Rust extension is loaded lazily and
fail-softly: if it cannot be built or loaded, the scheduler silently
falls back to the pure-Python path (a warning is logged once).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MODULE_NAME = "sglang.srt.rust_extensions._scheduler"
_STAGES = ("off", "radix", "planner", "core", "stream")
_RANK = {s: i for i, s in enumerate(_STAGES)}

_module: Any = None
_load_attempted = False
_load_warned = False


def rust_scheduler_stage() -> str:
    """The active stage, normalized and validated (``off`` on bad values)."""
    stage = (envs.SGLANG_RUST_SCHEDULER.get() or "off").strip().lower()
    return stage if stage in _STAGES else "off"


def stage_at_least(stage: str) -> bool:
    """True when the active stage is ``stage`` or a later stage."""
    return _RANK[rust_scheduler_stage()] >= _RANK[stage]


def load_module() -> Optional[Any]:
    """Lazily load the ``_scheduler`` extension; None when unavailable."""
    global _module, _load_attempted, _load_warned
    if _load_attempted:
        return _module
    _load_attempted = True
    try:
        from sglang.srt.rust_extensions import load_rust_extension

        _module = load_rust_extension(MODULE_NAME)
    except Exception:
        if not _load_warned:
            _load_warned = True
            logger.warning(
                "SGLANG_RUST_SCHEDULER=%s requested but the _scheduler Rust "
                "extension could not be loaded; falling back to Python. "
                "See SGLANG_RUST_BUILD_MODE for local builds.",
                rust_scheduler_stage(),
                exc_info=True,
            )
    return _module


# ------------------------------------------------------------------ config


def build_rust_config(server_args, tree_cache, *, page_size: int) -> Dict[str, Any]:
    """The full 20-key ``Config`` dict (the module's strict schema).

    Values mirror what the Python scheduler reads; missing attributes fall
    back to the Rust defaults (``Config::default`` in sglang-scheduler).
    """
    from sglang.srt.managers.schedule_policy import (
        IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD,
        IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD,
    )

    schedule = _get_schedule_config()
    return {
        "policy": getattr(server_args, "schedule_policy", "fcfs") or "fcfs",
        "page_size": page_size,
        "max_prefill_tokens": int(getattr(server_args, "max_prefill_tokens", 0) or 0)
        or 16384,
        "chunked_prefill_size": getattr(server_args, "chunked_prefill_size", None),
        "mixed_chunk": bool(getattr(server_args, "enable_mixed_chunk", False)),
        "priority_scheduling": bool(
            getattr(server_args, "enable_priority_scheduling", False)
        ),
        "low_priority_values_first": bool(
            getattr(server_args, "schedule_low_priority_values_first", False)
        ),
        "clip_max_new_tokens": int(
            envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get() or 4096
        ),
        "in_batch_check_threshold": int(IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD),
        "in_batch_deprioritize_threshold": int(
            IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
        ),
        "prefill_max_requests": schedule.get("prefill_max_requests"),
        "truncation_align_size": getattr(
            server_args, "truncation_align_size", None
        ),
        "lpm_queue_degrade_at": 128,
        "random_seed": int(getattr(server_args, "random_seed", 0) or 0),
        "disable_tree": bool(getattr(server_args, "disable_radix_cache", False)),
        "ntr_init_raw": float(envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()),
        "schedule_conservativeness": schedule.get(
            "schedule_conservativeness", 1.0
        ),
        "ntr_min_factor": float(envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get()),
        "ntr_decay_steps": int(envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()),
        "retract_decode_steps": int(envs.SGLANG_RETRACT_DECODE_STEPS.get()),
    }


_schedule_config_cache: Optional[Dict[str, Any]] = None


def _get_schedule_config() -> Dict[str, Any]:
    """Best-effort read of the dynamic schedule knobs (defaults on failure)."""
    global _schedule_config_cache
    if _schedule_config_cache is not None:
        return _schedule_config_cache
    try:
        from sglang.srt.utils import get_schedule

        sched = get_schedule()
        _schedule_config_cache = {
            "prefill_max_requests": getattr(
                sched, "prefill_max_requests", None
            ),
            "schedule_conservativeness": float(
                getattr(sched, "schedule_conservativeness", 1.0)
            ),
        }
    except Exception:
        _schedule_config_cache = {
            "prefill_max_requests": None,
            "schedule_conservativeness": 1.0,
        }
    return _schedule_config_cache


# ------------------------------------------------------------------ snapshots


def plan_req_dict(req, *, committed_len: int, prefix_len: int, last_node: int = 0) -> Dict[str, Any]:
    """One ``PlanReq`` dict (the module's strict schema: all keys present).

    ``pool_idx`` is ``0`` while the request holds no pool row yet — waiting
    reqs always (the row is allocated at batch construction) and hybrid-SSM
    reqs until their KV row lands; the core's ingress placeholder is 0 too,
    and the planner treats it as opaque.
    """
    sampling = req.sampling_params
    pool_idx = req.req_pool_idx
    return {
        "pool_idx": int(pool_idx) if pool_idx is not None else 0,
        "origin_len": len(req.origin_input_ids),
        "out_len": len(req.output_ids),
        "committed_len": int(committed_len),
        "prefix_len": int(prefix_len),
        "last_node": int(last_node),
        "priority": int(getattr(req, "priority", 0) or 0),
        "arrival_seq": 0,  # waiting-list order is the arrival order
        "max_new_tokens": int(getattr(sampling, "max_new_tokens", 0) or 0),
        "routing_key": 0,  # string routing keys are not used by the u64 policy
        "ignore_eos": bool(getattr(sampling, "ignore_eos", False)),
        "finished": bool(req.finished()),
        "retracted_stain": bool(getattr(req, "retracted", False)),
        "host_hit_length": 0,
    }


def step_env_dict(sched, running_batch) -> Dict[str, Any]:
    """The ``StepEnv`` dict (all five keys, CPU-mirror values only)."""
    running_bs = len(running_batch.reqs) if running_batch is not None else 0
    return {
        "allocator_avail_tokens": int(
            sched.token_to_kv_pool_allocator.available_size()
        ),
        "tree_evictable_tokens": int(sched.tree_cache.evictable_size()),
        "num_allocatable_reqs": int(sched.get_num_allocatable_reqs(running_bs)),
        "batch_is_full": bool(running_batch.batch_is_full)
        if running_batch is not None
        else False,
        "mixed_chunk_allowed": bool(
            getattr(sched.server_args, "enable_mixed_chunk", False)
        ),
    }


# ------------------------------------------------------------------- trace


class TraceRecorder:
    """Per-iteration JSONL capture (plan §4.2 lossless-replay backbone)."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self.iter = 0

    def record(self, **fields: Any) -> None:
        self.iter += 1
        fields["iter"] = self.iter
        fields["ts"] = time.time_ns()
        try:
            self._fh.write(json.dumps(fields, default=_jsonable) + "\n")
            self._fh.flush()  # crash-safe capture: lines must not sit buffered
        except Exception:
            logger.exception("rust scheduler trace write failed")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return repr(v)


# ---------------------------------------------------------------- shadow planner


class RustPlannerShadow:
    """Phase-2 A/B driver: shadow the Rust decision against Python's.

    The shadow call happens at the top of ``get_new_batch_prefill`` (after
    the last-batch merge, so the snapshot is exactly the pre-decision
    state). The diff is finalized where the Python decision is known
    (``get_next_batch_to_run``).

    Known soft-diff sources (documented in the trace as ``soft``):
    - LPM scores reuse the previous pass' per-req match
      (``num_matched_prefix_tokens``); the shadow does no extra tree
      matches because the Python ``match_prefix`` mutates LRU clocks and
      splits nodes — re-running it would desync the dual-written Rust
      tree.
    - The LPM in-batch-deprioritize set is not re-derived (it only
      reorders near-duplicate prefixes).
    - The routing-key policy is not shadowed (string keys vs u64).
    """

    def __init__(self, sched):
        self.sched = sched
        self.cfg = build_rust_config(
            sched.server_args,
            sched.tree_cache,
            page_size=sched.page_size,
        )
        self.trace: Optional[TraceRecorder] = None  # attach() wires the shared recorder
        self._pending: Optional[Tuple[Any, List[str], float]] = None
        self._iter = 0
        self.mismatches = 0
        self.soft_mismatches = 0
        logger.info("rust scheduler: planner shadow enabled (stage=%s)",
                    rust_scheduler_stage())

    # -- input snapshot ------------------------------------------------

    def _ntr(self) -> float:
        tracker = getattr(self.sched, "new_token_ratio_tracker", None)
        if tracker is None:
            return 0.7
        return float(tracker.current)

    def shadow(self, running_batch) -> None:
        """Run the Rust planner on the current pre-decision snapshot."""
        mod = load_module()
        if mod is None:
            return
        dm = getattr(self.sched, "disaggregation_mode", None)
        if dm is not None and "null" not in str(getattr(dm, "value", dm)).lower():
            return  # PD-disaggregated nodes are out of scope
        waiting = self.sched.waiting_queue
        running = running_batch.reqs if running_batch is not None else []
        chunked = self.sched.chunked_req

        waiting_dicts = [
            plan_req_dict(
                r,
                committed_len=len(r.origin_input_ids) + len(r.output_ids),
                prefix_len=int(getattr(r, "num_matched_prefix_tokens", 0) or 0),
            )
            for r in waiting
        ]
        running_dicts = [
            plan_req_dict(
                r,
                committed_len=int(getattr(r, "kv_committed_len", 0) or 0),
                prefix_len=len(r.prefix_indices) if r.prefix_indices is not None else 0,
            )
            for r in running
        ]
        chunked_dict = None
        if chunked is not None:
            chunked_dict = plan_req_dict(
                chunked,
                committed_len=int(
                    getattr(chunked, "kv_committed_len", 0)
                    or len(chunked.origin_input_ids) + len(chunked.output_ids)
                ),
                prefix_len=len(chunked.prefix_indices)
                if chunked.prefix_indices is not None
                else 0,
            )

        scores = [
            int(getattr(r, "num_matched_prefix_tokens", 0) or 0) for r in waiting
        ]
        deprio = [False] * len(waiting)
        env = step_env_dict(self.sched, running_batch)

        try:
            plan = mod.plan_next_batch(
                self.cfg,
                self._ntr(),
                waiting_dicts,
                running_dicts,
                chunked_dict,
                scores,
                deprio,
                env,
                self._iter,
            )
        except Exception:
            logger.exception("rust shadow planner failed; skipping this iteration")
            return

        self._iter += 1
        self._pending = (plan, [r.rid for r in waiting], time.perf_counter())

    # -- diff ----------------------------------------------------------

    def finalize(self, plan) -> None:
        """Diff the pending Rust plan against Python's decision (NextBatchPlan)."""
        if self._pending is None:
            return
        rust_plan, waiting_rids, _t0 = self._pending
        self._pending = None
        ret = plan.batch_to_run
        running_batch = plan.running_batch

        if ret is None:
            py_mode, py_rids, py_ext = "none", [], []
        else:
            if ret.forward_mode.is_extend():
                py_mode = "prefill"
                py_rids = [r.rid for r in ret.reqs]
                py_ext = [
                    [len(r.prefix_indices) if r.prefix_indices is not None else 0,
                     int(getattr(r, "extend_range", None) and r.extend_range.end
                         or len(r.origin_input_ids))]
                    for r in ret.reqs
                ]
            else:
                py_mode = "decode"
                py_rids = [r.rid for r in ret.reqs]
                py_ext = []

        mode, bif, prefill, decode = rust_plan
        if mode == 1:  # MODE_PREFILL
            rust_mode = "prefill"
            admitted, _chunked, _mixed, _ext_tok, _pages = prefill
            rust_rids = [waiting_rids[i] if i < len(waiting_rids) else -1
                         for i, _p, _es, _ee in admitted]
            rust_ext = [
                [plen, ee] for _wi, plen, _es, ee in admitted
            ]
        elif mode == 2:  # MODE_DECODE
            rust_mode = "decode"
            survivors, _fin, _retract, _abort, _evict, _pages, _ntr = decode
            running = running_batch.reqs if running_batch is not None else []
            rust_rids = [
                running[i].rid if i < len(running) else -1 for i in survivors
            ]
            rust_ext = []
        else:
            rust_mode = "none"
            rust_rids, rust_ext = [], []

        hard = rust_mode != py_mode
        soft = not hard
        if not hard:
            if rust_mode == "prefill" and rust_rids != py_rids:
                # LPM tie-breaks / stale scores reorder equal-budget admits.
                if sorted(rust_rids) == sorted(py_rids):
                    soft = True
                else:
                    hard = True
            elif rust_mode == "decode" and rust_rids != py_rids:
                hard = True
            elif rust_mode == "prefill" and py_ext and rust_ext != py_ext:
                soft = True  # extend-range drift comes from the score source
        if hard:
            self.mismatches += 1
        elif soft:
            self.soft_mismatches += 1

        if self.trace is not None:
            self.trace.record(
                stage=rust_scheduler_stage(),
                py={"mode": py_mode, "reqs": py_rids, "ext": py_ext,
                    "batch_is_full": bool(running_batch.batch_is_full)
                    if running_batch is not None else False},
                rust={"mode": rust_mode, "batch_is_full": bool(bif),
                      "admitted": [
                          [waiting_rids[i] if i < len(waiting_rids) else -1,
                           p, es, ee]
                          for i, p, es, ee in (prefill[0] if prefill else [])
                      ] if prefill else None,
                      "decode": list(decode) if decode else None},
                match=not hard,
                soft=soft,
                ntr=self._ntr(),
            )
        if hard:
            logger.debug(
                "rust planner hard mismatch: py=%s %s vs rust=%s %s",
                py_mode, py_rids, rust_mode, rust_rids,
            )

    def stats(self) -> Dict[str, Any]:
        return {
            "stage": rust_scheduler_stage(),
            "iterations": self._iter,
            "hard_mismatches": self.mismatches,
            "soft_mismatches": self.soft_mismatches,
        }


# ---------------------------------------------------------------- core driver


class RustCoreDriver:
    """Phase-3 bookkeeping driver: the core keeps lock-step state.

    The core ingests every request, applies every result, and plans every
    iteration; its plans and events are recorded in the trace (A/B).
    Python still executes the batches — and therefore still performs the
    frees / row writes the core's events describe. Applying those events
    here too would double-free, so in bookkeeping mode they are only
    traced. The cutover (Python executing from the core's plans and
    letting the driver apply the events) is behind
    ``SGLANG_RUST_CORE_APPLY=1`` and is not the default.

    ``SGLANG_RUST_CORE_VALUES=1`` traces exact KV index values (for
    lossless replay); by default zero-filled rows are traced — the core's
    planning only needs structure, sizes, and lock state, and copying
    full rows to the CPU on every decode step would defeat the point.
    """

    def __init__(self, sched):
        mod = load_module()
        assert mod is not None, "core stage requires the _scheduler module"
        self._mod = mod
        self.sched = sched
        self.cfg = build_rust_config(
            sched.server_args,
            sched.tree_cache,
            page_size=sched.page_size,
        )
        self.tree_policy = getattr(sched.tree_cache, "eviction_policy", "lru")
        self.core = mod.SchedulerCore(self.cfg, self.tree_policy)
        self.rid_to_core: Dict[str, int] = {}
        self.pool_idx: Dict[int, int] = {}  # core_idx -> real req pool idx
        self.core_apply = str(envs.SGLANG_RUST_CORE_APPLY.get()) == "1"
        self.exact_values = str(envs.SGLANG_RUST_CORE_VALUES.get()) == "1"
        self.trace: Optional[TraceRecorder] = None  # attach() wires the shared recorder
        self.arrival = 0
        self.mismatches = 0
        logger.info("rust scheduler: core bookkeeping enabled (apply=%s)", self.core_apply)

    def _rid_to(self, rid: str) -> Optional[int]:
        return self.rid_to_core.get(rid)

    def reset(self) -> None:
        """Recreate the core and clear per-request bookkeeping.

        Test-only: the live replay A/B (plan §4.2) re-drives a session
        after flushing the Python engine state, so the core must come
        back from a clean tree to stay in lock-step with the flushed
        Python tree cache.
        """
        self.core = self._mod.SchedulerCore(self.cfg, self.tree_policy)
        self.rid_to_core.clear()
        self.pool_idx.clear()
        self.arrival = 0

    def _trace_events(self, events, **extra) -> None:
        if self.trace is not None and events:
            rec = {"kind": "core", "events": [_jsonable(e) for e in events]}
            rec.update(extra)
            self.trace.record(**rec)

    def on_ingress(self, req) -> None:
        if req.rid in self.rid_to_core:
            return
        origin = [int(t) for t in req.origin_input_ids]
        # pool_idx is allocated later (batch construction); 0 is a
        # placeholder — the real index is tracked in self.pool_idx.
        result = self.core.ingest(
            [
                {
                    "rid": _stable_rid(req.rid),
                    "pool_idx": 0,
                    "origin": origin,
                    "max_new_tokens": int(
                        getattr(req.sampling_params, "max_new_tokens", 0) or 0
                    ),
                    "priority": int(getattr(req, "priority", 0) or 0),
                    "arrival_seq": self.arrival,
                    "routing_key": 0,
                    "ignore_eos": bool(
                        getattr(req.sampling_params, "ignore_eos", False)
                    ),
                }
            ]
        )
        self.arrival += 1
        self.rid_to_core[req.rid] = int(result[0])
        if self.trace is not None:
            self.trace.record(
                kind="ingress",
                rid=req.rid,
                origin=self._origin_for_trace(origin),
                origin_len=len(origin),
                max_new_tokens=int(
                    getattr(req.sampling_params, "max_new_tokens", 0) or 0
                ),
                priority=int(getattr(req, "priority", 0) or 0),
                ignore_eos=bool(
                    getattr(req.sampling_params, "ignore_eos", False)
                ),
                arrival_seq=self.arrival - 1,
            )

    def _origin_for_trace(self, origin: List[int]) -> Any:
        """Raw token ids by default; `SGLANG_TRACE_SCHEDULER_TOKENS=hash`
        stores a sha256 fingerprint + length instead (keeps captures small
        and content-free, plan §4.2)."""
        mode = str(envs.SGLANG_TRACE_SCHEDULER_TOKENS.get()).lower()
        if mode == "hash":
            import hashlib

            return hashlib.sha256(
                "".join(str(t) for t in origin).encode()
            ).hexdigest()
        return origin

    def on_abort(self, rid: str) -> None:
        """A waiting request was removed from the queue (abort)."""
        core_idx = self._rid_to(rid)
        if core_idx is None:
            return
        events = self.core.drop(core_idx)
        if self.core_apply:
            for event in events:
                apply_event(self.sched, event)
        else:
            # rid so the replay feeder knows which request to drop.
            self._trace_events(events, op="drop", rid=rid)
        self.rid_to_core.pop(rid, None)

    def apply_result(self, batch, result=None) -> None:
        """Fold one executed batch's results into the core."""
        rows, kv_rows, rids, kv_lens = [], [], [], []
        req_to_token = self.sched.req_to_token_pool.req_to_token
        is_decode = batch.forward_mode.is_decode()
        # Spec-v2 (plan §9): the result processor (Rust branch) records the
        # resolved accepted runs + the pre-step settle gate on `result`.
        is_spec = is_decode and not batch.spec_algorithm.is_none()
        spec_runs = getattr(result, "resolved_spec_runs", None)
        spec_settled = getattr(result, "resolved_spec_settled", None)
        per_req_cpu = getattr(result, "num_correct_drafts_per_req_cpu", None)
        spec_block = getattr(result, "resolved_spec_block_accept_lens", None)
        spec_cap = getattr(result, "resolved_spec_cap_lens", None)
        for i, req in enumerate(batch.reqs):
            core_idx = self._rid_to(req.rid)
            if core_idx is None:
                continue
            if req.req_pool_idx is not None:
                self.pool_idx[core_idx] = int(req.req_pool_idx)
            spec_meta: Optional[Dict[str, Any]] = None
            if is_spec and spec_runs is not None and i < len(spec_runs):
                settled = bool(spec_settled[i]) if spec_settled is not None else True
                # Settle mirrors Python: retracted / pre-finished rows
                # committed nothing (the raw slice is dead data downstream).
                accepted = list(spec_runs[i]) if settled else []
                accept_len = (
                    int(per_req_cpu[i]) + 1
                    if per_req_cpu is not None
                    else len(accepted)
                )
                spec_meta = {
                    "accept_len": accept_len,
                    "settled": settled,
                    "block_accept_len": (
                        int(spec_block[i]) if spec_block is not None else None
                    ),
                    "cap_len": int(spec_cap[i]) if spec_cap is not None else None,
                }
            else:
                accepted = [int(req.output_ids[-1])] if req.output_ids else []
            rids.append(req.rid)
            rows.append(
                {
                    "accepted": accepted,
                    "finished": bool(req.finished()),
                    "finish_reason": _finish_reason_int(req),
                    "spec": spec_meta,
                }
            )
            kv_len: Optional[int] = None
            if req.req_pool_idx is not None:
                if is_decode:
                    fill = len(req.origin_input_ids) + len(req.output_ids)
                else:
                    # Extend / mixed: every req carries a fresh
                    # extend_range (init_next_round_input / mix_with_running);
                    # the token sampled this step is not in the row yet.
                    er = getattr(req, "extend_range", None)
                    fill = (
                        int(er.end)
                        if er is not None
                        else len(req.origin_input_ids) + len(req.output_ids) - 1
                    )
                kv_len = fill
                if self.exact_values:
                    values = [int(x) for x in req_to_token[req.req_pool_idx, :fill].tolist()]
                else:
                    values = [0] * fill
                kv_rows.append({"core_idx": core_idx, "row": values})
            kv_lens.append(kv_len)
        if rows:
            events = self.core.apply_result(rows, kv_rows)
            if self.core_apply:
                for event in events:
                    apply_event(self.sched, event)
            if self.trace is not None:
                self.trace.record(
                    kind="core",
                    op="apply_result",
                    rids=rids,
                    # The accepted token VALUES (not just the count): they are
                    # the tree-key tail (origin + out) the replayer needs to
                    # reproduce prefix matches.
                    result=[[list(r["accepted"]), r["finished"],
                             r["finish_reason"], r.get("spec")] for r in rows],
                    # Per-row committed KV fill length (None = no pool row);
                    # the replay feeder rebuilds zero-filled kv_rows from it.
                    kv_lens=kv_lens,
                    events=[_jsonable(e) for e in events],
                )

    def plan(self, running_batch) -> None:
        """Plan this iteration in the core (state + trace).

        Events are NOT applied in bookkeeping mode: Python performs the
        same frees/evictions as part of executing its own batch.
        """
        env = step_env_dict(self.sched, running_batch)
        plan, events = self.core.plan(env)
        if self.core_apply:
            for event in events:
                apply_event(self.sched, event)
        if self.trace is not None:
            self.trace.record(
                kind="core",
                op="plan",
                plan=_jsonable(plan),
                env=_jsonable(env),
                events=[_jsonable(e) for e in events],
            )


def _stable_rid(rid: str) -> int:
    """A deterministic u64 for a request id (string -> stable hash)."""
    import hashlib

    return int.from_bytes(
        hashlib.blake2b(rid.encode(), digest_size=8).digest(), "big"
    )


def _finish_reason_int(req) -> int:
    reason = getattr(req, "finished_reason", None)
    if reason is None:
        return 0
    try:
        return int(reason.value)
    except AttributeError:
        try:
            return int(reason)
        except (TypeError, ValueError):
            return 0


def _kv_tensor(sched, values) -> Any:
    import torch

    dev = getattr(sched.req_to_token_pool, "device", None)
    if dev is None:
        dev = getattr(sched, "device", "cuda")
    return torch.tensor([int(v) for v in values], dtype=torch.int64, device=dev)


def apply_event(sched, event: Tuple[Any, ...]) -> None:
    """Execute one core event against the Python allocator / row pool.

    Only called in core-apply (cutover) mode; in bookkeeping mode the
    events are traced instead.
    """
    kind = event[0]
    if kind == "evict":
        for run in event[1]:
            if run:
                sched.token_to_kv_pool_allocator.free_segment(
                    _kv_tensor(sched, run), start_pos=0
                )
    elif kind == "free_segments":
        pool_idx, ranges = event[1], event[2]
        row = sched.req_to_token_pool.req_to_token[pool_idx]
        for start, end in ranges:
            if end > start:
                sched.token_to_kv_pool_allocator.free_segment(
                    row[start:end], start_pos=start
                )
    elif kind == "stash_row_write":
        pool_idx, start, new_indices = event[1], event[2], event[3]
        sched.req_to_token_pool.req_to_token[
            pool_idx, start : start + len(new_indices)
        ] = _kv_tensor(sched, new_indices)
    elif kind == "finished":
        # Core-mode abort notices: the Python side already streams finished
        # reqs through the normal result path; nothing to do for now.
        pass


# -------------------------------------------------------------------- attach


def attach(sched) -> Dict[str, Any]:
    """Attach the active drivers to a ``Scheduler``. Returns their handles."""
    stage = rust_scheduler_stage()
    drivers: Dict[str, Any] = {"stage": stage}
    mod = load_module()
    if _RANK[stage] >= _RANK["planner"] and mod is not None:
        drivers["shadow"] = RustPlannerShadow(sched)
    if _RANK[stage] >= _RANK["core"] and mod is not None:
        try:
            drivers["core"] = RustCoreDriver(sched)
        except Exception:
            logger.exception("failed to attach the Rust core driver")
    trace_path = envs.SGLANG_TRACE_SCHEDULER.get()
    if trace_path:
        recorder = TraceRecorder(trace_path)
        for driver in drivers.values():
            if isinstance(driver, (RustPlannerShadow, RustCoreDriver)):
                driver.trace = recorder
        core = drivers.get("core")
        if isinstance(core, RustCoreDriver):
            # Session header: the replay feeder (scripted_runtime/replay.py)
            # rebuilds the exact core config from this line.
            recorder.record(
                kind="cfg",
                cfg=core.cfg,
                tree_policy=core.tree_policy,
                stage=stage,
            )
    return drivers
