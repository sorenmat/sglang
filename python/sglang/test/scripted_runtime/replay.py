"""Lossless scheduler-trace replay (plan §4.2 — the correctness backbone).

A capture written by ``SGLANG_TRACE_SCHEDULER=<path>`` (see
``sglang.srt.managers.rust_scheduler.TraceRecorder``) is a JSONL stream of
the Rust core driver's inputs and outputs::

    {"kind": "cfg", "cfg": {...20-key Config...}, "tree_policy": "lru", ...}
    {"kind": "ingress", "rid": "...", "origin": [token, ...],
     "origin_len": N, "max_new_tokens": N, "priority": 0,
     "ignore_eos": false, "arrival_seq": K, ...}
    {"kind": "core", "op": "plan", "plan": [...], "env": {...},
     "events": [...], ...}
    {"kind": "core", "op": "apply_result", "rids": [...],
     "result": [[accepted, finished, finish_reason, spec_meta | None], ...],
     "kv_lens": [fill_or_null, ...], "events": [...], ...}
    {"kind": "core", "op": "drop", "rid": "...", "events": [...], ...}

Shadow-planner lines (``stage`` + ``py``/``rust`` payload) are part of the
capture but are not core operations; the replayer skips them.

``replay_core`` feeds the recorded ingress/plan/apply_result/drop sequence
into a **fresh** ``SchedulerCore`` and diffs every recomputed plan against
the recorded one, field for field. Replaying a lossless trace through the
same engine must reproduce the original plans exactly — that is the
backbone every phase gate builds on:

- a capture that fails to replay proves a fidelity hole in the trace
  schema (a field the plan depends on was not recorded);
- a capture that replays cleanly proves the Rust engine is deterministic
  over that session, so the recorded Python-vs-Rust diffs (the shadow
  lines) are the complete disagreement report;
- ``live_replay_script`` (target host) re-drives a *live* Python
  scheduler from the same ingress sequence with a mocked ``run_batch``
  and lets the attached Rust core replay in lock-step.

The offline replayer is torch-free: it only needs the ``_scheduler``
extension (built with ``cargo build -p sglang-scheduler --features python``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "load_module",
    "load_trace",
    "replay_core",
    "diff_plans",
    "diff_sessions",
    "plan_view",
    "plans_match",
    "sessions_match",
    "synthesize_session_trace",
    "stable_rid",
    "ReplayError",
    "TraceSession",
    "IngressOp",
    "PlanOp",
    "ApplyOp",
    "DropOp",
    "PlanStep",
    "ReplayResult",
    "live_replay_script",
]

_MODES = {0: "none", 1: "prefill", 2: "decode"}


class ReplayError(RuntimeError):
    """The trace could not be replayed (schema hole or state divergence)."""


# ------------------------------------------------------------------- module


_MOD: Optional[Any] = None


def load_module() -> Any:
    """Load the ``_scheduler`` PyO3 module.

    Prefers the package loader (``load_rust_extension``) when the sglang
    package is importable; otherwise loads the built cdylib directly so the
    replayer works on hosts without torch/numpy. The cdylib must have been
    built with the ``python`` feature (``cargo build -p sglang-scheduler
    --features python``) or it lacks the module init symbol.
    """
    global _MOD
    if _MOD is not None:
        return _MOD
    try:
        from sglang.srt.rust_extensions import load_rust_extension

        _MOD = load_rust_extension("sglang.srt.rust_extensions._scheduler")
        return _MOD
    except Exception:
        pass

    import importlib.util

    so = os.environ.get("SGLANG_SCHEDULER_SO") or str(
        Path(__file__).resolve().parents[4]
        / "rust"
        / "target"
        / "debug"
        / "libsglang_scheduler.so"
    )
    if not os.path.exists(so):
        raise ReplayError(
            f"the _scheduler extension is unavailable and {so} does not exist; "
            "build it with `cargo build -p sglang-scheduler --features python` "
            "or set SGLANG_SCHEDULER_SO"
        )
    spec = importlib.util.spec_from_file_location("_scheduler", so)
    if spec is None or spec.loader is None:
        raise ReplayError(f"cannot create an import spec for {so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MOD = module
    return module


def stable_rid(rid: str) -> int:
    """The u64 rid the core driver hashes request ids to (must match
    ``rust_scheduler._stable_rid``)."""
    return int.from_bytes(
        hashlib.blake2b(rid.encode(), digest_size=8).digest(), "big"
    )


# ------------------------------------------------------------- trace schema


@dataclass(frozen=True)
class IngressOp:
    rid: str
    origin: Tuple[int, ...]
    max_new_tokens: int
    priority: int
    ignore_eos: bool
    arrival_seq: int


@dataclass(frozen=True)
class PlanOp:
    plan: Any  # recorded JSON view of the raw pybind plan
    env: Dict[str, Any]
    events: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class ApplyOp:
    rids: Tuple[str, ...]
    # [[accepted, finished, finish_reason, spec_meta | None], ...] — the
    # 4th element (spec-v2 metadata, plan §9) is absent in pre-M6 captures.
    result: List[Any]
    kv_lens: List[Optional[int]]
    events: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class DropOp:
    rid: str
    events: List[Any] = field(default_factory=list)


@dataclass
class TraceSession:
    cfg: Dict[str, Any]
    tree_policy: str
    ops: List[Any] = field(default_factory=list)
    n_shadow_lines: int = 0


def load_trace(path: str) -> TraceSession:
    """Parse a JSONL capture into an ordered op sequence.

    Raises :class:`ReplayError` when the session has no ``cfg`` header (the
    capture predates the replay schema or the core stage was off).
    """
    cfg: Dict[str, Any] = {}
    tree_policy = "lru"
    ops: List[Any] = []
    n_shadow = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ReplayError(f"{path}:{line_no}: bad JSON line: {e}") from e
            kind = rec.get("kind")
            if kind == "cfg":
                if cfg:
                    raise ReplayError(
                        f"{path}:{line_no}: duplicate cfg header line"
                    )
                cfg = rec.get("cfg") or {}
                tree_policy = str(rec.get("tree_policy") or "lru")
            elif kind == "ingress":
                origin = rec.get("origin")
                if not isinstance(origin, (list, tuple)):
                    raise ReplayError(
                        f"{path}:{line_no}: ingress origin is not raw token ids "
                        "(capture taken with SGLANG_TRACE_SCHEDULER_TOKENS=hash "
                        "is not live-replayable)"
                    )
                ops.append(
                    IngressOp(
                        rid=str(rec["rid"]),
                        origin=tuple(int(t) for t in origin),
                        max_new_tokens=int(rec.get("max_new_tokens") or 0),
                        priority=int(rec.get("priority") or 0),
                        ignore_eos=bool(rec.get("ignore_eos") or False),
                        arrival_seq=int(rec.get("arrival_seq") or 0),
                    )
                )
            elif kind == "core":
                op = rec.get("op")
                if op == "plan":
                    ops.append(
                        PlanOp(
                            plan=rec.get("plan"),
                            env=dict(rec.get("env") or {}),
                            events=list(rec.get("events") or []),
                        )
                    )
                elif op == "apply_result":
                    rids = tuple(str(r) for r in (rec.get("rids") or []))
                    result = list(rec.get("result") or [])
                    kv_lens = list(rec.get("kv_lens") or [None] * len(result))
                    if len(rids) != len(result) or len(rids) != len(kv_lens):
                        raise ReplayError(
                            f"{path}:{line_no}: apply_result rids/result/kv_lens "
                            f"misaligned ({len(rids)}/{len(result)}/{len(kv_lens)})"
                        )
                    ops.append(
                        ApplyOp(
                            rids=rids,
                            result=result,
                            kv_lens=kv_lens,
                            events=list(rec.get("events") or []),
                        )
                    )
                elif op == "drop":
                    ops.append(
                        DropOp(
                            rid=str(rec.get("rid") or ""),
                            events=list(rec.get("events") or []),
                        )
                    )
                else:
                    raise ReplayError(f"{path}:{line_no}: unknown core op {op!r}")
            elif "stage" in rec and ("py" in rec or "rust" in rec):
                n_shadow += 1  # shadow-planner diff line; not a core op
            # anything else: ignore (forward compatibility)
    if not cfg:
        raise ReplayError(
            f"{path}: no cfg header line — capture the session with "
            "SGLANG_RUST_SCHEDULER=core SGLANG_TRACE_SCHEDULER=<path>"
        )
    return TraceSession(cfg=cfg, tree_policy=tree_policy, ops=ops, n_shadow_lines=n_shadow)


# ----------------------------------------------------------- plan normalize


def plan_view(plan: Any) -> Dict[str, Any]:
    """Normalize a raw pybind plan (or its recorded JSON view) to a dict.

    Shape: ``{"mode", "batch_is_full", "prefill", "decode"}`` where
    ``prefill = {"admitted": [(waiting_idx, prefix_len, extend_start,
    extend_end), ...], "chunked", "mixed", "extend_tokens",
    "alloc_extend_pages"}`` and
    ``decode = {"decode", "finished_removed", "retract", "abort",
    "evict_tokens", "alloc_decode_pages", "ntr"}`` (absent when None).
    """
    mode, batch_is_full, prefill, decode = plan
    view: Dict[str, Any] = {
        "mode": int(mode),
        "batch_is_full": bool(batch_is_full),
        "prefill": None,
        "decode": None,
    }
    if prefill is not None:
        admitted, chunked, mixed, extend_tokens, alloc_extend_pages = prefill
        view["prefill"] = {
            "admitted": [
                tuple(int(x) for x in a) for a in (admitted or [])
            ],
            "chunked": int(chunked),
            "mixed": bool(mixed),
            "extend_tokens": int(extend_tokens),
            "alloc_extend_pages": int(alloc_extend_pages),
        }
    if decode is not None:
        (
            dec,
            finished_removed,
            retract,
            abort,
            evict_tokens,
            alloc_decode_pages,
            ntr,
        ) = decode
        view["decode"] = {
            "decode": [int(x) for x in (dec or [])],
            "finished_removed": [int(x) for x in (finished_removed or [])],
            "retract": [int(x) for x in (retract or [])],
            "abort": [int(x) for x in (abort or [])],
            "evict_tokens": int(evict_tokens),
            "alloc_decode_pages": int(alloc_decode_pages),
            "ntr": float(ntr),
        }
    return view


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _diff_lists(
    field: str, rec: List[Any], got: List[Any], out: List[Dict[str, Any]]
) -> None:
    if _same(rec, got):
        return
    out.append({"field": field, "recorded": rec, "recomputed": got})


def diff_plans(recorded: Any, recomputed: Any) -> List[Dict[str, Any]]:
    """Field-for-field diff of two plans (recorded JSON view vs raw pybind).

    Each entry is ``{"field", "recorded", "recomputed", "severity"}`` where
    ``severity`` is ``"hard"`` (a decision disagreement — the backbone
    signal) or ``"soft"`` (the shadow driver's tolerated drift: prefill
    admission reordering of the same set, extend-range drift, and ntr float
    noise). An empty list means the plans are identical.
    """
    r = plan_view(recorded)
    g = plan_view(recomputed)
    diffs: List[Dict[str, Any]] = []

    def add(field: str, rec: Any, got: Any, severity: str = "hard") -> None:
        if not _same(rec, got):
            diffs.append(
                {"field": field, "recorded": rec, "recomputed": got, "severity": severity}
            )

    add("mode", r["mode"], g["mode"])
    add("batch_is_full", r["batch_is_full"], g["batch_is_full"])

    rp, gp = r["prefill"], g["prefill"]
    if (rp is None) != (gp is None):
        add("prefill.presence", rp is not None, gp is not None)
    elif rp is not None:
        ra, ga = rp["admitted"], gp["admitted"]
        if not _same(ra, ga):
            rpos = [a[0] for a in ra]
            gpos = [a[0] for a in ga]
            # Same admitted set, different order (LPM tie-breaks) or only
            # extend-range drift: soft. A different set: hard.
            severity = "soft" if sorted(rpos) == sorted(gpos) else "hard"
            diffs.append(
                {"field": "prefill.admitted", "recorded": ra, "recomputed": ga,
                 "severity": severity}
            )
        add("prefill.chunked", rp["chunked"], gp["chunked"])
        add("prefill.mixed", rp["mixed"], gp["mixed"])
        add("prefill.extend_tokens", rp["extend_tokens"], gp["extend_tokens"])
        add("prefill.alloc_extend_pages", rp["alloc_extend_pages"], gp["alloc_extend_pages"])

    rd, gd = r["decode"], g["decode"]
    if (rd is None) != (gd is None):
        add("decode.presence", rd is not None, gd is not None)
    elif rd is not None:
        add("decode.decode", rd["decode"], gd["decode"])
        add("decode.finished_removed", rd["finished_removed"], gd["finished_removed"])
        add("decode.retract", rd["retract"], gd["retract"])
        add("decode.abort", rd["abort"], gd["abort"])
        add("decode.evict_tokens", rd["evict_tokens"], gd["evict_tokens"])
        add("decode.alloc_decode_pages", rd["alloc_decode_pages"], gd["alloc_decode_pages"])
        add("decode.ntr", rd["ntr"], gd["ntr"], severity="soft")

    return diffs


def plans_match(recorded: Any, recomputed: Any) -> bool:
    """True when the plans agree on every hard field."""
    return not any(d["severity"] == "hard" for d in diff_plans(recorded, recomputed))


# ----------------------------------------------------------------- replayer


@dataclass
class PlanStep:
    index: int
    recorded_plan: Any
    recomputed_plan: Any
    diffs: List[Dict[str, Any]]
    recorded_events: List[Any]
    recomputed_events: List[Any]
    tree_stats: Tuple[int, int, int]

    @property
    def ok(self) -> bool:
        return not any(d["severity"] == "hard" for d in self.diffs)


@dataclass
class ReplayResult:
    n_ops: int
    n_plans: int
    steps: List[PlanStep]
    final_tree_stats: Tuple[int, int, int]
    final_ntr: float
    # rid -> the replayed core's spec-v2 counters (plan §9). Empty when no
    # live req is spec-carrying; lets the test assert the spec metadata
    # round-tripped through the trace.
    final_spec_counters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def all_plans_ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def hard_diffs(self) -> List[Tuple[int, Dict[str, Any]]]:
        out: List[Tuple[int, Dict[str, Any]]] = []
        for s in self.steps:
            for d in s.diffs:
                if d["severity"] == "hard":
                    out.append((s.index, d))
        return out


def replay_core(
    trace: TraceSession,
    *,
    mod: Optional[Any] = None,
) -> ReplayResult:
    """Replay a session through a fresh ``SchedulerCore``.

    Every recorded plan line is recomputed and diffed field-for-field
    against the capture. Raises :class:`ReplayError` on a schema hole or a
    hard state divergence (e.g. the replayer's core rejects a recorded op —
    which means the captured state and the replayed state diverged before
    that point).
    """
    mod = mod or load_module()
    core = mod.SchedulerCore(trace.cfg, trace.tree_policy)
    rid_to_core: Dict[str, int] = {}
    steps: List[PlanStep] = []
    plan_index = 0

    for op in trace.ops:
        if isinstance(op, IngressOp):
            result = core.ingest(
                [
                    {
                        "rid": stable_rid(op.rid),
                        "pool_idx": 0,  # the driver ingests with a 0 placeholder
                        "origin": list(op.origin),
                        "max_new_tokens": op.max_new_tokens,
                        "priority": op.priority,
                        "arrival_seq": op.arrival_seq,
                        "routing_key": 0,
                        "ignore_eos": op.ignore_eos,
                    }
                ]
            )
            rid_to_core[op.rid] = int(result[0])
        elif isinstance(op, PlanOp):
            out = core.plan(op.env)
            plan, got_events = out[0], out[1]
            diffs = diff_plans(op.plan, plan)
            hard = [d for d in diffs if d["severity"] == "hard"]
            if hard:
                # A hard disagreement means the recorded decision was not
                # reproduced; the recorded apply (built from the original
                # last_batch) would desync the core, so stop cleanly rather
                # than panic inside the engine.
                raise ReplayError(
                    f"plan divergence at step {plan_index}: {hard[0]['field']} "
                    f"recorded={hard[0]['recorded']} recomputed={hard[0]['recomputed']}"
                )
            steps.append(
                PlanStep(
                    index=plan_index,
                    recorded_plan=op.plan,
                    recomputed_plan=plan,
                    diffs=diffs,
                    recorded_events=op.events,
                    recomputed_events=_jsonable(got_events),
                    tree_stats=tuple(core.tree_stats()),
                )
            )
            plan_index += 1
        elif isinstance(op, ApplyOp):
            rows = [
                {
                    "accepted": [int(t) for t in row[0]],
                    "finished": bool(row[1]),
                    "finish_reason": int(row[2]) if row[2] else 0,
                    # Spec-v2 metadata (plan §9); absent in pre-M6 captures.
                    "spec": (
                        dict(row[3])
                        if len(row) > 3 and row[3] is not None
                        else None
                    ),
                }
                for row in op.result
            ]
            kv_rows = []
            for rid, kv_len in zip(op.rids, op.kv_lens):
                core_idx = rid_to_core.get(rid)
                if core_idx is not None and kv_len is not None:
                    # KV values are opaque to the planner (only structure,
                    # sizes and locks matter) — zero-filled rows suffice.
                    kv_rows.append({"core_idx": core_idx, "row": [0] * int(kv_len)})
            got_events = _jsonable(core.apply_result(rows, kv_rows))
            if got_events != op.events:
                raise ReplayError(
                    f"apply_result state divergence: replayed events "
                    f"{got_events} != recorded {op.events}"
                )
        elif isinstance(op, DropOp):
            core_idx = rid_to_core.pop(op.rid, None)
            if core_idx is None:
                if op.events:
                    raise ReplayError(
                        f"drop of unknown rid {op.rid!r} recorded events "
                        f"{op.events}"
                    )
                continue
            got_events = _jsonable(core.drop(core_idx))
            if got_events != op.events:
                raise ReplayError(
                    f"drop state divergence: replayed events {got_events} "
                    f"!= recorded {op.events}"
                )

    final_spec_counters: Dict[str, Dict[str, Any]] = {}
    for rid, idx in rid_to_core.items():
        sc = core.spec_counters(int(idx))
        if sc is not None:
            final_spec_counters[rid] = _jsonable(sc)
    return ReplayResult(
        n_ops=len(trace.ops),
        n_plans=plan_index,
        steps=steps,
        final_tree_stats=tuple(core.tree_stats()),
        final_ntr=core.new_token_ratio(),
        final_spec_counters=final_spec_counters,
    )


# ------------------------------------------------------- reference session


def _jsonable(v: Any) -> Any:
    """Best-effort JSON view of a raw pybind output (tuples -> lists)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def synthesize_session_trace(
    mod: Any,
    *,
    path: str,
    max_prefill_tokens: int = 64,
) -> TraceSession:
    """Drive a fresh ``SchedulerCore`` through a fixed reference session and
    record it in the driver's JSONL schema at ``path``. Returns the parsed
    :class:`TraceSession` (via :func:`load_trace` on the written file).

    This is the test-support half of the correctness backbone: a trace the
    engine itself produced must replay cleanly through a *second* core
    instance (``replay_core``). It exercises prefill admission with a
    shared prefix, decode, a forced finish (tree insert + release), a
    running-request abort, and the idle tail — all without a live engine,
    so it runs on hosts that only have the built ``.so``.

    The session (page_size=1, fcfs, chunked off, mixed off):

    - iter 0:  ``rA`` (20 distinct tokens) ingested + prefilled;
    - iter 1:  ``rB`` (shares ``rA``'s first 12 tokens) + ``rC`` ingested;
    - decodes: one sampled token per running req per step, except iter 3
      where ``rA`` takes a spec-v2 row (2 accepted tokens, accept_len 3);
    - ``rC`` finishes after 2 decode steps, ``rA`` after 3 (length);
    - ``rB`` is aborted after iter 4's result is folded in;
    - remaining iterations are the idle tail.
    """
    cfg = {
        "policy": "fcfs",
        "page_size": 1,
        "max_prefill_tokens": int(max_prefill_tokens),
        "chunked_prefill_size": None,
        "mixed_chunk": False,
        "priority_scheduling": False,
        "low_priority_values_first": False,
        "clip_max_new_tokens": 4096,
        "in_batch_check_threshold": 32,
        "in_batch_deprioritize_threshold": 32,
        "prefill_max_requests": None,
        "truncation_align_size": None,
        "lpm_queue_degrade_at": 128,
        "random_seed": 0,
        "disable_tree": False,
        "ntr_init_raw": 0.7,
        "schedule_conservativeness": 1.0,
        "ntr_min_factor": 0.14,
        "ntr_decay_steps": 604,
        "retract_decode_steps": 20,
    }
    env = {
        "allocator_avail_tokens": 100_000,
        "tree_evictable_tokens": 0,  # the core overrides with the live size
        "num_allocatable_reqs": 1000,
        "batch_is_full": False,  # the core overrides with its internal flag
        "mixed_chunk_allowed": False,
    }
    tree_policy = "lru"

    origins = {
        "rA": list(range(1000, 1020)),
        "rB": list(range(1000, 1012)) + list(range(2000, 2008)),
        "rC": list(range(3000, 3008)),
    }
    max_new = {"rA": 16, "rB": 16, "rC": 8}
    ingest_at = {0: ["rA"], 1: ["rB", "rC"]}
    finish_after_decodes = {"rA": 3, "rC": 2}
    drop_after_iter = {4: "rB"}
    # Spec-v2 step (plan §9): iter 3 rA accepts 2 tokens (grammar-truncated
    # from accept_len 3), exercising the spec row metadata round-trip.
    spec_decode_iter = {3: "rA"}
    n_iters = 10

    core = mod.SchedulerCore(cfg, tree_policy)
    lines: List[Dict[str, Any]] = [
        {"kind": "cfg", "cfg": cfg, "tree_policy": tree_policy,
         "stage": "core", "iter": 0, "ts": 0}
    ]
    rid_to_core: Dict[str, int] = {}
    core_rid_to_str: Dict[str, str] = {str(stable_rid(r)): r for r in origins}
    origin_len: Dict[str, int] = {}
    decodes_seen: Dict[str, int] = {r: 0 for r in origins}
    finished: set = set()

    def _emit(rec: Dict[str, Any]) -> None:
        rec.setdefault("iter", len(lines))
        rec.setdefault("ts", 0)
        lines.append(rec)

    for it in range(n_iters):
        for rid in ingest_at.get(it, []):
            idx = int(core.ingest([
                {
                    "rid": stable_rid(rid),
                    "pool_idx": 0,
                    "origin": list(origins[rid]),
                    "max_new_tokens": int(max_new[rid]),
                    "priority": 0,
                    "arrival_seq": it,
                    "routing_key": 0,
                    "ignore_eos": False,
                }
            ])[0])
            rid_to_core[rid] = idx
            origin_len[rid] = len(origins[rid])
            _emit({
                "kind": "ingress", "rid": rid, "origin": list(origins[rid]),
                "origin_len": len(origins[rid]), "max_new_tokens": int(max_new[rid]),
                "priority": 0, "ignore_eos": False, "arrival_seq": it,
            })

        out = core.plan(env)
        plan, plan_events = out[0], out[1]
        _emit({"kind": "core", "op": "plan",
               "plan": _jsonable(plan), "env": dict(env),
               "events": _jsonable(plan_events)})

        lb = list(core.last_batch())
        if lb:
            mode = int(plan[0])
            pre_entries = (
                list(plan[2][0]) if (mode == 1 and plan[2] is not None) else []
            )
            rids: List[str] = []
            rows: List[Dict[str, Any]] = []
            kv_lens: List[int] = []
            for i, core_idx in enumerate(lb):
                rid = core_rid_to_str[str(core.req_rid(core_idx))]
                tok = 5000 + it
                is_spec_row = spec_decode_iter.get(it) == rid
                n_accept = 2 if is_spec_row else 1
                if mode == 1:  # prefill admit: fill = the extend_end
                    fill = int(pre_entries[i][3])
                else:  # decode: origin + out + the accepted run
                    fill = (
                        origin_len[rid] + int(core.req_out_len(core_idx))
                        + n_accept
                    )
                is_fin = (
                    mode == 2
                    and rid in finish_after_decodes
                    and decodes_seen[rid] >= finish_after_decodes[rid]
                )
                if mode == 2:
                    decodes_seen[rid] += 1
                if is_fin:
                    finished.add(rid)
                rows.append({
                    "accepted": [tok, tok + 1][:n_accept],
                    "finished": bool(is_fin),
                    "finish_reason": 1 if is_fin else 0,
                    "spec": (
                        {"accept_len": 3, "settled": True,
                         "block_accept_len": None, "cap_len": None}
                        if is_spec_row else None
                    ),
                })
                kv_lens.append(fill)
                rids.append(rid)
            apply_events = core.apply_result(
                rows,
                [
                    {"core_idx": int(ci), "row": [0] * fl}
                    for ci, fl in zip(lb, kv_lens)
                ],
            )
            _emit({
                "kind": "core", "op": "apply_result",
                "rids": rids,
                "result": [[r["accepted"], r["finished"], r["finish_reason"],
                            r.get("spec")] for r in rows],
                "kv_lens": kv_lens,
                "events": _jsonable(apply_events),
            })

        if drop_after_iter.get(it) in rid_to_core:
            dropped = drop_after_iter[it]
            drop_events = core.drop(int(rid_to_core.pop(dropped)))
            _emit({"kind": "core", "op": "drop", "rid": dropped,
                   "events": _jsonable(drop_events)})

    with open(path, "w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec, default=_jsonable) + "\n")
    return load_trace(path)


# ------------------------------------------------------- live replay (e2e)


def diff_sessions(
    recorded: TraceSession, replayed: TraceSession
) -> List[Tuple[int, Dict[str, Any]]]:
    """Field-for-field diff of two captured sessions' Rust plans.

    Compares the *recorded* plan of each plan line in ``recorded`` against
    the corresponding line in ``replayed``. Used by the live A/B gate:
    replay a canonical session on the live engine (capturing a fresh trace)
    and assert the two sessions' plans agree on every hard field. Returns
    a list of ``(plan_index, diff)`` for disagreements; empty when the
    sessions match. A length mismatch is itself a hard diff.
    """
    a = [op for op in recorded.ops if isinstance(op, PlanOp)]
    b = [op for op in replayed.ops if isinstance(op, PlanOp)]
    out: List[Tuple[int, Dict[str, Any]]] = []
    if len(a) != len(b):
        out.append(
            (
                -1,
                {
                    "field": "plan_count",
                    "recorded": len(a),
                    "recomputed": len(b),
                    "severity": "hard",
                },
            )
        )
    for i, (pa, pb) in enumerate(zip(a, b)):
        for d in diff_plans(pa.plan, pb.plan):
            out.append((i, d))
    return out


def sessions_match(recorded: TraceSession, replayed: TraceSession) -> bool:
    """True when the two sessions agree on every hard plan field."""
    return not any(d[1]["severity"] == "hard" for d in diff_sessions(recorded, replayed))


def live_replay_script(
    trace_path: str, *, settle_steps: int = 0, timeout_s: float = 30.0
):
    """Build a ``ScriptedContext`` generator that re-drives a *live*
    scheduler from a recorded session (target host: needs the full engine).

    The returned callable takes the ``ScriptedContext`` ``t`` and returns a
    generator. The scripted-runtime dispatch loop advances that generator
    once per scheduler event-loop iteration (``hook.step`` runs at the top
    of ``recv_requests``), so **one ``yield`` == one iteration == one
    recorded plan line**. The script walks the capture in order:

    - an ``ingress`` line → submit the request via ``t.start_req`` with its
      exact prompt tokens; because the submit happens in the same iteration
      as the plan line that follows it in the capture, the request is
      visible to that plan (ingress is recorded just before the plan that
      first sees it);
    - a ``drop`` line → abort the request;
    - a ``plan`` line → ``yield`` (let the live scheduler run one
      iteration);
    - an ``apply_result`` line → nothing: the engine folds the mocked
      ``run_batch`` result in on its own.

    ``scheduler.run_batch`` is swapped for a mock that returns the recorded
    sampled tokens (looked up by ``rid``), so the live Python scheduler —
    with the Rust core attached at stage ``core`` — replays the session in
    lock-step and the attached drivers write a fresh trace (set
    ``SGLANG_TRACE_SCHEDULER``). The caller then diffs the two captures
    with :func:`diff_sessions`. The original ``run_batch`` is restored in a
    ``finally``.

    ``settle_steps`` defaults to 0: the script ends immediately after the
    last recorded plan line, so a capture-and-replay on the same engine
    yields a 1:1 plan-line alignment for :func:`diff_sessions`. Pass a
    positive value to drain in-flight finishes before the script ends.
    """

    def script(t):
        import torch

        from sglang.srt.managers.utils import GenerationBatchResult

        trace = load_trace(trace_path)
        ops = trace.ops
        sched = t.scheduler

        # rid -> recorded result rows [[accepted, finished, finish_reason], ...]
        # in step order, consumed left-to-right by the mocked run_batch.
        results: Dict[str, List[Any]] = {}
        for op in ops:
            if isinstance(op, ApplyOp):
                for rid, row in zip(op.rids, op.result):
                    results.setdefault(rid, []).append(row)

        rid_to_handle: Dict[str, Any] = {}
        orig_run_batch = sched.run_batch

        def mocked_run_batch(batch, *args, **kwargs):
            reqs = list(getattr(batch, "reqs", None) or [])
            next_tokens: List[int] = []
            extend_lens: List[int] = []
            for req in reqs:
                rid = getattr(req, "rid", None)
                queue = results.get(rid, [])
                if queue:
                    accepted = queue.pop(0)[0]
                    next_tokens.append(int(accepted[0]) if accepted else 0)
                else:
                    next_tokens.append(0)  # parked chunk: no sampled token
                er = getattr(req, "extend_range", None)
                if er is not None:
                    extend_lens.append(int(er.end - er.start))
                else:
                    extend_lens.append(int(getattr(req, "extend_input_len", 0) or 0))
            return GenerationBatchResult(
                next_token_ids=torch.tensor(next_tokens, dtype=torch.int64),
                extend_input_len_per_req=extend_lens,
            )

        sched.run_batch = mocked_run_batch
        try:
            for op in ops:
                if isinstance(op, IngressOp):
                    handle = t.start_req(
                        rid=op.rid,
                        prompt_ids=list(op.origin),
                        max_new_tokens=op.max_new_tokens,
                        ignore_eos=op.ignore_eos,
                        priority=op.priority or None,
                        timeout_s=timeout_s,
                    )
                    rid_to_handle[op.rid] = handle
                elif isinstance(op, DropOp):
                    handle = rid_to_handle.pop(op.rid, None)
                    if handle is not None:
                        t.abort(handle)
                elif isinstance(op, PlanOp):
                    yield  # one scheduler iteration per recorded plan line
                # ApplyOp: folded in by the engine from the mocked result.
            # Settle: let in-flight finishes stream out before the script ends.
            for _ in range(settle_steps):
                if t.is_fully_idle():
                    break
                yield
        finally:
            sched.run_batch = orig_run_batch

    return script
