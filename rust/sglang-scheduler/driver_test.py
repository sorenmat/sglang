#!/usr/bin/env python3
"""Standalone test of the Python driver layer (sglang.srt.managers.rust_scheduler)
against the real `_scheduler` extension.

Runs without torch/numpy: the `sglang.srt.environ` / `sglang.srt.rust_extensions`
imports are stubbed, the `.so` is loaded directly (like smoke_test.py), and a
fake Scheduler/Req tree drives the driver's hooks (attach / on_ingress /
on_abort / apply_result / plan / shadow / finalize / trace).

Usage:  python3 rust/sglang-scheduler/driver_test.py [path-to-.so]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "target", "debug", "libsglang_scheduler.so"
)

CHECKS = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        print(f"FAIL [{name}] {extra}")
        sys.exit(1)
    print(f"ok   [{name}]")


# --------------------------------------------------------------- .so loader


def load_so(path: str):
    spec = importlib.util.spec_from_file_location("_scheduler", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- stub modules


class _FakeEnv:
    def __init__(self, default):
        self.v = default

    def get(self):
        return self.v


class _FakeEnvs:
    SGLANG_RUST_SCHEDULER = _FakeEnv("core")
    SGLANG_TRACE_SCHEDULER = _FakeEnv("")  # set later to a tmp path
    SGLANG_RUST_CORE_APPLY = _FakeEnv("0")
    SGLANG_RUST_CORE_VALUES = _FakeEnv("0")
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = _FakeEnv(4096)
    SGLANG_INIT_NEW_TOKEN_RATIO = _FakeEnv(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = _FakeEnv(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = _FakeEnv(604)
    SGLANG_RETRACT_DETRACT_STEPS = _FakeEnv(20)
    SGLANG_RETRACT_DECODE_STEPS = _FakeEnv(20)


so = load_so(SO)


def _load_rust_extension(_name):
    return so


def _mk_pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


_mk_pkg("sglang")
_mk_pkg("sglang.srt")
environ_mod = types.ModuleType("sglang.srt.environ")
environ_mod.envs = _FakeEnvs
sys.modules["sglang.srt.environ"] = environ_mod
_mk_pkg("sglang.srt.managers")
sched_policy_mod = types.ModuleType("sglang.srt.managers.schedule_policy")
sched_policy_mod.IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = 32
sched_policy_mod.IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = 32
sys.modules["sglang.srt.managers.schedule_policy"] = sched_policy_mod
rust_ext_mod = types.ModuleType("sglang.srt.rust_extensions")
rust_ext_mod.load_rust_extension = _load_rust_extension
sys.modules["sglang.srt.rust_extensions"] = rust_ext_mod

# now the driver module itself, by file path
spec = importlib.util.spec_from_file_location(
    "sglang.srt.managers.rust_scheduler",
    os.path.join(HERE, "..", "..", "python", "sglang", "srt", "managers", "rust_scheduler.py"),
)
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


# ------------------------------------------------------------- fake objects


class _RowPool:
    """req_to_token stand-in: [pool, :fill] -> list of ints."""

    def __init__(self):
        self.rows = {}
        self.device = "cpu"

    def __getitem__(self, key):
        pool, sl = key
        return list(self.rows.get(pool, []))[: sl.stop]

    def set(self, pool, values):
        self.rows[pool] = values


class _ForwardMode:
    def __init__(self, is_decode=False, is_extend=True):
        self._d, self._e = is_decode, is_extend

    def is_decode(self):
        return self._d

    def is_extend(self):
        return self._e


class _Sampling:
    def __init__(self, max_new_tokens=64, ignore_eos=False):
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos


class _FinishReason:
    value = 1


class _Req:
    n = 0

    def __init__(self, rid, pool_idx, origin_len, out=0, finished=False,
                 finish_reason=None, extend_end=None, prefix_len=0):
        self.rid = rid
        self.req_pool_idx = pool_idx
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = [9000 + i for i in range(out)]
        self.sampling_params = _Sampling()
        self.priority = 0
        self.num_matched_prefix_tokens = 0
        self.prefix_indices = list(range(prefix_len)) if prefix_len else None
        self.last_node = 0
        self._finished = finished
        self.finished_reason = finish_reason
        self.extend_range = (
            types.SimpleNamespace(end=extend_end) if extend_end is not None else None
        )

    def finished(self):
        return self._finished


class _Batch:
    def __init__(self, reqs, decode=False):
        self.reqs = reqs
        self.forward_mode = _ForwardMode(is_decode=decode, is_extend=not decode)
        self.batch_is_full = False


class _Scheduler:
    def __init__(self):
        self.server_args = types.SimpleNamespace(
            schedule_policy="fcfs",
            max_prefill_tokens=16384,
            chunked_prefill_size=None,
            enable_mixed_chunk=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            random_seed=0,
            disable_radix_cache=False,
            truncation_align_size=None,
        )
        self.tree_cache = types.SimpleNamespace(
            eviction_policy="lru", evictable_size=lambda: 0
        )
        self.page_size = 1
        self.token_to_kv_pool_allocator = types.SimpleNamespace(
            available_size=lambda: 100_000
        )
        self.req_to_token_pool = types.SimpleNamespace(
            req_to_token=_RowPool(), device="cpu"
        )
        self.waiting_queue = []
        self.chunked_req = None
        self.disaggregation_mode = None
        self.new_token_ratio_tracker = types.SimpleNamespace(current=0.7)

    def get_num_allocatable_reqs(self, running_bs):
        return 1000


def main():
    sched = _Scheduler()

    # trace to a tmp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    _FakeEnvs.SGLANG_TRACE_SCHEDULER.v = tmp.name

    # ---- attach ---------------------------------------------------------
    drivers = rs.attach(sched)
    check("attach-stage", drivers.get("stage") == "core", str(drivers.get("stage")))
    shadow = drivers.get("shadow")
    core = drivers.get("core")
    check("attach-shadow", shadow is not None)
    check("attach-core", core is not None)
    check("trace-wired", shadow.trace is not None and core.trace is shadow.trace)

    # ---- ingress --------------------------------------------------------
    rids = []
    for i in range(2):
        rid = f"req-{i}"
        req = _Req(rid, pool_idx=100 + i, origin_len=40, out=0)
        sched.waiting_queue.append(req)
        core.on_ingress(req)
        rids.append(rid)
    check("core-waiting", core.core.waiting() == [0, 1],
          str(core.core.waiting()))

    # ---- prefill plan ---------------------------------------------------
    running = _Batch([], decode=True)
    core.plan(running)
    # shadow plan on the same snapshot
    shadow.shadow(running)
    check("core-last-batch", core.core.last_batch() == [0, 1],
          str(core.core.last_batch()))
    check("core-waiting-empty", core.core.waiting() == [],
          str(core.core.waiting()))

    # ---- results: extend batch (both admitted, not finished) ------------
    rowpool = sched.req_to_token_pool.req_to_token
    for i in range(2):
        rowpool.set(100 + i, list(range(i * 50, i * 50 + 40)))
    ext_reqs = [
        _Req(rids[0], 100, 40, out=1, extend_end=40),
        _Req(rids[1], 101, 40, out=1, extend_end=40),
    ]
    core.apply_result(_Batch(ext_reqs, decode=False))
    check("core-last-batch-cleared", core.core.last_batch() == [],
          str(core.core.last_batch()))

    # ---- next plan: merge pending into running, decode ------------------
    core.plan(running)
    check("core-running-after-merge", core.core.running() == [0, 1],
          str(core.core.running()))
    shadow.shadow(running)

    # ---- results: decode step, req1 finishes ---------------------------
    dec_reqs = [
        _Req(rids[0], 100, 40, out=2, extend_end=None),
        _Req(rids[1], 101, 40, out=2, finished=True,
             finish_reason=_FinishReason()),
    ]
    core.apply_result(_Batch(dec_reqs, decode=True))

    # ---- next plan: decode filters finished req ------------------------
    core.plan(running)
    check("core-running-after-finish", core.core.running() == [0],
          str(core.core.running()))

    # ---- abort the survivor --------------------------------------------
    core.on_abort(rids[0])
    check("core-empty-after-abort", core.core.running() == [],
          str(core.core.running()))
    check("core-mapping-popped", rids[0] not in core.rid_to_core)

    # ---- planner shadow diff -------------------------------------------
    class _Plan:
        def __init__(self, batch, running):
            self.batch_to_run = batch
            self.running_batch = running

    py_batch = _Batch(ext_reqs, decode=False)
    py_running = _Batch(dec_reqs, decode=True)
    shadow.finalize(_Plan(py_batch, py_running))
    # waiting rids fed to the shadow were rids[0], rids[1]; py admitted the
    # same two in the same order -> prefill/pre-fill mode must match.
    check("shadow-no-pending", shadow._pending is None)
    check("shadow-no-hard-mismatch", shadow.mismatches == 0,
          f"hard={shadow.mismatches} soft={shadow.soft_mismatches}")

    # a deliberately different py decision -> hard mismatch
    py_batch2 = _Batch(ext_reqs[:1], decode=False)  # only one admitted
    shadow.shadow(running)
    shadow.finalize(_Plan(py_batch2, py_running))
    check("shadow-hard-mismatch-counted", shadow.mismatches == 1,
          f"hard={shadow.mismatches}")

    # ---- trace file -----------------------------------------------------
    lines = []
    with open(tmp.name) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    check("trace-lines", len(lines) >= 3, f"{len(lines)} lines")
    kinds = {(ln.get("kind"), ln.get("op")) for ln in lines}
    check("trace-core-entries", ("core", "plan") in kinds, str(kinds))
    check("trace-shadow-entries", any("py" in ln and "rust" in ln for ln in lines),
          str(kinds))
    os.unlink(tmp.name)

    # ---- ntr helpers (regression, already in smoke test) ----------------
    # Ntr::from_config: min = init*min_factor, decay = (init - min)/steps,
    # next_after_decay = (current - decay).max(min).
    init, min_f, steps = 0.7, 0.14, 604
    decay = (init - init * min_f) / steps
    got = so.ntr_next_after_decay(init, init, 1.0, min_f, steps)
    check("ntr-decay", abs(got - (init - decay)) < 1e-12, f"{got} != {init - decay}")
    est = so.ntr_estimate_after_retract([1, 3], [100, 200], 20)
    check("ntr-estimate", abs(est - 44.0 / 301.0) < 1e-9, str(est))

    # ---- facade stage gate (no torch available: verify the gate logic) --
    check(
        "stage-gate",
        rs.stage_at_least("radix") and rs.stage_at_least("planner")
        and rs.stage_at_least("core") and not rs.stage_at_least("stream"),
    )

    print(f"\nALL DRIVER TESTS PASSED ({CHECKS} checks)")


if __name__ == "__main__":
    main()
