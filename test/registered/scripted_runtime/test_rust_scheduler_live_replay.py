"""Live scheduler-trace replay on the engine (plan §4.2 e2e gate).

One engine process, two phases in a single script:

1. **Capture** — the live Python scheduler (with the Rust core attached at
   stage ``core``) drives a small request set with the *real* model
   (``run_batch`` unmocked) and every core plan/apply lands in
   ``SGLANG_TRACE_SCHEDULER``.
2. **Replay** — the engine state is reset (abort-all, drain, cache flush,
   and ``RustCoreDriver.reset()`` so the core starts from a clean tree),
   then ``replay.live_replay_script`` re-drives the *recorded* session
   with a mocked ``run_batch`` that returns the recorded sampled tokens.
   The attached drivers write the replayed plans into the same trace.

The test splits the trace at the phase boundary and asserts the two
sessions' Rust plans agree on every hard field (``replay.diff_sessions``).
Because both phases start from a clean engine + clean core, the recorded
and replayed sessions are 1:1 in lock-step: one script yield == one
scheduler iteration == one recorded plan line.

Target host only (needs the full engine + a GPU + the built
``libsglang_scheduler.so``). The offline, torch-free half of the
backbone lives in ``test/registered/rust/test_rust_trace_replay.py``.
"""

import json
import os
import tempfile

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime import replay
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    base_engine_kwargs,
    run_until_all_finished,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

_ENGINE_KWARGS = base_engine_kwargs()

# Two prompts sharing a 6-token prefix so the capture exercises a radix
# prefix match on the second admission. All ids are inside the Qwen3
# vocab; the sampled tokens are whatever the model emits — the replay
# re-samples the *recorded* values, so finish logic sees identical input.
_PROMPT_1 = [1000 + i for i in range(14)]
_PROMPT_2 = [1000 + i for i in range(6)] + [2000 + i for i in range(6)]
_MAX_NEW_TOKENS = 6


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _first_ingress_after(path: str, after: int):
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < after:
                continue
            if line.strip() and json.loads(line).get("kind") == "ingress":
                return i
    return None


def _script_capture_then_replay(
    t: ScriptedContext, trace_path: str, sidecar_path: str
):
    from sglang.test.scripted_runtime import replay as _replay

    sched = t.scheduler
    driver = sched.rust_drivers.get("core")
    assert driver is not None, "SGLANG_RUST_SCHEDULER=core did not attach"

    def _reset_engine_and_core():
        if sched._engine_paused:
            t.continue_generation()
        t._release_exhausted_pools()
        t.abort_all()
        for _ in range(200):
            yield
            if sched.is_fully_idle():
                break
        t.flush_cache()
        yield
        driver.reset()

    # Wipe any warmup residue from the core before the session starts, so
    # the capture begins from the same clean state the replay will.
    driver.reset()

    # ---- Phase 1: capture with the real run_batch.
    r1 = t.start_req(prompt_ids=_PROMPT_1, max_new_tokens=_MAX_NEW_TOKENS)
    r2 = t.start_req(prompt_ids=_PROMPT_2, max_new_tokens=_MAX_NEW_TOKENS)
    yield from run_until_all_finished([r1, r2])

    capture_end = _count_lines(trace_path)
    assert capture_end > 0, "capture produced no trace lines"

    # ---- Phase 2: clean slate, then replay the recorded session with a
    # mocked run_batch (recorded sampled tokens).
    yield from _reset_engine_and_core()
    yield from _replay.live_replay_script(trace_path, settle_steps=0)(t)

    replay_start = _first_ingress_after(trace_path, capture_end)
    assert replay_start is not None, "replay produced no ingresses in the trace"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"capture_end": capture_end, "replay_start": replay_start}, fh
        )


def _write_lines(path: str, lines) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))


class TestRustSchedulerLiveReplay(ScriptedTestCase):
    ENGINE_KWARGS = _ENGINE_KWARGS

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="rust_live_replay_")
        cls._trace_path = os.path.join(cls._tmp, "trace.jsonl")
        cls._sidecar_path = os.path.join(cls._tmp, "markers.json")
        # The spawned engine process inherits os.environ, so both knobs
        # must be in place before ScriptedHttpServer.start().
        os.environ["SGLANG_RUST_SCHEDULER"] = "core"
        os.environ["SGLANG_TRACE_SCHEDULER"] = cls._trace_path
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        try:
            os.environ.pop("SGLANG_RUST_SCHEDULER", None)
            os.environ.pop("SGLANG_TRACE_SCHEDULER", None)
        finally:
            super().tearDownClass()

    def test_live_replay_reproduces_recorded_plans(self):
        self.server.execute_script(
            _script_capture_then_replay,
            args=(self._trace_path, self._sidecar_path),
            timeout_s=300,
        )

        with open(self._sidecar_path, "r", encoding="utf-8") as fh:
            markers = json.load(fh)
        with open(self._trace_path, "r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]

        cap_end = markers["capture_end"]
        rep_start = markers["replay_start"]
        cap_start = None
        for i, line in enumerate(lines[:cap_end]):
            if json.loads(line).get("kind") == "ingress":
                cap_start = i
                break
        assert cap_start is not None, "capture session has no ingress lines"
        assert cap_start < cap_end <= rep_start < len(lines)

        # Each session = the cfg header plus its own op lines; the
        # startup plans (warmup drive / reset drains) are trimmed away.
        cap_file = os.path.join(self._tmp, "capture.jsonl")
        rep_file = os.path.join(self._tmp, "replay.jsonl")
        _write_lines(cap_file, [lines[0]] + lines[cap_start:cap_end])
        _write_lines(rep_file, [lines[0]] + lines[rep_start:])

        captured = replay.load_trace(cap_file)
        replayed = replay.load_trace(rep_file)
        diffs = replay.diff_sessions(captured, replayed)
        hard = [d for d in diffs if d[1]["severity"] == "hard"]
        self.assertEqual(
            hard,
            [],
            f"live replay diverged from the capture:\n"
            + "\n".join(f"plan {i}: {d}" for i, d in hard),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
