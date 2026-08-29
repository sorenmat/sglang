"""Lossless scheduler-trace replay — the correctness backbone (plan.md §4.2).

Drives the ``_scheduler`` PyO3 extension's ``SchedulerCore`` through a fixed
reference session (``replay.synthesize_session_trace``), records it in the
driver's JSONL schema, and replays it through a **second** core instance
(``replay.replay_core``). Replaying a trace the engine itself produced must
reproduce every recorded plan field-for-field — that is what makes the
capture a valid backbone:

- it proves the trace schema is **lossless** (every field a plan depends on
  is recorded: ingress tokens, plan env, sampled-token values, KV fills);
- it proves the Rust engine is **deterministic** over the session;
- it gives ``diff_plans`` / ``diff_sessions`` a known-good oracle to detect
  real disagreements (hard vs. soft classification).

``synthesize_session_trace`` and the offline replayer are torch-free, so this
suite also runs on CPU-only CI (the extension is built with the ``python``
feature); no GPU or live engine is needed. The live end-to-end feeder
(``replay.live_replay_script``) is a separate, target-host test.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime import replay
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _write_trace(path, records) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _read_trace(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestRustTraceReplay(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mod = load_rust_extension("sglang.srt.rust_extensions._scheduler")

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory(prefix="rust_trace_replay_")
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def path(self, name):
        return os.path.join(self.tmpdir, name)

    def test_replay_reproduces_plans(self):
        session = replay.synthesize_session_trace(self.mod, path=self.path("a.jsonl"))
        n_plans = sum(isinstance(o, replay.PlanOp) for o in session.ops)
        self.assertGreaterEqual(n_plans, 5, "session should include several plans")

        result = replay.replay_core(session)
        self.assertTrue(result.all_plans_ok, result.hard_diffs)
        # A lossless replay of the engine's own trace has ZERO diffs (hard or
        # soft) and finishes in the recorded cache state.
        for step in result.steps:
            self.assertEqual(
                step.diffs, [], f"plan[{step.index}] diverged: {step.diffs}"
            )
        # The session was genuinely exercised: it ends with a non-trivial
        # resident cache (finished requests were stashed / released into it).
        self.assertGreater(result.final_tree_stats[0], 0)

    def test_replay_detects_hard_plan_diff(self):
        replay.synthesize_session_trace(self.mod, path=self.path("h.jsonl"))
        records = _read_trace(self.path("h.jsonl"))
        flipped = False
        for rec in records:
            if (
                rec.get("kind") == "core"
                and rec.get("op") == "plan"
                and rec["plan"][0] == 1
            ):
                rec["plan"][0] = 0  # record an idle plan where one was prefilled
                flipped = True
                break
        self.assertTrue(flipped)
        _write_trace(self.path("h2.jsonl"), records)
        with self.assertRaises(replay.ReplayError):
            replay.replay_core(replay.load_trace(self.path("h2.jsonl")))

    def test_diff_plans_classification(self):
        prefill = [1, True, [[[0, 5, 0, 512]], -1, False, 512, 1], None]
        # identical -> no diff
        self.assertEqual(replay.diff_plans(prefill, prefill), [])
        # a different admitted set -> hard
        different = [1, True, [[[2, 0, 0, 16]], -1, False, 16, 1], None]
        self.assertTrue(
            any(d["severity"] == "hard" for d in replay.diff_plans(prefill, different))
        )
        # a mode disagreement -> hard
        self.assertTrue(
            any(
                d["severity"] == "hard"
                for d in replay.diff_plans(prefill, [2, False, None, None])
            )
        )
        self.assertTrue(replay.plans_match(prefill, prefill))
        self.assertFalse(replay.plans_match(prefill, different))

    def test_sessions_match_clean_and_detect_mismatch(self):
        a = replay.synthesize_session_trace(self.mod, path=self.path("s1.jsonl"))
        b = replay.synthesize_session_trace(self.mod, path=self.path("s2.jsonl"))
        self.assertTrue(replay.sessions_match(a, b))
        self.assertEqual(replay.diff_sessions(a, b), [])

        records = _read_trace(self.path("s2.jsonl"))
        touched = False
        for rec in records:
            if (
                rec.get("kind") == "core"
                and rec.get("op") == "plan"
                and rec["plan"][0] == 2
                and rec["plan"][3]
                and len(rec["plan"][3][0]) > 1
            ):
                # drop a decode survivor from the recorded plan -> hard diff
                rec["plan"][3][0] = rec["plan"][3][0][:-1]
                touched = True
                break
        self.assertTrue(touched, "expected a multi-req decode plan to corrupt")
        _write_trace(self.path("s3.jsonl"), records)
        self.assertFalse(
            replay.sessions_match(a, replay.load_trace(self.path("s3.jsonl")))
        )

    def test_load_trace_validation(self):
        # no cfg header -> ReplayError
        with open(self.path("empty.jsonl"), "w", encoding="utf-8"):
            pass
        with self.assertRaises(replay.ReplayError):
            replay.load_trace(self.path("empty.jsonl"))
        # a hash-origin capture is not live-replayable
        with open(self.path("hash.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"kind": "cfg", "cfg": self._minimal_cfg(),
                             "tree_policy": "lru"})
                + "\n"
            )
            fh.write(
                json.dumps({"kind": "ingress", "rid": "r", "origin": "sha256…",
                             "origin_len": 8, "max_new_tokens": 8, "priority": 0,
                             "ignore_eos": False, "arrival_seq": 0})
                + "\n"
            )
        with self.assertRaises(replay.ReplayError):
            replay.load_trace(self.path("hash.jsonl"))

    def _minimal_cfg(self):
        return {
            "policy": "fcfs", "page_size": 1, "max_prefill_tokens": 64,
            "chunked_prefill_size": None, "mixed_chunk": False,
            "priority_scheduling": False, "low_priority_values_first": False,
            "clip_max_new_tokens": 4096, "in_batch_check_threshold": 32,
            "in_batch_deprioritize_threshold": 32, "prefill_max_requests": None,
            "truncation_align_size": None, "lpm_queue_degrade_at": 128,
            "random_seed": 0, "disable_tree": False, "ntr_init_raw": 0.7,
            "schedule_conservativeness": 1.0, "ntr_min_factor": 0.14,
            "ntr_decay_steps": 604, "retract_decode_steps": 20,
        }


if __name__ == "__main__":
    unittest.main()
