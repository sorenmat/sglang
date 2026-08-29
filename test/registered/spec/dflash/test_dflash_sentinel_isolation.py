"""DFlash2 concurrent-state isolation stress test.

Deterministic reproduction harness for reported request-state mixing under
concurrency (e.g. Qwen3.8-27B + DFlash2 on RTX PRO 6000): N concurrent
requests, each carrying a unique sentinel, assert that no request's output
ever contains another request's sentinel. Variations cover the agent-shaped
triggers: shared prefixes vs distinct prefixes, mid-flight cancellation
churn, radix-cache reuse, and overlap scheduling.

The test is model-agnostic: any DFLASH target/draft pair works, since the
assertion only needs foreign-sentinel absence, not answer quality.
"""

import random
import string
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-small")

CONCURRENCY = 16
FILLER_SENTENCES = 64  # ~2-4K prompt tokens per request


def _sentinel(seed: int) -> str:
    rng = random.Random(seed)
    return "QZ" + "".join(rng.choices(string.ascii_uppercase, k=8))


def _agent_prompt(seed: int, shared_prefix: str, shared: bool) -> str:
    own = _sentinel(seed)
    rng = random.Random(seed)
    filler = " ".join(
        rng.choice(
            [
                "The build finished without warnings.",
                "Two tests failed on the parser edge case.",
                "The cache layer was refactored last week.",
                "Latency improved after the scheduler change.",
            ]
        )
        for _ in range(FILLER_SENTENCES)
    )
    prefix = shared_prefix if shared else f"Session {seed} private scratchpad. "
    return (
        f"{prefix}"
        f"Agent notebook #{seed}: {filler}\n"
        f"Your unique agent marker is {own}. "
        f"Never mention any other marker. Reply with exactly: {own}"
    )


class TestDFlashSentinelIsolation(CustomTestCase):
    """Overlap scheduling is the concurrency-sensitive path (accept/bonus
    buffers rotate between in-flight steps), so it runs by default."""

    model = DEFAULT_TARGET_MODEL_DFLASH
    draft_model = DEFAULT_DRAFT_MODEL_DFLASH
    max_running_requests = 32

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--max-running-requests",
                str(cls.max_running_requests),
                "--mem-fraction-static",
                "0.7",
                "--cuda-graph-bs",
                *[str(i) for i in range(1, cls.max_running_requests + 1)],
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _run_batch(self, prompts, max_new_tokens=48, temperature=0):
        def one(i):
            res = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompts[i],
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=300,
            )
            assert res.status_code == 200, res.text
            return res.json()["text"]

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            return list(pool.map(one, range(len(prompts))))

    def _assert_isolation(self, seeds, outputs, shared: bool):
        for i, (seed, out) in enumerate(zip(seeds, outputs)):
            own = _sentinel(seed)
            for j, other_seed in enumerate(seeds):
                if i == j:
                    continue
                other = _sentinel(other_seed)
                self.assertNotIn(
                    other,
                    out,
                    f"request {i} (seed {seed}) leaked sentinel of request {j} "
                    f"(seed {other_seed}) -- concurrent state mixing; "
                    f"shared_prefix={shared}",
                )

    def test_distinct_prefix_isolation(self):
        seeds = list(range(CONCURRENCY))
        prompts = [_agent_prompt(s, "", shared=False) for s in seeds]
        outputs = self._run_batch(prompts)
        self._assert_isolation(seeds, outputs, shared=False)

    def test_shared_prefix_isolation(self):
        """Tool-return traffic: many agents share a long common prefix and
        branch late -- exercises radix-cache reuse of draft/target state."""
        shared_prefix = (
            "You are one of several coding agents sharing a repository README. "
            + "The README describes a build system, a test suite and a release "
            "process. " * 128
        )
        seeds = list(range(CONCURRENCY, 2 * CONCURRENCY))
        prompts = [_agent_prompt(s, shared_prefix, shared=True) for s in seeds]
        # First round populates the radix cache; the second round runs with
        # shared-prefix hits while concurrent decoding is in flight.
        outputs = self._run_batch(prompts)
        self._assert_isolation(seeds, outputs, shared=True)
        outputs2 = self._run_batch(prompts)
        self._assert_isolation(seeds, outputs2, shared=True)

    def test_isolation_after_cancellation_churn(self):
        """Cancel half the in-flight requests, then immediately re-check
        isolation: freed req-pool / mamba slots are reused by fresh requests."""
        seeds = list(range(2 * CONCURRENCY, 3 * CONCURRENCY))
        prompts = [_agent_prompt(s, "", shared=False) for s in seeds]

        def cancel_one(i):
            try:
                requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompts[i],
                        "sampling_params": {"max_new_tokens": 512},
                    },
                    timeout=0.5,
                )
            except requests.exceptions.Timeout:
                pass  # the point: abandon mid-prefill/decode

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(cancel_one, range(CONCURRENCY)))

        fresh_seeds = list(range(3 * CONCURRENCY, 4 * CONCURRENCY))
        fresh_prompts = [_agent_prompt(s, "", shared=False) for s in fresh_seeds]
        outputs = self._run_batch(fresh_prompts)
        self._assert_isolation(fresh_seeds, outputs, shared=False)


if __name__ == "__main__":
    unittest.main()
