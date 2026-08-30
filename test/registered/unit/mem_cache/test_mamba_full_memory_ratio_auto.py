"""CPU-only unit tests for --mamba-full-memory-ratio auto.

Pins the workload-derived sizing: the state (mamba/GDN) pool is sized for
exactly the target concurrency (its per-request slot count, including the
spec-decode intermediate states) and the KV cache receives the remainder of
the rest-memory budget, instead of splitting by a fixed ratio. A fixed ratio
either strands KV memory or silently caps concurrency when the state pool is
too small -- both failure modes are easy to hit on hybrid models with
speculative decoding, where each running request holds several state slots.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.kv_cache_configurator import (
    AUTO_MAMBA_MAX_REST_SHARE,
    KVCacheConfigurator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GiB = 1 << 30


def _fake_kvc(*, cell_size: int = 0):
    """A KVCacheConfigurator shell with just the fields the auto-sizing path
    touches; _kv_cell_size_bytes is stubbed so no model config is needed.
    The stub is patched on the class (the dataclass uses slots, so
    per-instance shadowing is not possible)."""
    kvc = object.__new__(KVCacheConfigurator)
    kvc.model_config = SimpleNamespace(
        context_len=262144, num_hidden_layers=64, hf_config=SimpleNamespace()
    )
    kvc.ps = SimpleNamespace(attn_dp_size=1, pp_size=1)
    kvc.spec_algorithm = SpeculativeAlgorithm.NONE
    kvc.hybrid_gdn_config = None
    kvc.is_draft_worker = False
    kvc.mambaish_config = SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(
            layers=list(range(48)), mamba_cache_per_req=16 * (1 << 20)
        )
    )
    return kvc, cell_size


def _patch_cell_size(cell_size: int):
    return mock.patch.object(
        KVCacheConfigurator, "_kv_cell_size_bytes", return_value=cell_size
    )


class TestRatioParsing(unittest.TestCase):
    def test_cli_type_parser(self):
        from sglang.srt.server_args import _parse_mamba_full_memory_ratio as p

        self.assertEqual(p("auto"), "auto")
        self.assertEqual(p("AUTO"), "auto")
        self.assertEqual(p("0.5"), 0.5)
        self.assertEqual(p(0.9), 0.9)
        self.assertEqual(p(2), 2.0)
        with self.assertRaises(ValueError):
            p("bogus")

    def test_hook_normalization(self):
        from sglang.srt.arg_groups.mamba_hook import _normalize_mamba_full_memory_ratio

        for raw, expected in [("auto", "auto"), ("AUTO", "auto"), ("0.25", 0.25)]:
            args = SimpleNamespace(mamba_full_memory_ratio=raw)
            _normalize_mamba_full_memory_ratio(args)
            self.assertEqual(args.mamba_full_memory_ratio, expected)
        args = SimpleNamespace(mamba_full_memory_ratio=0.9)
        _normalize_mamba_full_memory_ratio(args)
        self.assertEqual(args.mamba_full_memory_ratio, 0.9)
        for bad in ["bogus", -0.5, 0, None]:
            with self.assertRaises(ValueError):
                _normalize_mamba_full_memory_ratio(
                    SimpleNamespace(mamba_full_memory_ratio=bad)
                )


class TestAutoSizing(unittest.TestCase):
    def _ctx(self, **fields):
        from sglang.srt import runtime_context as rc

        return rc.get_context().override_server_args(**fields)

    def test_sizes_for_target_concurrency(self):
        """With --max-running-requests set, the pool is exactly
        slots-per-request * target concurrency (5 slots/req with overlap
        extra_buffer), and no spec intermediates without spec decoding."""
        from sglang.srt.environ import envs

        kvc, cell_size = _fake_kvc()
        with self._ctx(
            disable_radix_cache=False,
            disable_overlap_schedule=False,
            mamba_radix_cache_strategy="extra_buffer",
            mamba_full_memory_ratio="auto",
            max_running_requests=16,
            max_mamba_cache_size=None,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                with _patch_cell_size(cell_size):
                    k, intermediate = kvc._auto_mamba_pool_size(
                        total_rest_memory=80.0,
                        stage_per_req=16 * (1 << 20),
                        replayssm_ring_per_req=0,
                        has_spec_dec=False,
                        replayssm_active=False,
                    )
        self.assertEqual(k, 16 * 5)
        self.assertEqual(intermediate, 0)

    def test_pool_capped_by_max_share(self):
        """An unreachably high concurrency target gets capped at
        AUTO_MAMBA_MAX_REST_SHARE of the budget instead of consuming it all."""
        from sglang.srt.environ import envs

        kvc, cell_size = _fake_kvc()
        with self._ctx(
            disable_radix_cache=True,
            mamba_full_memory_ratio="auto",
            max_running_requests=1 << 20,
            max_mamba_cache_size=None,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                with _patch_cell_size(cell_size):
                    k, intermediate = kvc._auto_mamba_pool_size(
                        total_rest_memory=80.0,
                        stage_per_req=16 * (1 << 20),
                        replayssm_ring_per_req=0,
                        has_spec_dec=False,
                        replayssm_active=False,
                    )
        per_slot = 16 * (1 << 20)
        expected_k = (int(80.0 * GiB * AUTO_MAMBA_MAX_REST_SHARE) - per_slot) // per_slot
        self.assertEqual(k, expected_k)
        self.assertEqual(intermediate, 0)

    def test_derives_concurrency_when_unset(self):
        """Without --max-running-requests, target concurrency solves from the
        per-request footprint (state slots + a full-context KV row)."""
        from sglang.srt.environ import envs

        cell_size = 32 * 1024  # bytes/token across full-attn layers
        kvc, _ = _fake_kvc(cell_size=cell_size)
        per_req = 16 * (1 << 20)
        with self._ctx(
            disable_radix_cache=True,
            mamba_full_memory_ratio="auto",
            max_running_requests=None,
            max_mamba_cache_size=None,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                with _patch_cell_size(cell_size):
                    k, intermediate = kvc._auto_mamba_pool_size(
                        total_rest_memory=80.0,
                        stage_per_req=per_req,
                        replayssm_ring_per_req=0,
                        has_spec_dec=False,
                        replayssm_active=False,
                    )
        # ratio=1 slot/req (disable_radix_cache); per-request total =
        # 16 MiB state + 262144 * 32 KiB KV; solve R, K = R slots.
        per_request_total = per_req + 262144 * cell_size
        total_bytes = int(80.0 * GiB)
        r = total_bytes // per_request_total
        self.assertEqual(k, r)
        self.assertEqual(intermediate, 0)

    def test_spec_intermediates_reserved(self):
        """Under spec decoding the pool carries the intermediate-state bytes
        for the target concurrency: (R + 1) * draft_tokens * per-request."""
        from sglang.srt.environ import envs

        kvc, cell_size = _fake_kvc()
        with self._ctx(
            disable_radix_cache=False,
            disable_overlap_schedule=False,
            mamba_radix_cache_strategy="extra_buffer",
            mamba_full_memory_ratio="auto",
            max_running_requests=16,
            max_mamba_cache_size=None,
            speculative_num_draft_tokens=4,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                with _patch_cell_size(cell_size):
                    k, intermediate = kvc._auto_mamba_pool_size(
                        total_rest_memory=80.0,
                        stage_per_req=16 * (1 << 20),
                        replayssm_ring_per_req=0,
                        has_spec_dec=True,
                        replayssm_active=False,
                    )
        self.assertEqual(k, 16 * 5)
        self.assertEqual(intermediate, (16 + 1) * 4 * 16 * (1 << 20))

    def test_effective_ratio_override(self):
        """After auto sizing, the published ratio leaf carries the derived
        numeric ratio so downstream readbacks (e.g. /server_info) see a float."""
        from sglang.srt import runtime_context as rc
        from sglang.srt.environ import envs

        kvc, cell_size = _fake_kvc()
        per_slot = 16 * (1 << 20)
        with self._ctx(
            disable_radix_cache=True,
            mamba_full_memory_ratio="auto",
            max_running_requests=16,
            max_mamba_cache_size=None,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                with _patch_cell_size(cell_size):
                    k, intermediate = kvc._auto_mamba_pool_size(
                        total_rest_memory=80.0,
                        stage_per_req=per_slot,
                        replayssm_ring_per_req=0,
                        has_spec_dec=False,
                        replayssm_active=False,
                    )
                value = rc.get_context().config_leaf("mamba_full_memory_ratio")
                self.assertIsInstance(value, float)
                mamba_bytes = (k + 1) * per_slot + intermediate
                expected = mamba_bytes / (80.0 * GiB - mamba_bytes)
                self.assertAlmostEqual(value, round(expected, 4), places=3)



class TestAutoDeploymentShape(unittest.TestCase):
    """The production-reported failure mode: auto sized for the fleet cap
    (64) instead of the observed concurrency (<5) and left the KV pool too
    small for long-context agent prompts. Pins the fixes: explicit target
    concurrency, and a KV floor that shrinks the state pool rather than
    starving KV."""

    def _ctx(self, **fields):
        from sglang.srt import runtime_context as rc

        return rc.get_context().override_server_args(**fields)

    def _pool(self, *, cell_size, total_gb=54.0, per_req_mb=147, **publish):
        from sglang.srt.environ import envs

        kvc, _ = _fake_kvc(cell_size=cell_size)
        with self._ctx(
            disable_radix_cache=True,
            mamba_full_memory_ratio="auto",
            max_mamba_cache_size=None,
            **publish,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                with _patch_cell_size(cell_size):
                    return kvc._auto_mamba_pool_size(
                        total_rest_memory=total_gb,
                        stage_per_req=per_req_mb * (1 << 20),
                        replayssm_ring_per_req=3 * (1 << 20),
                        has_spec_dec=False,
                        replayssm_active=False,
                    )

    def test_explicit_target_overrides_fleet_cap(self):
        """--mamba-auto-target-concurrency wins over --max-running-requests:
        sizing for the observed concurrency, not the fleet cap."""
        k, _ = self._pool(
            cell_size=0,
            max_running_requests=64,
            mamba_auto_target_concurrency=8,
        )
        # disable_radix -> 1 slot/req: 8 slots, not 64
        self.assertEqual(k, 8)

    def test_kv_floor_shrinks_state_pool(self):
        """The reported production case: 64-request target on ~54 GB rest
        took 37.6 GB of state and left KV starving. With a floor of 8 whole
        106k contexts the state pool shrinks to fit."""
        cell = 32 * 1024  # 32 KB/token (16 full-attn layers, fp8)
        floor_gb = 8 * 106000 * cell / (1 << 30)  # ~26.5 GB
        k, _ = self._pool(
            cell_size=cell,
            total_gb=54.0,
            max_running_requests=64,
            mamba_auto_kv_floor_contexts=8,
            mamba_auto_kv_floor_context_tokens=106000,
        )
        per_slot = (147 + 3) * (1 << 20)
        mamba_gb = (k + 1) * per_slot / (1 << 30)
        kv_gb = 54.0 - mamba_gb
        self.assertGreaterEqual(kv_gb, floor_gb - 0.01)
        # and the state pool is materially smaller than the 37.6 GB failure
        self.assertLess(mamba_gb, 30.0)

    def test_floor_default_uses_capped_model_context(self):
        """Default floor tokens = min(context_len, 131072)."""
        cell = 32 * 1024
        floor_gb = 8 * 131072 * cell / (1 << 30)
        k, _ = self._pool(
            cell_size=cell,
            total_gb=54.0,
            max_running_requests=64,
        )
        per_slot = (147 + 3) * (1 << 20)
        self.assertGreaterEqual(54.0 - (k + 1) * per_slot / (1 << 30), floor_gb - 0.01)

    def test_unachievable_floor_falls_back(self):
        """A floor larger than rest memory degrades to 75% instead of
        leaving no room for state at all."""
        k, _ = self._pool(
            cell_size=32 * 1024,
            total_gb=10.0,
            max_running_requests=2,
            mamba_auto_kv_floor_contexts=64,
            mamba_auto_kv_floor_context_tokens=131072,
        )
        self.assertGreaterEqual(k, 1)

    def test_small_target_gives_kv_the_rest(self):
        """Agent shape: target 8 on 54 GB -> state ~4.7 GB, KV ~49 GB (far
        above the 1.08-ratio equivalent of ~26 GB; auto only ever helps KV
        when the target is honest)."""
        cell = 32 * 1024
        k, _ = self._pool(
            cell_size=cell,
            total_gb=54.0,
            max_running_requests=64,
            mamba_auto_target_concurrency=8,
        )
        per_slot = (147 + 3) * (1 << 20)
        self.assertEqual(k, 8)
        kv_gb = 54.0 - (k + 1) * per_slot / (1 << 30)
        self.assertGreater(kv_gb, 45.0)


class TestAutoJointSolve(unittest.TestCase):
    """End-to-end through _handle_max_mamba_cache: the auto pool lands on
    exactly slots-per-req * target concurrency state slots (no float-floor
    shortfall), so no concurrency is spuriously capped and none is stranded."""

    def _ctx(self, **fields):
        from sglang.srt import runtime_context as rc

        return rc.get_context().override_server_args(**fields)

    def test_auto_pool_matches_target_concurrency(self):
        from sglang.srt import runtime_context as rc
        from sglang.srt.environ import envs

        with self._ctx(
            disable_radix_cache=False,
            disable_overlap_schedule=False,
            mamba_radix_cache_strategy="extra_buffer",
            mamba_full_memory_ratio="auto",
            max_running_requests=16,
            max_mamba_cache_size=None,
            speculative_num_draft_tokens=4,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                kvc, _ = _fake_kvc()
                kvc.spec_algorithm = SpeculativeAlgorithm.EAGLE
                total_rest = kvc._handle_max_mamba_cache(80.0)
                k = rc.get_context().config_leaf("max_mamba_cache_size")
                # 5 slots/req * 16 reqs = 80 persistent slots for the state
                self.assertEqual(k, 80)
                # and everything the state pool actually uses came off the
                # KV budget: (K + 1) * per_req persistent bytes plus
                # (capped_reqs + 1) * D * per_req intermediate bytes.
                per_req = 16 * (1 << 20)
                self.assertAlmostEqual(
                    total_rest,
                    80.0 - (k + 1) * per_req / GiB - (16 + 1) * 4 * per_req / GiB,
                    places=6,
                )

    def test_fixed_ratio_path_unchanged(self):
        """The numeric-ratio path keeps its existing arithmetic."""
        from sglang.srt import runtime_context as rc
        from sglang.srt.environ import envs

        with self._ctx(
            disable_radix_cache=True,
            mamba_full_memory_ratio=1.0,
            max_running_requests=None,
            max_mamba_cache_size=None,
        ):
            with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
                kvc, _ = _fake_kvc()
                kvc._handle_max_mamba_cache(80.0)
                k = rc.get_context().config_leaf("max_mamba_cache_size")
                # ratio 1.0 -> half of 80 GiB; disable_radix -> 1 slot/req
                per_req = 16 * (1 << 20)
                expected = int((40.0 * GiB - per_req) // per_req)
                self.assertEqual(k, expected)


if __name__ == "__main__":
    unittest.main()
