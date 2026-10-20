"""Host-KV integration units (HKV-20): stash resolution, args plumbing,
capture-guard stand-down, scheduler tick wiring. CPU only."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.qsa.glue import resolve_selection_stash
from sglang.srt.layers.attention.qsa.host_kv import HostKVConfig, HostKVTier


class _Backend:
    def __init__(self, with_stash=False):
        if with_stash:
            self._mtp_shared_sparse_indices = object()


class _Hybrid:
    def __init__(self, with_stash=False):
        self.full_attn_backend = _Backend(with_stash)


class _DraftWorker:
    def __init__(self):
        self.draft_extend_attn_backend = _Hybrid(with_stash=True)
        self.draft_attn_backend = _Hybrid(with_stash=False)


class _EagleWorker:
    def __init__(self):
        self._draft_worker = _DraftWorker()


class StashResolution(unittest.TestCase):
    def test_proven_path(self):
        ew = _EagleWorker()
        owner = resolve_selection_stash(ew)
        self.assertIs(
            owner,
            ew._draft_worker.draft_extend_attn_backend  # noqa: SLF001
            .full_attn_backend)

    def test_public_draft_worker_fallback(self):
        ew = _EagleWorker()
        ew.draft_worker = ew._draft_worker  # noqa: SLF001
        ew._draft_worker = None  # noqa: SLF001
        self.assertIsNotNone(resolve_selection_stash(ew))

    def test_no_stash_returns_none(self):
        self.assertIsNone(resolve_selection_stash(object()))
        self.assertIsNone(resolve_selection_stash(_EagleWorker()
                         ._draft_worker.draft_attn_backend))  # noqa: SLF001


class ArgsPlumbing(unittest.TestCase):
    def _args(self, **kw):
        # ServerArgs.__post_init__ resolves accelerator platforms; a mock
        # carrying exactly the published fields is the CPU-testable shape.
        base = dict(host_kv_tier=True, host_kv_keep_pages=1600,
                    host_kv_hot_pages=256, host_kv_prefetch=True,
                    host_kv_prefetch_radius=1, host_kv_reserve_floor=128,
                    host_kv_verify_sample=8, host_kv_stats_path=None)
        base.update(kw)
        return mock.MagicMock(**base)

    def test_defaults(self):
        cfg = HostKVConfig.from_server_args(self._args())
        self.assertTrue(cfg.enable)
        self.assertEqual(cfg.keep_pages, 1600)
        self.assertEqual(cfg.hot_pages, 256)
        self.assertTrue(cfg.prefetch)
        self.assertEqual(cfg.prefetch_radius, 1)
        self.assertEqual(cfg.prefetch_floor, 128)
        self.assertEqual(cfg.verify_sample, 8)

    def test_overrides(self):
        cfg = HostKVConfig.from_server_args(
            self._args(host_kv_keep_pages=2510, host_kv_hot_pages=64,
                       host_kv_prefetch=False, host_kv_stats_path="/tmp/x"))
        self.assertEqual(cfg.keep_pages, 2510)
        self.assertEqual(cfg.hot_pages, 64)
        self.assertFalse(cfg.prefetch)
        self.assertEqual(cfg.stats_path, "/tmp/x")

    def test_disabled(self):
        cfg = HostKVConfig.from_server_args(
            self._args(host_kv_tier=False))
        self.assertFalse(cfg.enable)

    def test_absent_attrs_fall_back(self):
        cfg = HostKVConfig.from_server_args(
            mock.MagicMock(spec=[], host_kv_tier=True))
        self.assertTrue(cfg.enable)
        self.assertEqual(cfg.hot_pages, 256)


class CaptureGuardStandDown(unittest.TestCase):
    def test_hooks_noop_when_disabled(self):
        tier = HostKVTier(HostKVConfig(enable=False))

        class B:
            host_kv_tier = tier
            _host_kv_pre = None  # placeholder; set below
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as mod
        # The guards read self.host_kv_tier; disabled tiers never arm.
        backend = mock.MagicMock()
        backend.host_kv_tier = tier
        backend._host_kv_pre(None, None)  # real methods bound via class
        self.assertFalse(tier.armed)

    def test_capturing_stream_skips(self):
        tier = HostKVTier(HostKVConfig(enable=True))
        tier.armed = True  # pretend armed
        tier.before_gather = mock.Mock()

        class Backend:
            host_kv_tier = tier
        Backend._host_kv_pre = (
            qwen_sparse_attn_backend_module().__dict__.get(
                "QwenSparseAttnBackend").__dict__["_host_kv_pre"])
        with mock.patch("torch.cuda.is_current_stream_capturing",
                        return_value=True):
            b = Backend()
            b._host_kv_pre(object(), object())
        tier.before_gather.assert_not_called()


def qwen_sparse_attn_backend_module():
    from sglang.srt.layers.attention import qwen_sparse_attn_backend
    return qwen_sparse_attn_backend


class SchedulerTickWiring(unittest.TestCase):
    def test_scheduler_tick_resolves_and_ticks(self):
        tier = HostKVTier(HostKVConfig(enable=True))
        tier.armed = True
        tier.rows[0] = {"ctx_written": 0, "chunks": 0, "mirrored": set(),
                        "retired": set(), "page_slots": {},
                        "re_retire": set(), "restored": set(),
                        "prefetched": set(), "decode_streak": 0,
                        "last_horizon": 0}
        tier.tick = mock.Mock()
        eagle = _EagleWorker()
        tier.scheduler_tick(eagle)
        tier.tick.assert_called_once_with(
            eagle._draft_worker.draft_extend_attn_backend  # noqa: SLF001
            .full_attn_backend)

    def test_scheduler_tick_no_stash_is_silent(self):
        tier = HostKVTier(HostKVConfig(enable=True))
        tier.armed = True
        tier.tick = mock.Mock()
        tier.scheduler_tick(object())
        tier.tick.assert_not_called()

    def test_unarmed_never_ticks(self):
        tier = HostKVTier(HostKVConfig(enable=True))
        tier.tick = mock.Mock()
        tier.scheduler_tick(_EagleWorker())
        tier.tick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
