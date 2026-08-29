"""CPU-only unit tests for the incremental NVFP4 dequant mirror.

Pins the invariant the mirror exists for: KV slot content is written once
and read-only until the slot is reused, so a slot-keyed FP8 mirror with
per-(layer, slot) epochs only re-dequantizes rows whose backing FP4 data
was rewritten. Repeated prefix reads (every chunked-prefill chunk, every
speculative verify cycle) then cost O(newly written slots) instead of
O(prefix) per step.
"""

import unittest

import torch

from sglang.srt.layers.quantization.nvfp4_dequant_mirror import NVFP4DequantMirror
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class CountingDequant:
    """Stand-in for NVFP4KVCacheMethod.dequantize_prev_kv: deterministic
    per-slot values plus a call/row counter to observe what got recomputed."""

    def __init__(self, payload: torch.Tensor, head_num=2, head_dim=16):
        self.payload = payload  # [size] float -- one distinct value per slot
        self.head_num = head_num
        self.head_dim = head_dim
        self.calls = 0
        self.rows = 0

    def __call__(self, idx: torch.Tensor):
        self.calls += 1
        self.rows += int(idx.shape[0])
        value = self.payload[idx].float() / 2971.0
        row = value[:, None, None].expand(-1, self.head_num, self.head_dim)
        return row.to(torch.float8_e4m3fn), row.clone().to(torch.float8_e4m3fn)


def _mirror(size=64, mirror_size=None, layers=2, head_num=2, head_dim=16):
    payload = torch.arange(size, dtype=torch.float32) + 1.0
    mirror = NVFP4DequantMirror(
        size=size,
        mirror_size=mirror_size if mirror_size is not None else size,
        layer_num=layers,
        head_num=head_num,
        head_dim=head_dim,
        device="cpu",
    )
    return mirror, CountingDequant(payload, head_num, head_dim), payload


class TestDequantMirror(unittest.TestCase):
    def test_first_refresh_dequantizes_then_cached(self):
        mirror, dq, _ = _mirror()
        idx = torch.tensor([1, 2, 3, 4, 5])
        k1, v1 = mirror.refresh(0, idx, dq)
        self.assertEqual(dq.rows, 5)
        k2, v2 = mirror.refresh(0, idx, dq)
        self.assertEqual(dq.rows, 5)  # no re-dequant without writes
        self.assertTrue(torch.equal(k1, k2))
        self.assertTrue(torch.equal(v1, v2))
        self.assertEqual(mirror.stats()["dequantized_rows"], 5)
        self.assertEqual(mirror.stats()["requested_rows"], 10)

    def test_write_invalidates_only_written_slots(self):
        mirror, dq, payload = _mirror()
        idx = torch.tensor([1, 2, 3, 4, 5])
        mirror.refresh(0, idx, dq)
        payload[3] = 999.0  # slot 3 rewritten by another request
        mirror.note_kv_write(torch.tensor([3]))
        dq.rows = 0
        k, _ = mirror.refresh(0, idx, dq)
        self.assertEqual(dq.rows, 1)  # only the rewritten slot
        expected, _ = dq(torch.tensor([3]))
        self.assertTrue(torch.equal(k[2], expected[0]))
        # and the untouched rows still serve cached copies
        cached, _ = dq(torch.tensor([1]))
        self.assertTrue(torch.equal(k[0], cached[0]))

    def test_slot_reuse_returns_new_content(self):
        """The reported corruption shape: a freed slot reused by another
        request must never serve the previous occupant's dequantized row."""
        mirror, dq, payload = _mirror()
        idx = torch.tensor([7])
        mirror.refresh(0, idx, dq)
        payload[7] = 1234.0
        mirror.note_kv_write(torch.tensor([7]))
        k, _ = mirror.refresh(0, idx, dq)
        expected, _ = dq(idx)
        self.assertTrue(torch.equal(k, expected))

    def test_layers_are_independent(self):
        mirror, dq, _ = _mirror()
        idx = torch.tensor([1, 2, 3])
        mirror.refresh(0, idx, dq)
        mirror.refresh(1, idx, dq)
        self.assertEqual(dq.rows, 6)  # each layer dequantizes its own rows

    def test_partial_mirror_mixed_rows(self):
        size, mirror_size = 8, 4
        mirror, dq, payload = _mirror(size=size, mirror_size=mirror_size)
        idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        k, _ = mirror.refresh(0, idx, dq)
        # first 4 rows via mirror, rest dequantized on the fly: all 8 once
        self.assertEqual(dq.rows, 8)
        expected, _ = CountingDequant(payload)(idx)
        self.assertTrue(torch.equal(k, expected))
        # second refresh: mirror rows cached, out-of-mirror rows re-dequantized
        dq.rows = 0
        mirror.refresh(0, idx, dq)
        self.assertEqual(dq.rows, 4)

    def test_padding_rows_never_index_out_of_bounds(self):
        # Full mirror: pool padding rows (slot ids past `size`) clamp safely.
        mirror, dq, _ = _mirror(size=8)
        idx = torch.tensor([0, 1, 8, 9])  # 8/9 are padding ids
        k, _ = mirror.refresh(0, idx, dq)
        self.assertEqual(k.shape[0], 4)

    def test_note_kv_write_clamps_out_of_range(self):
        mirror, dq, _ = _mirror(size=8)
        mirror.note_kv_write(torch.tensor([100]))  # beyond the epoch tensor
        idx = torch.tensor([0, 1])
        mirror.refresh(0, idx, dq)  # must not raise

    def test_invalidate_all_forces_full_refresh(self):
        mirror, dq, _ = _mirror(size=8)
        idx = torch.tensor([0, 1, 2])
        mirror.refresh(0, idx, dq)
        dq.rows = 0
        mirror.invalidate_all()
        mirror.refresh(0, idx, dq)
        self.assertEqual(dq.rows, 3)

    def test_matches_unmirrored_reference(self):
        """The mirrored result equals what plain on-the-fly dequant would
        return, for arbitrary interleavings of writes and refreshes."""
        torch.manual_seed(0)
        size = 32
        payload = torch.rand(size) * 100
        mirror = NVFP4DequantMirror(
            size=size, mirror_size=size, layer_num=2, head_num=2, head_dim=16,
            device="cpu",
        )
        dq = CountingDequant(payload)
        all_idx = torch.arange(size)
        reference, _ = CountingDequant(payload)(all_idx)
        for _ in range(5):
            # rewrite a few random slots, then read everything
            rewritten = torch.randperm(size)[:5]
            payload[rewritten] = torch.rand(5) * 100
            mirror.note_kv_write(rewritten)
            k, _ = mirror.refresh(0, all_idx, dq)
            ref_k, _ = CountingDequant(payload)(all_idx)
            self.assertTrue(torch.equal(k, ref_k))


class TestQuantMethodWiring(unittest.TestCase):
    def test_create_buffers_allocates_mirror_when_enabled(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        method = NVFP4KVCacheMethod(num_layers=2, device="cpu")
        with envs.SGLANG_NVFP4_DQ_MIRROR_FRACTION.override(0.5):
            bufs = method.create_buffers(
                size=64, head_num=2, head_dim=32, layer_num=2, device="cpu"
            )
            self.assertIsNotNone(bufs["dequant_mirror"])
            self.assertEqual(bufs["dequant_mirror"].mirror_size, 32)
            self.assertEqual(
                bufs["dequant_mirror"].mirror_k.shape, (2, 32, 2, 32)
            )
        # disabled by default -> no mirror, unchanged buffer set
        method2 = NVFP4KVCacheMethod(num_layers=2, device="cpu")
        bufs2 = method2.create_buffers(
            size=64, head_num=2, head_dim=32, layer_num=2, device="cpu"
        )
        self.assertIsNone(bufs2["dequant_mirror"])
        self.assertIsNone(method2.dequant_mirror)

    def test_cell_size_accounts_for_mirror(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        method = NVFP4KVCacheMethod(num_layers=2, device="cpu")
        base = method.compute_cell_size(head_num=2, head_dim=32, num_layers=2, kv_size=64)
        with envs.SGLANG_NVFP4_DQ_MIRROR_FRACTION.override(1.0):
            with_mirror = method.compute_cell_size(
                head_num=2, head_dim=32, num_layers=2, kv_size=64
            )
        # mirror term: 2 heads * 32 dim * 2 (K+V) * 2 layers * 64 slots * 1B
        self.assertEqual(with_mirror - base, 2 * 32 * 2 * 2 * 64)


if __name__ == "__main__":
    unittest.main()
