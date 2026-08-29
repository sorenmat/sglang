# SPDX-License-Identifier: Apache-2.0
"""Incremental FP8 mirror of packed-NVFP4 KV slots.

Quantized KV pools store packed FP4 plus per-block scales; attention
backends that consume FP8 dequantize the *whole* cached prefix into a
workspace on every use -- chunked prefill re-dequantizes the prefix for
every chunk and every layer, and a speculative verify path would do the
same for every verify cycle, an O(context) cost per step that grows
linearly with agent context lengths.

KV slot content is written once when its tokens are appended and stays
read-only until the slot is freed and reused by another request, so a
dequantized FP8 copy keyed by slot remains valid as long as it reflects
the latest write. Per-(layer, slot) integer epochs track that: writers
stamp ``kv_write_epoch[slot]`` with a monotonically increasing counter,
and the mirror stamps ``mirror_epoch[layer, slot]`` when it dequantizes
a row. ``refresh`` dequantizes only rows whose epochs disagree, turning
per-step O(prefix) dequant into O(newly written slots), at the cost of
``mirror_size * layer_num * 2 * head_num * head_dim`` bytes of FP8
workspace. The mirror covers slots ``[0, mirror_size)``; slots outside
it dequantize on the fly (partial-mirror configurations stay correct,
just less cached).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


class NVFP4DequantMirror:
    """Per-layer FP8 mirror over the front of an NVFP4 KV pool.

    ``dequant_rows(idx)`` must return ``(k_fp8_rows, v_fp8_rows)`` for the
    given slot ids -- the pool passes its existing ``dequantize_prev_kv``
    so scale handling stays in one place.
    """

    def __init__(
        self,
        *,
        size: int,
        mirror_size: int,
        layer_num: int,
        head_num: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        assert 0 < mirror_size <= size
        self.size = size
        self.mirror_size = mirror_size
        self.layer_num = layer_num
        self.mirror_k = torch.zeros(
            (layer_num, mirror_size, head_num, head_dim), dtype=dtype, device=device
        )
        self.mirror_v = torch.zeros(
            (layer_num, mirror_size, head_num, head_dim), dtype=dtype, device=device
        )
        self.kv_write_epoch = torch.zeros((size,), dtype=torch.int64, device=device)
        # -1 so any real write (epoch >= 1) -- including a write that happened
        # before the first refresh -- makes the row stale; zero-init would let
        # never-dequantized rows masquerade as fresh.
        self.mirror_epoch = torch.full(
            (layer_num, mirror_size), -1, dtype=torch.int64, device=device
        )
        # Host-side write counter; mirror_epoch mirrors kv_write_epoch values.
        self.write_counter = 0
        # Epoch of kv_write_epoch at the last refresh per layer, used only for
        # stats/debug (the epoch tensors themselves drive correctness).
        self.last_refresh_counter = [-1] * layer_num
        # stats
        self.total_refresh_rows = 0
        self.total_requested_rows = 0
        self.refresh_calls = 0

    def note_kv_write(self, loc: torch.Tensor) -> None:
        """Stamp the written slots. Writers must call this for every FP4
        mutation (quantized store, prefix-valid commit, slot move, offload
        restore); over-stamping is safe (extra dequant), missing one is not.
        Out-of-range ids (e.g. an expandable pool's reserved tail beyond the
        epoch tensor) are clamped: the spurious stamp only over-invalidates
        the last slot."""
        self.write_counter += 1
        if loc.numel() == 0:
            return
        loc = loc.to(self.kv_write_epoch.device, non_blocking=True).view(-1).long()
        loc = loc.clamp(0, self.size - 1)
        self.kv_write_epoch[loc] = self.write_counter

    def _stale_mask(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        clamped = indices.clamp(max=self.mirror_size - 1)
        return (indices < self.mirror_size) & (
            self.mirror_epoch[layer_id][clamped] < self.kv_write_epoch[clamped]
        )

    def refresh(
        self,
        layer_id: int,
        indices: torch.Tensor,
        dequant_rows: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(k_fp8, v_fp8)`` rows for ``indices``, dequantizing only
        mirror rows that are stale; rows for slots beyond the mirror are
        dequantized on the fly."""
        self.refresh_calls += 1
        self.total_requested_rows += int(indices.shape[0])
        indices = indices.to(self.kv_write_epoch.device).view(-1).long()

        stale = self._stale_mask(layer_id, indices)
        stale_count = int(stale.sum().item())
        if stale_count > 0:
            stale_idx = indices[stale]
            k_new, v_new = dequant_rows(stale_idx)
            self.mirror_k[layer_id][stale_idx] = k_new
            self.mirror_v[layer_id][stale_idx] = v_new
            self.mirror_epoch[layer_id][stale_idx] = self.kv_write_epoch[stale_idx]
            self.total_refresh_rows += stale_count
        self.last_refresh_counter[layer_id] = self.write_counter

        if self.mirror_size >= self.size:
            # Full mirror: every requested row is cached. Clamp the gather so
            # pool padding rows (slot ids past `size`) never index out of
            # bounds; padding rows are not meaningfully read, matching the
            # un-mirrored path's zero-filled workspace rows.
            gather_idx = indices.clamp(max=self.mirror_size - 1)
            return self.mirror_k[layer_id][gather_idx], self.mirror_v[layer_id][gather_idx]

        # Partial mirror: out-of-mirror rows dequantize on the fly and
        # overwrite the (garbage) gathered rows.
        k_out = self.mirror_k[layer_id][indices.clamp(max=self.mirror_size - 1)]
        v_out = self.mirror_v[layer_id][indices.clamp(max=self.mirror_size - 1)]
        outside = indices >= self.mirror_size
        if int(outside.sum().item()) > 0:
            out_idx = indices[outside]
            k_direct, v_direct = dequant_rows(out_idx)
            k_out[outside] = k_direct
            v_out[outside] = v_direct
        return k_out, v_out

    def invalidate_all(self) -> None:
        """Force full re-dequant on the next refresh (e.g. after bulk state
        surgery on the underlying pool)."""
        self.mirror_epoch.fill_(-1)

    def stats(self) -> dict:
        return {
            "refresh_calls": self.refresh_calls,
            "requested_rows": self.total_requested_rows,
            "dequantized_rows": self.total_refresh_rows,
            "dequant_ratio": (
                self.total_refresh_rows / max(self.total_requested_rows, 1)
            ),
        }
