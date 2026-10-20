"""Host-RAM KV memory tier for QSA sparse-attention models.

Retires a request's cold full-attention KV pages to a pinned host mirror
and restores them on demand, so the GPU pool serves contexts far larger
than its token budget. Enabled with ``--host-kv-tier``.

Design (validated end to end on Qwen3.8-Flash-Next NVFP4, 2x RTX PRO
6000; see benchmark/context_certificate in the repo history for the
measurement artifacts):

- **Shrunk pool + reserve**: at arm time, KEEP_PAGES pages stay in the
  runtime allocator (the admission budget: input + max_new + page must
  fit KEEP_PAGES * page_size tokens) and the rest become the tier's
  reserve of staging frames. The runtime's pool-accounting identity
  (available + evictable + protected + uncached == allocator.size)
  stays balanced: the allocator's size is lowered to the kept budget,
  frees of reserve-owned frames are filtered, and frees already returned
  by retirement are deduplicated (the runtime's finish path re-frees the
  full original extent).
- **Streaming prefill**: each prefill chunk's pages are mirrored to the
  pinned host buffer, then pages below the hot horizon are logically
  retired - columns zeroed, frames returned. Availability only grows.
- **Decode with CUDA graphs**: replays skip Python entirely, so the tier
  runs BETWEEN replays from the scheduler loop
  (``TierManager.tick``): it reads each row's stashed selection (the
  backend's own QSAMTPSharedSparseIndices, written by capture() inside
  the draft-extend graph), restores those pages for the next iteration,
  and re-retires the previous iteration's pages - the current selection
  is exempt (selections live below the horizon by construction). The
  hot horizon (default 16k tokens) absorbs the one-iteration selection
  drift; residual misses are counted.
- **Correctness instrumentation**: bitwise verify of staged restores
  (sampled), a coverage audit proving every selected page was resident
  at forward time, and counters for every free path.

Known limits (documented in HOST_KV_PRODUCTION_PLAN.md): the admission
budget caps the reserve (~1.5k pages at 101k contexts), so concurrent
prefill chunk-gathers (~440 pages each) can starve restores - starved
pages stay retired and are counted, never silently corrupted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import torch

_PAGE = 64  # must equal the allocator's page size; asserted at arm time


@dataclass
class HostKVConfig:
    enable: bool = False
    # Pages the runtime allocator keeps (admission budget in tokens is
    # keep_pages * page_size). The remainder becomes the staging reserve.
    keep_pages: int = 1600
    # Pages (x page_size tokens) at the context tail that are never
    # retired: absorbs one MTP iteration of selection drift (measured:
    # 64 pages leaked recent-band misses; 256 eliminates them).
    hot_pages: int = 256
    prefetch: bool = True
    prefetch_radius: int = 1
    # Reserve frames kept free for synchronous restores.
    prefetch_floor: int = 128
    # Verify every Nth staged page bitwise against the mirror.
    verify_sample: int = 8
    verify: bool = True
    stats_path: Optional[str] = None
    stats_interval_s: float = 20.0
    arm_after: int = 4

    @staticmethod
    def from_server_args(server_args: Any) -> "HostKVConfig":
        return HostKVConfig(
            enable=bool(getattr(server_args, "host_kv_tier", False)),
            keep_pages=int(getattr(server_args, "host_kv_keep_pages", 1600)),
            hot_pages=int(getattr(server_args, "host_kv_hot_pages", 256)),
            prefetch=bool(getattr(server_args, "host_kv_prefetch", True)),
            prefetch_radius=int(getattr(
                server_args, "host_kv_prefetch_radius", 1)),
            prefetch_floor=int(getattr(
                server_args, "host_kv_reserve_floor", 128)),
            verify_sample=int(getattr(
                server_args, "host_kv_verify_sample", 8)),
            stats_path=getattr(server_args, "host_kv_stats_path", None),
        )


def _log(msg: str) -> None:
    print(f"[host-kv {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class HostKVTier:
    """Per-scheduler-process host KV tier over one full-attention pool.

    The tier owns: the pinned host mirror (slot-indexed), the reserve
    frame pool, per-request row state, and the allocator accounting
    wraps that keep the runtime's pool identity balanced while pages
    move between the GPU pool and the host.
    """

    def __init__(self, config: HostKVConfig):
        self.cfg = config
        self.armed = False
        self.pool: Any = None
        self.allocator: Any = None
        self.table: Any = None
        self.layers: list = []
        self.mirror: Dict[int, Any] = {}
        self.reserve: Optional[torch.Tensor] = None
        self.reserve_tokens = 0
        self.sink_page: Optional[int] = None
        self.tier_frames: Set[int] = set()
        self.returned_tokens = 0
        # Wrapped allocator originals for the accounting mode.
        self._orig_free: Any = None
        self._orig_free_seg: Any = None
        self._orig_available: Any = None
        self._orig_size: int = 0
        # Rows: row -> {ctx_written, mirrored, retired, page_slots,
        # re_retire, restored, prefetched, decode_streak, last_sel_pages}
        self.rows: Dict[int, Dict[str, Any]] = {}
        self.max_rows = 32
        self._arm_calls = 0
        self._last_key: Any = None
        # Counters.
        self.counters: Dict[str, Any] = {
            "staged_pages": 0, "verify_mismatches": 0,
            "prefetch_issued": 0, "prefetch_hits": 0,
            "coverage_misses": 0, "restore_starved": 0,
            "clamp_skips": 0, "dedup_total": 0, "dedup_dropped": 0,
            "dedup_freed": 0, "ff_total": 0, "ff_dropped_tier": 0,
            "ff_dropped_reserve": 0, "br_ticks": 0, "br_errors": 0,
            "ws_sum": 0, "ws_n": 0, "drift_sum": 0, "drift_n": 0,
            "miss_recent": 0, "miss_deep": 0,
        }
        self.selfcheck: Optional[Dict[str, Any]] = None
        self._stream: Any = None
        self._stats_written_at = 0.0

    # ------------------------------------------------------------------
    # arm / accounting mode

    def arm(self, backend: Any, row: int) -> None:
        """Idempotent arm on a hooked forward; confiscates on the
        arm_after'th call (warmup requests share row 0; arming on them
        corrupts accounting)."""
        if self.armed or not self.cfg.enable:
            return
        self._arm_calls += 1
        if self._arm_calls < self.cfg.arm_after:
            return
        pool = getattr(backend, "token_to_kv_pool", None)
        layers = []
        if pool is not None and getattr(
                pool, "full_attention_layer_id_mapping", None):
            layers = sorted(int(l)
                            for l in pool.full_attention_layer_id_mapping)
        if len(layers) < 8:
            return
        allocator = self._discover_allocator(backend)
        table = getattr(backend, "req_to_token", None)
        if allocator is None or table is None:
            _log("arm: allocator/table not found; dormant")
            self.armed = True
            return
        page = int(getattr(allocator, "page_size", _PAGE) or _PAGE)
        if page != _PAGE:
            _log(f"arm: allocator page_size={page} != {_PAGE}; dormant")
            self.armed = True
            return
        self._confiscate(allocator)
        self.mirror = {
            layer: (
                torch.zeros_like(pool.get_key_buffer(layer), device="cpu")
                .pin_memory()
                if pool.get_key_buffer(layer).is_cuda
                else torch.zeros_like(pool.get_key_buffer(layer)),
                torch.zeros_like(pool.get_value_buffer(layer), device="cpu")
                .pin_memory()
                if pool.get_value_buffer(layer).is_cuda
                else torch.zeros_like(pool.get_value_buffer(layer)),
            )
            for layer in layers
        }
        self.pool, self.allocator, self.table, self.layers = (
            pool, allocator, table, layers)
        self.armed = True
        self._install_allocator_wraps()
        _log(f"armed: {len(layers)} layers, budget="
             f"{allocator.size} tokens, reserve={self.reserve_tokens}")

    def _discover_allocator(self, backend: Any) -> Any:
        for owner in (backend, getattr(backend, "runner", None),
                      getattr(getattr(backend, "runner", None),
                              "scheduler", None)):
            if owner is None:
                continue
            for attr in ("token_to_kv_pool_allocator", "allocator"):
                obj = getattr(owner, attr, None)
                if obj is not None and hasattr(obj, "free_pages"):
                    return obj
        return None

    def _confiscate(self, allocator: Any) -> None:
        free = allocator.free_pages
        keep = self.cfg.keep_pages
        if int(free.numel()) <= keep:
            _log(f"confiscate: pool already <= keep "
                 f"({int(free.numel())}); nothing taken")
            self.reserve = torch.empty(0, dtype=free.dtype,
                                       device=free.device)
            self.reserve_tokens = 0
            self.sink_page = None
            return
        self.reserve = free[keep:].clone()
        allocator.free_pages = free[:keep].clone()
        self.sink_page = int(self.reserve[0].item())
        self.reserve_tokens = int(self.reserve.numel()) * _PAGE
        # + page 0 (the padding page): retired columns read as zeros and
        # the finish path would otherwise "free" page 0 into circulation.
        self.tier_frames = set(self.reserve.tolist()) | {0}
        self._orig_size = allocator.size
        allocator.size = keep * _PAGE
        if hasattr(allocator, "full_tokens_per_layer"):
            allocator.full_tokens_per_layer = allocator.size

    def _install_allocator_wraps(self) -> None:
        allocator = self.allocator
        self._orig_free = allocator.free
        self._orig_free_seg = allocator.free_segment
        self._orig_available = allocator.available_size

        def _tier_available(_orig=self._orig_available):
            return _orig() - self.returned_tokens

        allocator.available_size = _tier_available

        def _filtered_free(free_index, _orig=self._orig_free, _self=self):
            return _self._filtered_free(free_index, _orig)

        def _filtered_free_seg(free_index, *, start_pos,
                               _orig=self._orig_free_seg, _self=self):
            keep = _self._dedup_filter(free_index)
            if int(keep.numel()) == 0:
                return None
            return _orig(keep, start_pos=start_pos)

        allocator.free = _filtered_free
        allocator.free_segment = _filtered_free_seg

    def _reserve_frames(self) -> Set[int]:
        if self.reserve is None:
            return set()
        return set(self.reserve.tolist())

    def _dedup_filter(self, free_index: torch.Tensor) -> torch.Tensor:
        """Slots whose frames are tier-owned never re-enter circulation."""
        pages = (free_index // _PAGE).tolist()
        reserve_now = self._reserve_frames()
        keep = [f for f, p in zip(free_index.tolist(), pages)
                if p not in self.tier_frames and p not in reserve_now]
        self.counters["ff_total"] += len(pages)
        self.counters["ff_dropped_tier"] += sum(
            1 for p in pages if p in self.tier_frames)
        self.counters["ff_dropped_reserve"] += sum(
            1 for p in pages if p in reserve_now)
        return torch.tensor(keep, dtype=free_index.dtype,
                            device=free_index.device)

    def _filtered_free(self, free_index, _orig):
        keep = self._dedup_filter(free_index)
        if int(keep.numel()) == 0:
            return None
        return self._dedup_free(keep, _orig)

    def _dedup_free(self, slots: torch.Tensor, _orig=None) -> int:
        """Free frames not already free; returns pages freed."""
        if _orig is None:
            _orig = self._orig_free
        already = set(self.allocator.free_pages.tolist()) | \
            self._reserve_frames()
        frames = set(int(s) // _PAGE for s in slots.tolist())
        fresh = [f * _PAGE for f in frames if f not in already]
        self.counters["dedup_total"] += len(frames)
        self.counters["dedup_dropped"] += len(frames) - len(fresh)
        self.counters["dedup_freed"] += len(fresh)
        if not fresh:
            return 0
        tok = torch.tensor(fresh, dtype=self.allocator.free_pages.dtype,
                           device=self.allocator.free_pages.device)
        _orig(tok)
        return len(fresh)

    # ------------------------------------------------------------------
    # hooks (eager prefill path)

    def _stream_obj(self):
        if self._stream is None and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()
        return self._stream

    def _row_state(self, row: int) -> Optional[Dict[str, Any]]:
        st = self.rows.get(row)
        if st is None:
            if len(self.rows) >= self.max_rows:
                return None
            st = self.rows[row] = {
                "ctx_written": 0, "chunks": 0,
                "mirrored": set(), "retired": set(),
                "page_slots": {}, "re_retire": set(),
                "restored": set(), "prefetched": set(),
                "decode_streak": 0, "last_horizon": 0,
            }
        return st

    def before_gather(self, topk_indices: torch.Tensor,
                      forward_batch: Any) -> None:
        """Pre-forward: restore the selected retired pages (bulk)."""
        if not self.armed or not self.rows:
            return
        rows = getattr(forward_batch, "req_pool_indices", None)
        if rows is None or rows.numel() == 0:
            return
        positions = topk_indices
        if positions.dim() > 1:
            positions = (positions[0] if rows.numel() == 1
                         else positions.flatten())
        positions = positions[positions >= 0]
        if positions.numel() == 0:
            return
        ext = getattr(forward_batch, "extend_seq_lens", None)
        if (rows.numel() > 1 and ext is not None
                and ext.numel() == rows.numel()):
            flat = positions.flatten()
            bounds = torch.cumsum(ext, 0).tolist()
            lo = 0
            for i, row_id in enumerate(rows.tolist()):
                hi = int(min(bounds[i], flat.numel()))
                st = self._row_state(int(row_id))
                if st is not None and hi > lo:
                    self._restore_row_pages(int(row_id), st, flat[lo:hi])
                lo = hi
            return
        st = self._row_state(int(rows[0].item()))
        if st is not None:
            self._restore_row_pages(int(rows[0].item()), st, positions)

    def after_write(self, forward_batch: Any) -> None:
        """Post-forward: mirror fresh pages, retire below the horizon."""
        if not self.armed:
            return
        out_locs = getattr(forward_batch, "out_cache_loc", None)
        if out_locs is None or out_locs.numel() == 0:
            return
        rows = getattr(forward_batch, "req_pool_indices", None)
        if rows is None or rows.numel() == 0:
            return
        # Dedupe by content, globally: one forward serves all layers and
        # the same (first-slot, size) recurs per layer.
        key = (int(out_locs[0].item()), int(out_locs.numel()))
        if key == self._last_key:
            return
        self._last_key = key
        seq = getattr(forward_batch, "seq_lens", None)
        ext = getattr(forward_batch, "extend_seq_lens", None)

        def _total(i: int) -> Optional[int]:
            # seq_lens is the per-row RUNNING total; extend_seq_lens is
            # per-batch delta on the single-row paths of this build.
            if seq is not None and seq.numel() > i:
                return int(seq[i].item())
            if ext is not None and ext.numel() > i:
                return int(ext[i].item())
            return None

        if (ext is not None and ext.numel() == rows.numel()
                and int(rows.numel()) > 1):
            bounds = torch.cumsum(ext, 0).tolist()
            lo = 0
            for i, row_id in enumerate(rows.tolist()):
                hi = int(min(bounds[i], out_locs.numel()))
                st = self._row_state(int(row_id))
                if st is not None and hi > lo:
                    self._write_row(int(row_id), st, out_locs[lo:hi],
                                    total_ctx=_total(i))
                lo = hi
        else:
            row = int(rows[0].item())
            st = self._row_state(row)
            if st is not None:
                self._write_row(row, st, out_locs, total_ctx=_total(0))
        self._maybe_dump_stats()

    def audit_coverage(self, topk_indices: torch.Tensor,
                       forward_batch: Any) -> None:
        """Post-forward: every selected page must be resident."""
        if not self.armed or not self.rows:
            return
        rows = getattr(forward_batch, "req_pool_indices", None)
        if rows is None or rows.numel() == 0 or topk_indices is None:
            return
        positions = topk_indices
        if positions.dim() > 1:
            positions = (positions[0] if rows.numel() == 1
                         else positions.flatten())
        positions = positions[positions >= 0]
        if positions.numel() == 0:
            return
        ext = getattr(forward_batch, "extend_seq_lens", None)
        if (rows.numel() > 1 and ext is not None
                and ext.numel() == rows.numel()):
            flat = positions.flatten()
            bounds = torch.cumsum(ext, 0).tolist()
            lo = 0
            for i, row_id in enumerate(rows.tolist()):
                hi = int(min(bounds[i], flat.numel()))
                st = self._row_state(int(row_id))
                if st is not None and hi > lo:
                    self._audit_row(int(row_id), st, flat[lo:hi])
                lo = hi
        else:
            st = self._row_state(int(rows[0].item()))
            if st is not None:
                self._audit_row(int(rows[0].item()), st, positions)

    def _audit_row(self, row: int, st: Dict[str, Any],
                   positions: torch.Tensor) -> None:
        positions = positions[positions >= 0]
        if positions.numel() == 0:
            return
        pages = torch.unique(positions // _PAGE).tolist()
        miss = [p for p in pages if p in st["retired"]]
        if miss:
            self.counters["coverage_misses"] += len(miss)
            if self.counters["coverage_misses"] <= 20:
                _log(f"COVERAGE MISS row={row} pages={miss[:6]} "
                     f"(total {self.counters['coverage_misses']})")

    # ------------------------------------------------------------------
    # write path

    def _write_row(self, row: int, st: Dict[str, Any], locs: torch.Tensor,
                   total_ctx: Optional[int] = None) -> None:
        delta = int(locs.numel())
        # Decode/verify batches carry extend_seq_lens as tiny per-row
        # deltas; validating the staleness check against them would wipe
        # live row state every step (the retired=0 regression). Only
        # prefill-scale writes carry a trustworthy total.
        if (delta >= _PAGE and total_ctx is not None
                and st["ctx_written"] > total_ctx + 8 * _PAGE):
            # Rows recycle across sequential requests.
            _log(f"row {row}: stale ctx {st['ctx_written']} > "
                 f"{total_ctx}; resetting recycled row")
            st.clear()
            st.update({"ctx_written": 0, "chunks": 0, "mirrored": set(),
                       "retired": set(), "page_slots": {},
                       "re_retire": set(), "restored": set(),
                       "prefetched": set(), "decode_streak": 0,
                       "last_horizon": 0})
        if delta >= _PAGE:
            st["decode_streak"] = 0
        start, end = st["ctx_written"], st["ctx_written"] + delta
        st["ctx_written"] = end
        st["chunks"] += 1
        table = self.table
        for page in range(start // _PAGE, (end + _PAGE - 1) // _PAGE):
            if page in st["mirrored"]:
                continue
            cols = torch.arange(page * _PAGE, (page + 1) * _PAGE,
                                device=table.device)
            cols = cols[cols < end]
            if cols.numel() == 0:
                continue
            slots = table[row, cols].long()
            valid = slots >= 0
            if not bool(valid.any()):
                continue
            with torch.no_grad():
                for layer in self.layers:
                    mk, mv = self.mirror[layer]
                    mk[slots[valid]] = self.pool.get_key_buffer(layer)[
                        slots[valid]].to("cpu", non_blocking=False)
                    mv[slots[valid]] = self.pool.get_value_buffer(layer)[
                        slots[valid]].to("cpu", non_blocking=False)
            st["mirrored"].add(page)
        if st["chunks"] % 8 == 1:
            _log(f"row {row} chunk #{st['chunks']}: ctx={end} "
                 f"mirrored={len(st['mirrored'])} "
                 f"retired={len(st['retired'])}")
        self._retire(row, st, end)

    # ------------------------------------------------------------------
    # retire (bulk, frontier-tracked)

    def _retire(self, row: int, st: Dict[str, Any], ctx_end: int) -> None:
        horizon = ctx_end // _PAGE - self.cfg.hot_pages
        cands = st["re_retire"]
        last_h = st["last_horizon"]
        for page in range(max(last_h, 0), max(horizon, 0)):
            if page in st["mirrored"]:
                cands.add(page)
        st["last_horizon"] = max(horizon, last_h)
        cands &= st["mirrored"]
        cands -= st["retired"]
        if not cands:
            return
        pages = sorted(cands)
        restored_now = st["restored"]

        remirror = [p for p in pages if p not in restored_now]
        if remirror:
            col_list = [table_cols(p, self.table.device) for p in remirror]
            src_slots = torch.cat(
                [self.table[row, c].long().clone() for c in col_list])
            with torch.no_grad():
                for layer in self.layers:
                    mk, mv = self.mirror[layer]
                    self._copy_gpu_to_host(
                        self.pool.get_key_buffer(layer), mk, src_slots)
                    self._copy_gpu_to_host(
                        self.pool.get_value_buffer(layer), mv, src_slots)
            for p, c in zip(remirror, col_list):
                st["page_slots"][p] = self.table[row, c].long().clone()

        reserve_now = self._reserve_frames()
        to_reserve, to_free_cols = [], []
        for p in pages:
            cols = table_cols(p, self.table.device)
            slots = self.table[row, cols].long().clone()
            if bool((slots < 0).any()):
                continue
            st["page_slots"][p] = slots
            frame = int(slots[0].item()) // _PAGE
            if (self.sink_page is not None
                    and (frame in reserve_now or frame in self.tier_frames)):
                # A staged page re-retiring: its frame is tier-owned.
                to_reserve.append(frame)
            else:
                to_free_cols.append(slots)

        if to_reserve and self.reserve is not None:
            self.reserve = torch.cat([
                self.reserve,
                torch.tensor(to_reserve, dtype=self.reserve.dtype,
                             device=self.reserve.device)])
            self.reserve_tokens = int(self.reserve.numel()) * _PAGE
        if to_free_cols:
            all_slots = torch.cat(to_free_cols)
            self._dedup_free(all_slots)
            # Keep kv_allocated_len above the committed window so the
            # runtime's finish free range never inverts.
            req = self._req_of(row)
            if req is not None and getattr(req, "kv", None) is not None:
                floor = self._committed_floor(req)
                budget = _PAGE * len(to_free_cols)
                if req.kv.kv_allocated_len - budget >= floor:
                    req.kv.kv_allocated_len -= budget
                else:
                    self.counters["clamp_skips"] += 1

        pages_t = torch.tensor(pages, dtype=torch.int64,
                               device=self.table.device)
        cols_all = (pages_t[:, None] * _PAGE
                    + torch.arange(_PAGE,
                                   device=self.table.device)[None, :]
                    ).flatten()
        with torch.no_grad():
            self.table[row, cols_all] = 0
        st["retired"].update(pages)
        restored_now -= set(pages)

    def _committed_floor(self, req: Any) -> int:
        try:
            return int(req.effective_kv_committed_len())
        except Exception:
            return len(getattr(req, "output_ids", []) or [])

    _scheduler: Any = None

    def _req_of(self, row: int) -> Any:
        sched = self._scheduler
        if sched is None:
            return None
        for batch in (getattr(sched, "running_batch", None),
                      getattr(sched, "chunked_req", None)):
            if batch is None:
                continue
            reqs = getattr(batch, "reqs", None) or (
                [batch] if hasattr(batch, "req_pool_idx") else [])
            for req in reqs:
                if getattr(req, "req_pool_idx", None) == row:
                    return req
        return None

    # ------------------------------------------------------------------
    # restore (bulk, src != dst)

    def _restore_row_pages(self, row: int, st: Dict[str, Any],
                           positions: torch.Tensor) -> None:
        positions = positions[positions >= 0]
        if positions.numel() == 0:
            return
        pages = torch.unique(positions // _PAGE).tolist()
        prefetched = st["prefetched"]
        need = [p for p in pages if p in st["retired"]]
        device = self.table.device

        fast = [p for p in need if p in prefetched]
        sync_pages = [p for p in need if p not in prefetched]

        if fast:
            if torch.cuda.is_available():
                self._stream_obj().synchronize()
            dst_list = [st["page_slots"][p] for p in fast]
            pages_t = torch.tensor(fast, dtype=torch.int64, device=device)
            cols_all = (pages_t[:, None] * _PAGE
                        + torch.arange(_PAGE, device=device)[None, :]
                        ).flatten()
            slots_all = torch.cat(dst_list)
            with torch.no_grad():
                self.table[row, cols_all] = slots_all.to(self.table.dtype)
            for p in fast:
                prefetched.discard(p)
                st["retired"].discard(p)
                st["restored"].add(p)
                st["re_retire"].add(p)
            self.counters["staged_pages"] += len(fast)
            self.counters["prefetch_hits"] += len(fast)

        if sync_pages:
            # Never restore in place: original slots were freed at
            # retirement and may belong to another row. Starved pages
            # stay retired (a visible, counted coverage miss).
            room = (int(self.reserve.numel())
                    if self.reserve is not None else 0)
            if room < len(sync_pages):
                self.counters["restore_starved"] += (
                    len(sync_pages) - max(room, 0))
                sync_pages = sync_pages[:max(room, 0)]
            if sync_pages:
                n = len(sync_pages)
                frames = self.reserve[:n]
                self.reserve = self.reserve[n:]
                self.reserve_tokens = int(self.reserve.numel()) * _PAGE
                src_list, dst_list = [], []
                for p, frame_page in zip(sync_pages, frames.tolist()):
                    src_list.append(st["page_slots"][p])
                    frame_slots = torch.arange(
                        int(frame_page) * _PAGE, (int(frame_page) + 1)
                        * _PAGE, dtype=src_list[-1].dtype, device=device)
                    st["page_slots"][p] = frame_slots
                    dst_list.append(frame_slots)
                src_cpu = torch.cat(src_list).cpu()
                dst = torch.cat(dst_list)
                with torch.no_grad():
                    for layer in self.layers:
                        mk, mv = self.mirror[layer]
                        if torch.cuda.is_available():
                            stream = self._stream_obj()
                            with torch.cuda.stream(stream):
                                self.pool.get_key_buffer(layer)[dst] = \
                                    mk[src_cpu].to(dst.device,
                                                   non_blocking=True)
                                self.pool.get_value_buffer(layer)[dst] = \
                                    mv[src_cpu].to(dst.device,
                                                   non_blocking=True)
                        else:
                            self.pool.get_key_buffer(layer)[dst] = \
                                mk[src_cpu].to(dst.device)
                            self.pool.get_value_buffer(layer)[dst] = \
                                mv[src_cpu].to(dst.device)
                    pages_t = torch.tensor(sync_pages, dtype=torch.int64,
                                           device=device)
                    cols_all = (pages_t[:, None] * _PAGE
                                + torch.arange(_PAGE, device=device)
                                [None, :]).flatten()
                    self.table[row, cols_all] = dst.to(self.table.dtype)
                if torch.cuda.is_available():
                    self._stream_obj().synchronize()
                if (self.cfg.verify and self.selfcheck is None
                        and self.counters["staged_pages"]
                        % max(self.cfg.verify_sample, 1) == 0):
                    self._verify_staged(sync_pages[0], src_list[0],
                                        dst_list[0], row)
                for p in sync_pages:
                    st["retired"].discard(p)
                    st["restored"].add(p)
                    st["re_retire"].add(p)
                self.counters["staged_pages"] += n

        if (self.cfg.prefetch and self.cfg.prefetch_radius > 0
                and st["retired"] and self.reserve is not None):
            cand = set()
            for p in pages:
                for d in range(1, self.cfg.prefetch_radius + 1):
                    cand.add(p - d)
                    cand.add(p + d)
            want = sorted(p for p in cand
                          if p >= 0 and p in st["retired"]
                          and p not in prefetched)
            room = int(self.reserve.numel()) - self.cfg.prefetch_floor
            if want and room > 0:
                take = want[:room]
                frames = self.reserve[:len(take)]
                self.reserve = self.reserve[len(take):]
                self.reserve_tokens = int(self.reserve.numel()) * _PAGE
                src_list, dst_list = [], []
                for p, frame_page in zip(take, frames.tolist()):
                    src_list.append(st["page_slots"][p])
                    frame_slots = torch.arange(
                        int(frame_page) * _PAGE, (int(frame_page) + 1)
                        * _PAGE, dtype=src_list[-1].dtype, device=device)
                    st["page_slots"][p] = frame_slots
                    dst_list.append(frame_slots)
                src_cpu = torch.cat(src_list).cpu()
                dst = torch.cat(dst_list)
                with torch.no_grad():
                    for layer in self.layers:
                        mk, mv = self.mirror[layer]
                        if torch.cuda.is_available():
                            stream = self._stream_obj()
                            with torch.cuda.stream(stream):
                                self.pool.get_key_buffer(layer)[dst] = \
                                    mk[src_cpu].to(dst.device,
                                                   non_blocking=True)
                                self.pool.get_value_buffer(layer)[dst] = \
                                    mv[src_cpu].to(dst.device,
                                                   non_blocking=True)
                        else:
                            self.pool.get_key_buffer(layer)[dst] = \
                                mk[src_cpu].to(dst.device)
                            self.pool.get_value_buffer(layer)[dst] = \
                                mv[src_cpu].to(dst.device)
                for p in take:
                    prefetched.add(p)
                self.counters["prefetch_issued"] += len(take)

    def _verify_staged(self, page: int, src_slots: torch.Tensor,
                       dst_slots: torch.Tensor, row: int) -> None:
        ok = True
        src_cpu = src_slots.cpu()
        for layer in self.layers:
            mk, mv = self.mirror[layer]
            gk = self.pool.get_key_buffer(layer)[dst_slots]
            if gk.is_cuda:
                gk = gk.cpu()
            ok &= torch.equal(gk.view(torch.uint16),
                              mk[src_cpu].view(torch.uint16))
        self.selfcheck = {"page": page, "row": row, "verified": bool(ok)}
        self.counters["verify_mismatches"] += int(not ok)
        _log(f"selfcheck row {row} page {page}: verified={ok}")

    def _copy_gpu_to_host(self, gpu_tensor, host_tensor, slots) -> None:
        # Gather on the GPU first, one contiguous D2H per call.
        gathered = gpu_tensor[slots]
        host_tensor[slots.cpu()] = (
            gathered.to("cpu", non_blocking=True)
            if gathered.is_cuda else gathered)
        if gathered.is_cuda and torch.cuda.is_available():
            self._stream_obj().synchronize()

    # ------------------------------------------------------------------
    # between-replay tick (scheduler-owned)

    def tick(self, selection_owner: Any) -> None:
        """Between CUDA graph replays: restore each row's stashed
        selection for the next iteration, re-retire the previous one.

        ``selection_owner`` is the backend object carrying
        ``_mtp_shared_sparse_indices`` (see glue.resolve_selection_stash).
        """
        if not self.armed or not self.rows:
            return
        state = getattr(selection_owner, "_mtp_shared_sparse_indices",
                        None)
        if state is None:
            return
        try:
            rows = sorted(self.rows.keys())
            rows_t = torch.tensor(rows, dtype=torch.long,
                                  device=state.indices.device)
            idx = state.indices[0, rows_t].cpu()
            lens = state.captured_len[0, rows_t].cpu()
        except Exception as e:  # noqa: BLE001 - diagnostic only
            self.counters["br_errors"] += 1
            if self.counters["br_errors"] <= 5:
                _log(f"tick read error: {e!r}")
            return
        for i, row in enumerate(rows):
            st = self.rows[row]
            # QSAMTPSharedSparseIndices initializes captured_len to ONES:
            # a row reading <= 1 is recycled or never captured. Its stash
            # slot may already hold ANOTHER request's selection; restoring
            # or auditing against it drains the reserve into a dead row
            # and starves the live one (the N=4 native wedge: 2.4B
            # coverage misses, zero progress). Drop the dead state.
            cap_len = int(lens[i].item())
            if cap_len <= 1:
                # Not yet captured: warmup dummy or a request still in
                # PREFILL (its first draft-extend capture comes only
                # after prefill). Row lifecycle is owned by forget_row.
                continue
            if abs(cap_len - st["ctx_written"]) > 4 * _PAGE + 256:
                # Stale capture on a recycled slot: the stash still
                # holds the PREVIOUS request's selection while this row
                # prefills a new one - restoring it services dead pages
                # every iteration and wedges the loop.
                continue
            pos = idx[i]
            sel = pos[pos >= 0]
            sel_set = set(torch.unique(sel // _PAGE).tolist())
            self.counters["ws_sum"] += len(sel_set)
            self.counters["ws_n"] += 1
            prev = st.get("last_sel_pages")
            if prev is not None:
                self.counters["drift_sum"] += len(sel_set - prev)
                self.counters["drift_n"] += 1
            st["last_sel_pages"] = sel_set
            self._restore_row_pages(int(row), st, sel)
            # The selection IS the next replay's working set - its pages
            # live below the retire horizon by construction; keep them
            # out of the re-retire queue or this tick would retire what
            # it just restored.
            st["re_retire"].difference_update(sel_set)
            self._retire(int(row), st, int(lens[i].item()))
            pages = sorted(sel_set)
            miss = [p for p in pages if p in st["retired"]]
            if miss:
                self.counters["coverage_misses"] += len(miss)
                end_page = int(lens[i].item()) // _PAGE
                for p in miss:
                    if end_page - p <= 256:
                        self.counters["miss_recent"] += 1
                    else:
                        self.counters["miss_deep"] += 1
                if self.counters["coverage_misses"] <= 20:
                    _log(f"TICK COVERAGE MISS row={row} pages={miss[:6]} "
                         f"(total {self.counters['coverage_misses']})")
        self.counters["br_ticks"] += 1
        self._maybe_dump_stats()

    # ------------------------------------------------------------------
    # scheduler integration

    def attach_to_backends(self, tp_worker: Any) -> None:
        """Register self on the QSA backend(s) so the forward hooks can
        reach the tier, and pre-arm against the first rows seen."""
        mr = getattr(tp_worker, "model_runner", None)
        for backend in (getattr(mr, "attn_backend", None),):
            if backend is None:
                continue
            for resolved in (backend,
                             getattr(backend, "full_attn_backend", None)):
                if resolved is not None and hasattr(
                        resolved, "forward_extend"):
                    resolved.host_kv_tier = self

    def forget_row(self, row: int) -> None:
        """Scheduler-driven row cleanup at request finish - the only
        reliable death signal. The stash-row heuristics were wrong: a
        live request mid-prefill ALSO reads captured_len==1 (capture
        happens at the first draft-extend, after prefill), so the tick
        was deleting live rows every iteration and exploding starvation
        (406k starved restores on a single prefiling request).

        Every live page holds a RESERVE frame (restored or prefetched);
        dropping state without returning them leaks the reserve - across
        sequential requests the pool drained empty and later requests
        starved every restore (1.39B starved at N=8). Return all live
        frames before dropping the row."""
        st = self.rows.pop(row, None)
        if st is None:
            return
        frames = []
        table = self.table
        for page in list(st.get("restored", set())) + list(
                st.get("prefetched", set())):
            slots = st["page_slots"].get(page)
            if slots is None or int(slots.numel()) == 0:
                continue
            frames.append(int(slots[0].item()) // _PAGE)
            if page in st.get("restored", set()):
                cols = torch.arange(page * _PAGE, (page + 1) * _PAGE,
                                    device=table.device)
                with torch.no_grad():
                    table[row, cols] = 0
        if frames and self.reserve is not None:
            self.reserve = torch.cat([
                self.reserve,
                torch.tensor(sorted(set(frames)),
                             dtype=self.reserve.dtype,
                             device=self.reserve.device)])
            self.reserve_tokens = int(self.reserve.numel()) * _PAGE
        self.counters["forgotten_rows"] = \
            self.counters.get("forgotten_rows", 0) + 1
        self.counters["reclaimed_frames"] = \
            self.counters.get("reclaimed_frames", 0) + len(set(frames))
        if self.cfg.stats_path:
            self.dump_stats()

    def scheduler_tick(self, model_worker: Any) -> None:
        """Scheduler-side between-replay entry: resolve the selection
        stash owner and run one tier tick."""
        if not self.armed:
            return
        try:
            from sglang.srt.layers.attention.qsa.glue import (
                resolve_selection_stash,
            )
        except ImportError:
            return
        owner = resolve_selection_stash(model_worker)
        if owner is not None:
            self.tick(owner)

    # ------------------------------------------------------------------
    # stats

    def _maybe_dump_stats(self) -> None:
        path = self.cfg.stats_path
        if not path:
            return
        now = time.monotonic()
        if now - self._stats_written_at < self.cfg.stats_interval_s:
            return
        self._stats_written_at = now
        self.dump_stats()

    def dump_stats(self) -> None:
        if not self.cfg.stats_path:
            return
        payload = {
            "armed": self.armed,
            "layers_n": len(self.layers),
            "requests": len(self.rows),
            "ws_avg": round(self.counters["ws_sum"]
                            / max(self.counters["ws_n"], 1), 1),
            "drift_avg": round(self.counters["drift_sum"]
                               / max(self.counters["drift_n"], 1), 1),
            **{k: v for k, v in self.counters.items()},
            "selfcheck": self.selfcheck,
            "per_request": {
                str(r): {"ctx": st["ctx_written"],
                         "mirrored": len(st["mirrored"]),
                         "retired": len(st["retired"])}
                for r, st in self.rows.items()
            },
        }
        try:
            with open(self.cfg.stats_path, "w") as handle:
                json.dump(payload, handle, indent=2)
        except OSError:
            pass


def table_cols(page: int, device: torch.device) -> torch.Tensor:
    return torch.arange(page * _PAGE, (page + 1) * _PAGE, device=device)
