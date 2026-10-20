"""P2.3 host tier (G1: single request): shrunk pool, streaming, staging.

Correct model (see HOST_KV_DESIGN.md P2.3):

- The request's context positions [0, len) are divided into logical pages
  (page = 64 context positions). req_to_token[row, ctx_pos] holds the
  pool SLOT storing that token.
- The host mirror is keyed by CONTEXT page; its content is the byte image
  of that page wherever it last lived on GPU.
- Streaming (write path, post-call): after a prefill chunk's writes are
  mirrored, old context pages are freed — their slots return to the
  runtime allocator via allocator.free(slot_range) and their table columns
  are cleared to 0.
- Staging (gather path, pre-call): a selected context page whose columns
  read 0 is re-allocated (allocator.alloc(64) → a fresh, page-aligned
  frame), its bytes copied from the mirror INTO THE FRAME'S SLOTS, and its
  table columns rewritten to the frame's slots. Relocation, never
  in-place: freed slots may already belong to other pages.
- Byte-verifies the first staged page (self-check) and counts everything.

Env: SGLANG_HOSTTIER_ENABLE=1, SGLANG_HOSTTIER_LOG=path,
SGLANG_HOSTTIER_KEEP_FREE=pages (stream until the allocator has this many
free pages; default: half the pool), SGLANG_HOSTTIER_VERIFY=1.
"""

import atexit
import json
import os
import time

import torch

_PREFIX = "SGLANG_HOSTTIER_"
ENABLE = os.environ.get(_PREFIX + "ENABLE", "") == "1"
LOG_PATH = os.environ.get(_PREFIX + "23_LOG",
                          "/root/phase0/hosttier23-stats.json")
KEEP_FREE = int(os.environ.get(_PREFIX + "KEEP_FREE", 0))
VERIFY = os.environ.get(_PREFIX + "VERIFY", "1") == "1"
PAGE = 64

_state = {"armed": False, "layers": None, "host_mirror": None, "pool": None,
          "allocator": None, "table": None, "row": None,
          "ctx_written": 0, "mirrored_pages": set(), "freed_pages": set(),
          "staged_pages": 0, "verify_mismatches": 0, "selfcheck": None,
          "last_exception": None}


def _info(msg):
    print(f"[hosttier23 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump_stats():
    def _plain(v):
        if isinstance(v, set):
            return sorted(x if not torch.is_tensor(x) else int(x) for x in v)
        if torch.is_tensor(v):
            return int(v.numel())
        return v
    payload = {k: _plain(v) for k, v in _state.items()
               if k not in ("host_mirror", "pool", "allocator", "table", "layers",
                            "reserve_pages", "page_slots")}
    payload["page_slots_n"] = len(_state.get("page_slots") or {})
    payload["layers_n"] = len(_state["layers"] or [])
    try:
        with open(LOG_PATH, "w") as handle:
            json.dump(payload, handle, indent=2)
    except OSError:
        pass


if ENABLE:
    _info(f"P2.3 module loaded (page={PAGE}, verify={VERIFY})")
    atexit.register(dump_stats)


def _discover_allocator(carrier):
    for obj_owner in (carrier, getattr(carrier, "runner", None),
                      getattr(getattr(carrier, "runner", None), "scheduler", None)):
        if obj_owner is None:
            continue
        for attr in ("token_to_kv_pool_allocator", "allocator"):
            obj = getattr(obj_owner, attr, None)
            if obj is not None and hasattr(obj, "free_pages"):
                return obj
    return None


ARM_AFTER = int(os.environ.get(_PREFIX + "ARM_AFTER", "4"))
_call_count = {"n": 0}


def arm(backend, row):
    if _state["armed"] or not ENABLE:
        return
    _call_count["n"] += 1
    if _call_count["n"] < ARM_AFTER:
        return  # warmup requests share row 0; arming on them corrupts
                # ctx accounting (the residual double-count)
    pool = getattr(backend, "token_to_kv_pool", None)
    layers = (sorted(int(l) for l in pool.full_attention_layer_id_mapping)
              if pool is not None and getattr(pool, "full_attention_layer_id_mapping", None) else [])
    if len(layers) < 8:
        return
    allocator = _discover_allocator(backend)
    table = getattr(backend, "req_to_token", None)
    if allocator is None or table is None:
        _info(f"discovery: allocator={allocator is not None} table={table is not None}; dormant")
        _state["armed"] = True
        return
    mirror = {layer: (torch.zeros_like(pool.get_key_buffer(layer), device="cpu"),
                      torch.zeros_like(pool.get_value_buffer(layer), device="cpu"))
              for layer in layers}
    slots = pool.get_key_buffer(layers[0]).shape[0]
    # No confiscation: the runtime's pool-accounting watchdog flags pages
    # that vanish from free_pages without an owning request. Instead the
    # pool launches full-size (admission-fair), and streaming FREES old
    # pages at decode pressure — availability only grows, which the
    # accounting tolerates. The G1 memory-savings claim is carried by
    # freed_pages: context pages resident on host instead of GPU.
    _state.update(armed=True, layers=layers, host_mirror=mirror, pool=pool,
                  allocator=allocator, table=table, row=int(row))
    _info(f"armed row={row}: {len(layers)} layers, pool slots={slots}, "
          f"free_pages={len(allocator.free_pages)}")


# -- write path ----------------------------------------------------------

def after_write(forward_batch):
    """Post-call: mirror the chunk's context pages, then stream."""
    if not _state["armed"]:
        return
    row, table = _state["row"], _state["table"]
    # Chunk delta = number of slots the runtime just wrote. extend_seq_lens
    # carries per-row TOTAL lengths (its sum overcounts by the prefix), and
    # target-verify steps also present as extends: both inflated ctx_written
    # and the retirement totals (window-4's 3962 > 2621 pages).
    out_locs = getattr(forward_batch, "out_cache_loc", None)
    delta = int(out_locs.numel()) if out_locs is not None else 0
    if delta == 0:
        return  # decode appends flow through the same path next extend
    # The wrapper fires once PER full-attention layer for the same chunk:
    # dedupe by the out_cache_loc identity, or ctx_written inflates ~1.6x
    # and reads past the request's real end (the window-5 device assert).
    # Dedupe by CONTENT: all layers of one forward receive the same
    # out_cache_loc values, while successive chunks allocate different
    # slots (object identity fails — the backend builds per-layer batch
    # objects; buffer reuse defeats data_ptr).
    key = (int(out_locs[0].item()), delta)
    if key == _state.get("last_chunk_key"):
        return
    _state["last_chunk_key"] = key
    start, end = _state["ctx_written"], _state["ctx_written"] + delta
    _state["ctx_written"] = end
    _state["chunks"] = _state.get("chunks", 0) + 1
    if _state["chunks"] % 4 == 1:
        _info(f"chunk #{_state['chunks']}: delta={delta} ctx_written={end} "
              f"mirrored={len(_state['mirrored_pages'])} retired={len(_state['freed_pages'])}")
    pool, layers, mirror = _state["pool"], _state["layers"], _state["host_mirror"]
    for page in range(start // PAGE, (end + PAGE - 1) // PAGE):
        if page in _state["mirrored_pages"]:
            continue
        cols = torch.arange(page * PAGE, (page + 1) * PAGE, device=table.device)
        cols = cols[cols < end]
        slots = table[row, cols].long()
        valid = slots >= 0
        if not bool(valid.any()):
            continue
        with torch.no_grad():
            # slot-addressed mirror: a page's bytes live at its own rows
            for layer in layers:
                k_gpu = pool.get_key_buffer(layer)[slots[valid]]
                v_gpu = pool.get_value_buffer(layer)[slots[valid]]
                mk, mv = mirror[layer]
                mk[slots[valid]] = k_gpu.to("cpu", non_blocking=False)
                mv[slots[valid]] = v_gpu.to("cpu", non_blocking=False)
        _state["mirrored_pages"].add(page)
    _stream(end)


def _stream(ctx_end):
    """Retire old context pages: mirror (already done per chunk), remember
    their slots, and clear the table rows. No allocator interaction — the
    runtime's own request cleanup owns those frees (double-free assert
    otherwise). The pages stay slot-owned by the request; retirement is a
    logical eviction the gather path restores in place on demand."""
    table, row = _state["table"], _state["row"]
    hot_pages = int(os.environ.get(_PREFIX + "HOT_PAGES", "512"))
    # 512 pages = 32k tokens: retire only below the recent-selection
    # horizon observed in this workload (selections reached ~mid-context).
    retired = 0
    pool, layers, mirror = _state["pool"], _state["layers"], _state["host_mirror"]
    for page in sorted(_state["mirrored_pages"]):
        if page in _state["freed_pages"] or page >= ctx_end // PAGE - hot_pages:
            continue
        cols = torch.arange(page * PAGE, (page + 1) * PAGE, device=table.device)
        slots = table[row, cols].long().clone()
        if bool((slots < 0).any()):
            continue
        with torch.no_grad():  # retire-time re-mirror: authoritative bytes
            for layer in layers:
                mk, mv = mirror[layer]
                mk[slots] = pool.get_key_buffer(layer)[slots].to("cpu", non_blocking=False)
                mv[slots] = pool.get_value_buffer(layer)[slots].to("cpu", non_blocking=False)
        _state.setdefault("page_slots", {})[page] = slots
        table[row, cols] = 0
        _state["freed_pages"].add(page)
        retired += 1
    if retired:
        _info(f"retired {retired} context pages to host "
              f"(total {len(_state['freed_pages'])})")


# -- gather path ---------------------------------------------------------

def before_gather(topk_indices, sequence_length):
    """Pre-call: restore selected retired pages in place from the mirror."""
    if not _state["armed"] or not _state["freed_pages"]:
        return
    positions = topk_indices[topk_indices >= 0].flatten()
    if positions.numel() == 0:
        return
    pages = torch.unique(positions // PAGE).tolist()
    need = [p for p in pages if p in _state["freed_pages"]]
    if not need:
        return
    st = _state
    table, row = st["table"], st["row"]
    pool, layers, mirror = st["pool"], st["layers"], st["host_mirror"]
    for page in need:
        slots = st["page_slots"][page]
        slots_cpu = slots.cpu()
        with torch.no_grad():
            cols = torch.arange(page * PAGE, (page + 1) * PAGE, device=table.device)
            for layer in layers:
                mk, mv = mirror[layer]
                pool.get_key_buffer(layer)[slots] = mk[slots_cpu].to(slots.device)
                pool.get_value_buffer(layer)[slots] = mv[slots_cpu].to(slots.device)
            table[row, cols] = slots.to(table.dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if VERIFY and st["selfcheck"] is None:
            ok = True
            for layer in layers:
                gk = pool.get_key_buffer(layer)[slots]
                if gk.is_cuda:
                    gk = gk.cpu()
                # BIT-level equality: torch.equal says NaN != NaN, so a
                # faithful copy of NaN-bearing bytes would falsely fail.
                ok &= torch.equal(gk.view(torch.uint16), mk[slots_cpu].view(torch.uint16))
            st["selfcheck"] = {"page": page, "mode": "in-place", "verified": bool(ok)}
            st["verify_mismatches"] += int(not ok)
            _info(f"selfcheck restored page {page}: verified={ok}")
        st["staged_pages"] += 1
        st["freed_pages"].discard(page)



