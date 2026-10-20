"""P2.3 host tier, MULTI-REQUEST (G2): the G1 proven model, per row.

G1 (hosttier23.py) proved the mechanism single-request: slot-addressed
host mirror, per-chunk write-through, logical retirement (clear table
rows, remember slots), in-place restore on demand, byte verification.
G2 runs the same model for up to `MAX_REQUESTS` concurrent requests
sharing one pool and one host mirror:

- Discovery/arming is global (pool, allocator, table, 12-layer mirror —
  the mirror is slot-addressed, so requests share it naturally: each
  request's pages live at its own slots).
- All write/retire/restore state is PER-ROW: ctx counters, mirrored and
  retired page sets, remembered slots, chunk-dedupe keys.
- The gather path restores only the requesting row's pages; one request's
  retirement never touches another's columns (verified by CPU tests).

Env: as hosttier23, plus SGLANG_HOSTTIER_MAX_REQUESTS (default 16).
"""

import atexit
import json
import os
import time

import torch

_PREFIX = "SGLANG_HOSTTIER_"
ENABLE = os.environ.get(_PREFIX + "ENABLE", "") == "1"
LOG_PATH = os.environ.get(_PREFIX + "23M_LOG",
                          "/root/phase0/hosttier23m-stats.json")
VERIFY = os.environ.get(_PREFIX + "VERIFY", "1") == "1"
VERIFY_SAMPLE = int(os.environ.get(_PREFIX + "VERIFY_SAMPLE", "0"))  # 0=all
ASYNC = os.environ.get(_PREFIX + "ASYNC", "0") == "1"
PAGE = 64
MAX_REQUESTS = int(os.environ.get(_PREFIX + "MAX_REQUESTS", "16"))
SHRUNK = os.environ.get(_PREFIX + "SHRUNK", "") == "1"
KEEP_PAGES = int(os.environ.get(_PREFIX + "KEEP_PAGES", "1500"))
NO_RADIX = os.environ.get(_PREFIX + "NO_RADIX", "") == "1"
PREFETCH = os.environ.get(_PREFIX + "PREFETCH", "0") == "1"
PREFETCH_RADIUS = int(os.environ.get(_PREFIX + "PREFETCH_RADIUS", "1"))
PREFETCH_FLOOR = int(os.environ.get(_PREFIX + "PREFETCH_FLOOR", "128"))

# Global (shared) state: discovery artifacts only.
G = {"armed": False, "layers": None, "host_mirror": None, "pool": None,
     "allocator": None, "table": None, "calls": 0,
     "staged_pages": 0, "verify_mismatches": 0, "selfcheck": None,
     "scheduler": None, "reserve": None, "sink_page": None,
     "reserve_tokens": 0, "orig_free": None, "orig_free_seg": None,
     "orig_protected": None, "orig_full_protected": None, "tree_cache": None}

# Per-request state keyed by req_to_token row id.
R = {}


def _info(msg):
    print(f"[hosttier23m {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _tier_stream():
    """A dedicated CUDA stream for tier copies: the runtime's compute
    stream never waits on our transfers unless we explicitly sync."""
    s = G.get("_stream")
    if s is None and torch.cuda.is_available():
        s = G["_stream"] = torch.cuda.Stream()
    return s


def _copy_gpu_to_host(gpu_tensor, host_tensor, slots):
    # slots arrive on the GPU: gather there first, one contiguous D2H.
    gathered = gpu_tensor[slots]
    if ASYNC and torch.cuda.is_available():
        stream = _tier_stream()
        with torch.cuda.stream(stream):
            host_tensor[slots.cpu()] = gathered.to("cpu", non_blocking=True)
        return stream
    host_tensor[slots.cpu()] = gathered.to("cpu", non_blocking=False)
    return None


def _copy_host_to_gpu(host_tensor, gpu_tensor, slots_cpu):
    # gather on the CPU side (host tensor + CPU indices), then a single
    # contiguous H2D copy under the side stream; indexing a GPU tensor
    # with CPU indices in one expression mixes devices and raises.
    gathered = host_tensor[slots_cpu]
    if ASYNC and torch.cuda.is_available():
        stream = _tier_stream()
        with torch.cuda.stream(stream):
            gpu_tensor[slots_cpu.to(gpu_tensor.device)] = gathered.to(
                gpu_tensor.device, non_blocking=True)
        return stream
    gpu_tensor[slots_cpu.to(gpu_tensor.device)] = gathered.to(
        gpu_tensor.device, non_blocking=False)
    return None


def _row_state(row):
    st = R.get(row)
    if st is None:
        if len(R) >= MAX_REQUESTS:
            return None  # beyond budget: this row runs untiered
        st = R[row] = {"ctx_written": 0, "chunks": 0, "mirrored": set(),
                       "retired": set(), "page_slots": {}, "last_key": None}
    return st


def dump_stats():
    payload = {
        "armed": G["armed"], "layers_n": len(G["layers"] or []),
        "requests": len(R), "max_requests": MAX_REQUESTS,
        "staged_pages": G["staged_pages"],
        "prefetch_issued": G.get("prefetch_issued", 0),
        "prefetch_hits": G.get("prefetch_hits", 0),
        "verify_mismatches": G["verify_mismatches"], "selfcheck": G["selfcheck"],
        "per_request": {str(row): {"ctx": st["ctx_written"],
                                   "mirrored": len(st["mirrored"]),
                                   "retired": len(st["retired"])}
                        for row, st in R.items()},
    }
    try:
        with open(LOG_PATH, "w") as handle:
            json.dump(payload, handle, indent=2)
    except OSError:
        pass


if ENABLE:
    _info(f"P2.3m loaded (page={PAGE}, max_requests={MAX_REQUESTS}, verify={VERIFY})")
    atexit.register(dump_stats)


ARM_AFTER = int(os.environ.get(_PREFIX + "ARM_AFTER", "4"))


def _discover_allocator(carrier):
    for owner in (carrier, getattr(carrier, "runner", None),
                  getattr(getattr(carrier, "runner", None), "scheduler", None)):
        if owner is None:
            continue
        for attr in ("token_to_kv_pool_allocator", "allocator"):
            obj = getattr(owner, attr, None)
            if obj is not None and hasattr(obj, "free_pages"):
                return obj
    return None


def _discover_scheduler(carrier):
    runner = getattr(carrier, "runner", None)
    return getattr(runner, "scheduler", None)


def _req_of(row):
    scheduler = G["scheduler"]
    if scheduler is None:
        return None
    for batch in (getattr(scheduler, "running_batch", None),
                  getattr(scheduler, "chunked_req", None)):
        if batch is None:
            continue
        reqs = getattr(batch, "reqs", None) or ([batch] if hasattr(batch, "req_pool_idx") else [])
        for req in reqs:
            if getattr(req, "req_pool_idx", None) == row:
                return req
    return None


def _reserve_frames():
    r = G.get("reserve")
    if r is None:
        return set()
    return set(r.tolist())


def _dedup_free(slots):
    """Free frames that are not already in free_pages.

    The runtime's finish path re-frees pages the tier already returned at
    retire time (its per-request extent covers retired positions); letting
    both frees through double-lists frames and the invariant checker kills
    the scheduler on available > total (the +8,128-token finish crash)."""
    alloc = G["allocator"]
    already = set(alloc.free_pages.tolist()) | _reserve_frames()
    frames = set(int(s) // PAGE for s in slots.tolist())
    fresh = [f * PAGE for f in frames if f not in already]
    if not fresh:
        return None
    tok = torch.tensor(fresh, dtype=alloc.free_pages.dtype,
                       device=alloc.free_pages.device)
    return G["orig_free"](tok)


def _arm_shrunk(allocator):
    """HKV-4: confiscate pages and shrink allocator.size to the kept
    budget. The invariant checker's `total` IS allocator.size, so the
    identity balances without reaching the scheduler's tree_cache — which
    is unreachable from the model-runner side (ModelRunner holds no
    scheduler back-pointer; window finding). Admission stays permissive
    because the HTTP-side limit was computed at startup from the original
    pool size. See HOST_KV_SHRUNK_POOL.md."""
    free = allocator.free_pages
    if int(free.numel()) <= KEEP_PAGES:
        _info(f"shrunk: pool already <= KEEP_PAGES ({int(free.numel())}); nothing to take")
        return
    G["reserve"] = free[KEEP_PAGES:].clone()
    allocator.free_pages = free[:KEEP_PAGES].clone()
    G["sink_page"] = int(G["reserve"][0].item())
    G["reserve_tokens"] = int(G["reserve"].numel()) * PAGE
    G["orig_size"] = allocator.size
    allocator.size = KEEP_PAGES * PAGE  # the invariant's total
    if hasattr(allocator, "full_tokens_per_layer"):
        allocator.full_tokens_per_layer = allocator.size
    # free filter: the sink page must never enter circulation via the
    # finish path freeing sink-filled table columns
    G["orig_free"] = allocator.free
    G["orig_free_seg"] = allocator.free_segment
    tier_pages = set(G["reserve"].tolist()) | {0}  # + padding page
    G["tier_pages"] = tier_pages
    # The radix's chunk-stash holds ORIGINAL indices: at finish it claims
    # the full context as evictable while our retired pages are already
    # back in free_pages. Report available minus the tier's outstanding
    # returns so the identity balances until the radix reconciles.
    G["returned_tokens"] = 0
    G["orig_available"] = allocator.available_size
    def _tier_available(_orig=G["orig_available"]):
        return _orig() - G["returned_tokens"]
    allocator.available_size = _tier_available
    def _filtered_free(free_index, _orig=G["orig_free"], _tier=tier_pages):
        pages = (free_index // PAGE).tolist()
        reserve_frames = _reserve_frames()
        keep = torch.tensor([f for f, p in zip(free_index.tolist(), pages)
                             if p not in _tier and p not in reserve_frames],
                            dtype=free_index.dtype, device=free_index.device)
        if int(keep.numel()):
            keep = _dedup_free(keep)
        return keep
    def _filtered_free_seg(free_index, *, start_pos, _orig=G["orig_free_seg"], _tier=tier_pages):
        pages = (free_index // PAGE).tolist()
        keep = torch.tensor([f for f, p in zip(free_index.tolist(), pages)
                             if p not in _tier], dtype=free_index.dtype,
                            device=free_index.device)
        if int(keep.numel()) == 0:
            return
        return _orig(keep, start_pos=start_pos)
    allocator.free = _filtered_free
    allocator.free_segment = _filtered_free_seg
    _info(f"shrunk armed: keep={KEEP_PAGES} pages, reserve="
          f"{int(G['reserve'].numel())} pages ({G['reserve_tokens']} tokens), "
          f"sink_page={G['sink_page']}")
    import threading as _threading

    def _watch():
        import time as _time
        n = 0
        while True:
            _time.sleep(2)
            n += 1
            try:
                _alloc = G["allocator"]
                _info(
                    f"watch#{n}: size={_alloc.size} free_pages={int(_alloc.free_pages.numel())} "
                    f"avail={_alloc.available_size()} returned={G.get('returned_tokens', 0)}"
                )
            except Exception as _e:
                _info(f"watch#{n} err={_e!r}")
    _threading.Thread(target=_watch, daemon=True).start()


def arm(backend, row):
    if not ENABLE:
        return
    G["calls"] += 1
    if not G["armed"]:
        if G["calls"] < ARM_AFTER:
            return  # server warmup precedes real traffic
        pool = getattr(backend, "token_to_kv_pool", None)
        layers = (sorted(int(l) for l in pool.full_attention_layer_id_mapping)
                  if pool is not None and getattr(pool, "full_attention_layer_id_mapping", None)
                  else [])
        if len(layers) < 8:
            return
        allocator = _discover_allocator(backend)
        table = getattr(backend, "req_to_token", None)
        if allocator is None or table is None:
            _info("discovery incomplete; dormant")
            G["armed"] = True
            return
        mirror = {layer: (torch.zeros_like(pool.get_key_buffer(layer), device="cpu"),
                          torch.zeros_like(pool.get_value_buffer(layer), device="cpu"))
                  for layer in layers}
        G.update(armed=True, layers=layers, host_mirror=mirror, pool=pool,
                 allocator=allocator, table=table)
        G["scheduler"] = _discover_scheduler(backend)
        _info(f"armed: {len(layers)} layers, "
              f"pool slots={pool.get_key_buffer(layers[0]).shape[0]}, "
              f"free_pages={len(allocator.free_pages)}, "
              f"scheduler={'yes' if G['scheduler'] else 'no'}")
        if SHRUNK:
            _arm_shrunk(allocator)
    elif SHRUNK and G.get("reserve") is None and G.get("armed"):
        # tree_cache may only appear after init; retry on later arms
        G["scheduler"] = G["scheduler"] or _discover_scheduler(backend)
        _arm_shrunk(G["allocator"])
    _row_state(int(row))  # register the requesting row


# -- write path (post-call, per row) -------------------------------------

def after_write(forward_batch):
    if not G["armed"]:
        return
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None:
        decodeish = False
        try:
            decodeish = bool(mode.is_decode())
        except Exception:
            decodeish = False
        if not decodeish and hasattr(mode, "is_target_verify"):
            try:
                decodeish = bool(mode.is_target_verify())
            except Exception:
                decodeish = False
        if decodeish:
            # HKV-6: the delta<PAGE heuristic never fired in this build
            # (the G3 volley: zero freezes, 5,038 re-staged pages, decode
            # at 6.4 tok/s from the retire/restore ping-pong). Freeze by
            # MODE: once a row decodes, its working set is fixed.
            for row_id in getattr(forward_batch, "req_pool_indices",
                                  torch.empty(0)).tolist():
                st = R.get(int(row_id))
                if st is not None and not st.get("decode_frozen"):
                    st["decode_frozen"] = True
                    _info(f"row {int(row_id)}: retire frozen by mode at "
                          f"ctx={st['ctx_written']} (retired={len(st['retired'])})")
    table = G["table"]
    out_locs = getattr(forward_batch, "out_cache_loc", None)
    if out_locs is None or out_locs.numel() == 0:
        return
    rows = getattr(forward_batch, "req_pool_indices", None)
    if rows is None or rows.numel() == 0:
        return
    # Dedupe by content, GLOBALLY: one forward serves all layers, so the
    # same (first-slot, size) recurs per layer; per-row keys would let
    # layer repeats leak across accounting.
    key = (int(out_locs[0].item()), int(out_locs.numel()))
    if key == G.get("last_key"):
        return
    G["last_key"] = key
    # Row attribution: extend batches carry one flat out_cache_loc whose
    # row boundaries come from extend_seq_lens (cumulative); a single-row
    # batch or missing lens attribute everything to the first row (the
    # G1-proven behavior).
    ext = getattr(forward_batch, "extend_seq_lens", None)
    if ext is not None and ext.numel() == rows.numel() and int(rows.numel()) > 1:
        bounds = torch.cumsum(ext, 0).tolist()
        lo = 0
        for i, row_id in enumerate(rows.tolist()):
            hi = int(min(bounds[i], out_locs.numel()))
            st = _row_state(int(row_id))
            if st is not None and hi > lo:
                _write_row(int(row_id), st, out_locs[lo:hi], table,
                           total_ctx=int(ext[i].item()))
            lo = hi
    else:
        row = int(rows[0].item())
        st = _row_state(row)
        if st is not None:
            total = (int(ext[0].item()) if ext is not None and ext.numel() else None)
            _write_row(row, st, out_locs, table, total_ctx=total)


def _write_row(row, st, locs, table, total_ctx=None):
    delta = int(locs.numel())
    if (total_ctx is not None and st["ctx_written"] > total_ctx + 8 * PAGE):
        # Rows recycle across sequential requests; extend_seq_lens carries
        # each row's true total context. Stale ctx_written from a previous
        # request on this row otherwise accumulates (159,373 + 106,490 =
        # 265,863) and indexes past the table width.
        _info(f"row {row}: stale ctx {st['ctx_written']} > total {total_ctx}"
              f" - resetting recycled row state")
        st.clear()
        st.update({"ctx_written": 0, "chunks": 0, "mirrored": set(),
                   "retired": set(), "page_slots": {}, "last_key": None})
    st["max_delta"] = max(st.get("max_delta", 0), delta)
    if (delta < PAGE and st["max_delta"] >= PAGE
            and not st.get("decode_frozen")):
        # A sub-page delta is a verify/decode step: prefill is complete,
        # the decode working set is fixed — retire no more (the phase-1
        # finding: re-retiring restored pages every step dominated cost).
        st["decode_frozen"] = True
        _info(f"row {row}: retire frozen at ctx={st['ctx_written']} "
              f"(retired={len(st['retired'])})")
    start, end = st["ctx_written"], st["ctx_written"] + delta
    st["ctx_written"] = end
    st["chunks"] += 1
    pool, layers, mirror = G["pool"], G["layers"], G["host_mirror"]
    for page in range(start // PAGE, (end + PAGE - 1) // PAGE):
        if page in st["mirrored"]:
            continue
        cols = torch.arange(page * PAGE, (page + 1) * PAGE, device=table.device)
        cols = cols[cols < end]
        if cols.numel() == 0:
            continue
        slots = table[row, cols].long()
        valid = slots >= 0
        if not bool(valid.any()):
            continue
        with torch.no_grad():
            for layer in layers:
                mk, mv = mirror[layer]
                mk[slots[valid]] = pool.get_key_buffer(layer)[slots[valid]].to("cpu", non_blocking=False)
                mv[slots[valid]] = pool.get_value_buffer(layer)[slots[valid]].to("cpu", non_blocking=False)
        st["mirrored"].add(page)
    if st["chunks"] % 8 == 1:
        _info(f"row {row} chunk #{st['chunks']}: ctx={end} "
              f"mirrored={len(st['mirrored'])} retired={len(st['retired'])}")
    _retire(row, st, end)


def _retire(row, st, ctx_end):
    if st.get("decode_frozen"):
        return
    """Per-row logical retirement, with retire-time re-mirroring: copy
    each victim page GPU->host BEFORE clearing the row, so the mirror is
    authoritative even for pages that received decode/verify writes after
    their prefill-time mirror (the G2 window's verified=False cause).
    Another request's columns are never touched."""
    table = G["table"]
    pool, layers, mirror = G["pool"], G["layers"], G["host_mirror"]
    hot_pages = int(os.environ.get(_PREFIX + "HOT_PAGES", "512"))
    retired = 0
    for page in sorted(st["mirrored"]):
        if page in st["retired"] or page >= ctx_end // PAGE - hot_pages:
            continue
        cols = torch.arange(page * PAGE, (page + 1) * PAGE, device=table.device)
        slots = table[row, cols].long().clone()
        if bool((slots < 0).any()):
            continue
        with torch.no_grad():
            for layer in layers:
                mk, mv = mirror[layer]
                _copy_gpu_to_host(pool.get_key_buffer(layer), mk, slots)
                _copy_gpu_to_host(pool.get_value_buffer(layer), mv, slots)
        st["page_slots"][page] = slots
        if SHRUNK and G["sink_page"] is not None:
            frame = int(slots[0].item()) // PAGE
            if frame in G.get("tier_pages", set()):
                # re-retiring a STAGED page: its frame is tier-owned.
                # Freeing it through the allocator would double-count
                # (the oracle's available=1.16M-in-51.2k signal); return
                # the frame to the reserve instead.
                G["reserve"] = torch.cat([G["reserve"],
                                          torch.tensor([frame], dtype=G["reserve"].dtype,
                                                       device=G["reserve"].device)])
            else:
                # original runtime slots: return to circulation and drop
                # the request's uncached by the same amount
                _dedup_free(slots)
                if not NO_RADIX:
                    # The subtraction compensates the radix tree's stale
                    # evictable claim on returned pages; without a radix
                    # cache it undercounts availability forever (the
                    # post-finish avail=5376-in-160640 catatonia).
                    G["returned_tokens"] = G.get("returned_tokens", 0) + PAGE
                req = _req_of(row)
                if req is not None and getattr(req, "kv", None) is not None:
                    req.kv.kv_allocated_len -= PAGE
            # Retire to 0, NOT to sink slots: the radix cache's
            # finish-insert reads table content and would absorb sink
            # columns as phantom evictable pages (the 33,280-token
            # over-count). Full-pool mode retired to 0 with the radix
            # silent across 14 concurrent finishes.
            table[row, cols] = 0
        else:
            table[row, cols] = 0
        st["retired"].add(page)
        retired += 1
    if retired:
        _info(f"row {row}: retired {retired} pages (total {len(st['retired'])})")


# -- gather path (pre-call, per row) -------------------------------------

def before_gather(topk_indices, forward_batch):
    """Restore the REQUESTING row's selected retired pages in place.

    HKV-6: one stream sync per gather (not per page), a table-rewrite-only
    fast path for pages prefetched by a previous gather, and async
    neighbor prefetch into reserve frames that never touches the table
    until the page is actually selected."""
    if not G["armed"] or not R:
        return
    rows = getattr(forward_batch, "req_pool_indices", None)
    if rows is None or rows.numel() == 0:
        return
    positions = topk_indices
    if positions.dim() > 1:
        positions = positions[0] if rows.numel() == 1 else positions.flatten()
    positions = positions[positions >= 0]
    if positions.numel() == 0:
        return
    pages = torch.unique(positions // PAGE).tolist()
    row = int(rows[0].item())
    st = R.get(row)
    if st is None:
        return
    prefetched = st.setdefault("prefetched", set())
    need = [p for p in pages if p in st["retired"]]
    table, pool, layers, mirror = G["table"], G["pool"], G["layers"], G["host_mirror"]
    copies_ran = False
    verify_page = None
    for page in need:
        if page in prefetched:
            # Copy was issued asynchronously during an earlier gather; it
            # must be complete before the table publishes the page.
            if ASYNC and torch.cuda.is_available():
                _tier_stream().synchronize()
            prefetched.discard(page)
            G["prefetch_hits"] = G.get("prefetch_hits", 0) + 1
            with torch.no_grad():
                cols = torch.arange(page * PAGE, (page + 1) * PAGE,
                                    device=table.device)
                table[row, cols] = st["page_slots"][page].to(table.dtype)
            st["retired"].discard(page)
            G["staged_pages"] += 1
            continue
        slots = st["page_slots"][page]
        slots_cpu = slots.cpu()
        if SHRUNK and G["reserve"] is not None and int(G["reserve"].numel()) > 0:
            # stage into a RESERVE frame: the old slots were returned to
            # circulation at retirement and may belong to someone else now
            frame_page = int(G["reserve"][0].item())
            G["reserve"] = G["reserve"][1:]
            G["reserve_tokens"] = int(G["reserve"].numel()) * PAGE
            frame_slots = torch.arange(frame_page * PAGE, (frame_page + 1) * PAGE,
                                       dtype=slots.dtype, device=slots.device)
            # No kv_allocated_len change: reserve frames are outside the
            # pool identity under size-shrink (the runtime never allocated
            # them; its uncached counter must not grow for them).
            st["page_slots"][page] = frame_slots
            slots = frame_slots
            slots_cpu = slots.cpu()
        with torch.no_grad():
            cols = torch.arange(page * PAGE, (page + 1) * PAGE, device=table.device)
            for layer in layers:
                mk, mv = mirror[layer]
                _copy_host_to_gpu(mk, pool.get_key_buffer(layer), slots_cpu)
                _copy_host_to_gpu(mv, pool.get_value_buffer(layer), slots_cpu)
            table[row, cols] = slots.to(table.dtype)
        copies_ran = True
        if verify_page is None and VERIFY and G["selfcheck"] is None:
            verify_page = (page, slots, slots_cpu)
        G["staged_pages"] += 1
        st["retired"].discard(page)
    if copies_ran:
        # ONE sync per gather: the attention kernel that consumes these
        # pages runs after this return; per-page syncs (the previous
        # behavior, despite its comment) serialized every restore.
        if ASYNC and torch.cuda.is_available():
            _tier_stream().synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        if verify_page is not None:
            page, slots, slots_cpu = verify_page
            ok = True
            for layer in layers:
                mk, mv = mirror[layer]  # rebind: the restore loop above
                # held the LAST layer's mirror
                gk = pool.get_key_buffer(layer)[slots]
                if gk.is_cuda:
                    gk = gk.cpu()
                # BIT-level equality: torch.equal says NaN != NaN, so a
                # faithful copy of NaN-bearing bytes would falsely fail.
                same = torch.equal(gk.view(torch.uint16),
                                   mk[slots_cpu].view(torch.uint16))
                ok &= same
                if not same:
                    diff = (gk.view(torch.uint16) != mk[slots_cpu].view(torch.uint16))
                    n = int(diff.sum())
                    idx = torch.nonzero(diff.flatten())[:4].flatten().tolist()
                    gv = gk.flatten()[idx].tolist() if idx else []
                    mvv = mk[slots_cpu].flatten()[idx].tolist() if idx else []
                    _info(f"VERIFY-DIFF layer={layer} n={n}/{diff.numel()} "
                          f"first_idx={idx} gpu={gv} mirror={mvv} "
                          f"gpu_nan={int(gk.isnan().sum())} "
                          f"mirror_nan={int(mk[slots_cpu].isnan().sum())}")
            G["selfcheck"] = {"page": page, "row": row, "mode": "in-place",
                              "verified": bool(ok)}
            G["verify_mismatches"] += int(not ok)
            _info(f"selfcheck row {row} page {page}: verified={ok}")
    if PREFETCH and PREFETCH_RADIUS > 0 and st["retired"]:
        # Selections drift slowly across steps; restore the neighbors of
        # this step's selections asynchronously so a later gather's fast
        # path only rewrites the table. Prefetched pages stay out of the
        # table, so nothing reads them before their copy completes.
        cand = set()
        for p in pages:
            for d in range(1, PREFETCH_RADIUS + 1):
                cand.add(p - d)
                cand.add(p + d)
        for page in sorted(cand):
            if page < 0 or page in prefetched or page not in st["retired"]:
                continue
            if (G["reserve"] is None
                    or int(G["reserve"].numel()) <= PREFETCH_FLOOR):
                break  # keep a staging floor for synchronous restores
            slots = st["page_slots"].get(page)
            if slots is None:
                continue
            frame_page = int(G["reserve"][0].item())
            G["reserve"] = G["reserve"][1:]
            G["reserve_tokens"] = int(G["reserve"].numel()) * PAGE
            frame_slots = torch.arange(frame_page * PAGE,
                                       (frame_page + 1) * PAGE,
                                       dtype=slots.dtype, device=slots.device)
            st["page_slots"][page] = frame_slots
            slots_cpu = frame_slots.cpu()
            with torch.no_grad():
                for layer in layers:
                    mk, mv = mirror[layer]
                    _copy_host_to_gpu(mk, pool.get_key_buffer(layer), slots_cpu)
                    _copy_host_to_gpu(mv, pool.get_value_buffer(layer), slots_cpu)
            prefetched.add(page)
            G["prefetch_issued"] = G.get("prefetch_issued", 0) + 1
