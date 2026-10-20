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
FREEZE_STREAK = int(os.environ.get(_PREFIX + "FREEZE_STREAK", "8"))
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
        "coverage_misses": G.get("coverage_misses", 0),
        "clamp_skips": G.get("clamp_skips", 0),
        "br_ticks": G.get("br_ticks", 0),
        "br_nostash": G.get("br_nostash", 0),
        "ws_avg": round(G.get("ws_sum", 0) / max(G.get("ws_n", 1), 1), 1),
        "drift_avg": round(G.get("drift_sum", 0) / max(G.get("drift_n", 1), 1), 1),
        "miss_recent": G.get("miss_recent", 0),
        "miss_deep": G.get("miss_deep", 0),
        "br_errors": G.get("br_errors", 0),
        "restore_starved": G.get("restore_starved", 0),
        "dedup_total": G.get("dedup_total", 0),
        "dedup_dropped": G.get("dedup_dropped", 0),
        "dedup_freed": G.get("dedup_freed", 0),
        "ff_total": G.get("ff_total", 0),
        "ff_dropped_tier": G.get("ff_dropped_tier", 0),
        "ff_dropped_reserve": G.get("ff_dropped_reserve", 0),
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
    G["dedup_total"] = G.get("dedup_total", 0) + len(frames)
    G["dedup_dropped"] = G.get("dedup_dropped", 0) + (len(frames) - len(fresh))
    G["dedup_freed"] = G.get("dedup_freed", 0) + len(fresh)
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
        keep = []
        n_tier = n_res = 0
        for f, p in zip(free_index.tolist(), pages):
            if p in _tier:
                n_tier += 1
            elif p in reserve_frames:
                n_res += 1
            else:
                keep.append(f)
        G["ff_total"] = G.get("ff_total", 0) + len(pages)
        G["ff_dropped_tier"] = G.get("ff_dropped_tier", 0) + n_tier
        G["ff_dropped_reserve"] = G.get("ff_dropped_reserve", 0) + n_res
        keep = torch.tensor(keep, dtype=free_index.dtype,
                            device=free_index.device)
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
                if n % 10 == 0:
                    # Periodic stats dump: hard kills (SIGKILL during
                    # docker stop) never reach atexit, and the measurement
                    # counters must survive them.
                    dump_stats()
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
        _locs = getattr(forward_batch, "out_cache_loc", None)
        _rows_now = getattr(forward_batch, "req_pool_indices", torch.empty(0))
        if (decodeish and _locs is not None
                and int(_locs.numel()) < int(_rows_now.numel()) * PAGE):
            # HKV-6: the delta<PAGE heuristic never fired in this build
            # (the G3 volley: zero freezes, 5,038 re-staged pages, decode
            # at 6.4 tok/s from the retire/restore ping-pong). Freeze by
            # MODE, but only when the ctx has grown by less than a page
            # since the previous decode-ish batch: the MTP draft also runs
            # decode-shaped verify batches DURING chunked prefill (the
            # ctx=3719 false freeze that disabled retirement for the
            # whole request), and those are followed by +8192-token
            # chunks, while real decode steps grow ctx by a few tokens.
            for row_id in _rows_now.tolist():
                st = R.get(int(row_id))
                if st is None or st.get("decode_frozen"):
                    continue
                streak = st.get("decode_streak", 0) + 1
                st["decode_streak"] = streak
                if streak >= FREEZE_STREAK:
                    st["decode_frozen"] = True
                    _info(f"row {int(row_id)}: retire frozen after decode "
                          f"streak {streak} at ctx={st['ctx_written']} "
                          f"(retired={len(st['retired'])})")
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
    seq = getattr(forward_batch, "seq_lens", None)
    ext = getattr(forward_batch, "extend_seq_lens", None)
    # seq_lens is the per-row RUNNING TOTAL - unlike extend_seq_lens,
    # which this build fills with the batch's own delta (8192 per prefill
    # chunk, 4-6 per verify step) on the single-row paths.
    def _total(i):
        if seq is not None and seq.numel() > i:
            return int(seq[i].item())
        if ext is not None and ext.numel() > i:
            return int(ext[i].item())
        return None

    if ext is not None and ext.numel() == rows.numel() and int(rows.numel()) > 1:
        bounds = torch.cumsum(ext, 0).tolist()
        lo = 0
        for i, row_id in enumerate(rows.tolist()):
            hi = int(min(bounds[i], out_locs.numel()))
            st = _row_state(int(row_id))
            if st is not None and hi > lo:
                _write_row(int(row_id), st, out_locs[lo:hi], table,
                           total_ctx=_total(i))
            lo = hi
    else:
        row = int(rows[0].item())
        st = _row_state(row)
        if st is not None:
            _write_row(row, st, out_locs, table, total_ctx=_total(0))


def _write_row(row, st, locs, table, total_ctx=None):
    delta = int(locs.numel())
    # Decode/verify batches carry extend_seq_lens as tiny per-row DELTAS
    # (4-6 tokens), not totals; validating the staleness check against
    # them wiped live row state at every decode step (ctx=3987/retired=0),
    # killing retirement for the rest of the request and parking ~89k
    # tokens resident - the concurrent-admission blocker. Only prefill-
    # scale writes carry a trustworthy total.
    if (delta >= PAGE and total_ctx is not None
            and st["ctx_written"] > total_ctx + 8 * PAGE):
        # Rows recycle across sequential requests; extend_seq_lens carries
        # each row's true total context. Stale ctx_written from a previous
        # request on this row otherwise accumulates (159,373 + 106,490 =
        # 265,863) and indexes past the table width.
        _info(f"row {row}: stale ctx {st['ctx_written']} > total {total_ctx}"
              f" - resetting recycled row state")
        st.clear()
        st.update({"ctx_written": 0, "chunks": 0, "mirrored": set(),
                   "retired": set(), "page_slots": {}, "last_key": None})
    if delta >= PAGE:
        st["decode_streak"] = 0  # a prefill chunk breaks the decode streak
    st["max_delta"] = max(st.get("max_delta", 0), delta)
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
    """Per-row logical retirement (HKV-11 bulk form).

    Candidates are the pages newly below the horizon since the last call
    (frontier-tracked: O(churn), never the O(context) scan that cost
    ~2 tok/s once retirement ran during decode) plus pages restored by
    gathers since then. All frees, table zeroing, reserve returns and
    the kv_allocated_len clamp are batched per call. Retire-time
    re-mirroring still runs per non-restored page - restored pages' bytes
    ARE the mirror's."""
    if st.get("decode_frozen"):
        return
    table = G["table"]
    hot_pages = int(os.environ.get(_PREFIX + "HOT_PAGES", "512"))
    horizon = ctx_end // PAGE - hot_pages
    cands = st.setdefault("re_retire", set())
    last_h = st.get("last_horizon", 0)
    for page in range(max(last_h, 0), max(horizon, 0)):
        if page in st["mirrored"]:
            cands.add(page)
    st["last_horizon"] = max(horizon, st.get("last_horizon", 0))
    cands &= st["mirrored"]
    cands -= st["retired"]
    if not cands:
        return
    pages = sorted(cands)
    restored_now = st.get("restored", set())

    # 1. re-mirror only pages the mirror may not already hold
    remirror = [p for p in pages if p not in restored_now]
    if remirror:
        col_list = []
        for p in remirror:
            cols = torch.arange(p * PAGE, (p + 1) * PAGE, device=table.device)
            col_list.append(table[row, cols].long().clone())
        src_slots = torch.cat(col_list)
        pool, mirror = G["pool"], G["host_mirror"]
        with torch.no_grad():
            for layer in G["layers"]:
                mk, mv = mirror[layer]
                _copy_gpu_to_host(pool.get_key_buffer(layer), mk, src_slots)
                _copy_gpu_to_host(pool.get_value_buffer(layer), mv, src_slots)
        for p, sl in zip(remirror, col_list):
            st["page_slots"][p] = sl

    # 2. classify frames: reserve-owned vs runtime, batched
    reserve_now = _reserve_frames()
    tier_static = G.get("tier_pages", set())
    to_reserve, to_free_cols = [], []
    for p in pages:
        cols = torch.arange(p * PAGE, (p + 1) * PAGE, device=table.device)
        slots = table[row, cols].long().clone()
        if bool((slots < 0).any()):
            continue
        st["page_slots"][p] = slots
        frame = int(slots[0].item()) // PAGE
        if SHRUNK and G["sink_page"] is not None and (
                frame in reserve_now or frame in tier_static):
            to_reserve.append(frame)
        else:
            to_free_cols.append(slots)

    # 3. batched frame returns
    if to_reserve:
        G["reserve"] = torch.cat([
            G["reserve"],
            torch.tensor(to_reserve, dtype=G["reserve"].dtype,
                         device=G["reserve"].device)])
    freed_pages = 0
    if to_free_cols:
        all_slots = torch.cat(to_free_cols)
        freed_pages = _dedup_free(all_slots)
        if not NO_RADIX:
            G["returned_tokens"] = (G.get("returned_tokens", 0)
                                    + PAGE * len(to_free_cols))
        req = _req_of(row)
        if req is not None and getattr(req, "kv", None) is not None:
            committed = None
            try:
                committed = int(req.effective_kv_committed_len())
            except Exception:
                committed = None
            floor = (committed if committed is not None
                     else len(getattr(req, "output_ids", []) or []))
            budget = PAGE * len(to_free_cols)
            if req.kv.kv_allocated_len - budget >= floor:
                req.kv.kv_allocated_len -= budget
            else:
                G["clamp_skips"] = G.get("clamp_skips", 0) + 1

    # 4. one table-zeroing op for every retired page this call
    pages_t = torch.tensor(pages, dtype=torch.int64, device=table.device)
    cols_all = (pages_t[:, None] * PAGE
                + torch.arange(PAGE, device=table.device)[None, :]).flatten()
    with torch.no_grad():
        table[row, cols_all] = 0
    st["retired"].update(pages)
    restored_now -= set(pages)
    G["retire_calls"] = G.get("retire_calls", 0) + 1
    G["retire_pages"] = G.get("retire_pages", 0) + len(pages)
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
    # Multi-row decode batches: attribute each token's selections to its
    # own row via extend_seq_lens deltas (per-row token counts, in the
    # same token order as out_cache_loc). The old code flattened every
    # row's positions into row 0's page map, restoring the wrong pages
    # for every other row the moment two requests decode together.
    ext = getattr(forward_batch, "extend_seq_lens", None)
    if rows.numel() > 1 and ext is not None and ext.numel() == rows.numel():
        flat = positions.flatten()
        bounds = torch.cumsum(ext, 0).tolist()
        lo = 0
        for i, row_id in enumerate(rows.tolist()):
            hi = int(min(bounds[i], flat.numel()))
            st = R.get(int(row_id))
            if st is not None and hi > lo:
                _restore_row_pages(int(row_id), st,
                                   flat[lo:hi][flat[lo:hi] >= 0])
            lo = hi
        return
    row = int(rows[0].item())
    st = R.get(row)
    if st is None:
        return
    _restore_row_pages(row, st, positions)


def between_replay(backend):
    """HKV-12: lag-one-iteration tier tick, called from the scheduler
    between MTP iterations (outside any CUDA graph).

    The backend's QSAMTPSharedSparseIndices holds each row's freshest
    target-aligned selection in persistent GPU buffers written by
    capture() inside the draft-extend graph; restore those pages now so
    the next iteration's decode graphs read them via req_to_token, and
    re-retire the previous iteration's restores (their bytes are the
    mirror's, so the re-mirror is skipped). Selections drift slowly
    across iterations - the measured prefetch hit rate was 100%."""
    if not G.get("armed") or not R:
        return
    state = getattr(backend, "_mtp_shared_sparse_indices", None)
    if state is None:
        G["br_nostash"] = G.get("br_nostash", 0) + 1
        return
    try:
        rows = sorted(R.keys())
        rows_t = torch.tensor(rows, dtype=torch.long, device=state.indices.device)
        idx = state.indices[0, rows_t].cpu()
        lens = state.captured_len[0, rows_t].cpu()
    except Exception as e:  # noqa: BLE001 - diagnostic only
        G["br_errors"] = G.get("br_errors", 0) + 1
        if G["br_errors"] <= 5:
            _info(f"between_replay read error: {e!r}")
        return
    for i, row in enumerate(rows):
        st = R[row]
        if st is None or st.get("decode_frozen"):
            continue
        pos = idx[i]
        sel = pos[pos >= 0]
        sel_set = set(torch.unique(sel // PAGE).tolist())
        # HKV-17 instrumentation: working-set width and inter-tick drift
        # drive the reserve formula (reserve >= N * (width + drift)).
        G["ws_sum"] = G.get("ws_sum", 0) + len(sel_set)
        G["ws_n"] = G.get("ws_n", 0) + 1
        prev = st.get("last_sel_pages")
        if prev is not None:
            G["drift_sum"] = G.get("drift_sum", 0) + len(sel_set - prev)
            G["drift_n"] = G.get("drift_n", 0) + 1
        st["last_sel_pages"] = sel_set
        _restore_row_pages(int(row), st, sel)
        # The selection IS the next replay's working set - its pages live
        # below the retire horizon by construction, so keep them out of
        # the re-retire queue or the same tick would retire what it just
        # restored (the 116k-coverage-miss churn) and the replay would
        # read zeroed columns.
        sel_pages = set(torch.unique(sel // PAGE).tolist())
        st.setdefault("re_retire", set()).difference_update(sel_pages)
        _retire(int(row), st, int(lens[i].item()))
        # In-tick coverage audit (the wrapper's audit cannot run during
        # replays): every stash-selected page must be resident now.
        pages = sorted(sel_set)
        miss = [p for p in pages if p in st["retired"]]
        if miss:
            n = G.get("coverage_misses", 0) + len(miss)
            G["coverage_misses"] = n
            # HKV-18 instrumentation: are misses in the recent band
            # (fixable by a wider hot horizon) or deep-sparse (lag)?
            end_page = int(lens[i].item()) // PAGE
            for p in miss:
                dist = end_page - p
                if dist <= 256:
                    G["miss_recent"] = G.get("miss_recent", 0) + 1
                else:
                    G["miss_deep"] = G.get("miss_deep", 0) + 1
            if n <= 20:
                _info(f"BR COVERAGE MISS row={row} pages={miss[:6]} (total {n})")
    G["br_ticks"] = G.get("br_ticks", 0) + 1


def audit_coverage(topk_indices, forward_batch):
    """Post-forward check: every page this forward SELECTED must have
    been resident (not retired). A miss means the attention kernel read
    a retired page's zeroed columns - the restore hook never fired for
    that path. Output-token diffs are unusable as a gate here (temp=0
    runs differ from themselves on this build), so coverage is the
    correctness signal."""
    if not G["armed"] or R is not None and not R:
        return
    rows = getattr(forward_batch, "req_pool_indices", None)
    if rows is None or rows.numel() == 0 or topk_indices is None:
        return
    positions = topk_indices
    if positions.dim() > 1:
        positions = positions[0] if rows.numel() == 1 else positions.flatten()
    positions = positions[positions >= 0]
    if positions.numel() == 0:
        return
    ext = getattr(forward_batch, "extend_seq_lens", None)
    if rows.numel() > 1 and ext is not None and ext.numel() == rows.numel():
        flat = positions.flatten()
        bounds = torch.cumsum(ext, 0).tolist()
        lo = 0
        for i, row_id in enumerate(rows.tolist()):
            hi = int(min(bounds[i], flat.numel()))
            st = R.get(int(row_id))
            if st is not None and hi > lo:
                _audit_row(int(row_id), st, flat[lo:hi])
            lo = hi
    else:
        st = R.get(int(rows[0].item()))
        if st is not None:
            _audit_row(int(rows[0].item()), st, positions)


def _audit_row(row, st, positions):
    positions = positions[positions >= 0]
    if positions.numel() == 0:
        return
    pages = torch.unique(positions // PAGE).tolist()
    miss = [p for p in pages if p in st["retired"]]
    if miss:
        n = G.get("coverage_misses", 0) + len(miss)
        G["coverage_misses"] = n
        if n <= 20:
            _info(f"COVERAGE MISS row={row} pages={miss[:6]} "
                  f"(total {n})")


def _restore_row_pages(row, st, positions):
    """Restore the row's selected retired pages (HKV-11 bulk form).

    Correctness first: the mirror holds a page's bytes at the slots
    recorded in page_slots BEFORE remap (original slots for a prefill-
    mirrored page; frame slots once re-retired while staged). The copy
    must read mirror[src] and write gpu[dst] with src != dst when staging
    into a reserve frame - the previous code read mirror[frame], which is
    empty on first restore and staged zeros (coherent-but-deadened
    outputs since the shrunk-pool era; the old verify compared
    mirror[frame] vs gpu[frame] and passed vacuously).

    All pages restore per call with ONE reserve slice, ONE host gather
    and H2D per layer, ONE table write, and ONE stream sync."""
    positions = positions[positions >= 0]
    if positions.numel() == 0:
        return
    pages = torch.unique(positions // PAGE).tolist()
    prefetched = st.setdefault("prefetched", set())
    need = [p for p in pages if p in st["retired"]]
    table, pool, layers, mirror = G["table"], G["pool"], G["layers"], G["host_mirror"]
    device = table.device

    # Partition BEFORE any mutation: the fast path discards pages from
    # `prefetched`, and computing sync_pages after that would re-stage
    # the same pages through both paths.
    fast = [p for p in need if p in prefetched]
    sync_pages = [p for p in need if p not in prefetched]

    # A. prefetched pages: copies issued by an earlier gather already
    # targeted the page's CURRENT frame; sync once, publish in one write.
    if fast:
        if ASYNC and torch.cuda.is_available():
            _tier_stream().synchronize()
        dst_list = [st["page_slots"][p] for p in fast]
        pages_t = torch.tensor(fast, dtype=torch.int64, device=device)
        cols_all = (pages_t[:, None] * PAGE
                    + torch.arange(PAGE, device=device)[None, :]).flatten()
        slots_all = torch.cat(dst_list)
        with torch.no_grad():
            table[row, cols_all] = slots_all.to(table.dtype)
        for p in fast:
            prefetched.discard(p)
            st["retired"].discard(p)
            st.setdefault("restored", set()).add(p)
            st.setdefault("re_retire", set()).add(p)
        G["staged_pages"] += len(fast)
        G["prefetch_hits"] = G.get("prefetch_hits", 0) + len(fast)

    # B. synchronous restores: one reserve slice, batched per-layer copies
    verify_page = None
    if sync_pages:
        if SHRUNK and G["reserve"] is not None:
            # Never restore in place under SHRUNK: original slots were
            # freed at retirement and may belong to another row. Pages
            # beyond the reserve simply stay retired (a visible coverage
            # miss) instead of corrupting a foreign frame.
            room = int(G["reserve"].numel())
            if room < len(sync_pages):
                G["restore_starved"] = G.get("restore_starved", 0) + (
                    len(sync_pages) - room)
                sync_pages = sync_pages[:max(room, 0)]
        n = len(sync_pages)
        use_frames = (SHRUNK and G["reserve"] is not None
                      and int(G["reserve"].numel()) >= n and n > 0)
        frames = None
        if use_frames:
            frames = G["reserve"][:n]
            G["reserve"] = G["reserve"][n:]
            G["reserve_tokens"] = int(G["reserve"].numel()) * PAGE
        restored_sync = list(sync_pages)
        src_list, dst_list = [], []
        for i, p in enumerate(sync_pages):
            src_list.append(st["page_slots"][p])   # where the mirror holds it
            if use_frames:
                frame_page = int(frames[i].item())
                frame_slots = torch.arange(
                    frame_page * PAGE, (frame_page + 1) * PAGE,
                    dtype=src_list[-1].dtype, device=device)
                st["page_slots"][p] = frame_slots  # remap AFTER capturing src
                dst_list.append(frame_slots)
            else:
                # full-pool mode: the page still owns its original slots,
                # restore in place (dst == src).
                dst_list.append(src_list[-1])
        src_cpu = torch.cat(src_list).cpu()
        dst = torch.cat(dst_list)
        with torch.no_grad():
            for layer in layers:
                mk, mv = mirror[layer]
                gathered_k = mk[src_cpu]
                gathered_v = mv[src_cpu]
                if ASYNC and torch.cuda.is_available():
                    stream = _tier_stream()
                    with torch.cuda.stream(stream):
                        pool.get_key_buffer(layer)[dst] = gathered_k.to(
                            dst.device, non_blocking=True)
                        pool.get_value_buffer(layer)[dst] = gathered_v.to(
                            dst.device, non_blocking=True)
                else:
                    pool.get_key_buffer(layer)[dst] = gathered_k.to(dst.device)
                    pool.get_value_buffer(layer)[dst] = gathered_v.to(dst.device)
            pages_t = torch.tensor(sync_pages, dtype=torch.int64, device=device)
            cols_all = (pages_t[:, None] * PAGE
                        + torch.arange(PAGE, device=device)[None, :]).flatten()
            table[row, cols_all] = dst.to(table.dtype)
        if ASYNC and torch.cuda.is_available():
            _tier_stream().synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        if VERIFY and G["selfcheck"] is None:
            verify_page = (sync_pages[0], dst_list[0], src_list[0])
        for p in restored_sync:
            st["retired"].discard(p)
            st.setdefault("restored", set()).add(p)
            # Re-queue for the next _retire call: without this, restored
            # pages stay resident forever, the reserve monotonically
            # drains, and the (now removed) in-place fallback would write
            # onto frames the allocator may have handed to another row.
            st.setdefault("re_retire", set()).add(p)
        G["staged_pages"] += n

    if verify_page is not None:
        page, dst_slots, src_slots = verify_page
        src_cpu = src_slots.cpu()
        ok = True
        for layer in layers:
            mk, mv = mirror[layer]
            gk = pool.get_key_buffer(layer)[dst_slots]
            if gk.is_cuda:
                gk = gk.cpu()
            ok &= torch.equal(gk.view(torch.uint16),
                               mk[src_cpu].view(torch.uint16))
        G["selfcheck"] = {"page": page, "row": row, "mode": "staged",
                          "verified": bool(ok)}
        G["verify_mismatches"] += int(not ok)
        _info(f"selfcheck row {row} page {page}: verified={ok} "
              f"(src!=dst staged restore)")

    # C. neighbor prefetch: src is the pre-remap mapping (same rule);
    #    no sync, no table write.
    if (PREFETCH and PREFETCH_RADIUS > 0 and st["retired"]
            and G["reserve"] is not None):
        cand = set()
        for p in pages:
            for d in range(1, PREFETCH_RADIUS + 1):
                cand.add(p - d)
                cand.add(p + d)
        want = sorted(p for p in cand
                      if p >= 0 and p in st["retired"] and p not in prefetched)
        room = int(G["reserve"].numel()) - PREFETCH_FLOOR
        if want and room > 0:
            take = want[:room]
            frames = G["reserve"][:len(take)]
            G["reserve"] = G["reserve"][len(take):]
            G["reserve_tokens"] = int(G["reserve"].numel()) * PAGE
            src_list, dst_list = [], []
            for p, frame_page in zip(take, frames.tolist()):
                src_list.append(st["page_slots"][p])
                frame_slots = torch.arange(
                    int(frame_page) * PAGE, (int(frame_page) + 1) * PAGE,
                    dtype=src_list[-1].dtype, device=device)
                st["page_slots"][p] = frame_slots
                dst_list.append(frame_slots)
            src_cpu = torch.cat(src_list).cpu()
            dst = torch.cat(dst_list)
            with torch.no_grad():
                for layer in layers:
                    mk, mv = mirror[layer]
                    stream = _tier_stream() if (ASYNC and torch.cuda.is_available()) else None
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            pool.get_key_buffer(layer)[dst] = mk[src_cpu].to(
                                dst.device, non_blocking=True)
                            pool.get_value_buffer(layer)[dst] = mv[src_cpu].to(
                                dst.device, non_blocking=True)
                    else:
                        pool.get_key_buffer(layer)[dst] = mk[src_cpu].to(dst.device)
                        pool.get_value_buffer(layer)[dst] = mv[src_cpu].to(dst.device)
            for p in take:
                prefetched.add(p)
            G["prefetch_issued"] = G.get("prefetch_issued", 0) + len(take)
