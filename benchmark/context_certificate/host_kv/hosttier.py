"""Runtime host-tier v0: snapshot, evict-in-place, demand-stage pages.

P2.2 correctness slice. The pool keeps its full size; "eviction" poisons a
page's GPU contents and "staging" copies the page back from a pinned host
mirror to the SAME slots (page-aligned allocation makes a page's slots
contiguous, so slot // page_size identifies the page frame). Outputs are
bit-identical to the un-evicted run iff staging is correct; a silent
failure reads the canary and corrupts the token stream loudly.

Config (env): SGLANG_HOSTTIER_ENABLE=1, SGLANG_HOSTTIER_EVICT_FRACTION
(default 0.9), SGLANG_HOSTTIER_HOT_PAGES (recent pages kept, default 8),
SGLANG_HOSTTIER_LOG (stats path), SGLANG_HOSTTIER_PAGE (tokens per page,
must match the pool's page_size, default 64).
"""

import atexit
import json
import os
import time

import torch

_PREFIX = "SGLANG_HOSTTIER_"
ENABLE = os.environ.get(_PREFIX + "ENABLE", "") == "1"
EVICT_FRACTION = float(os.environ.get(_prefix := _PREFIX + "EVICT_FRACTION", 0.9))
HOT_PAGES = int(os.environ.get(_PREFIX + "HOT_PAGES", 8))
PAGE = int(os.environ.get(_PREFIX + "PAGE", 64))
LOG_PATH = os.environ.get(_PREFIX + "LOG", "/root/phase0/hosttier-stats.json")

if ENABLE:
    print(f"[hosttier] module loaded (fraction={EVICT_FRACTION}, hot={HOT_PAGES})", flush=True)

_state = {"mirror": None, "pool": None, "kv_layers": None, "last_exception": None,
          "snapshotted": False,
          "evicted": set(), "staged_pages": 0, "staged_bytes": 0,
          "stage_calls": 0, "stage_seconds": 0.0, "syncs": 0,
          "verify_mismatches": 0, "verified_page_layers": 0}


def _attn_layers(pool):
    """Accepted KV layer ids for this pool, or [] if not the main pool.

    The hybrid pool's layer_num counts linear/GDN layers (no KV buffers),
    and the draft model's pool carries a single layer; both must yield []
    rather than poison the cache."""
    cached = _state.setdefault("kv_layers_by_pool", {}).get(id(pool))
    if cached:
        return cached
    # HybridLinearKVPool translates model layer ids (3, 7, ... 47) through
    # full_attention_layer_id_mapping and raises for every other id; naive
    # range() enumeration probes the wrong id space (window-2 lesson).
    mapping = getattr(pool, "full_attention_layer_id_mapping", None)
    if mapping:
        layers = sorted(int(layer) for layer in mapping)
    else:
        layers = []
        for layer in range(getattr(pool, "layer_num", 12)):
            try:
                pool.get_key_buffer(layer)
                pool.get_value_buffer(layer)
            except Exception:  # noqa: BLE001 - layer enumeration probe
                continue
            layers.append(layer)
    if len(layers) >= 8:
        _state.setdefault("kv_layers_by_pool", {})[id(pool)] = layers
        return layers
    return []


def on_pool(pool):
    """Allocate the pinned host mirror for the MAIN model's KV pool.

    The draft model extends the same backend and carries a single-layer KV
    pool; it must be ignored, or its one-layer snapshot blocks the main
    pool's staging for the rest of the run. Only pools with a full KV layer
    set (>= 8 layers here) are accepted."""
    if not ENABLE or _state["mirror"] is not None:
        return
    layers = _attn_layers(pool)
    if not layers:
        return  # draft-sized pool; do not arm
    def _host_clone(buffer):
        clone = buffer.cpu() if buffer.is_cuda else buffer.clone()
        try:
            return clone.pin_memory()
        except RuntimeError:  # CPU-only torch (tests): mirror unpinned
            return clone

    mirror = []
    for layer in layers:
        mirror.append({"k": _host_clone(pool.get_key_buffer(layer)),
                       "v": _host_clone(pool.get_value_buffer(layer)),
                       "shape": pool.get_key_buffer(layer).shape})
    _state["kv_layers"] = layers
    _state["mirror"] = mirror
    _state["pool"] = pool
    _info(f"mirror allocated: {len(mirror)} layers, "
          f"{sum(m['k'].numel() * m['k'].element_size() * 2 for m in mirror) / 2**30:.2f} GiB")


def _info(msg):
    print(f"[hosttier {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _page_of(slots):
    return slots // PAGE


def _snapshot_and_evict(slots_flat):
    """Once per run: mirror every touched page, then poison all but the
    most recent HOT_PAGES pages of the (single) request."""
    pool, mirror = _state["pool"], _state["mirror"]
    assert pool is not None and mirror, "snapshot before mirror armed"
    pages = torch.unique(_page_of(slots_flat))
    last_page = int(pages.max().item())
    with torch.no_grad():
        for layer_i, layer in enumerate(_attn_layers(pool)):
            mirror[layer_i]["k"].copy_(pool.get_key_buffer(layer), non_blocking=True)
            mirror[layer_i]["v"].copy_(pool.get_value_buffer(layer), non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Evict across the request's FULL page span (the mirror already holds
    # the whole pool): evicting only the arming step's own selection pages
    # never forces staging, because decode selections are recency-biased
    # and land elsewhere (window-3 finding).
    span = int(_page_of(slots_flat).max().item())
    evict = [p for p in range(1, span + 1) if p < span - HOT_PAGES + 1]
    keep = max(1, int(round(len(evict) * (1 - EVICT_FRACTION))))
    evict = sorted(evict[: len(evict) - keep]) if keep else evict
    canary = float("nan")
    for page in evict:
        lo, hi = page * PAGE, (page + 1) * PAGE
        with torch.no_grad():
            for layer in _attn_layers(pool):
                pool.get_key_buffer(layer)[lo:hi].fill_(canary)
                pool.get_value_buffer(layer)[lo:hi].fill_(canary)
        _state["evicted"].add(page)
    _info(f"snapshot {len(pages)} pages, evicted {len(_state['evicted'])} "
          f"(fraction={EVICT_FRACTION}, hot={HOT_PAGES})")
    # Deterministic hardware exercise of the stage path: real decode traffic
    # is recency-biased and may never demand a cold page (window-4 finding),
    # so stage one evicted page through the identical copy+verify code path
    # at arm time. This is the v0 hardware proof, independent of selection
    # luck.
    if _state["evicted"]:
        page = min(_state["evicted"])
        lo, hi = page * PAGE, (page + 1) * PAGE
        for layer_i, layer in enumerate(_attn_layers(pool)):
            pool.get_key_buffer(layer)[lo:hi].copy_(mirror[layer_i]["k"][lo:hi],
                                                    non_blocking=True)
            pool.get_value_buffer(layer)[lo:hi].copy_(mirror[layer_i]["v"][lo:hi],
                                                      non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        verified = True
        for layer_i, layer in enumerate(_attn_layers(pool)):
            gpu_k = pool.get_key_buffer(layer)[lo:hi]
            gpu_v = pool.get_value_buffer(layer)[lo:hi]
            if gpu_k.is_cuda:
                gpu_k, gpu_v = gpu_k.cpu(), gpu_v.cpu()
            verified &= (torch.equal(gpu_k, mirror[layer_i]["k"][lo:hi])
                         and torch.equal(gpu_v, mirror[layer_i]["v"][lo:hi]))
        _state["evicted"].discard(page)
        _state["selfcheck"] = {"page": page, "verified": bool(verified)}
        _info(f"selfcheck staged page {page}: verified={verified}")
    _state["snapshotted"] = True
    dump_stats()


def stage_physical_slots(slots, pool):
    """Demand path: bring evicted pages of these physical slots back."""
    if not ENABLE:
        return
    try:
        _stage_inner(slots, pool)
    except Exception as error:  # noqa: BLE001 - diagnose, persist, re-raise
        _state["last_exception"] = repr(error)[:2000]
        try:
            dump_stats()
        except Exception:
            pass
        raise


def _stage_inner(slots, pool):
    on_pool(pool)
    if _state["mirror"] is None:
        return
    if not _state["snapshotted"]:
        valid = slots[slots >= 0]
        if valid.numel() == 0 or int(valid.max()) == 0:
            return
        # Arm only on a genuinely long context: the server's own warmup
        # request spans ~128 pages and clears a small-page guard (window-3
        # finding: arming on warmup evicts warmup slots the real request
        # never revisits). 2,000 pages = 128k tokens: arm only when
        # selections reach the deep-recent region so the eviction span
        # covers where decode actually selects.
        if int(_page_of(valid).max().item()) < 2000:
            return
        _snapshot_and_evict(valid)
    if not _state["evicted"]:
        return
    pages = torch.unique(_page_of(slots[slots >= 0]))
    missing = [int(p) for p in pages.tolist() if int(p) in _state["evicted"]]
    if not missing:
        return
    start = time.monotonic()
    mirror = _state["mirror"]
    with torch.no_grad():
        for page in missing:
            lo, hi = page * PAGE, (page + 1) * PAGE
            for layer_i, layer in enumerate(_attn_layers(pool)):
                pool.get_key_buffer(layer)[lo:hi].copy_(mirror[layer_i]["k"][lo:hi],
                                                        non_blocking=True)
                pool.get_value_buffer(layer)[lo:hi].copy_(mirror[layer_i]["v"][lo:hi],
                                                          non_blocking=True)
            _state["evicted"].discard(page)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Memory-level correctness gate: every restored page must be
    # byte-identical to the pre-eviction mirror. This is immune to the
    # sampling nondeterminism that breaks end-to-end token comparison.
    verified = 0
    for page in missing:
        lo, hi = page * PAGE, (page + 1) * PAGE
        for layer_i, layer in enumerate(_attn_layers(pool)):
            gpu_k = pool.get_key_buffer(layer)[lo:hi]
            gpu_v = pool.get_value_buffer(layer)[lo:hi]
            if gpu_k.is_cuda:  # torch.equal needs one device; mirror is host-pinned
                gpu_k, gpu_v = gpu_k.cpu(), gpu_v.cpu()
            if not (torch.equal(gpu_k, mirror[layer_i]["k"][lo:hi])
                    and torch.equal(gpu_v, mirror[layer_i]["v"][lo:hi])):
                _state["verify_mismatches"] += 1
                _info(f"VERIFY MISMATCH page={page} layer={layer}")
            else:
                verified += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _state["staged_pages"] += len(missing)
    _state["verified_page_layers"] = verified
    _state["staged_bytes"] += len(missing) * PAGE * 24576
    _state["stage_calls"] += 1
    _state["syncs"] += 1
    _state["stage_seconds"] += time.monotonic() - start
    _info(f"staged {len(missing)} pages ({_state['staged_pages']} total, "
          f"{_state['stage_seconds']:.2f}s cumulated)")


def dump_stats():
    if not ENABLE:
        return
    payload = {k: (sorted(v) if isinstance(v, set) else v) for k, v in _state.items()
               if k not in ("mirror", "pool")}
    payload["still_evicted_pages"] = len(_state["evicted"])
    try:
        with open(LOG_PATH, "w") as handle:
            json.dump(payload, handle, indent=2)
        _info(f"stats written to {LOG_PATH}")
    except OSError:
        pass


if ENABLE:
    atexit.register(dump_stats)


def _capturing():
    """True while CUDA graph capture records this forward.

    Tier work (host syncs, side-stream copies, table writes keyed to a
    live batch) is illegal or meaningless inside capture; replays skip
    Python entirely, so the tier must run between replays at the
    scheduler level instead (HKV-12). During capture the tier stands
    down."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def probe_install(ns):
    """Wrap every forward-ish method of the backend classes in `ns` with a
    call counter, and attempt staging in any of them that carries
    topk_indices + forward_batch. One decode run then reveals the live
    verify entrypoint AND may arm staging wherever it actually flows."""
    import inspect

    counts = {}
    _state["entrypoint_counts"] = counts

    def make_wrapper(name, orig):
        def wrapper(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            if counts[name] <= 2:
                _info(f"entrypoint {name} call #{counts[name]} "
                      f"kwargs={sorted(kwargs)}")
            self_obj = args[0] if args else None
            topk = kwargs.get("topk_indices")
            batch = kwargs.get("forward_batch")
            if batch is None:
                batch = next((a for a in args[1:]
                              if hasattr(a, "forward_mode")), None)
            if (topk is not None and batch is not None and self_obj is not None
                    and not _capturing()):
                try:
                    import os as _os
                    from sglang.srt.layers.attention.qsa.hosttier import stage_physical_slots
                    _mode = _os.environ.get("SGLANG_HOSTTIER_MODE", "")
                    if _mode in ("P23", "P23M"):
                        # P2.3 single- or multi-request tier: pre-restore
                        # retired pages, run the original, then mirror +
                        # retire the freshly written chunk.
                        if _mode == "P23M":
                            from sglang.srt.layers.attention.qsa import hosttier23m as _p23
                        else:
                            from sglang.srt.layers.attention.qsa import hosttier23 as _p23
                        rows = getattr(batch, "req_pool_indices", None)
                        if _p23.ENABLE and rows is not None and rows.numel() > 0:
                            _p23.arm(self_obj, int(rows[0].item()))
                            _foreign = (
                                getattr(self_obj, "token_to_kv_pool", None)
                                is not _p23.G.get("pool"))
                            _nl = _state.get("wrap_logs", 0)
                            if _nl < 80:
                                _state["wrap_logs"] = _nl + 1
                                _mb = getattr(batch, "out_cache_loc", None)
                                _mode = getattr(batch, "forward_mode", None)
                                _info(
                                    f"wrap#{_nl} cls={type(self_obj).__name__}"
                    f" foreign={_foreign} mode={getattr(_mode, 'value', _mode)}"
                    f" locs={int(_mb.numel()) if _mb is not None else None}"
                    f" rows={int(rows.numel())}"
                    f" ext={[int(x) for x in getattr(batch, 'extend_seq_lens', [])[:3]]}"
                                )
                            if not _foreign:
                                if _mode == "P23M":
                                    _p23.before_gather(topk, batch)
                                else:
                                    _p23.before_gather(
                                        topk, int(getattr(batch, "seq_lens", [0])[0]))
                                out = orig(*args, **kwargs)
                                try:
                                    _p23.audit_coverage(topk, batch)
                                except Exception:
                                    pass
                                _p23.after_write(batch)
                                return out
                    if _mode in ("P23", "P23M"):
                        pass  # the tier owns staging in these modes; the
                              # v0 path must not run with a globally cached
                              # layer list against a foreign (draft) pool
                    else:
                        resolve = getattr(self_obj, "_resolve_metadata", None)
                        pool = getattr(self_obj, "token_to_kv_pool", None)
                        if resolve is not None and pool is not None:
                            stage_physical_slots(
                                self_obj._logical_to_physical(topk, resolve(batch)).long(), pool)
                except (ImportError, AttributeError, TypeError, ValueError):
                    pass
            return orig(*args, **kwargs)
        return wrapper

    for cname, cls in list(ns.items()):
        if not inspect.isclass(cls) or cls.__module__ != ns["__name__"]:
            continue
        for name, member in vars(cls).items():
            # _logical_to_physical is a staticmethod utility: wrapping it
            # changes its calling convention and breaks every caller.
            if name.startswith("forward") or name == "_forward_paged_attention":
                if callable(member):
                    wrapped = make_wrapper(f"{cname}.{name}", member)
                    if isinstance(inspect.getattr_static(cls, name), staticmethod):
                        wrapped = staticmethod(wrapped)  # preserve descriptor
                    setattr(cls, name, wrapped)
                    _info(f"probing {cname}.{name}")

