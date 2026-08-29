"""Rust dual-write shadow for the base ``RadixCache``.

Activated by ``SGLANG_RUST_SCHEDULER=radix`` (or a later stage). The
Python ``RadixCache`` remains the source of truth for serving — this
facade mirrors every hot operation into the Rust ``RadixTree``
(``sglang-radix`` via the ``_scheduler`` extension) and checks the two
against each other:

- every ``match_prefix`` result length is compared (lengths only, so no
  GPU sync is added on the hot path);
- every ``insert`` prefix length is compared;
- sizes / evict counts are compared on the cheap int paths.

On divergence the Rust tree is resynced from the Python tree (walk the
tree, reinsert leaf paths, reapply locks). Resyncs are rate-limited;
mismatches are counted forever and surfaced through ``stats()``.

Known soft-diff sources (do not indicate a bug):
- LRU victim choice: Python orders by ``time.monotonic()`` floats, the
  Rust tree by a deterministic walk clock — ties and near-ties can pick
  different victims, which then diverges match lengths until the next
  resync.
- ``cache_unfinished_req`` runs an extra internal match in Python that
  the mirror cannot observe without desyncing LRU clocks, so the mirror
  maps the post-insert terminal node instead.

Namespaced keys (``extra_key`` / ``cache_salt``) are not representable in
the current Rust tree API; the shadow disables itself on first sight.
EAGLE (bigram) trees are supported: raw tokens are passed and the Rust
tree applies the bigram view, mirroring ``maybe_to_bigram_view``.
"""

from __future__ import annotations

import logging
import time
import weakref
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_RESYNC_MIN_INTERVAL = 5.0  # seconds between resyncs
_RESYNC_SOFT_THRESHOLD = 8  # cheap-check divergences before a resync
_WARN_INTERVAL = 30.0  # seconds between divergence warnings


def wrap_if_rust(cache: Any) -> Any:
    """Wrap ``cache`` in the dual-write Rust shadow when the active
    ``SGLANG_RUST_SCHEDULER`` stage is ``radix`` or above and the cache
    is a base-MHA ``RadixCache``. Returns the original object otherwise.
    """
    if cache is None:
        return cache
    try:
        from sglang.srt.managers.rust_scheduler import load_module, stage_at_least

        if not stage_at_least("radix"):
            return cache
        from sglang.srt.mem_cache.radix_cache import RadixCache

        if type(cache) is not RadixCache or cache.disable:
            return cache
        mod = load_module()
        if mod is None:
            return cache
        return RustRadixShadow(mod, cache)
    except Exception:
        logger.exception(
            "failed to attach the Rust radix shadow; using Python only"
        )
        return cache


class RustRadixShadow:
    """Dual-write facade: Python ``RadixCache`` (source of truth) plus a
    mirrored Rust ``RadixTree`` that is verified on every hot op."""

    def __init__(self, mod: Any, cache: Any):
        self._mod = mod
        self._py = cache
        self.tree = mod.RadixTree(
            page_size=cache.page_size, is_eagle=cache.is_eagle, policy=cache.eviction_policy
        )
        self._handles = weakref.WeakKeyDictionary()  # TreeNode -> rust NodeId
        # counters
        self.ops = 0
        self.mismatches = 0
        self.soft_diverges = 0
        self.resyncs = 0
        self.resync_skipped = 0
        self.handle_misses = 0
        self.namespaced_skips = 0
        self._dead = False  # set when namespacing is detected
        self._last_resync = 0.0
        self._last_warn = 0.0
        logger.info(
            "rust radix shadow enabled (page_size=%d, is_eagle=%s, policy=%s)",
            cache.page_size,
            cache.is_eagle,
            cache.eviction_policy,
        )

    # ------------------------------------------------------------- helpers

    def _key_ids(self, key: Any) -> Optional[List[int]]:
        """Raw token ids for the Rust tree; None when the key is
        namespaced (which disables the shadow) or capped beyond what the
        tree can represent."""
        if getattr(key, "extra_key", None) is not None or getattr(
            key, "cache_salt", None
        ) is not None:
            self.namespaced_skips += 1
            self._dead = True
            return None
        raw = key.raw_token_ids()
        return [int(t) for t in raw]

    def _rid(self, node: Any) -> Optional[int]:
        """Rust node id for a Python TreeNode (ROOT for the root)."""
        if node is self._py.root_node:
            return self._mod.ROOT
        rid = self._handles.get(node)
        if rid is None:
            self.handle_misses += 1
        return rid

    def _logical_len(self, key: Any) -> int:
        """Page-aligned logical length after the bigram view, exactly as
        Python's insert/match preprocessing computes it."""
        key, _ = key.maybe_to_bigram_view(self._py.is_eagle)
        return len(key.page_aligned(self._py.page_size))

    def _diverge(self, msg: str, *, hard: bool = False) -> None:
        self.mismatches += 1
        if hard:
            self._resync()
        else:
            self.soft_diverges += 1
            if self.soft_diverges >= _RESYNC_SOFT_THRESHOLD:
                self._resync()
        now = time.monotonic()
        if now - self._last_warn > _WARN_INTERVAL:
            self._last_warn = now
            logger.warning(
                "rust radix shadow diverged: %s (total mismatches=%d)",
                msg,
                self.mismatches,
            )

    def _resync(self) -> None:
        """Rebuild the Rust tree from the Python tree."""
        now = time.monotonic()
        if now - self._last_resync < _RESYNC_MIN_INTERVAL:
            self.resync_skipped += 1
            return
        self._last_resync = now
        self.resyncs += 1
        py = self._py
        self.tree = self._mod.RadixTree(
            page_size=py.page_size, is_eagle=py.is_eagle, policy=py.eviction_policy
        )
        self._handles = weakref.WeakKeyDictionary()
        root = py.root_node
        leaves: List[tuple] = []
        locked: List[tuple] = []

        def walk(node, toks: List[int], vals: List[int]) -> None:
            if node is not root and node.value is None:
                return  # evicted edge: subtree unreachable
            edge = [int(t) for t in node.key.raw_token_ids()]
            edge_vals = (
                [int(x) for x in node.value]
                if node is not root and node.value is not None
                else []
            )
            nt, nv = toks + edge, vals + edge_vals
            if node.children:
                if node is not root and node.lock_ref > 0:
                    locked.append((nt, nv, node.lock_ref, node))
                for child in list(node.children.values()):
                    walk(child, nt, nv)
            elif node is not root:
                leaves.append((nt, nv, int(node.priority), node))
                if node.lock_ref > 0:
                    locked.append((nt, nv, node.lock_ref, node))

        walk(root, [], [])
        for toks, vals, prio, node in leaves:
            _p, rid = self.tree.insert(toks, vals, prio, False)
            self._handles[node] = rid
        for toks, vals, ref, node in locked:
            rid = self._handles.get(node)
            if rid is None:
                # Materialize the boundary (splits the mid-edge, like a
                # Python match at this path would) and lock from there.
                _p, rid = self.tree.insert(toks, vals, 0, False)
                self._handles[node] = rid
            for _ in range(int(ref)):
                self.tree.inc_lock_ref(rid)

    # ------------------------------------------------------- mirrored ops

    def match_prefix(self, params: Any) -> Any:
        result = self._py.match_prefix(params)
        if self._dead:
            return result
        self.ops += 1
        ids = self._key_ids(params.key)
        if ids is None:
            return result
        # Fast path: length + terminal handle only (no KV-index list
        # conversion, which dominates on long prompts).
        rust_len, rust_node = self.tree.match_prefix_meta(ids)
        if rust_node != self._mod.ROOT:
            self._handles[result.last_device_node] = rust_node
        n_py = len(result.device_indices)
        if rust_len != n_py:
            self._diverge(f"match_len py={n_py} rust={rust_len}", hard=True)
        return result

    def insert(self, params: Any) -> Any:
        result = self._py.insert(params)
        if self._dead or params.value is None:
            return result
        self.ops += 1
        ids = self._key_ids(params.key)
        if ids is None:
            return result
        n = self._logical_len(params.key)
        if n == 0:
            return result
        vals = [int(x) for x in params.value[:n].tolist()]
        priority = int(getattr(params, "priority", 0) or 0)
        chunked = bool(getattr(params, "chunked", False))
        rust_prefix, rust_node = self.tree.insert(ids, vals, priority, chunked)
        if rust_node != self._mod.ROOT:
            self._handles[result.last_device_node] = rust_node
        if rust_prefix != result.prefix_len:
            self._diverge(
                f"insert_prefix_len py={result.prefix_len} rust={rust_prefix}",
                hard=True,
            )
        return result

    def cache_finished_req(self, req: Any, is_insert: bool = True, **kw) -> Any:
        result = self._py.cache_finished_req(req, is_insert=is_insert, **kw)
        if (
            self._dead
            or not is_insert
            or self._py.disable_finished_insert
            or req.req_pool_idx is None
        ):
            return result
        self.ops += 1
        kv_len_to_handle = kw.get("kv_len_to_handle")
        if kv_len_to_handle is None:
            return result
        key = _key_from_tokens(
            (req.origin_input_ids + req.output_ids)[:kv_len_to_handle],
            req,
            self._py.is_eagle,
        )
        ids = self._key_ids(key)
        if ids is None:
            return result
        n = self._logical_len(key)
        if n == 0:
            return result
        row = self._py.req_to_token_pool.req_to_token[req.req_pool_idx]
        vals = [int(x) for x in row[:n].tolist()]
        priority = int(getattr(req, "priority", 0) or 0)
        self.tree.insert(ids, vals, priority, False)
        # Python released the request's old prefix lock internally
        # (`if req.last_node is not None: self.dec_lock_ref(req.last_node)`).
        if req.last_node is not None:
            rid = self._rid(req.last_node)
            if rid is not None:
                self.tree.dec_lock_ref(rid)
        return result

    def cache_unfinished_req(self, req: Any, chunked: bool = False) -> Any:
        old_node = req.last_node
        result = self._py.cache_unfinished_req(req, chunked=chunked)
        if self._dead or req.req_pool_idx is None:
            return result
        self.ops += 1
        key = _key_from_tokens(req.get_fill_ids(), req, self._py.is_eagle)
        ids = self._key_ids(key)
        if ids is None:
            return result
        n = self._logical_len(key)
        if n == 0:
            return result
        row = self._py.req_to_token_pool.req_to_token[req.req_pool_idx]
        vals = [int(x) for x in row[:n].tolist()]
        priority = int(getattr(req, "priority", 0) or 0)
        _rust_prefix, rust_node = self.tree.insert(ids, vals, priority, chunked)
        # req.last_node was just set to the terminal node of the inserted
        # key (Python's post-insert match returns the same boundary; the
        # mirror insert returns it directly).
        if req.last_node is not None and rust_node != self._mod.ROOT:
            self._handles[req.last_node] = rust_node
        # Mirror Python's internal lock handoff: release the old node (the
        # one the adder locked at admission), lock the new terminal node.
        if old_node is not None and old_node is not req.last_node:
            old_rid = self._handles.get(old_node)
            if old_rid is not None:
                self.tree.dec_lock_ref(old_rid)
            else:
                self.handle_misses += 1
        if rust_node != self._mod.ROOT:
            self.tree.inc_lock_ref(rust_node)
        return result

    def inc_lock_ref(self, node: Any) -> Any:
        result = self._py.inc_lock_ref(node)
        if self._dead:
            return result
        rid = self._rid(node)
        if rid is not None:
            self.tree.inc_lock_ref(rid)
        return result

    def dec_lock_ref(self, node: Any, params: Any = None) -> Any:
        result = self._py.dec_lock_ref(node, params)
        if self._dead:
            return result
        rid = self._rid(node)
        if rid is not None:
            self.tree.dec_lock_ref(rid)
        return result

    def evict(self, params: Any) -> Any:
        result = self._py.evict(params)
        if self._dead:
            return result
        n = int(getattr(params, "num_tokens", 0) or 0)
        if n > 0:
            _runs, rust_n = self.tree.evict(n)
            if rust_n != result.num_tokens_evicted:
                self._diverge(
                    f"evict_count py={result.num_tokens_evicted} rust={rust_n}"
                )
        return result

    def total_size(self) -> int:
        v = self._py.total_size()
        if not self._dead and self.tree.total_size() != v:
            self._diverge(f"total_size py={v} rust={self.tree.total_size()}")
        return v

    def evictable_size(self) -> int:
        v = self._py.evictable_size()
        if not self._dead and self.tree.evictable_size() != v:
            self._diverge(
                f"evictable_size py={v} rust={self.tree.evictable_size()}"
            )
        return v

    def protected_size(self) -> int:
        v = self._py.protected_size()
        if not self._dead and self.tree.protected_size() != v:
            self._diverge(
                f"protected_size py={v} rust={self.tree.protected_size()}"
            )
        return v

    def reset(self) -> None:
        self._py.reset()
        self.tree = self._mod.RadixTree(
            page_size=self._py.page_size,
            is_eagle=self._py.is_eagle,
            policy=self._py.eviction_policy,
        )
        self._handles = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------- proxy

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define; delegate
        # everything else to the wrapped Python cache.
        return getattr(self.__dict__["_py"], name)

    def __repr__(self) -> str:
        return f"RustRadixShadow({self._py!r})"

    def stats(self) -> dict:
        return {
            "ops": self.ops,
            "mismatches": self.mismatches,
            "soft_diverges": self.soft_diverges,
            "resyncs": self.resyncs,
            "resync_skipped": self.resync_skipped,
            "handle_misses": self.handle_misses,
            "namespaced_skips": self.namespaced_skips,
            "dead": self._dead,
        }


def _key_from_tokens(tokens: Any, req: Any, is_eagle: bool) -> Any:
    """Rebuild the RadixKey the Python cache would build for ``req`` over
    ``tokens`` (same extra_key / cache_salt / bigram view)."""
    from array import array

    from sglang.srt.mem_cache.radix_cache import RadixKey

    return RadixKey(
        array("q", [int(t) for t in tokens]),
        getattr(req, "extra_key", None),
        is_bigram=is_eagle,
        cache_salt=getattr(req, "cache_salt", None),
    )
