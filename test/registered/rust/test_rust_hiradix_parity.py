"""Differential parity test: Rust HiRadixTree vs Python HiRadixCache
(plan.md M2/1d).

Drives the ``_scheduler`` PyO3 extension's ``HiRadixTree`` and the
unmodified ``sglang.srt.mem_cache.hiradix_cache.HiRadixCache`` with the
same op sequence (page_size=1, non-eagle, write-through) and asserts the
observable state agrees after every op: device/host size bookkeeping,
match device indices + host-hit length + last-device/last-host node
position, insert prefix lengths + threshold-triggered backup sets,
write-through ack deltas, evict demote-vs-drop free runs, evict_host
skip/promote/delete behavior, two-phase load_back (threshold / quota /
round trip / permanent chain lock), insert_host (fresh / full-match /
split), the write-back facade primitives (detach_backuped,
drop_subtree_no_host, promote_parent, ordered leaves), and the
host-ref underflow error.

The Python cache is constructed without its real ``__init__`` (which
needs server args, a HiCacheController, torch.distributed and a host
pool): ``HiRadixCache.__new__`` + the attributes the tree ops actually
touch + the real ``reset()``. DMA stays caller-side on both
implementations: a recording fake controller hands out deterministic
arange host/device indices so the runs stay comparable, and the test
feeds every run the Python tree produced back into the Rust tree
(``begin_backup`` / ``finish_load_back``) — exactly what the Python
facade does with the controller's acks.

Node identity across implementations is by key-tuple (a token sequence
maps to a unique root path in both trees); node ids are not comparable.

Op orderings that depend on LRU tie-breaks are handled by construction:
the Rust tree stamps every node of one op with a single tick (Python
stamps each visited node with a fresh ``time.monotonic()``), so where
Rust sees a tie the test only relies on the node-id tie-break agreeing
with Python's within-walk stamp order, and otherwise only full drains
and single-leaf partials run.
"""

import heapq
import sys
import unittest
from array import array as qarr

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import get_eviction_strategy
from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DRAIN = 10**9


def token_run(seed: int, n: int) -> list:
    """Deterministic token-id run (mirrors the other parity tests)."""
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(s % 100_000)
    return out


def values_for(ids: list) -> list:
    return [i + 100_000 for i in ids]


class FakeSink:
    """Ordered recording of every DMA / free the Python tree issues."""

    def __init__(self):
        # device runs freed, in tree order (demotes and drops alike)
        self.device_frees = []
        # host runs freed by evict_host / drop-subtree
        self.host_frees = []
        # host runs the fake controller's write() returned
        self.host_writes = []
        # device runs the fake controller's load() returned
        self.device_loads = []

    def take(self, attr):
        out = getattr(self, attr)
        setattr(self, attr, [])
        return out


class FakeDeviceAllocator:
    """Stands in for the device KV allocator (the Python tree only
    frees through it; the parity test never allocates)."""

    def __init__(self, sink):
        self.sink = sink

    def free(self, index):
        self.sink.device_frees.append(list(index.tolist()))


class FakeController:
    """Stands in for HiCacheController: synchronous deterministic DMA.

    ``write`` / ``load`` return fresh arange runs (recorded in the
    sink so the test can feed the identical indices to the Rust tree);
    ``evict_device`` / ``evict_host`` record the freed runs.
    """

    def __init__(self, sink):
        self.write_policy = "write_through"
        self.mem_pool_device_allocator = FakeDeviceAllocator(sink)
        self._sink = sink
        self._host_next = 1_000_000
        self._device_next = 2_000_000

    def write(self, device_indices, node_id, **kwargs):
        n = device_indices.numel()
        host = torch.arange(
            self._host_next, self._host_next + n, dtype=torch.int64
        )
        self._host_next += n
        self._sink.host_writes.append(list(host.tolist()))
        return host

    def load(self, host_indices, node_id, **kwargs):
        n = host_indices.numel()
        device = torch.arange(
            self._device_next, self._device_next + n, dtype=torch.int64
        )
        self._device_next += n
        self._sink.device_loads.append(list(device.tolist()))
        return device

    def evict_device(self, device_indices):
        self._sink.device_frees.append(list(device_indices.tolist()))

    def evict_host(self, host_indices):
        self._sink.host_frees.append(list(host_indices.tolist()))
        return len(host_indices)

    def reset(self):
        pass


class FakeHostPool:
    def clear(self):
        pass

    def destroy(self):
        pass


class FakeEvents:
    """KVCacheEventRecorder stand-in: disabled, all recorders no-op."""

    enabled = False

    def record_store(self, *args, **kwargs):
        pass

    def record_remove(self, *args, **kwargs):
        pass

    def record_all_cleared(self, *args, **kwargs):
        pass


def make_py_hiradix(eviction_policy: str = "lru") -> tuple:
    """Minimal HiRadixCache: skip __init__, set the attributes the tree
    ops touch, run the real reset()."""
    cache = HiRadixCache.__new__(HiRadixCache)
    sink = FakeSink()
    cache.disable = False
    cache.req_to_token_pool = None
    cache.token_to_kv_pool_allocator = FakeDeviceAllocator(sink)
    cache.page_size = 1
    cache.is_eagle = False
    cache.disable_finished_insert = True
    cache.eviction_policy = eviction_policy
    cache.eviction_strategy = get_eviction_strategy(eviction_policy)
    cache.kv_events = FakeEvents()
    cache.enable_metrics = False
    cache.metrics_collector = None
    cache.device = torch.device("cpu")
    cache.evictable_leaves = set()
    cache.ongoing_write_through = {}
    cache.ongoing_load_back = {}
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.work_list = []
    cache.write_through_threshold = 1
    cache.load_back_threshold = 10
    cache.evictable_host_leaves = set()
    cache.enable_storage = False
    cache.hicache_storage_pass_prefix_keys = False
    cache.cache_controller = FakeController(sink)
    cache.token_to_kv_pool_host = FakeHostPool()
    cache.reset()
    return cache, sink


class TestRustHiRadixParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mod = load_rust_extension("sglang.srt.rust_extensions._scheduler")

    def make_pair(self, eviction_policy: str = "lru"):
        self.py, self.sink = make_py_hiradix(eviction_policy)
        self.rs = self.mod.HiRadixTree(1, False, "write_through", eviction_policy, 1, 10)
        return self.py, self.rs, self.sink

    def setUp(self):
        super().setUp()
        # Fresh pair per test: the op sequences are stateful.
        self.make_pair()

    # ---- op wrappers ----

    def _rs_match(self, ids):
        return self.rs.match_prefix(list(ids))

    def _py_match(self, ids):
        return self.py.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=qarr("q", ids)))
        )

    def _rs_insert(self, ids, values, priority=0):
        return self.rs.insert(list(ids), list(values), priority, False)

    def _py_insert(self, ids, values, priority=0):
        return self.py.insert(
            InsertParams(
                key=RadixKey(token_ids=qarr("q", ids)),
                value=torch.tensor(values, dtype=torch.int64),
                priority=priority,
            )
        )

    def _mirror_backups(self, rs_backup_needed):
        """Feed the host runs the Python tree just wrote to Rust."""
        runs = self.sink.take("host_writes")
        self.assertEqual(
            len(rs_backup_needed), len(runs), "backup trigger counts diverged"
        )
        for nid, run in zip(rs_backup_needed, runs):
            self.rs.begin_backup(nid, run, True)
        return runs

    def _ack(self, py_node, rs_node):
        """Process the write-through DMA ack on both sides; returns the
        Rust lock-release delta."""
        self.py._finish_write_through_ack(py_node.id, release_lock=True)
        self.rs.end_backup(rs_node)
        return self.rs.dec_lock_ref(rs_node)

    # ---- state comparison ----

    def _sizes(self):
        py = (
            self.py.evictable_size(),
            self.py.protected_size(),
            self.py.total_size(),
            self._py_total_host(),
        )
        rs = (
            self.rs.evictable_size(),
            self.rs.protected_size(),
            self.rs.total_size(),
            self.rs.total_host_size(),
        )
        self.assertEqual(py, rs, "size bookkeeping diverged")
        return rs

    def _py_total_host(self):
        total = 0
        stack = [self.py.root_node]
        while stack:
            n = stack.pop()
            if n is not self.py.root_node and n.host_value is not None:
                total += len(n.host_value)
            stack.extend(n.children.values())
        return total

    @staticmethod
    def _py_key(node):
        return tuple(node.key.token_ids)

    def _rs_key(self, nid):
        return () if nid == 0 else tuple(self.rs.node_key(nid))

    def _py_find(self, key_tuple):
        stack = [self.py.root_node]
        while stack:
            n = stack.pop()
            if tuple(n.key.token_ids) == key_tuple:
                return n
            stack.extend(n.children.values())
        raise AssertionError(f"python node {key_tuple} not found")

    def _rs_find(self, key_tuple):
        stack = [0]
        while stack:
            nid = stack.pop()
            if self._rs_key(nid) == key_tuple:
                return nid
            stack.extend(self.rs.node_children(nid))
        raise AssertionError(f"rust node {key_tuple} not found")

    def _leaves_eq(self, which="device"):
        if which == "device":
            self.assertEqual(
                sorted(self._py_key(n) for n in self.py.evictable_leaves),
                sorted(self._rs_key(i) for i in self.rs.evictable_leaves()),
                "device evictable leaves diverged",
            )
        else:
            self.assertEqual(
                sorted(self._py_key(n) for n in self.py.evictable_host_leaves),
                sorted(self._rs_key(i) for i in self.rs.evictable_host_leaves()),
                "host evictable leaves diverged",
            )

    def _match_eq(self, ids, dev_len, hhl, last_dev_key, last_host_key):
        idx_rs, ldev_rs, lhost_rs, hhl_rs, splits_rs = self._rs_match(ids)
        py = self._py_match(ids)
        self.assertEqual(hhl_rs, py.host_hit_length, "host hit length diverged")
        self.assertEqual(len(idx_rs), dev_len)
        self.assertEqual(list(idx_rs), py.device_indices.tolist())
        self.assertEqual(self._rs_key(ldev_rs), last_dev_key)
        self.assertEqual(self._py_key(py.last_device_node), last_dev_key)
        self.assertEqual(self._rs_key(lhost_rs), last_host_key)
        self.assertEqual(self._py_key(py.last_host_node), last_host_key)
        return idx_rs, ldev_rs, lhost_rs, splits_rs, py

    # ------------------------------------------------------------------

    def test_write_through_flow(self):
        py, rs, sink = self.py, self.rs, self.sink
        drain = DRAIN

        a = token_run(11, 12)
        b = token_run(22, 8)
        ext = token_run(33, 4)
        va, vb, vext = values_for(a), values_for(b), values_for(ext)
        aext_key = tuple(a + ext)

        # A) insert run A: the fresh leaf crosses the threshold-1
        #    write-through backup on its first hit; the protective lock
        #    stays held until the DMA ack.
        rs_plen, rs_a, rs_need, rs_splits = self._rs_insert(a, va)
        py_res = self._py_insert(a, va)
        self.assertEqual((rs_plen, rs_splits), (0, []))
        self.assertEqual(py_res.prefix_len, 0)
        host_a = self._mirror_backups(rs_need)[0]
        py_a = self._py_find(tuple(a))
        self.assertEqual(py_a.write_through_pending_id, py_a.id)
        self.assertEqual(py.ongoing_write_through, {py_a.id: (py_a, 12, [py_a])})
        self.assertTrue(rs.node_backup_pending(rs_a))
        self.assertEqual(self._sizes(), (0, 12, 12, 12))

        # B) insert disjoint run B: same, on its own leaf.
        rs_plen, rs_b, rs_need, rs_splits = self._rs_insert(b, vb)
        py_res = self._py_insert(b, vb)
        self.assertEqual((rs_plen, rs_splits), (0, []))
        self.assertEqual(py_res.prefix_len, 0)
        host_b = self._mirror_backups(rs_need)[0]
        py_b = self._py_find(tuple(b))
        self.assertEqual(self._sizes(), (0, 20, 20, 20))

        # C) matches: full device hits, no host suffix, no splits.
        idx_rs, ldev, lhost, splits_rs, py = self._match_eq(
            a, 12, 0, tuple(a), tuple(a)
        )
        self.assertEqual(list(idx_rs), va)
        self.assertEqual(splits_rs, [])
        self.assertEqual(len(py.device_indices), 12)
        idx_rs, _, _, _, py = self._match_eq(b, 8, 0, tuple(b), tuple(b))
        self.assertEqual(list(idx_rs), vb)

        # D) extend A: the live match increments A's hit count (already
        #    backuped -> no new backup); the new 4-token leaf backs up.
        rs_plen, rs_aext, rs_need, rs_splits = self._rs_insert(a + ext, va + vext)
        py_res = self._py_insert(a + ext, va + vext)
        self.assertEqual(rs_plen, py_res.prefix_len)
        self.assertEqual(rs_plen, 12, "live match counts the prefix")
        host_aext = self._mirror_backups(rs_need)[0]
        py_aext = self._py_find(aext_key)
        self.assertEqual(len(host_aext), 4)
        self.assertEqual(py_aext.parent, py_a)
        self.assertEqual(self._sizes(), (0, 24, 24, 24))

        idx_rs, _, _, _, _ = self._match_eq(a + ext, 16, 0, aext_key, aext_key)
        self.assertEqual(list(idx_rs), va + vext)

        # E) DMA acks: publish the backups, release the protective locks.
        self.assertEqual(self._ack(py_a, rs_a), 12)
        self.assertEqual(self._ack(py_b, rs_b), 8)
        self.assertEqual(self._ack(py_aext, rs_aext), 4)
        self.assertEqual(py.ongoing_write_through, {})
        self.assertFalse(rs.node_backup_pending(rs_a))
        self.assertFalse(rs.node_backup_pending(rs_aext))
        self.assertIsNone(py_a.write_through_pending_id)
        self.assertEqual(self._sizes(), (24, 0, 24, 24))
        self._leaves_eq("device")  # {b, aext} leaves; a is internal

        # F) lock B, then full drain: B survives, A + AEXT demote to
        #    host (backed up), leaf order AEXT -> A (promotion).
        self.assertEqual(rs.inc_lock_ref(rs_b), -8)
        self.assertEqual(py.inc_lock_ref(py_b).delta, -8)
        self.assertEqual(self._sizes(), (16, 8, 24, 24))

        rs_num, rs_frees = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 16)
        self.assertEqual(rs_frees, sink.take("device_frees"))
        self.assertEqual(rs_frees, [vext, va])
        self.assertEqual(self._sizes(), (0, 8, 8, 24))
        self._leaves_eq("host")  # {a, aext}

        idx_rs, ldev, lhost, _, py = self._match_eq(a + ext, 0, 16, (), aext_key)
        self.assertEqual(rs_num, py.host_hit_length)
        self.assertIs(py.last_device_node, py.root_node)

        # G) unlock B and drain it.
        self.assertEqual(rs.dec_lock_ref(rs_b), 8)
        self.assertEqual(py.dec_lock_ref(py_b).delta, 8)
        rs_num, rs_frees = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 8)
        self.assertEqual(rs_frees, sink.take("device_frees"))
        self.assertEqual(self._sizes(), (0, 0, 0, 24))
        idx_rs, ldev, lhost, _, py = self._match_eq(b, 0, 8, (), tuple(b))

        # H) load_back the 16-token evicted chain: two-phase on Rust,
        #    one synchronous call on Python; the controller hands out
        #    the same device run on both sides.
        py_vals = py.load_back(py_aext)
        self.assertIsNotNone(py_vals)
        dev_run = sink.take("device_loads")[0]
        self.assertEqual(list(py_vals.tolist()), dev_run)

        plan = rs.init_load_back(rs_aext, None)
        self.assertIsNotNone(plan)
        anc, last, nodes, host_idx = plan
        self.assertEqual(anc, 0)
        self.assertEqual(last, rs_aext)
        self.assertEqual([self._rs_key(n) for n in nodes], [tuple(a), aext_key])
        self.assertEqual(host_idx, host_a + host_aext)
        self.assertEqual(rs.finish_load_back(anc, last, nodes, dev_run), -16)

        self.assertEqual(list(py_a.value.tolist()), dev_run[:12])
        self.assertEqual(list(py_aext.value.tolist()), dev_run[12:])
        self.assertEqual(rs.node_value(rs_a), dev_run[:12])
        self.assertEqual(rs.node_value(rs_aext), dev_run[12:])
        self.assertEqual(py_a.lock_ref, py_aext.lock_ref, 1)
        self.assertEqual(rs.node_lock_ref(rs_a), 1)
        self.assertEqual(rs.node_lock_ref(rs_aext), 1)
        self.assertEqual(self._sizes(), (0, 16, 16, 24))

        # I) release the permanent chain lock; demote again; the values
        #    that come back are the loaded ones.
        self.assertEqual(rs.dec_lock_ref(rs_aext), 16)
        self.assertEqual(py.dec_lock_ref(py_aext).delta, 16)
        rs_num, rs_frees = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 16)
        self.assertEqual(rs_frees, sink.take("device_frees"))
        self.assertEqual(rs_frees, [dev_run[12:], dev_run[:12]])
        self.assertEqual(self._sizes(), (0, 0, 0, 24))

        # J) load_back skips: over-quota (16 > 10 + 0) and under the
        #    10-token threshold (B chain is 8).
        self.assertIsNone(py.load_back(py_aext, 10))
        self.assertIsNone(rs.init_load_back(rs_aext, 10))
        self.assertIsNone(py.load_back(py_b))
        self.assertIsNone(rs.init_load_back(rs_b, None))
        self.assertEqual(self._sizes(), (0, 0, 0, 24))

        # K) re-insert after demotion: re-attach is NOT counted and
        #    triggers no new backup.
        rs_plen, rs_last, rs_need, rs_splits = self._rs_insert(a + ext, va + vext)
        py_res = self._py_insert(a + ext, va + vext)
        self.assertEqual(rs_plen, py_res.prefix_len)
        self.assertEqual(rs_plen, 0, "re-attach must not count")
        self.assertEqual(rs_need, [])
        self.assertEqual(sink.take("host_writes"), [])
        self.assertEqual(self._sizes(), (16, 0, 16, 24))

        idx_rs, _, _, _, _ = self._match_eq(a + ext, 16, 0, aext_key, aext_key)
        self.assertEqual(list(idx_rs), va + vext)

        # L) demote again, then evict_host with B's host protected:
        #    B is skipped, A and AEXT are deleted (LRU: A before AEXT).
        rs_num, rs_frees = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 16)
        self.assertEqual(rs_frees, sink.take("device_frees"))

        py_b.protect_host()
        rs.protect_host(rs_b)
        self.assertIsNone(py.evict_host(drain))
        rs_num, rs_host_frees, rs_deleted = rs.evict_host(drain)
        self.assertEqual(rs_num, 16)
        self.assertEqual(rs_host_frees, sink.take("host_frees"))
        self.assertEqual(rs_host_frees, [host_a, host_aext])
        self.assertEqual([self._rs_key(i) for i in rs_deleted], [tuple(a), aext_key])
        self.assertEqual(self._sizes(), (0, 0, 0, 8))
        self._leaves_eq("host")  # only B remains

        idx_rs, ldev, lhost, _, py = self._match_eq(a + ext, 0, 0, (), ())
        self.assertIs(py.last_device_node, py.root_node)
        self.assertIs(py.last_host_node, py.root_node)
        idx_rs, _, _, _, _ = self._match_eq(b, 0, 8, (), tuple(b))

        py_b.release_host()
        rs.release_host(rs_b)

        # M) insert_host: a fresh host-only node (no device value) under
        #    the root, then a host-only extension of it.
        c = token_run(24, 6)
        h_c = list(range(7_000_000, 7_000_006))
        c2 = c + token_run(25, 3)
        h_c2 = list(range(7_000_010, 7_000_019))

        py_matched = py._insert_helper_host(
            py.root_node,
            RadixKey(token_ids=qarr("q", c)),
            torch.tensor(h_c, dtype=torch.int64),
            [],
        )
        rs_matched = rs.insert_host(0, list(c), list(h_c))
        self.assertEqual(py_matched, rs_matched)
        self.assertEqual(py_matched, 0)

        py_c = self._py_find(tuple(c))
        rs_c = self._rs_find(tuple(c))
        self.assertTrue(py_c.evicted)
        self.assertTrue(rs.node_evicted(rs_c))
        self.assertEqual(list(py_c.host_value.tolist()), h_c)
        self.assertEqual(rs.node_host_value(rs_c), h_c)
        self.assertEqual(py_c.lock_ref, rs.node_lock_ref(rs_c))
        self.assertEqual(py_c.host_ref_counter, rs.node_host_ref(rs_c))
        # priority inherited from the root on both sides
        self.assertEqual(py_c.priority, -sys.maxsize)
        self.assertEqual(rs.node_priority(rs_c), -2**31)
        self._leaves_eq("host")  # {b, c}

        py_matched = py._insert_helper_host(
            py.root_node,
            RadixKey(token_ids=qarr("q", c2)),
            torch.tensor(h_c2, dtype=torch.int64),
            [],
        )
        rs_matched = rs.insert_host(0, list(c2), list(h_c2))
        self.assertEqual(py_matched, rs_matched)
        self.assertEqual(py_matched, 6)

        py_c2 = self._py_find(tuple(c2[6:]))
        rs_c2 = self._rs_find(tuple(c2[6:]))
        self.assertEqual(list(py_c2.host_value.tolist()), h_c2[6:])
        self.assertEqual(rs.node_host_value(rs_c2), h_c2[6:])
        self.assertEqual(py_c2.parent, py_c)
        self._leaves_eq("host")  # {b, c2}

        # N) reset: both trees back to the empty root.
        py.reset()
        rs.reset()
        self.assertEqual(self._sizes(), (0, 0, 0, 0))
        self._leaves_eq("host")
        self._leaves_eq("device")
        idx_rs, _, _, _, _ = self._match_eq(a, 0, 0, (), ())

    def test_evict_host_and_drop_subtree(self):
        py, rs, sink = self.py, self.rs, self.sink
        drain = DRAIN

        p = token_run(41, 4)
        q = token_run(42, 4)
        vp, vq = values_for(p), values_for(q)

        # A) chain P -> Q, both backed up and acked, then demoted.
        rs_plen, rs_p, rs_need, _ = self._rs_insert(p, vp)
        py_res = self._py_insert(p, vp)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_p = self._mirror_backups(rs_need)[0]

        rs_plen, rs_q, rs_need, _ = self._rs_insert(p + q, vp + vq)
        py_res = self._py_insert(p + q, vp + vq)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_q = self._mirror_backups(rs_need)[0]

        py_p = self._py_find(tuple(p))
        py_q = self._py_find(tuple(q))
        self.assertEqual(self._ack(py_p, rs_p), 4)
        self.assertEqual(self._ack(py_q, rs_q), 4)
        rs_num, rs_frees = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 8)
        self.assertEqual(rs_frees, sink.take("device_frees"))
        self.assertEqual(self._sizes(), (0, 0, 0, 8))

        # B) drop_subtree is REFUSED while Q holds a host reference.
        py_q.protect_host()
        rs.protect_host(rs_q)
        self.assertEqual(py._drop_subtree_no_host(py_p), 0)
        self.assertEqual(rs.drop_subtree_no_host(rs_p), (0, [], []))
        self.assertEqual(self._sizes(), (0, 0, 0, 8))

        # C) release the reference: the whole subtree is freed
        #    (preorder: P's device + host, then Q's).
        py_q.release_host()
        rs.release_host(rs_q)
        py_freed = py._drop_subtree_no_host(py_p)
        rs_freed, rs_dev_frees, rs_host_frees = rs.drop_subtree_no_host(rs_p)
        self.assertEqual(py_freed, rs_freed)
        self.assertEqual(py_freed, 8)
        self.assertEqual(rs_dev_frees, [vp, vq])
        self.assertEqual(rs_host_frees, [host_p, host_q])
        self.assertEqual(rs_dev_frees, sink.take("device_frees"))
        self.assertEqual(rs_host_frees, sink.take("host_frees"))
        self.assertEqual(self._sizes(), (0, 0, 0, 0))
        self.assertEqual(rs.node_children(0), [])
        self.assertEqual(len(py.root_node.children), 0)

        # D) evict_host skip-by-reference, then delete after release.
        r = token_run(43, 6)
        s = token_run(44, 4)
        vr, vs = values_for(r), values_for(s)

        rs_plen, rs_r, rs_need, _ = self._rs_insert(r, vr)
        py_res = self._py_insert(r, vr)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_r = self._mirror_backups(rs_need)[0]

        rs_plen, rs_s, rs_need, _ = self._rs_insert(r + s, vr + vs)
        py_res = self._py_insert(r + s, vr + vs)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_s = self._mirror_backups(rs_need)[0]

        py_r = self._py_find(tuple(r))
        py_s = self._py_find(tuple(s))
        self._ack(py_r, rs_r)
        self._ack(py_s, rs_s)
        rs_num, rs_frees = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 10)

        py_s.protect_host()
        rs.protect_host(rs_s)
        self.assertIsNone(py.evict_host(drain))
        rs_num, rs_host_frees, rs_deleted = rs.evict_host(drain)
        self.assertEqual(rs_num, 6, "protected S must be skipped")
        self.assertEqual(rs_host_frees, sink.take("host_frees"))
        self.assertEqual(rs_host_frees, [host_r])
        self.assertEqual([self._rs_key(i) for i in rs_deleted], [tuple(r)])

        py_s.release_host()
        rs.release_host(rs_s)
        self.assertIsNone(py.evict_host(drain))
        rs_num, rs_host_frees, rs_deleted = rs.evict_host(drain)
        self.assertEqual(rs_num, 4)
        self.assertEqual(rs_host_frees, sink.take("host_frees"))
        self.assertEqual(rs_host_frees, [host_s])
        self.assertEqual([self._rs_key(i) for i in rs_deleted], [tuple(r + s)])
        self.assertEqual(self._sizes(), (0, 0, 0, 0))

        # E) release_host underflow: Python RuntimeError <-> Rust panic
        #    (surfaced as a Python exception).
        d = token_run(45, 5)
        vd = values_for(d)
        rs_plen, rs_d, rs_need, _ = self._rs_insert(d, vd)
        py_res = self._py_insert(d, vd)
        self.assertEqual(rs_plen, py_res.prefix_len)
        self._mirror_backups(rs_need)
        py_d = self._py_find(tuple(d))
        self._ack(py_d, rs_d)
        rs_num, _ = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 5)

        py_d.protect_host()
        rs.protect_host(rs_d)
        py_d.release_host()
        rs.release_host(rs_d)
        with self.assertRaises(RuntimeError):
            py_d.release_host()
        with self.assertRaises(Exception):
            rs.release_host(rs_d)

    def test_write_back_primitives(self):
        """The write_back facade primitives, on a priority-strategy
        pair (distinct node priorities keep the heap order
        deterministic on both sides)."""
        py, rs, sink = self.make_pair(eviction_policy="priority")
        drain = DRAIN

        a = token_run(51, 4)
        b = token_run(52, 4)
        va, vb = values_for(a), values_for(b)

        # A) two leaves with distinct priorities, backed up and acked.
        rs_plen, rs_a, rs_need, _ = self._rs_insert(a, va, priority=5)
        py_res = self._py_insert(a, va, priority=5)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_a = self._mirror_backups(rs_need)[0]

        rs_plen, rs_b, rs_need, _ = self._rs_insert(b, vb, priority=1)
        py_res = self._py_insert(b, vb, priority=1)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_b = self._mirror_backups(rs_need)[0]

        py_a = self._py_find(tuple(a))
        py_b = self._py_find(tuple(b))
        self._ack(py_a, rs_a)
        self._ack(py_b, rs_b)

        # B) eviction order: priority 1 (B) before priority 5 (A).
        py_heap = [
            (py.eviction_strategy.get_priority(n), n) for n in py.evictable_leaves
        ]
        heapq.heapify(py_heap)
        _, py_first = heapq.heappop(py_heap)
        _, py_second = heapq.heappop(py_heap)
        rs_order = rs.evictable_leaves_ordered()
        self.assertEqual(len(rs_order), 2)
        self.assertEqual(self._rs_key(rs_order[0]), tuple(py_first.key.token_ids))
        self.assertEqual(self._rs_key(rs_order[1]), tuple(py_second.key.token_ids))
        self.assertEqual(self._rs_key(rs_order[0]), tuple(b))

        # C) detach_backuped: demote A to host-only, device run handed
        #    to the caller (staged DMA), A leaves the device leaf set.
        py_num = py._detach_backuped(py_a)
        rs_run = rs.detach_backuped(rs_a)
        self.assertEqual(py_num, 4)
        self.assertEqual(rs_run, va)
        self.assertTrue(py_a.evicted)
        self.assertTrue(rs.node_evicted(rs_a))
        self.assertEqual(list(py_a.host_value.tolist()), host_a)
        self.assertEqual(rs.node_host_value(rs_a), host_a)
        self.assertEqual(self._sizes(), (4, 0, 4, 8))
        self._leaves_eq("device")  # {b}
        self._leaves_eq("host")  # {a, b}

        # D) promote_parent: A's parent is the root -> nothing.
        py_heap = []
        py._promote_parent(py_a, py_heap)
        self.assertEqual(py_heap, [])
        self.assertIsNone(rs.promote_parent(rs_a))

        # E) chain C -> D (priority 3); detach D promotes C to the
        #    device leaf set on both sides.
        c = token_run(53, 4)
        d = token_run(54, 4)
        vc, vd = values_for(c), values_for(d)

        rs_plen, rs_c, rs_need, _ = self._rs_insert(c, vc, priority=3)
        py_res = self._py_insert(c, vc, priority=3)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_c = self._mirror_backups(rs_need)[0]

        rs_plen, rs_d, rs_need, _ = self._rs_insert(c + d, vc + vd, priority=3)
        py_res = self._py_insert(c + d, vc + vd, priority=3)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_d = self._mirror_backups(rs_need)[0]

        py_c = self._py_find(tuple(c))
        py_d = self._py_find(tuple(c + d))
        self._ack(py_c, rs_c)
        self._ack(py_d, rs_d)

        py_num = py._detach_backuped(py_d)
        rs_run = rs.detach_backuped(rs_d)
        self.assertEqual(py_num, 4)
        self.assertEqual(rs_run, vd)

        py_heap = []
        py._promote_parent(py_d, py_heap)
        self.assertEqual(len(py_heap), 1)
        self.assertIs(py_heap[0][1], py_c)
        self.assertEqual(rs.promote_parent(rs_d), rs_c)
        self.assertEqual(self._sizes(), (8, 0, 8, 16))

    def test_insert_host_split(self):
        py, rs, sink = self.py, self.rs, self.sink
        drain = DRAIN

        p = token_run(61, 6)
        q = token_run(62, 4)
        vp, vq = values_for(p), values_for(q)

        # A) P -> Q device chain, backed up, acked, demoted.
        rs_plen, rs_p, rs_need, _ = self._rs_insert(p, vp)
        py_res = self._py_insert(p, vp)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_p = self._mirror_backups(rs_need)[0]

        rs_plen, rs_q, rs_need, _ = self._rs_insert(p + q, vp + vq)
        py_res = self._py_insert(p + q, vp + vq)
        self.assertEqual(rs_plen, py_res.prefix_len)
        host_q = self._mirror_backups(rs_need)[0]

        py_p = self._py_find(tuple(p))
        py_q = self._py_find(tuple(q))
        self._ack(py_p, rs_p)
        self._ack(py_q, rs_q)
        rs_num, _ = rs.evict(drain)
        py_num = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(rs_num, py_num)
        self.assertEqual(rs_num, 10)

        # B) insert_host full-matching the existing P: advances through
        #    it, creates nothing.
        h2_p = list(range(8_000_000, 8_000_006))
        py_matched = py._insert_helper_host(
            py.root_node,
            RadixKey(token_ids=qarr("q", p)),
            torch.tensor(h2_p, dtype=torch.int64),
            [],
        )
        rs_matched = rs.insert_host(0, list(p), list(h2_p))
        self.assertEqual(py_matched, rs_matched)
        self.assertEqual(py_matched, 6)
        self.assertEqual(len(py.root_node.children), 1)
        self.assertEqual(len(rs.node_children(0)), 1)

        # C) insert_host ending inside Q: Q splits into a 2-token front
        #    and a 2-token tail, host value sliced with the split.
        key8 = p + q[:2]
        h2_8 = list(range(8_000_010, 8_000_018))
        py_matched = py._insert_helper_host(
            py.root_node,
            RadixKey(token_ids=qarr("q", key8)),
            torch.tensor(h2_8, dtype=torch.int64),
            [],
        )
        rs_matched = rs.insert_host(0, list(key8), list(h2_8))
        self.assertEqual(py_matched, rs_matched)
        self.assertEqual(py_matched, 8)

        py_front = self._py_find(tuple(q[:2]))
        py_tail = self._py_find(tuple(q[2:]))
        rs_front = self._rs_find(tuple(q[:2]))
        rs_tail = self._rs_find(tuple(q[2:]))
        self.assertEqual(list(py_front.host_value.tolist()), host_q[:2])
        self.assertEqual(list(py_tail.host_value.tolist()), host_q[2:])
        self.assertEqual(rs.node_host_value(rs_front), host_q[:2])
        self.assertEqual(rs.node_host_value(rs_tail), host_q[2:])
        # split inheritance: lock_ref / priority / hit_count / pending
        self.assertEqual(py_front.lock_ref, rs.node_lock_ref(rs_front))
        self.assertEqual(py_front.priority, rs.node_priority(rs_front))
        self.assertEqual(py_front.hit_count, rs.node_hit_count(rs_front))
        self.assertEqual(py_front.hit_count, 1)
        self.assertIsNone(py_front.write_through_pending_id)
        self.assertFalse(rs.node_backup_pending(rs_front))
        self.assertIs(py_front.parent, py_p)
        self.assertIs(py_tail.parent, py_front)

        # D) insert_host ending past the split: matches P + front, then
        #    appends a fresh host-only node under the front.
        z3 = token_run(63, 3)
        key9 = p + q[:2] + z3
        h2_9 = list(range(8_000_020, 8_000_029))
        py_matched = py._insert_helper_host(
            py.root_node,
            RadixKey(token_ids=qarr("q", key9)),
            torch.tensor(h2_9, dtype=torch.int64),
            [],
        )
        rs_matched = rs.insert_host(0, list(key9), list(h2_9))
        self.assertEqual(py_matched, rs_matched)
        self.assertEqual(py_matched, 8)

        py_z = self._py_find(tuple(z3))
        rs_z = self._rs_find(tuple(z3))
        self.assertTrue(py_z.evicted)
        self.assertTrue(rs.node_evicted(rs_z))
        self.assertEqual(list(py_z.host_value.tolist()), h2_9[8:])
        self.assertEqual(rs.node_host_value(rs_z), h2_9[8:])
        self.assertIs(py_z.parent, py_front)
        self.assertEqual(self._sizes(), (0, 0, 0, 11))
        # host leaves: the tail (no children) and z; the front and P
        #    have backuped children.
        self._leaves_eq("host")
        self.assertEqual(
            sorted(self._rs_key(i) for i in rs.evictable_host_leaves()),
            sorted([tuple(q[2:]), tuple(z3)]),
        )


if __name__ == "__main__":
    unittest.main()
