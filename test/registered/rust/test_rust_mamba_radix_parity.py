"""Differential parity test: Rust MambaRadixTree vs Python MambaRadixCache
(plan.md M2/1c).

Drives the ``_scheduler`` PyO3 extension's ``MambaRadixTree`` and the
unmodified ``sglang.srt.mem_cache.mamba_radix_cache.MambaRadixCache``
with the same op sequence (page_size=1, non-eagle) and asserts the
observable state agrees after every op: sizes, match lengths + KV
values + mamba branching point, insert prefix lengths / mamba_exist,
evict counts, allocator free runs (with ``start_pos``), and lock
deltas.

The Python cache runs against a recording fake
``TokenToKVPoolAllocator`` (its real ``__init__`` needs a KV pool; the
parity test never allocates — it drives tree ops and diffs the
``free_segment`` / mamba ``free`` calls against the runs the Rust tree
returns instead). Mamba states are freed through
``req_to_token_pool.mamba_allocator`` (the int8 checkpoint pool stays
absent, so the active-pool path is taken), so the fake pool records
those too.

The Python tree's ``mamba_cache_chunk_size`` comes from the server
args; it is patched to a fixed 64 (the FLA default) for the test.

Op orderings that depend on LRU tie-breaks are avoided on purpose:
partial evictions run on single-leaf trees, and multi-leaf trees are
fully drained.
"""

import unittest
from array import array as qarr

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache import mamba_radix_cache as mamba_module
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 64  # mamba_cache_chunk_size (FLA chunk default)
RUN_LEN = 128
N_RUNS = 4


def token_run(seed: int, n: int, salt: int) -> list:
    """Deterministic token-id run (mirrors the SWA parity test)."""
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(s % 100_000 + salt)
    return out


def values_for(ids: list) -> list:
    return [i + 100_000 for i in ids]


class RecordingMambaAllocator:
    """Records the mamba slot frees the Python tree makes."""

    def __init__(self):
        self.freed: list = []

    def free(self, mamba_value, *args, **kwargs):
        self.freed.append(list(mamba_value.tolist()))

    def take(self):
        out, self.freed = self.freed, []
        return out


class FakeMambaReqPool:
    """Stands in for ``req_to_token_pool`` in the mamba free path.

    No ``mamba_ckpt_pool`` attribute, so ``MambaRadixCache.int8_ckpt_pool``
    is None and ``_free_mamba_value`` uses the active allocator.
    """

    def __init__(self):
        self.mamba_allocator = RecordingMambaAllocator()


class RecordingMambaKVAllocator(TokenToKVPoolAllocator):
    """Records the ``free_segment`` calls the Python Mamba tree makes.

    Skips the real ``__init__`` (which needs a KV pool): the parity test
    never allocates, it only drives tree ops and diffs the freed runs
    against the Rust tree's returned runs.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.free_segment_calls: list = []

    def free_segment(self, free_index, *, start_pos):
        self.free_segment_calls.append((list(free_index.tolist()), start_pos))

    def take(self):
        out, self.free_segment_calls = self.free_segment_calls, []
        return out


def make_py_mamba() -> tuple:
    alloc = RecordingMambaKVAllocator()
    pool = FakeMambaReqPool()
    cache = MambaRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=alloc,
            page_size=1,
        )
    )
    return cache, alloc, pool


class TestRustMambaRadixParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mod = load_rust_extension("sglang.srt.rust_extensions._scheduler")
        # The chunk size comes from the (absent) server args in the test
        # environment; pin it to the FLA default.
        cls._orig_chunk_fn = mamba_module.mamba_cache_chunk_size
        mamba_module.mamba_cache_chunk_size = lambda: CHUNK

    @classmethod
    def tearDownClass(cls):
        mamba_module.mamba_cache_chunk_size = cls._orig_chunk_fn
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        # Fresh pair per test: the op sequences are stateful.
        self.py_cache, self.alloc, self.pool = make_py_mamba()
        self.rs_tree = self.mod.MambaRadixTree(1, False, CHUNK)

    def _rs_match(self, ids):
        return self.rs_tree.match_prefix(list(ids))

    def _py_match(self, ids):
        return self.py_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=qarr("q", ids)))
        )

    def _rs_insert(self, ids, values, mamba_slot, prev_prefix_len=0):
        return self.rs_tree.insert(
            list(ids), list(values), [mamba_slot], prev_prefix_len
        )

    def _py_insert(self, ids, values, mamba_slot, prev_prefix_len=0):
        return self.py_cache.insert(
            InsertParams(
                key=RadixKey(token_ids=qarr("q", ids)),
                value=torch.tensor(values, dtype=torch.int64),
                mamba_value=torch.tensor([mamba_slot], dtype=torch.int64),
                prev_prefix_len=prev_prefix_len,
            )
        )

    def _sizes(self):
        py = (
            self.py_cache.full_evictable_size(),
            self.py_cache.mamba_evictable_size(),
            self.py_cache.full_protected_size(),
            self.py_cache.mamba_protected_size(),
            self.py_cache.total_size(),
        )
        rs = (
            self.rs_tree.full_evictable_size(),
            self.rs_tree.mamba_evictable_size(),
            self.rs_tree.full_protected_size(),
            self.rs_tree.mamba_protected_size(),
            self.rs_tree.total_size(),
        )
        self.assertEqual(py, rs, "size bookkeeping diverged")
        return rs

    def _frees(self, rs_kv, rs_kv_start_pos, rs_mamba):
        """Assert the recorded Python allocator calls match the Rust runs."""
        py_calls = self.alloc.take()  # list of (values, start_pos)
        py_mamba = self.pool.mamba_allocator.take()
        py_kv = [c[0] for c in py_calls]
        py_pos = [c[1] for c in py_calls]
        self.assertEqual(rs_kv, py_kv, "free_segment runs diverged")
        self.assertEqual(rs_kv_start_pos, py_pos, "free_segment start_pos diverged")
        self.assertEqual(rs_mamba, py_mamba, "mamba free runs diverged")

    def _py_lock_delta(self, node):
        """Python inc_lock_ref returns delta=None; diff the sizes instead."""
        before = (
            self.py_cache.full_protected_size(),
            self.py_cache.mamba_protected_size(),
        )
        res = self.py_cache.inc_lock_ref(node)
        after = (
            self.py_cache.full_protected_size(),
            self.py_cache.mamba_protected_size(),
        )
        return res, (after[0] - before[0], after[1] - before[1])

    def test_parity_op_sequence(self):
        py, rs = self.py_cache, self.rs_tree
        drain = 10**9

        # A) four disjoint runs: insert / match / lock / full drain.
        runs = [token_run(100 + j, RUN_LEN, j * 1000) for j in range(N_RUNS)]
        for j, ids in enumerate(runs):
            rs_plen, _, kv, _, mamba = self._rs_insert(ids, values_for(ids), j + 1)
            py_res = self._py_insert(ids, values_for(ids), j + 1)
            self.assertEqual(rs_plen, py_res.prefix_len, f"insert run {j}")
            self.assertEqual(py_res.mamba_exist, False)
            self.assertEqual(kv, [])
            self.assertEqual(mamba, [])
            self._sizes()
        self.assertEqual(self._sizes()[4], (N_RUNS * RUN_LEN, N_RUNS))

        # full-leaf match: lengths and KV values agree; no branching
        # point (the match ends exactly on a mamba state).
        idx_rs, last_rs, branch_rs = self._rs_match(runs[0])
        py_res = self._py_match(runs[0])
        self.assertEqual(len(idx_rs), RUN_LEN)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())
        self.assertIsNone(branch_rs)
        self.assertIsNone(py_res.mamba_branching_seqlen)

        # match ending mid-segment (both trees split the node at 64; the
        # split front is a mamba tombstone, so the prefix falls back to
        # the root and the branching point is the chunk-aligned 64).
        key3 = runs[0][:64] + [999_999]
        idx_rs, last_rs, branch_rs = self._rs_match(key3)
        py_res = self._py_match(key3)
        self.assertEqual(len(idx_rs), 0)
        self.assertEqual(len(py_res.device_indices), 0)
        self.assertEqual(branch_rs, 64)
        self.assertEqual(py_res.mamba_branching_seqlen, 64)
        self.assertIs(py_res.last_device_node, py.root_node)
        self.assertEqual(last_rs, 0)

        # lock the leaf of the last run: full walk to the root + the
        # leaf's mamba state.
        _, rs_leaf = self._rs_match(runs[3])
        py_res = self._py_match(runs[3])
        py_leaf = py_res.last_device_node
        rs_fd, rs_md = rs.inc_lock_ref(rs_leaf)
        py_res_lock, py_delta = self._py_lock_delta(py_leaf)
        self.assertIsNone(py_res_lock.delta)  # Mamba inc_lock_ref: delta=None
        self.assertEqual((rs_fd, rs_md), (-RUN_LEN, -1))
        # Rust reports units moved evictable -> protected (negative);
        # the Python protected-size diff is the same amount, positive.
        self.assertEqual(py_delta, (-rs_fd, -rs_md))
        self._sizes()
        self.assertEqual(rs.full_protected_size(), RUN_LEN)
        self.assertEqual(rs.mamba_protected_size(), 1)

        # full drain: the locked leaf survives; the other three leaves
        # (and their mamba states) are evicted.
        _, _, kv, kv_pos, mamba = rs.evict(drain, 0)
        py_r = py.evict(EvictParams(num_tokens=drain, mamba_num=0))
        self._frees(kv, kv_pos, mamba)
        self.assertEqual(py_r.num_tokens_evicted, (N_RUNS - 1) * RUN_LEN)
        self.assertEqual(py_r.mamba_num_evicted, 0)
        self._sizes()
        self.assertEqual(self._sizes()[4], (RUN_LEN, 1))

        # unlock, drain the rest.
        rs.dec_lock_ref(rs_leaf)
        py.dec_lock_ref(py_leaf)
        _, _, kv, kv_pos, mamba = rs.evict(drain, drain)
        py_r = py.evict(EvictParams(num_tokens=drain, mamba_num=drain))
        self._frees(kv, kv_pos, mamba)
        self.assertEqual(py_r.num_tokens_evicted, RUN_LEN)
        # The full-phase leaf delete already freed the mamba state; the
        # mamba phase sees an empty list and counts nothing.
        self.assertEqual(py_r.mamba_num_evicted, 0)
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

        # B) single-leaf partial evict: both overfulfill to the leaf
        # boundary; the leaf's mamba state is freed too but does NOT
        # count toward the mamba budget.
        ids = token_run(900, 256, 50_000)
        self._rs_insert(ids, values_for(ids), 41)
        self._py_insert(ids, values_for(ids), 41)
        _, _, kv, kv_pos, mamba = rs.evict(100, 0)
        py_r = py.evict(EvictParams(num_tokens=100, mamba_num=0))
        self._frees(kv, kv_pos, mamba)
        self.assertEqual(kv, [values_for(ids)])
        self.assertEqual(mamba, [[41]])
        self.assertEqual(py_r.num_tokens_evicted, 256)
        self.assertEqual(py_r.mamba_num_evicted, 0)
        # drain is a no-op now; assert no stray allocator calls
        _, _, kv, kv_pos, mamba = rs.evict(drain, drain)
        py.evict(EvictParams(num_tokens=drain, mamba_num=drain))
        self._frees([], [], [])
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

        # C) overlapping insert: the 64-token key is fully in the tree —
        # the incoming overlap runs are freed on both sides, and the
        # incoming mamba value is a duplicate (mamba_exist on both).
        prefix = token_run(800, 64, 80_000)
        self._rs_insert(prefix, values_for(prefix), 61)
        self._py_insert(prefix, values_for(prefix), 61)
        new_prefix = [v + 1_000_000 for v in values_for(prefix)]
        rs_plen, _, kv, kv_pos, mamba = self._rs_insert(prefix, new_prefix, 62)
        py_res = self._py_insert(prefix, new_prefix, 62)
        self.assertEqual(rs_plen, 64)
        self.assertEqual(py_res.prefix_len, 64)
        self.assertTrue(py_res.mamba_exist)
        self.assertEqual(kv, [new_prefix])
        self.assertEqual(kv_pos, [0])
        self.assertEqual(mamba, [])
        self._frees(kv, kv_pos, mamba)
        self._sizes()
        rs.evict(drain, drain)
        py.evict(EvictParams(num_tokens=drain, mamba_num=drain))
        self._frees([], [], [])
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

        # D) prev_prefix_len partial free: the first 4 tokens of the
        # incoming value are already locked, so only [4, 10) is freed,
        # positioned at start_pos 4.
        ids10 = list(range(700_000, 700_010))
        self._rs_insert(ids10, values_for(ids10), 71)
        self._py_insert(ids10, values_for(ids10), 71)
        rs_plen, _, kv, kv_pos, mamba = self._rs_insert(
            ids10, values_for(ids10), 72, prev_prefix_len=4
        )
        py_res = self._py_insert(ids10, values_for(ids10), 72, prev_prefix_len=4)
        self.assertEqual(rs_plen, 10)
        self.assertEqual(py_res.prefix_len, 10)
        self.assertEqual(kv, [values_for(ids10)[4:]])
        self.assertEqual(kv_pos, [4])
        self.assertTrue(py_res.mamba_exist)
        self._frees(kv, kv_pos, mamba)
        self._sizes()

    def test_mamba_evict_internal_tombstone_and_branching(self):
        py, rs = self.py_cache, self.rs_tree
        # root -> A(100) -> B(100); mamba-evict tombstones A's state.
        a = list(range(0, 100))
        ab = list(range(0, 200))
        self._rs_insert(a, values_for(a), 1)
        self._py_insert(a, values_for(a), 1)
        self._rs_insert(ab, values_for(ab), 2)
        self._py_insert(ab, values_for(ab), 2)
        self._sizes()

        _, _, kv, kv_pos, mamba = rs.evict(0, 1)
        py_r = py.evict(EvictParams(num_tokens=0, mamba_num=1))
        self._frees(kv, kv_pos, mamba)
        self.assertEqual(kv, [])
        self.assertEqual(mamba, [[1]])
        self.assertEqual(py_r.mamba_num_evicted, 1)
        self.assertEqual(py_r.num_tokens_evicted, 0)
        self._sizes()
        self.assertEqual((rs.full_evictable_size(), rs.mamba_evictable_size()), (200, 1))

        # Full match of the 200 tokens: the path's last node (B) holds a
        # live state, so the whole full KV is reusable; no branching.
        idx_rs, last_rs, branch_rs = self._rs_match(ab)
        py_res = self._py_match(ab)
        self.assertEqual(len(idx_rs), 200)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())
        self.assertIsNone(branch_rs)
        self.assertIsNone(py_res.mamba_branching_seqlen)

        # 150-token match: ends inside B; the split front is a mamba
        # tombstone and A's state is gone, so the match falls back to
        # the root and reports the chunk-aligned branching point of the
        # whole 150-token run (150 // 64 * 64 = 128).
        idx_rs, last_rs, branch_rs = self._rs_match(ab[:150])
        py_res = self._py_match(ab[:150])
        self.assertEqual(len(idx_rs), 0)
        self.assertEqual(len(py_res.device_indices), 0)
        self.assertEqual(branch_rs, 128)
        self.assertEqual(py_res.mamba_branching_seqlen, 128)
        self.assertIs(py_res.last_device_node, py.root_node)
        self.assertEqual(last_rs, 0)

        # Drain everything (both phases). The full phase evicts B (leaf)
        # and cascades the tombstoned A; B's mamba state goes out with
        # the leaf delete, so the mamba phase counts nothing.
        _, _, kv, kv_pos, mamba = rs.evict(10**9, 10**9)
        py_r = py.evict(EvictParams(num_tokens=10**9, mamba_num=10**9))
        self._frees(kv, kv_pos, mamba)
        self.assertEqual(py_r.num_tokens_evicted, 200)
        self.assertEqual(py_r.mamba_num_evicted, 0)
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

    def test_insert_revives_mamba_tombstone(self):
        py, rs = self.py_cache, self.rs_tree
        a = list(range(300_000, 300_100))
        ab = list(range(300_000, 300_200))
        self._rs_insert(a, values_for(a), 1)
        self._py_insert(a, values_for(a), 1)
        self._rs_insert(ab, values_for(ab), 2)
        self._py_insert(ab, values_for(ab), 2)
        rs.evict(0, 1)  # tombstone A's state
        py.evict(EvictParams(num_tokens=0, mamba_num=1))
        self._sizes()

        # Re-insert A with a fresh state: the tombstone is revived (the
        # state attaches), the incoming full KV overlap is freed.
        new_a = [v + 1_000_000 for v in values_for(a)]
        rs_plen, _, kv, kv_pos, mamba = self._rs_insert(a, new_a, 9)
        py_res = self._py_insert(a, new_a, 9)
        self.assertEqual(rs_plen, 100)
        self.assertEqual(py_res.prefix_len, 100)
        self.assertFalse(py_res.mamba_exist)
        self.assertEqual(kv, [new_a])
        self.assertEqual(mamba, [])
        self._frees(kv, kv_pos, mamba)
        self._sizes()
        self.assertEqual((rs.full_evictable_size(), rs.mamba_evictable_size()), (200, 2))

        # The tree keeps ITS own A value.
        idx_rs, last_rs, branch_rs = self._rs_match(a)
        py_res = self._py_match(a)
        self.assertEqual(len(idx_rs), 100)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())
        self.assertEqual(idx_rs, values_for(a))
        self.assertIsNone(branch_rs)

    def test_split_lock_model(self):
        py, rs = self.py_cache, self.rs_tree
        # One 20-token node, locked; matching 10 tokens splits it.
        ids = list(range(500_000, 500_020))
        rs_leaf = self._rs_insert(ids, values_for(ids), 5)[1]
        py_leaf = self._py_insert(ids, values_for(ids), 5).last_device_node
        rs_fd, rs_md = rs.inc_lock_ref(rs_leaf)
        py_res_lock, py_delta = self._py_lock_delta(py_leaf)
        self.assertEqual((rs_fd, rs_md), (-20, -1))
        # The Python protected-size diff is the sign flip of the Rust
        # evictable->protected delta.
        self.assertEqual(py_delta, (-rs_fd, -rs_md))
        self._sizes()

        # The split: the front keeps the full lock, loses the mamba lock
        # and state; the tail (the original leaf object) keeps both.
        idx_rs, last_rs, branch_rs = self._rs_match(ids[:10])
        py_res = self._py_match(ids[:10])
        self.assertEqual(len(idx_rs), 0)
        self.assertEqual(len(py_res.device_indices), 0)
        self.assertIs(py_res.last_device_node, py.root_node)
        self.assertEqual(last_rs, 0)

        py_front = list(py.root_node.children.values())[0]
        py_tail = list(py_front.children.values())[0]
        self.assertIs(py_tail, py_leaf)  # the original node became the tail
        rs_front = rs.node_children(0)[0]
        rs_tail = rs.node_children(rs_front)[0]
        self.assertEqual(rs_front, 2)
        self.assertEqual(rs_tail, rs_leaf)
        self.assertTrue(rs.node_mamba_tombstone(rs_front))
        self.assertTrue(py_front.mamba_value is None)
        self.assertEqual(rs.node_full_lock_ref(rs_front), 1)
        self.assertEqual(py_front.full_lock_ref, 1)
        self.assertEqual(rs.node_mamba_lock_ref(rs_front), 0)
        self.assertEqual(py_front.mamba_lock_ref, 0)
        self.assertFalse(rs.node_mamba_tombstone(rs_tail))
        self.assertEqual(rs.node_mamba_value(rs_tail), [5])
        self.assertEqual(rs.node_mamba_lock_ref(rs_tail), 1)
        self.assertEqual(py_tail.mamba_lock_ref, 1)
        self._sizes()
        self.assertEqual((rs.full_protected_size(), rs.mamba_protected_size()), (20, 1))

        # Release from the original leaf id (now the tail).
        rs.dec_lock_ref(rs_leaf)
        py.dec_lock_ref(py_leaf)
        self._sizes()
        self.assertEqual((rs.full_protected_size(), rs.mamba_protected_size()), (0, 0))

    def test_mamba_evict_respects_locks(self):
        py, rs = self.py_cache, self.rs_tree
        a = list(range(0, 10))
        b = list(range(100, 110))
        rs_la = self._rs_insert(a, values_for(a), 1)[1]
        py_la = self._py_insert(a, values_for(a), 1).last_device_node
        rs_lb = self._rs_insert(b, values_for(b), 2)[1]
        py_lb = self._py_insert(b, values_for(b), 2).last_device_node
        rs.inc_lock_ref(rs_la)  # lock A (the LRU leaf)
        py.inc_lock_ref(py_la)
        self._sizes()

        # Mamba evict of 10: A is mamba-locked -> skipped; B is free ->
        # one state (plus its full KV via the leaf path).
        _, _, kv, kv_pos, mamba = rs.evict(0, 10)
        py_r = py.evict(EvictParams(num_tokens=0, mamba_num=10))
        self._frees(kv, kv_pos, mamba)
        self.assertEqual(mamba, [[2]])
        self.assertEqual(kv, [values_for(b)])
        self.assertEqual(py_r.mamba_num_evicted, 1)
        self._sizes()

        # Release A, drain the rest.
        rs.dec_lock_ref(rs_la)
        py.dec_lock_ref(py_la)
        _, _, kv, kv_pos, mamba = rs.evict(0, 10)
        py_r = py.evict(EvictParams(num_tokens=0, mamba_num=10))
        self._frees(kv, kv_pos, mamba)
        self.assertEqual(mamba, [[1]])
        self.assertEqual(kv, [values_for(a)])
        self.assertEqual(py_r.mamba_num_evicted, 1)
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))


if __name__ == "__main__":
    unittest.main()
