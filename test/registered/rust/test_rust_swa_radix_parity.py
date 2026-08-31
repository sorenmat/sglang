"""Differential parity test: Rust SWARadixTree vs Python SWARadixCache (plan.md M2/1b).

Drives the ``_scheduler`` PyO3 extension's ``SWARadixTree`` and the
unmodified ``sglang.srt.mem_cache.swa_radix_cache.SWARadixCache`` with the
same op sequence (page_size=1, non-eagle) and asserts the observable state
agrees after every op: sizes, match lengths + KV values, insert prefix
lengths, evict counts, allocator free runs, and lock/uuid outcomes.

The Python cache runs against a recording fake
``SWATokenToKVPoolAllocator`` (its real ``__init__`` needs a
``BaseSWAKVPool``; the parity test never allocates — it drives tree ops and
diffs the allocator calls against the runs the Rust tree returns instead).

Op orderings that depend on LRU tie-breaks are avoided on purpose:
partial evictions run on single-leaf trees, and multi-leaf trees are only
fully drained, so no assertion hinges on which of several equal-clock leaves
is chosen first.

The extension is loaded exactly like the server does:
``load_rust_extension`` (bundled wheel build, or built from source in
CI/dev).
"""

import unittest
from array import array as qarr

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RUN_LEN = 128
N_RUNS = 4
WINDOW = 64


def token_run(seed: int, n: int, salt: int) -> list:
    """Deterministic token-id run (mirrors the base radix parity test)."""
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(s % 100_000 + salt)
    return out


def values_for(ids: list) -> list:
    return [i + 100_000 for i in ids]


class RecordingSWAAllocator(SWATokenToKVPoolAllocator):
    """Records the allocator calls the Python SWA tree makes.

    Skips the real ``__init__`` (which needs a ``BaseSWAKVPool``): the
    parity test never allocates, it only drives tree ops and diffs the
    freed/re-mapped runs against the Rust tree's returned runs.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self._free_kv: list = []
        self._free_full: list = []
        self._free_swa: list = []
        self._set_mappings: list = []
        self._cleared: list = []

    def free(self, kv_indices, *args, **kwargs):
        self._free_kv.append(list(kv_indices.tolist()))

    def free_full(self, kv_indices, *args, **kwargs):
        self._free_full.append(list(kv_indices.tolist()))

    def free_swa(self, swa_indices, *args, **kwargs):
        self._free_swa.append(list(swa_indices.tolist()))

    def translate_loc_from_full_to_swa(self, full_indices):
        # Marker transform: the test inverts it to compare against the
        # Rust `recover` runs, which carry the raw incoming full values.
        return torch.tensor(
            [v + 500_000 for v in full_indices.tolist()], dtype=torch.int64
        )

    def set_full_to_swa_mapping(self, full_indices, swa_indices):
        self._set_mappings.append(
            (list(full_indices.tolist()), list(swa_indices.tolist()))
        )

    def clear_full_to_swa_mapping(self, full_indices, *args, **kwargs):
        self._cleared.append(list(full_indices.tolist()))

    def take(self):
        """Snapshot + clear the recorded calls (in call order)."""
        snap = (
            self._free_kv,
            self._free_full,
            self._free_swa,
            self._set_mappings,
            self._cleared,
        )
        self._free_kv, self._free_full, self._free_swa = [], [], []
        self._set_mappings, self._cleared = [], []
        return snap


def make_py_swa(window: int) -> tuple:
    alloc = RecordingSWAAllocator()
    cache = SWARadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=alloc,
            page_size=1,
            sliding_window_size=window,
        )
    )
    return cache, alloc


class TestRustSWARadixParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mod = load_rust_extension("sglang.srt.rust_extensions._scheduler")

    def setUp(self):
        super().setUp()
        # Fresh pair per test: the op sequences are stateful.
        self.py_cache, self.alloc = make_py_swa(WINDOW)
        self.rs_tree = self.mod.SWARadixTree(1, False, WINDOW)

    def _rs_match(self, ids):
        return self.rs_tree.match_prefix(list(ids))

    def _py_match(self, ids):
        return self.py_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=qarr("q", ids)))
        )

    def _rs_insert(self, ids, values, prev_prefix_len=0, swa_evicted_seqlen=0):
        return self.rs_tree.insert(
            list(ids), list(values), prev_prefix_len, swa_evicted_seqlen
        )

    def _py_insert(self, ids, values, prev_prefix_len=0, swa_evicted_seqlen=0):
        return self.py_cache.insert(
            InsertParams(
                key=RadixKey(token_ids=qarr("q", ids)),
                value=torch.tensor(values, dtype=torch.int64),
                prev_prefix_len=prev_prefix_len,
                swa_evicted_seqlen=swa_evicted_seqlen,
            )
        )

    def _sizes(self):
        py = (
            self.py_cache.full_evictable_size(),
            self.py_cache.swa_evictable_size(),
            self.py_cache.full_protected_size(),
            self.py_cache.swa_protected_size(),
            self.py_cache.total_size(),
        )
        rs = (
            self.rs_tree.full_evictable_size(),
            self.rs_tree.swa_evictable_size(),
            self.rs_tree.full_protected_size(),
            self.rs_tree.swa_protected_size(),
            self.rs_tree.total_size(),
        )
        self.assertEqual(py, rs, "size bookkeeping diverged")
        return rs

    def _frees(self, rs_kv, rs_full, rs_swa):
        """Assert the recorded Python allocator calls match the Rust runs."""
        py_kv, py_full, py_swa, _, _ = self.alloc.take()
        self.assertEqual(rs_kv, py_kv, "free(kv) runs diverged")
        self.assertEqual(rs_full, py_full, "free_full runs diverged")
        self.assertEqual(rs_swa, py_swa, "free_swa runs diverged")

    def test_parity_op_sequence(self):
        py, rs = self.py_cache, self.rs_tree
        drain = 10**9

        # A) four disjoint runs: insert / match / lock / full drain. The
        # lock+drain runs BEFORE the overlapping insert below, so the drain
        # arithmetic is exactly "all but the locked run".
        runs = [token_run(100 + j, RUN_LEN, j * 1000) for j in range(N_RUNS)]
        for j, ids in enumerate(runs):
            rs_plen, _, kv, full, swa, rec = self._rs_insert(ids, values_for(ids))
            py_res = self._py_insert(ids, values_for(ids))
            self.assertEqual(rs_plen, py_res.prefix_len, f"insert run {j}")
            self.assertEqual(kv + full + swa, [], "no frees on disjoint inserts")
            self.assertEqual(rec, [])
            self._sizes()
        self.assertEqual(self._sizes()[4], (N_RUNS * RUN_LEN, N_RUNS * RUN_LEN))

        # full-leaf match: lengths and KV values agree
        idx_rs, _ = self._rs_match(runs[0])
        py_res = self._py_match(runs[0])
        self.assertEqual(len(idx_rs), RUN_LEN)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())

        # match ending mid-segment (both trees split the node at 64)
        key3 = runs[0][:64] + [999_999]
        idx_rs, _ = self._rs_match(key3)
        py_res = self._py_match(key3)
        self.assertEqual(len(idx_rs), 64)
        self.assertEqual(len(py_res.device_indices), 64)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())

        # lock the leaf of the last run: the SWA lock covers the whole
        # node (128 > window 64) and gets a uuid on both sides.
        _, rs_leaf = self._rs_match(runs[3])
        py_res = self._py_match(runs[3])
        rs_uuid, rs_delta = rs.inc_lock_ref(rs_leaf)
        py_res_lock = py.inc_lock_ref(py_res.last_device_node)
        py_uuid = py_res_lock.swa_uuid_for_lock
        self.assertIsNone(py_res_lock.delta)  # SWA inc_lock_ref: delta=None
        self.assertEqual(rs_delta, -RUN_LEN)
        self.assertIsNotNone(rs_uuid)
        self.assertIsNotNone(py_uuid)
        self._sizes()
        self.assertEqual(rs.full_protected_size(), RUN_LEN)
        self.assertEqual(rs.swa_protected_size(), RUN_LEN)

        # full drain: both evict exactly the unlocked portion; the locked
        # leaf's SWA lock blocks the SWA-side scan.
        _, _, kv, full, swa = rs.evict(drain, drain)
        py_r = py.evict(EvictParams(num_tokens=drain, swa_num_tokens=drain))
        self._frees(kv, full, swa)
        self.assertEqual(py_r.num_tokens_evicted, (N_RUNS - 1) * RUN_LEN)
        self.assertEqual(py_r.swa_num_tokens_evicted, (N_RUNS - 1) * RUN_LEN)
        self._sizes()
        self.assertEqual(self._sizes()[4], (RUN_LEN, RUN_LEN))

        # unlock (with the uuid), drain the rest
        rs.dec_lock_ref(rs_leaf, rs_uuid, False)
        py.dec_lock_ref(
            py_res.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=py_uuid),
        )
        _, _, kv, full, swa = rs.evict(drain, drain)
        py_r = py.evict(EvictParams(num_tokens=drain, swa_num_tokens=drain))
        self._frees(kv, full, swa)
        self.assertEqual(py_r.num_tokens_evicted, RUN_LEN)
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

        # B) single-leaf partial evict: both overfulfill to the leaf boundary
        ids = token_run(900, 256, 50_000)
        self._rs_insert(ids, values_for(ids))
        self._py_insert(ids, values_for(ids))
        _, _, kv, full, swa = rs.evict(100, 100)
        py_r = py.evict(EvictParams(num_tokens=100, swa_num_tokens=100))
        self._frees(kv, full, swa)
        self.assertEqual(kv, [values_for(ids)])
        self.assertEqual(py_r.num_tokens_evicted, 256)
        self.assertEqual(py_r.swa_num_tokens_evicted, 256)
        # drain is a no-op now; assert no stray allocator calls
        _, _, kv, full, swa = rs.evict(drain, drain)
        py.evict(EvictParams(num_tokens=drain, swa_num_tokens=drain))
        self._frees([], [], [])
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

        # C) overlapping insert: shared 64-token prefix — the tree already
        # holds this KV, so the incoming overlap runs are freed on both
        # sides.
        prefix = token_run(800, 64, 80_000)
        self._rs_insert(prefix, values_for(prefix))
        self._py_insert(prefix, values_for(prefix))
        key = prefix + [999_998, 999_997]
        rs_plen, _, kv, full, swa, rec = self._rs_insert(key, values_for(key))
        py_res = self._py_insert(key, values_for(key))
        self.assertEqual(rs_plen, 64)
        self.assertEqual(py_res.prefix_len, 64)
        self.assertEqual(kv, [values_for(prefix)])
        self.assertEqual(rec, [])
        self._frees(kv, full, swa)
        self._sizes()
        rs.evict(drain, drain)
        py.evict(EvictParams(num_tokens=drain, swa_num_tokens=drain))
        self._frees([], [], [])
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

        # D) lock walk over a 32-token single-node chain (growing prefix:
        # each insert extends the chain by one token). 32 < window 64, so
        # no uuid is issued and the whole chain stays SWA-protected.
        chain = list(range(700_000, 700_032))
        leaf = 0
        for i in range(32):
            k = chain[: i + 1]
            leaf = self._rs_insert(k, values_for(k))[1]
        py_leaf = py.root_node
        for i in range(32):
            k = chain[: i + 1]
            self._py_insert(k, values_for(k))
            py_leaf = self._py_match(k).last_device_node
        rs_uuid, rs_delta = rs.inc_lock_ref(leaf)
        py_res_lock = py.inc_lock_ref(py_leaf)
        self.assertIsNone(rs_uuid)
        self.assertIsNone(py_res_lock.swa_uuid_for_lock)
        self.assertIsNone(py_res_lock.delta)
        self.assertEqual(rs_delta, -32)
        self._sizes()
        self.assertEqual(rs.swa_protected_size(), 32)
        self.assertEqual(rs.dec_lock_ref(leaf, rs_uuid, False), 32)
        py.dec_lock_ref(py_leaf)
        self._sizes()
        self.assertEqual(self._sizes()[3], 0)

    def test_swa_evict_tombstones_internal(self):
        py, rs = self.py_cache, self.rs_tree
        # root -> A(100) -> B(100); lock B so the SWA evict can only reach
        # A (internal): free SWA, tombstone.
        a = list(range(0, 100))
        ab = list(range(0, 200))
        self._rs_insert(a, values_for(a))
        self._py_insert(a, values_for(a))
        rs_leaf = self._rs_insert(ab, values_for(ab))[1]
        self._py_insert(ab, values_for(ab))
        py_leaf = self._py_match(ab).last_device_node

        # Lock the leaf B (window 64 <= 100 -> uuid at B).
        rs.inc_lock_ref(rs_leaf)
        py.inc_lock_ref(py_leaf)
        self._sizes()

        # SWA-evict 10: A is the LRU unlocked node, internal -> free_swa
        # only, tombstone (overfills to A's 100). B is skipped (swa lock).
        _, _, kv, full, swa = rs.evict(0, 10)
        py_r = py.evict(EvictParams(num_tokens=0, swa_num_tokens=10))
        self._frees([], [], swa)
        self.assertEqual(swa, [values_for(a)])
        self.assertEqual(py_r.swa_num_tokens_evicted, 100)
        self.assertEqual(py_r.num_tokens_evicted, 0)
        # Both full sides are locked (the lock walks to the root), so
        # nothing is full-evictable; the SWA side is fully accounted
        # (A tombstoned, B locked).
        self._sizes()
        self.assertEqual(
            (rs.full_evictable_size(), rs.swa_evictable_size()),
            (0, 0),
        )

        # Match of the full 200: A is a tombstone, but the distance since
        # it at B is 100 >= window 64, so both return all 200 indices.
        idx_rs, _ = self._rs_match(ab)
        py_res = self._py_match(ab)
        self.assertEqual(len(idx_rs), 200)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())
        self.assertEqual(idx_rs[:100], values_for(a))
        self.assertEqual(idx_rs[100:], values_for(ab)[100:])

    def test_dec_swa_lock_only_early_release(self):
        py, rs = self.py_cache, self.rs_tree
        # 10-token leaf, window 64 > 10: the whole leaf gets the SWA lock,
        # uuid at the leaf. Early-release the SWA side: the leaf is
        # tombstoned, its SWA slots freed, the full lock kept.
        ids = list(range(400_000, 400_010))
        rs_leaf = self._rs_insert(ids, values_for(ids))[1]
        self._py_insert(ids, values_for(ids))
        py_leaf = self._py_match(ids).last_device_node
        rs_uuid, _ = rs.inc_lock_ref(rs_leaf)
        py_res_lock = py.inc_lock_ref(py_leaf)
        py_uuid = py_res_lock.swa_uuid_for_lock
        # 10-token chain < window 64: no uuid boundary is reached on either
        # side (dec treats None as unlock-to-root, exclusive).
        self.assertIsNone(rs_uuid)
        self.assertIsNone(py_uuid)
        self._sizes()
        self.assertEqual(rs.swa_protected_size(), 10)

        free_swa_rs = rs.dec_swa_lock_only(rs_leaf, rs_uuid)
        py.dec_swa_lock_only(py_leaf, py_uuid)
        self.assertEqual(free_swa_rs, [values_for(ids)])
        self._frees([], [], [values_for(ids)])
        self._sizes()
        self.assertEqual(rs.swa_protected_size(), 0)
        self.assertEqual(rs.full_protected_size(), 10)

        # The tombstoned leaf's full side is released with skip_swa=True.
        rs.dec_lock_ref(rs_leaf, None, True)
        py.dec_lock_ref(py_leaf, skip_swa=True)
        self._sizes()
        self.assertEqual(self._sizes()[1], 0)
        self.assertEqual(self._sizes()[0], 10)

        # The tombstone blocks further matches of the full 10 tokens:
        # distance since the last tombstone is 0 < window 64.
        idx_rs, _ = self._rs_match(ids)
        py_res = self._py_match(ids)
        self.assertEqual(len(idx_rs), 0)
        self.assertEqual(len(py_res.device_indices), 0)

        # Drain the full side; the tombstone leaf is deleted with it.
        _, _, kv, full, swa = rs.evict(10**9, 0)
        py.evict(EvictParams(num_tokens=10**9, swa_num_tokens=0))
        self._frees([], full, [])
        self.assertEqual(full, [values_for(ids)])
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

    def test_insert_with_swa_evicted_seqlen(self):
        py, rs = self.py_cache, self.rs_tree
        # 20-token insert where SWA has evicted [5, 12): [0, 5) goes into
        # a tombstoned node, [12, 20) into a live leaf.
        ids = list(range(500_000, 500_020))
        rs_plen, _, kv, full, swa, rec = self._rs_insert(
            ids, values_for(ids), 0, 12
        )
        py_res = self._py_insert(ids, values_for(ids), 0, 12)
        self.assertEqual(rs_plen, py_res.prefix_len)
        self.assertEqual(kv + full + swa, [])
        self.assertEqual(rec, [])
        self._sizes()
        self.assertEqual((rs.full_evictable_size(), rs.swa_evictable_size()), (20, 8))
        self.assertEqual(py.total_size(), (20, 8))
        self.assertEqual(self._sizes()[4], (20, 8))

        # A match of the full 20 tokens reuses nothing at window 64: the
        # live tail [12, 20) is 8 tokens < window, so the distance-since-
        # tombstone rule fails. (At window 4 the tail would validate the
        # whole match and both sides would return 20.)
        idx_rs, _ = self._rs_match(ids)
        py_res = self._py_match(ids)
        self.assertEqual(len(idx_rs), 0)
        self.assertEqual(len(py_res.device_indices), 0)

        # Drain everything.
        _, _, kv, full, swa = rs.evict(10**9, 10**9)
        py.evict(EvictParams(num_tokens=10**9, swa_num_tokens=10**9))
        self._frees(kv, full, swa)
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))

    def test_tombstone_recover_locked_full(self):
        py, rs = self.py_cache, self.rs_tree
        # root -> A(100) -> B(100); lock B; SWA-evict tombstones A while
        # A's full side stays locked. Re-inserting the 200-token prefix
        # must adopt the incoming SWA for A (locked-full recover) and free
        # the incoming overlap for B.
        a = list(range(600_000, 600_100))
        ab = list(range(600_000, 600_200))
        self._rs_insert(a, values_for(a))
        self._py_insert(a, values_for(a))
        rs_leaf = self._rs_insert(ab, values_for(ab))[1]
        self._py_insert(ab, values_for(ab))
        py_leaf = self._py_match(ab).last_device_node

        first_uuid, _ = rs.inc_lock_ref(rs_leaf)
        first_py_uuid = py.inc_lock_ref(py_leaf).swa_uuid_for_lock
        self.assertIsNotNone(first_uuid)
        self.assertIsNotNone(first_py_uuid)
        self._sizes()

        # Tombstone A (internal, unlocked SWA) via a small SWA evict.
        _, _, kv, full, swa = rs.evict(0, 10)
        py.evict(EvictParams(num_tokens=0, swa_num_tokens=10))
        self._frees([], [], swa)
        self.assertEqual(swa, [values_for(a)])
        self._sizes()

        # Fresh incoming KV for the same 200 tokens (different values).
        new_ab = [v + 1_000_000 for v in values_for(ab)]
        rs_plen, _, kv, full, swa, rec = self._rs_insert(ab, new_ab, 0, 0)
        py_res = self._py_insert(ab, new_ab, 0, 0)
        # The full 200 tokens were already in the tree (A adopted, B
        # overwritten), so the prefix is the whole key.
        self.assertEqual(rs_plen, 200)
        self.assertEqual(py_res.prefix_len, 200)
        # The locked-full recover: Rust carries (tree_value, incoming);
        # the Python fake recorded the equivalent mapping calls.
        self.assertEqual(len(rec), 1)
        tree_value, incoming = rec[0]
        self.assertEqual(tree_value, values_for(a))
        self.assertEqual(incoming, new_ab[:100])
        # Python applied: remap A at the marker-transformed incoming SWA,
        # clear the incoming mapping, and free the incoming full slots.
        py_tree_value, py_swa = self.alloc.set_mappings[0]
        self.assertEqual(py_tree_value, tree_value)
        self.assertEqual(py_swa, [v + 500_000 for v in incoming])
        self.assertEqual(self.alloc.cleared, [incoming])
        # B's new incoming overlap was freed as full+swa on both sides,
        # A's incoming full side via free_full.
        self.assertEqual(kv, [new_ab[100:]])
        self.assertEqual(full, [incoming])
        self._frees(kv, full, swa)
        self._sizes()

        # A is live-SWA again: a match of the full 200 tokens returns the
        # ADOPTED tree values for A (the locked full slots stayed) and the
        # tree's OWN values for B (non-tombstone: the incoming overlap was
        # freed, not adopted).
        idx_rs, _ = self._rs_match(ab)
        py_res = self._py_match(ab)
        self.assertEqual(len(idx_rs), 200)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())
        self.assertEqual(idx_rs[:100], values_for(a))
        self.assertEqual(idx_rs[100:], values_for(ab)[100:])

        # Re-lock (the uuid is reused: B kept its marker across the
        # recover), release both locks, drain to zero.
        _, rs_leaf2 = self._rs_match(ab)
        py_res2 = self._py_match(ab)
        rs_uuid2, _ = rs.inc_lock_ref(rs_leaf2)
        py_uuid2 = py.inc_lock_ref(py_res2.last_device_node).swa_uuid_for_lock
        self.assertEqual(rs_uuid2, first_uuid)
        self._sizes()
        rs.dec_lock_ref(rs_leaf, first_uuid, False)
        py.dec_lock_ref(
            py_leaf, DecLockRefParams(swa_uuid_for_lock=first_py_uuid)
        )
        rs.dec_lock_ref(rs_leaf2, rs_uuid2, False)
        py.dec_lock_ref(
            py_res2.last_device_node, DecLockRefParams(swa_uuid_for_lock=py_uuid2)
        )
        self._sizes()
        _, _, kv, full, swa = rs.evict(10**9, 10**9)
        py.evict(EvictParams(num_tokens=10**9, swa_num_tokens=10**9))
        self._frees(kv, full, swa)
        self._sizes()
        self.assertEqual(self._sizes()[4], (0, 0))


if __name__ == "__main__":
    unittest.main()
