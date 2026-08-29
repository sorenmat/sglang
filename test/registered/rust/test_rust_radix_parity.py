"""Differential parity test: Rust RadixTree vs Python RadixCache (plan.md §5.3).

Drives the ``_scheduler`` PyO3 extension's ``RadixTree`` and the unmodified
``sglang.srt.mem_cache.radix_cache.RadixCache`` with the same op sequence
(page_size=1, LRU, non-eagle) and asserts the observable state agrees after
every op: sizes, match lengths + KV values, insert prefix lengths, and evict
counts.

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

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RUN_LEN = 128
N_RUNS = 4


def token_run(seed: int, n: int, salt: int) -> list:
    """Deterministic token-id run (mirrors the benches + driver_test)."""
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(s % 100_000 + salt)
    return out


def values_for(ids: list) -> list:
    return [i + 100_000 for i in ids]


class TestRustRadixParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mod = load_rust_extension("sglang.srt.rust_extensions._scheduler")

    def setUp(self):
        super().setUp()
        # Fresh pair per test: the op sequences are stateful.
        self.py_tree = RadixCache.create_simulated(page_size=1)
        self.rs_tree = self.mod.RadixTree(1, False, "lru")

    def _rs_match(self, ids):
        return self.rs_tree.match_prefix(list(ids))

    def _py_match(self, ids):
        return self.py_tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=qarr("q", ids)))
        )

    def _rs_insert(self, ids, values):
        return self.rs_tree.insert(list(ids), list(values), 0, False)

    def _py_insert(self, ids, values):
        return self.py_tree.insert(
            InsertParams(
                key=RadixKey(token_ids=qarr("q", ids)),
                value=torch.tensor(values, dtype=torch.int64),
            )
        )

    def test_parity_op_sequence(self):
        py, rs = self.py_tree, self.rs_tree
        drain = 10**9

        # A) four disjoint runs: insert / match / lock / full drain. The
        # lock+drain runs BEFORE the overlapping insert below, so the drain
        # arithmetic is exactly "all but the locked run".
        runs = [token_run(100 + j, RUN_LEN, j * 1000) for j in range(N_RUNS)]
        for j, ids in enumerate(runs):
            rs_plen, _ = self._rs_insert(ids, values_for(ids))
            py_res = self._py_insert(ids, values_for(ids))
            self.assertEqual(rs_plen, py_res.prefix_len, f"insert run {j}")
            self.assertEqual(py.total_size(), rs.total_size(), f"total after run {j}")
        self.assertEqual(py.total_size(), N_RUNS * RUN_LEN)

        # full-leaf match: lengths and KV values agree
        idx_rs, _ = self._rs_match(runs[0])
        py_res = self._py_match(runs[0])
        self.assertEqual(len(idx_rs), len(py_res.device_indices))
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())

        # match ending mid-segment (both trees split the node at 64)
        key3 = runs[0][:64] + [999_999]
        idx_rs, _ = self._rs_match(key3)
        py_res = self._py_match(key3)
        self.assertEqual(len(idx_rs), 64)
        self.assertEqual(len(py_res.device_indices), 64)
        self.assertEqual(list(idx_rs), py_res.device_indices.tolist())

        # lock the leaf of the last run: protected/evictable agree
        _, rs_leaf = self._rs_match(runs[3])
        py_res = self._py_match(runs[3])
        self.assertEqual(rs.inc_lock_ref(rs_leaf), -RUN_LEN)
        self.assertEqual(py.inc_lock_ref(py_res.last_device_node).delta, -RUN_LEN)
        self.assertEqual(py.protected_size(), rs.protected_size())
        self.assertEqual(py.evictable_size(), rs.evictable_size())
        self.assertEqual(rs.evictable_size(), (N_RUNS - 1) * RUN_LEN)

        # full drain: both evict exactly the unlocked portion
        _, n_rs = rs.evict(drain)
        n_py = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(n_rs, n_py)
        self.assertEqual(n_rs, (N_RUNS - 1) * RUN_LEN)
        self.assertEqual(py.total_size(), rs.total_size())
        self.assertEqual(py.evictable_size(), 0)
        self.assertEqual(rs.evictable_size(), 0)

        # unlock, drain the rest
        self.assertEqual(rs.dec_lock_ref(rs_leaf), RUN_LEN)
        py.dec_lock_ref(py_res.last_device_node)
        _, n_rs = rs.evict(drain)
        n_py = py.evict(EvictParams(num_tokens=drain)).num_tokens_evicted
        self.assertEqual(n_rs, n_py)
        self.assertEqual(py.total_size(), 0)
        self.assertEqual(rs.total_size(), 0)

        # B) single-leaf partial evict: both overfulfill to the leaf boundary
        ids = token_run(900, 256, 50_000)
        self._rs_insert(ids, values_for(ids))
        self._py_insert(ids, values_for(ids))
        _, n_rs = rs.evict(100)
        n_py = py.evict(EvictParams(num_tokens=100)).num_tokens_evicted
        self.assertGreaterEqual(n_rs, 100)
        self.assertEqual(n_rs, n_py)
        rs.evict(drain)
        py.evict(EvictParams(num_tokens=drain))
        self.assertEqual(py.total_size(), 0)
        self.assertEqual(rs.total_size(), 0)

        # C) overlapping insert: shared 64-token prefix (sentinel tail can't
        # collide with LCG tokens, which stay below 100_032)
        prefix = token_run(800, 64, 80_000)
        self._rs_insert(prefix, values_for(prefix))
        self._py_insert(prefix, values_for(prefix))
        key = prefix + [999_998, 999_997]
        rs_plen, _ = self._rs_insert(key, values_for(key))
        py_res = self._py_insert(key, values_for(key))
        self.assertEqual(rs_plen, 64)
        self.assertEqual(py_res.prefix_len, 64)
        self.assertEqual(py.total_size(), rs.total_size())
        rs.evict(drain)
        py.evict(EvictParams(num_tokens=drain))
        self.assertEqual(py.total_size(), 0)
        self.assertEqual(rs.total_size(), 0)

        # D) lock walk over a 32-token single-node chain (growing prefix:
        # each insert extends the chain by one token)
        chain = list(range(700_000, 700_032))
        leaf = 0
        for i in range(32):
            key = chain[: i + 1]
            leaf = self._rs_insert(key, values_for(key))[1]
        py_leaf = py.root_node
        for i in range(32):
            key = chain[: i + 1]
            py_leaf = self._py_insert(key, values_for(key)).last_device_node
        self.assertEqual(rs.inc_lock_ref(leaf), -32)
        self.assertEqual(py.inc_lock_ref(py_leaf).delta, -32)
        self.assertEqual(py.protected_size(), rs.protected_size())
        self.assertEqual(rs.dec_lock_ref(leaf), 32)
        self.assertEqual(py.dec_lock_ref(py_leaf).delta, 32)
        self.assertEqual(py.protected_size(), 0)
        self.assertEqual(rs.protected_size(), 0)

    def test_match_prefix_meta_matches_match_prefix(self):
        """The shadow fast path (match_prefix_meta) must agree with the full
        match on length + node, since the dual-write facade keys off it."""
        py, rs = self.py_tree, self.rs_tree
        ids = token_run(777, 512, 70_000)
        self._rs_insert(ids, values_for(ids))
        self._py_insert(ids, values_for(ids))
        for probe in (ids, ids[:100], ids[:100] + [42, 42], []):
            idx_rs, node_rs = self._rs_match(list(probe))
            py_res = self._py_match(list(probe))
            meta_len, meta_node = rs.match_prefix_meta(list(probe))
            self.assertEqual(meta_len, len(idx_rs))
            self.assertEqual(meta_node, node_rs)
            self.assertEqual(meta_len, len(py_res.device_indices))


if __name__ == "__main__":
    unittest.main()
