"""Differential parity test: Rust UnifiedRadixTree (sglang-radix) vs the
Python UnifiedTreeCore (plan.md M2/1e).

Drives both cores through the ``UnifiedTreeCoreInterface`` with the same op
sequence (page_size=1, non-eagle, FULL component only, LRU) and asserts the
observable state agrees after every op: match device indices + host-hit
lengths + boundary root paths + emitted actions, insert results (incl.
split-barrier actions, value-None and explicit-value, chunked, priority,
and extra_key namespaces), lock-ref deltas, full-drain eviction free
multisets, the write-back backup/demote round trip, insert_host host
paths, drive_host_eviction (incl. the write_back duplicate-host reclaim),
the structural snapshot (DFS key paths, device/host value lengths, FULL
lock refs, sizes, and the KV-canary walk), and the tree invariants via
sanity_check.

The cores are constructed with real ``CacheInitParams`` but a fake
cache/allocator: with FULL-only the tree core and the FullComponent only
touch the allocator through controller-driven actions, which the driver
records instead of applying. Node identity across implementations is by
root path (a token sequence maps to a unique path in both trees); node
ids are not comparable.

Op orderings that depend on LRU tie-breaks are handled by construction:
the Rust core stamps every node of one op with a single tick (Python
stamps each visited node with a fresh counter), so where the cores see a
tie their victims may differ, and the test therefore only runs full
drains, compares freed values as multisets, and keeps host-pool index
assignments out of the compared state.
"""

import unittest
from array import array as qarr

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    ComponentAction,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    ReplaceWriteThroughOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

FULL = ComponentType.FULL
DRAIN = 10**9


def token_run(seed: int, n: int) -> list:
    """Deterministic token-id run (mirrors the other parity tests)."""
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(s % 100_000)
    return out


class FakeAllocator:
    """Stands in for the device KV allocator; the tree only frees through
    it and the driver records those frees instead of applying them."""

    device = torch.device("cpu")

    def __init__(self):
        self.frees = []

    def free(self, index, *a, **k):
        self.frees.append(tuple(int(x) for x in index.tolist()))

    free_full = free_segment = free


class FakeCache:
    """The only cache surface the FULL component / tree core touch."""

    enable_session_radix_cache = False
    is_swa_enabled = False

    def __init__(self):
        self.token_to_kv_pool_allocator = FakeAllocator()


def make_params() -> CacheInitParams:
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=FakeAllocator(),
        page_size=1,
        is_eagle=False,
        eviction_policy="lru",
        enable_kv_cache_events=False,
    )


def build_python():
    params = make_params()
    full = FullComponent(cache=FakeCache(), params=params)
    return UnifiedTreeCore(params, {FULL: full})


def build_rust():
    from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
        create_tree_core,
    )

    params = make_params()
    full = FullComponent(cache=FakeCache(), params=params)
    return create_tree_core("rust", params, {FULL: full})


# ==== normalization (node identity = root path) ====


def path_key(core, node_id):
    """Tokens of the root->node path, or None for a missing node."""
    if node_id is None:
        return None
    node = core.node_by_id(node_id)
    toks = ()
    while node is not None and node.key is not None and len(node.key):
        toks = tuple(int(x) for x in node.key.raw_token_ids()) + toks
        node = node.parent
    return toks


def _chunks(tensors):
    return tuple(tuple(int(x) for x in t.tolist()) for t in tensors)


def norm_actions(core, actions):
    out = []
    for a in actions:
        if isinstance(a, BackupKV):
            out.append(("BackupKV", tuple(path_key(core, n) for n in a.node_ids)))
        elif isinstance(a, FreeDeviceKV):
            out.append(("FreeDeviceKV", _chunks(a.indices)))
        elif isinstance(a, FreeDeviceKVFullOnly):
            out.append(("FreeDeviceKVFullOnly", _chunks(a.indices)))
        elif isinstance(a, ReplaceWriteThroughOnNodeSplit):
            out.append(
                (
                    "RWT",
                    a.ack_id,
                    path_key(core, a.old_node_id),
                    path_key(core, a.new_node_id),
                    path_key(core, a.new_child_node_id),
                )
            )
        elif isinstance(a, ComponentAction):
            out.append((type(a).__name__, int(a.component_type)))
        else:
            raise AssertionError(f"unhandled cache action {type(a).__name__}")
    return tuple(out)


def norm_match(core, m):
    return (
        tuple(int(x) for x in m.device_indices.tolist()),
        m.host_hit_length,
        m.swa_host_hit_length,
        m.mamba_host_hit_length,
        m.mamba_branching_seqlen,
        m.full_kv_hit_length,
        path_key(core, m.last_device_node),
        path_key(core, m.last_host_node),
        path_key(core, m.best_match_node),
        norm_actions(core, m.cache_actions),
    )


def norm_insert(core, r):
    return (
        r.prefix_len,
        r.total_len,
        path_key(core, r.last_device_node),
        r.mamba_exist,
        path_key(core, r.inserted_host_node),
        r.host_insert_dropped,
        norm_actions(core, r.cache_actions),
    )


def norm_lock(r):
    skip = tuple(
        sorted((int(ct), tuple(sorted(ids))) for ct, ids in r.skip_lock_node_ids.items())
    )
    return (r.delta, r.swa_uuid_for_lock, r.swa_uuid_for_host_lock, skip)


def rust_path(t, nid):
    toks = ()
    n = nid
    while True:
        toks = tuple(int(x) for x in t.node_key(n)) + toks
        p = t.node_parent(n)
        if p is None:
            break
        n = p
    return toks


def structure(core):
    """(extra_key, root path, device len, host len, FULL lock, FULL host lock)."""
    rows = []
    if hasattr(core, "_tree"):
        t = core._tree
        for row in t.dump_nodes():
            nid = row[0]
            ns = t.node_ns(nid)
            ek = core._ns_pairs[ns][0] if ns else None
            rows.append(
                (
                    ek,
                    rust_path(t, nid),
                    0 if row[6] is None else len(row[6]),
                    0 if row[7] is None else len(row[7]),
                    row[12][0],
                    row[13][0],
                )
            )
    else:
        for node in _py_walk(core.root_node):
            cd = node.component_data[FULL]
            rows.append(
                (
                    node.key.extra_key,
                    path_key(core, node.id),
                    0 if cd.value is None else len(cd.value),
                    0 if cd.host_value is None else len(cd.host_value),
                    cd.lock_ref,
                    cd.host_lock_ref,
                )
            )
    return sorted(rows)


def _py_walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


# ==== the driver: one shared op script, run against each core ====


class Driver:
    def __init__(self, core):
        self.core = core
        self.dev_frees = []
        self.host_frees = []
        self.host_writes = []
        self.host_next = 1_000_000
        self.obs = []
        self.locks = {}
        self.last_match_id = None

    # -- plumbing ------------------------------------------------------

    def _drain_frees(self, res):
        # BaseEvictionResult.__del__ asserts the freed lists were consumed.
        for ct, items in list(res.device_frees.items()):
            for t in items:
                self.dev_frees.extend(int(x) for x in t.tolist())
            del res.device_frees[ct]
        for ct, items in list(res.host_frees.items()):
            for t in items:
                self.host_frees.extend(int(x) for x in t.tolist())
            del res.host_frees[ct]

    @staticmethod
    def _bump(tracker, delta):
        for ct, n in delta.items():
            tracker[ct] = tracker.get(ct, 0) + n

    def _take_frees(self):
        dev, self.dev_frees = self.dev_frees, []
        host, self.host_frees = self.host_frees, []
        return tuple(sorted(dev)), tuple(sorted(host))

    # -- ops -----------------------------------------------------------

    def ins(self, tokens, tag, value=None, extra_key=None, chunked=False, priority=0):
        core = self.core
        key = RadixKey(qarr("q", list(tokens)), extra_key=extra_key)
        step = core.begin_insert(
            InsertParams(
                key=key, value=value, chunked=chunked, priority=priority
            )
        )
        actions = []
        try:
            while True:
                actions.extend(norm_actions(core, step.actions))
                if step.result is not None:
                    res = step.result
                    break
                step = core.resume_insert()
        finally:
            actions.extend(norm_actions(core, core.end_insert()))
        self.obs.append((tag, norm_insert(core, res), tuple(actions)))
        return res

    def match(self, tokens, tag, extra_key=None):
        m = self.core.match_prefix(
            MatchPrefixParams(
                key=RadixKey(qarr("q", list(tokens)), extra_key=extra_key)
            )
        )
        self.last_match_id = m.best_match_node
        self.obs.append((tag, norm_match(self.core, m)))
        return m

    def lock(self, label, tokens):
        m = self.match(tokens, "m-" + label)
        res = self.core.inc_lock_ref(m.best_match_node)
        self.locks[label] = (m.best_match_node, res.to_dec_params())
        self.obs.append(("lock", label, norm_lock(res)))

    def unlock(self, label):
        node_id, params = self.locks.pop(label)
        res = self.core.dec_lock_ref(node_id, params)
        self.obs.append(("unlock", label, res.delta))

    def drain(self, label):
        """The facade's full device-eviction loop for FULL (DRAIN quota)."""
        core = self.core
        tracker = {FULL: 0}
        core.evict_device_start(FULL, DRAIN)
        try:
            while True:
                r = core.evict_device_next_node(FULL, tracker)
                self._drain_frees(r)
                self._bump(tracker, r.tracker)
                if r.node_id is None:
                    if r.made_progress:
                        continue
                    break
                res = core.evict_device_leaf(r.node_id, core.is_write_back)
                self._drain_frees(res)
                self._bump(tracker, res.tracker)
                if res.backup_kv is not None:
                    written = self._execute_backup(res.backup_kv)
                    if written > 0:
                        dem = core.demote(r.node_id)
                        self._drain_frees(dem)
                        self._bump(tracker, dem.tracker)
                    else:
                        drop = core.drop_subtree_no_host(r.node_id)
                        self._drain_frees(drop)
                        self._bump(tracker, drop.tracker)
                        if not drop.is_dropped:
                            # Locked victim stays device-resident; the facade
                            # would retry forever, so the test stops too.
                            break
        finally:
            core.evict_device_end(FULL)
        dev, host = self._take_frees()
        self.obs.append(("evict", label, tracker[FULL], dev, host))

    def _execute_backup(self, action):
        """The facade's _execute_and_commit_kv_backup without its controller."""
        written = 0
        for node_id in action.node_ids:
            dev, comp_xfers = self.core.build_backup_spec(node_id)
            if dev.numel() == 0 and not comp_xfers:
                continue
            host_idx = torch.arange(
                self.host_next, self.host_next + dev.numel(), dtype=torch.int64
            )
            self.host_next += dev.numel()
            self.host_writes.append(tuple(int(x) for x in host_idx.tolist()))
            self.core.commit_backup(node_id, host_idx, comp_xfers)
            written = len(host_idx)
        return written

    def host_drain(self, label):
        res = self.core.drive_host_eviction(FULL, DRAIN)
        self._drain_frees(res)
        dev, host = self._take_frees()
        self.obs.append(
            (
                "host-evict",
                label,
                tuple(sorted((int(ct), n) for ct, n in res.tracker.items())),
                dev,
                host,
            )
        )

    def insert_host(self, label, tokens):
        core = self.core
        key = RadixKey(qarr("q", list(tokens)))
        hv = torch.arange(500_000, 500_000 + len(tokens), dtype=torch.int64)
        hashes = [f"ph{i:02d}" for i in range(len(tokens))]
        res = core.insert_host(core.root_node_handle(), key, hv, hashes)
        self.obs.append(("host-insert", label, norm_insert(core, res)))
        return res

    def probe(self, label, collect=False):
        core = self.core
        nid = self.last_match_id
        vals = ()
        if collect:
            vals = tuple(
                int(x)
                for x in core.collect_full_device_indices(
                    nid, core.root_node_handle()
                ).tolist()
            )
        self.obs.append(
            (
                "probe",
                label,
                (
                    path_key(core, nid),
                    core.is_backuped(nid),
                    core.is_full_device_evicted(nid),
                    vals,
                ),
            )
        )

    def snap(self, label):
        core = self.core
        w = core.walk_for_kv_canary(False, False)
        core.sanity_check([], [])
        self.obs.append(
            (
                "snap",
                label,
                (
                    core.total_size(),
                    core.evictable_size(),
                    core.protected_size(),
                    core.full_evictable_size(),
                    core.full_protected_size(),
                    tuple(int(x) for x in core.all_values_flatten().tolist()),
                    tuple(int(x) for x in w.slot_indices.tolist()),
                    tuple(int(x) for x in w.positions.tolist()),
                    tuple(int(x) for x in w.prev_slot_indices.tolist()),
                    structure(core),
                ),
            )
        )

    def enable_hicache(self):
        if hasattr(self.core, "set_hicache_enabled"):
            self.core.set_hicache_enabled()
        else:
            self.core.enable_hicache = True
        self.core.is_write_back = True

    def reset(self, label):
        self.core.reset()
        self.locks.clear()
        self.last_match_id = None
        self.obs.append(("reset", label))
        self.snap(label + "-snap")


def main_script(d: Driver):
    d.ins(token_run(1, 6), "ins-A")
    d.match(token_run(1, 6), "m-A")
    d.ins(token_run(2, 7), "ins-B")
    d.ins(token_run(3, 4), "ins-C")
    b = token_run(2, 7)
    d.match(b, "m-B")
    # Split B's node mid-way with a new branch.
    d.ins(b[:3] + [900, 901], "ins-D")
    d.match(b[:3] + [900, 901], "m-D")
    d.match(b, "m-B2")
    # Extend the new leaf.
    d.ins(b[:3] + [900, 901, 902], "ins-E")
    d.match(b[:3] + [900, 901, 902], "m-E")
    d.match(token_run(1, 6), "m-A2")
    d.match([], "m-empty")
    # value=None insert: device values come from the key tokens.
    d.ins(token_run(4, 3), "ins-V")
    d.match(token_run(4, 3), "m-V")
    # Explicit tensor value: device values differ from the key tokens.
    d.ins(token_run(10, 4), "ins-XV", value=torch.tensor([7, 8, 9, 10]))
    d.match(token_run(10, 4), "m-XV")
    # Namespaces (extra_key) are isolated trees under one root.
    d.ins(token_run(5, 5), "ins-NX", extra_key="x")
    d.match(token_run(5, 5), "m-NX", extra_key="x")
    d.match(token_run(5, 5), "m-default-x")
    d.ins(token_run(6, 4), "ins-NS", extra_key="s")
    d.match(token_run(6, 4), "m-NS", extra_key="s")
    d.snap("s1")
    # Lock A's path, drain everything else, verify the lock shielded it.
    d.lock("L1", token_run(1, 6))
    d.match(token_run(1, 6), "m-locked")
    d.snap("s2")
    d.drain("ev-locked")
    d.match(token_run(1, 6), "m-A-ev")
    d.match(b, "m-B-ev")
    d.match(token_run(5, 5), "m-NX-ev", extra_key="x")
    d.snap("s3")
    d.unlock("L1")
    d.drain("ev-drain")
    d.match(token_run(1, 6), "m-A-gone")
    d.snap("s4")
    # Re-insert un-evicts the tombstoned paths with fresh device values.
    d.ins(token_run(1, 6), "ins-A2")
    d.ins(b, "ins-B2")
    d.match(token_run(1, 6), "m-A-restore")
    d.match(b, "m-B-restore")
    d.probe("probe-restore", collect=True)
    d.snap("s5")
    # chunked / priority inserts.
    d.ins(token_run(7, 5), "ins-CH", chunked=True)
    d.match(token_run(7, 5), "m-CH")
    d.ins(token_run(8, 5), "ins-PR", priority=7)
    d.match(token_run(8, 5), "m-PR")
    d.snap("s6")
    d.reset("reset")
    d.ins(token_run(9, 3), "ins-after-reset")
    d.match(token_run(9, 3), "m-after-reset")
    d.snap("s-reset")


def writeback_script(d: Driver):
    d.enable_hicache()
    a = token_run(11, 6)
    b = token_run(12, 4)
    d.ins(a, "wb-ins-A")
    d.ins(b, "wb-ins-B")
    d.match(a, "wb-m-A")
    d.snap("wb-s0")
    # Write-back eviction: every leaf backs up to host, then demotes.
    d.drain("wb-evict")
    d.match(a, "wb-m-host")
    d.match(b, "wb-m-host-b")
    d.probe("wb-probe-host")
    d.snap("wb-s1")
    # Re-insert: the host-backed paths un-evict onto the fresh device KV.
    d.ins(a, "wb-ins-A2")
    d.ins(b, "wb-ins-B2")
    d.match(a, "wb-m-device")
    d.match(b, "wb-m-device-b")
    d.probe("wb-probe-device", collect=True)
    d.snap("wb-s2")
    # A host-only path via insert_host (the prefetch commit shape).
    d.insert_host("wb-hi", token_run(13, 5))
    d.match(token_run(13, 5), "wb-m-hi")
    d.snap("wb-s3")
    # Host pressure: duplicate reclaim first, then the host-LRU leaves.
    d.host_drain("wb-host-evict")
    d.match(a, "wb-m-after")
    d.match(b, "wb-m-after-b")
    d.match(token_run(13, 5), "wb-m-hi-gone")
    d.snap("wb-s4")


def compare(py_obs, rust_obs, case):
    case.assertEqual(len(py_obs), len(rust_obs), "op count mismatch")
    for i, (p, r) in enumerate(zip(py_obs, rust_obs)):
        case.assertEqual(
            p,
            r,
            f"op #{i} ({p[0]!r}) diverged:\n  python = {p!r}\n  rust   = {r!r}",
        )


class UnifiedRadixParityTest(CustomTestCase):
    def _run_pair(self, script, build_py, build_rust):
        py = Driver(build_py())
        script(py)
        rust = Driver(build_rust())
        script(rust)
        compare(py.obs, rust.obs, self)

    def test_parity_main(self):
        self._run_pair(main_script, build_python, build_rust)

    def test_parity_writeback(self):
        self._run_pair(writeback_script, build_python, build_rust)

    def test_registry_factory(self):
        from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
            create_tree_core,
            registered_tree_core_backends,
        )
        from sglang.srt.mem_cache.unified_cache.tree_core_rust import (
            UnifiedTreeCoreRust,
        )

        self.assertIn("rust", registered_tree_core_backends())
        core = create_tree_core(
            "rust",
            make_params(),
            {FULL: FullComponent(cache=FakeCache(), params=make_params())},
        )
        self.assertIsInstance(core, UnifiedTreeCoreRust)
        self.assertEqual(core.total_size(), (0, 0))
        m = core.match_prefix(MatchPrefixParams(key=RadixKey(qarr("q", [1, 2]))))
        self.assertEqual(m.full_kv_hit_length, 0)


if __name__ == "__main__":
    unittest.main()
