"""Rust-backed TreeCore: the unified multi-pool radix tree running on the
``sglang_radix::unified::UnifiedRadixTree`` engine, exposed through the
``_scheduler`` PyO3 module and wrapped behind the
`UnifiedTreeCoreInterface`.

Selected via ``SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust``. The tree owns the
structure, per-node values (as pool-index lists), the per-component LRUs, the
size/leaf bookkeeping, and the tree-level FULL/SWA/Mamba hooks; the emitted
cache actions are converted back to the Python `CacheAction`/`ComponentAction`
objects the controller already applies against the real allocators.

Component values cross the boundary as plain index lists (the engine is
torch-free); this wrapper materializes them as int64 tensors on the tree
device. NodeIds are the engine's u32 arena ids.
"""

from __future__ import annotations

import array
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.rust_extensions import load_rust_extension
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    MambaEvictExcessPathStates,
    RecoverSWAWithLockedFull,
    RebuildFullToSWAMapping,
    ReplaceWriteThroughOnNodeSplit,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import StorageBackupSpec
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    DecSwaLockOnlyResult,
    DemoteResult,
    DriveHostEvictionResult,
    DropSubtreeNoHostResult,
    EvictDeviceLeafResult,
    EvictDeviceNextNodeResult,
    InsertStepResult,
    NodeId,
    RadixCacheWalkResult,
    UnifiedTreeCoreInterface,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import PoolTransferResult

logger = logging.getLogger(__name__)

MODULE_NAME = "sglang.srt.rust_extensions._scheduler"

_EMPTY_INT64_DEVICE: Optional[torch.Tensor] = None

_scheduler = load_rust_extension(MODULE_NAME)

# `CacheTransferPhase` (str enum) -> engine u8 phase id.
_PHASE = {
    CacheTransferPhase.BACKUP_HOST: _scheduler.PHASE_BACKUP_HOST,
    CacheTransferPhase.LOAD_BACK: _scheduler.PHASE_LOAD_BACK,
    CacheTransferPhase.BACKUP_STORAGE: _scheduler.PHASE_BACKUP_STORAGE,
    CacheTransferPhase.PREFETCH: _scheduler.PHASE_PREFETCH,
}
# `PoolName` -> engine pool id (matches the engine's POOL_* constants).
_POOL_ID = {PoolName.KV: 0, PoolName.SWA: 1, PoolName.MAMBA: 2}
_POOL_BY_ID = {v: k for k, v in _POOL_ID.items()}


def _ns_tuple(extra_key, cache_salt) -> tuple:
    return (extra_key, cache_salt)


class _RustNode:
    """Lightweight proxy over one engine node.

    Carries exactly the surface the KV-event recorder and the controller
    dereference: ``key`` (a real `RadixKey`), ``parent``, and the
    lazily-computable ``hash_value`` / ``event_hash_value`` (writes land in
    the engine arena).
    """

    __slots__ = ("id", "_core", "_key")

    def __init__(self, node_id: int, core: "UnifiedTreeCoreRust"):
        self.id = node_id
        self._core = core
        self._key = None

    @property
    def key(self) -> RadixKey:
        if self._key is None:
            ns = self._core._tree.node_ns(self.id)
            raw = array.array("q", self._core._tree.node_key(self.id))
            extra_key, cache_salt = self._core._ns_pairs[ns] if ns else (None, None)
            self._key = RadixKey(
                raw,
                extra_key=extra_key,
                is_bigram=self._core.is_eagle,
                cache_salt=cache_salt,
            )
        return self._key

    @property
    def parent(self) -> Optional["_RustNode"]:
        parent = self._core._tree.node_parent(self.id)
        if parent is None:
            return None
        return self._core.node_by_id(parent)

    @property
    def hash_value(self) -> Optional[list[str]]:
        return self._core._tree.get_hash_values_opt(self.id)

    @hash_value.setter
    def hash_value(self, values: list[str]) -> None:
        self._core._tree.set_hash_values(self.id, list(values))

    @property
    def event_hash_value(self) -> Optional[list[str]]:
        return self._core._tree.get_event_hash_values_opt(self.id)

    @event_hash_value.setter
    def event_hash_value(self, values: list[str]) -> None:
        self._core._tree.set_event_hash_values(self.id, list(values))

    @property
    def backuped(self) -> bool:
        return self._core._tree.is_backuped(self.id)

    @property
    def evicted(self) -> bool:
        return self._core._tree.is_evicted(self.id)

    def __repr__(self) -> str:
        return f"_RustNode(id={self.id}, key={self.key!r})"


class UnifiedTreeCoreRust(UnifiedTreeCoreInterface):
    """The unified tree core over the Rust engine (see module docstring)."""

    def __init__(
        self,
        params: "CacheInitParams",
        components: dict[ComponentType, TreeComponent],
    ):
        global _EMPTY_INT64_DEVICE
        self.params = params
        self.page_size = params.page_size
        self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components

        # ``device`` is derived from the construction-time allocator; the
        # allocator/pool themselves are owned by the cache, not the tree.
        if params.token_to_kv_pool_allocator:
            self.device = params.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")
        _EMPTY_INT64_DEVICE = torch.empty((0,), dtype=torch.int64, device=self.device)

        swa = components.get(ComponentType.SWA)
        mamba = components.get(ComponentType.MAMBA)
        if params.enable_session_radix_cache:
            raise NotImplementedError(
                "the Rust unified tree core does not support "
                "--enable-session-radix-cache"
            )
        if self.is_eagle:
            assert mamba is None, (
                "the Rust unified tree core does not support EAGLE bigram "
                "keys with Mamba"
            )
        self._tree = _scheduler.UnifiedRadixTree(
            page_size=self.page_size,
            is_eagle=self.is_eagle,
            sliding_window_size=swa.sliding_window_size if swa else 0,
            mamba_checkpoint_grid=mamba.mamba_checkpoint_grid if mamba else 0,
            mamba_max_states_per_path=(
                mamba.mamba_max_states_per_path if mamba else -1
            ),
            eviction_policy=params.eviction_policy.lower(),
            write_through_threshold=256,
            is_write_back=False,
            has_swa_host_pool=False,
            enable_session_radix_cache=False,
        )

        self.component_types = tuple(components.keys())
        self.components_by_type = components
        # The cache builds and owns the component drivers; the tree references
        # them to drive their facade-level hooks (apply_component_action,
        # prepare_prefetch). The tree-level hooks live inside the Rust engine.
        for component in components.values():
            component.tree_core = self
        self.components = tuple(components.values())

        self.kv_events = KVCacheEventRecorder(
            enabled=params.enable_kv_cache_events, page_size=self.page_size
        )

        # Namespace (extra_key, cache_salt) <-> engine u32 mapping; index 0 is
        # the root sentinel, so real namespaces start at 1.
        self._ns_pairs: list[tuple] = [None]
        self._ns_map: dict[tuple, int] = {}
        self._node_cache: dict[int, _RustNode] = {}
        self._empty_match_result: Optional[MatchResult] = None
        self._root_proxy: Optional[_RustNode] = None

        self.reset()

    # ---- tree-owned state the controller reads/writes ----

    @property
    def enable_hicache(self) -> bool:
        return self._hicache

    @property
    def enable_storage(self) -> bool:
        return self._storage

    @property
    def write_through_threshold(self) -> int:
        return self._tree.write_through_threshold()

    @write_through_threshold.setter
    def write_through_threshold(self, value: int) -> None:
        self._tree.set_write_through_threshold(value)

    @property
    def is_write_back(self) -> bool:
        return self._tree.is_write_back()

    @is_write_back.setter
    def is_write_back(self, value: bool) -> None:
        # The engine flag is one-way (the facade sets it once at wiring time).
        if value:
            self._tree.set_write_back()

    @property
    def has_swa_host_pool(self) -> bool:
        return self._swa_host_pool

    @has_swa_host_pool.setter
    def has_swa_host_pool(self, value: bool) -> None:
        self._swa_host_pool = value
        self._tree.set_swa_host_pool(value)

    @property
    def write_back_duplicate_reclaim_digest(self) -> int:
        return self._tree.reclaim_digest()

    @property
    def root_node(self) -> _RustNode:
        return self.node_by_id(self._tree.root_id())

    # ---- namespace plumbing ----

    def _ns_for(self, extra_key, cache_salt) -> int:
        key = _ns_tuple(extra_key, cache_salt)
        ns = self._ns_map.get(key)
        if ns is None:
            ns = len(self._ns_pairs)
            self._ns_pairs.append(key)
            self._ns_map[key] = ns
        return ns

    # ---- helpers ----

    def _dev_tensor(self, values: Sequence[int]) -> torch.Tensor:
        if not values:
            return _EMPTY_INT64_DEVICE
        return torch.tensor(list(values), dtype=torch.int64, device=self.device)

    @staticmethod
    def _cpu_tensor(values: Sequence[int]) -> torch.Tensor:
        if not values:
            return torch.empty((0,), dtype=torch.int64, device="cpu")
        return torch.tensor(list(values), dtype=torch.int64, device="cpu")

    @staticmethod
    def _tensors(values: Sequence[Sequence[int]], device: torch.device) -> list:
        return [
            (
                torch.empty((0,), dtype=torch.int64, device=device)
                if not chunk
                else torch.tensor(list(chunk), dtype=torch.int64, device=device)
            )
            for chunk in values
        ]

    def _action(self, a) -> object:
        """Engine action tuple -> Python CacheAction/ComponentAction."""
        tag = a[0]
        if tag == "ReplaceWT":
            return ReplaceWriteThroughOnNodeSplit(
                ack_id=a[1], old_node_id=a[2], new_node_id=a[3], new_child_node_id=a[4]
            )
        if tag == "FreeDeviceKV":
            return FreeDeviceKV(
                indices=self._tensors(a[1], self.device)
            )
        if tag == "FreeDeviceKVFullOnly":
            return FreeDeviceKVFullOnly(indices=self._tensors(a[1], self.device))
        if tag == "BackupKV":
            return BackupKV(node_ids=[int(n) for n in a[1]])
        if tag == "FreeComponentDeviceSlot":
            return FreeComponentDeviceSlot(
                indices=self._tensors(a[2], self.device),
                component_type=ComponentType(a[1]),
            )
        if tag == "FreeComponentHostSlot":
            return FreeComponentHostSlot(
                host_indices=self._tensors(a[2], torch.device("cpu")),
                component_type=ComponentType(a[1]),
            )
        if tag == "MambaEvictExcess":
            return MambaEvictExcessPathStates(tail_node_id=a[1])
        if tag == "RebuildFullToSWAMapping":
            return RebuildFullToSWAMapping(
                full_indices=self._tensors(a[1], self.device),
                swa_indices=self._tensors(a[2], self.device),
            )
        if tag == "RecoverSWAWithLockedFull":
            return RecoverSWAWithLockedFull(
                node_id=a[1],
                kept_full=self._dev_tensor(a[2]),
                incoming_full=self._dev_tensor(a[3]),
            )
        if tag == "SWARebuild":
            return SWARebuild(node_id=a[1], source_value=self._dev_tensor(a[2]))
        raise AssertionError(f"unhandled engine action: {tag!r}")

    def _actions(self, raw) -> list:
        return [self._action(a) for a in raw]

    @staticmethod
    def _xfers(raw) -> list[tuple[int, list]]:
        """Engine comp_xfers (list of (ct, [transfers])) -> plain pairs."""
        return [(int(ct), list(xfers)) for ct, xfers in raw]

    def _transfer(self, x) -> PoolTransfer:
        pool, device, host, keys, nodes, hit_policy = x
        return PoolTransfer(
            name={0: PoolName.KV, 1: PoolName.SWA, 2: PoolName.MAMBA}[pool],
            host_indices=None if host is None else self._cpu_tensor(host),
            device_indices=None if device is None else self._dev_tensor(device),
            keys=keys,
            hit_policy=(
                PoolHitPolicy.ALL_PAGES
                if hit_policy == 0
                else PoolHitPolicy.TRAILING_PAGES
            ),
            nodes_to_load=None if not nodes else [int(n) for n in nodes],
        )

    @staticmethod
    def _skip_to_pairs(skip: Optional[dict]) -> list:
        if not skip:
            return []
        return [(int(ct), sorted(ids)) for ct, ids in skip.items()]

    @staticmethod
    def _skip_to_dict(raw: list) -> dict:
        out: dict = {}
        for ct, ids in raw:
            out[ComponentType(ct)] = set(int(n) for n in ids)
        return out

    @staticmethod
    def _insert_result_tuple(result: Optional[InsertResult]) -> tuple:
        """Python InsertResult -> engine 8-tuple (cache_actions dropped; the
        tree does not need caller-side actions on the commit path)."""
        if result is None:
            return (0, 0, 0, False, None, False, [], [])
        return (
            result.prefix_len,
            result.total_len,
            result.last_device_node,
            result.mamba_exist,
            result.inserted_host_node,
            result.host_insert_dropped,
            [],
            [],
        )

    # ---- tree API ----

    def reset(self) -> None:
        """Drop the entire tree and reinitialize empty state."""
        self._hicache = False
        self._storage = False
        self._swa_host_pool = False
        self._tree.reset()
        self._node_cache.clear()
        root = self._tree.root_id()
        self._root_proxy = _RustNode(root, self)
        self._empty_match_result = MatchResult(
            device_indices=_EMPTY_INT64_DEVICE,
            last_device_node=root,
            last_host_node=root,
            best_match_node=root,
            cache_actions=[],
        )

    def node_by_id(self, node_id: NodeId) -> _RustNode:
        node = self._node_cache.get(node_id)
        if node is None:
            node = _RustNode(node_id, self)
            self._node_cache[node_id] = node
        return node

    def is_backuped(self, node_id: NodeId) -> bool:
        return self._tree.is_backuped(node_id)

    def is_root(self, node_id: NodeId) -> bool:
        return self._tree.is_root(node_id)

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        return self._tree.get_last_hash_value(node_id)

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        return self._tree.get_prefix_hash_values(node_id)

    def get_hash_values(self, node_id: NodeId) -> list[str]:
        values = self._tree.get_hash_values_opt(node_id)
        return values if values is not None else []

    def backfill_missing_hash_values(self) -> int:
        """Hash every node built while storage was disabled; return how many."""
        from sglang.srt.mem_cache.utils import compute_node_hash_values

        filled = 0
        root = self._tree.root_id()
        for dump in self._tree.dump_nodes():  # root first, parent before child
            node_id = dump[0]
            if node_id == root or self._tree.get_hash_values_opt(node_id) is not None:
                continue
            node = self.node_by_id(node_id)
            self._tree.set_hash_values(
                node_id, compute_node_hash_values(node, self.page_size)
            )
            filled += 1
        return filled

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        """The NodeId anchoring matches; the single root serves every namespace."""
        return self._tree.root_id()

    def take_events(self) -> list:
        """Replay the engine's raw store/remove log through the recorder."""
        for op, node_id, medium in self._tree.take_kv_events():
            node = self.node_by_id(node_id)
            medium = StorageMedium.GPU if medium == 1 else StorageMedium.CPU
            if op == 1:
                self.kv_events.record_store(node, medium=medium)
            else:
                self.kv_events.record_remove(node, medium=medium)
        return self.kv_events.take()

    # ---- locks ----

    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        delta, swa_uuid, swa_host_uuid, skip = self._tree.inc_lock_ref(
            node_id, [int(ct) for ct in skip_lock_components]
        )
        return IncLockRefResult(
            delta=delta,
            swa_uuid_for_lock=swa_uuid,
            swa_uuid_for_host_lock=swa_host_uuid,
            skip_lock_node_ids=self._skip_to_dict(skip),
        )

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        self._tree.dec_lock_ref(
            node_id,
            params.swa_uuid_for_lock if params else None,
            params.swa_uuid_for_host_lock if params else None,
            self._skip_to_pairs(params.skip_lock_node_ids if params else None),
            skip_swa,
        )
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        skip_lock_node_ids: Optional[dict] = None,
    ) -> DecSwaLockOnlyResult:
        result = DecSwaLockOnlyResult()
        tracker, device_frees, host_frees, _dropped, _backup = self._tree.dec_swa_lock_only(
            node_id, swa_uuid_for_lock, self._skip_to_pairs(skip_lock_node_ids)
        )
        for ct, n in tracker:
            result.tracker[ComponentType(ct)] += n
        for ct, chunks in device_frees:
            result.device_frees[ComponentType(ct)].extend(
                self._tensors(chunks, self.device)
            )
        for ct, chunks in host_frees:
            result.host_frees[ComponentType(ct)].extend(
                self._tensors(chunks, torch.device("cpu"))
            )
        return result

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        delta, swa_uuid, swa_host_uuid, skip = self._tree.inc_host_lock_ref(node_id)
        return IncLockRefResult(
            delta=delta,
            swa_uuid_for_lock=swa_uuid,
            swa_uuid_for_host_lock=swa_host_uuid,
            skip_lock_node_ids=self._skip_to_dict(skip),
        )

    def dec_host_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
    ) -> DecLockRefResult:
        self._tree.dec_host_lock_ref(
            node_id,
            params.swa_uuid_for_lock if params else None,
            params.swa_uuid_for_host_lock if params else None,
            self._skip_to_pairs(params.skip_lock_node_ids if params else None),
        )
        return DecLockRefResult()

    # ---- device eviction ----

    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:
        self._tree.evict_device_start(int(component_type), request_cnt)

    def evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> EvictDeviceNextNodeResult:
        ct = int(component_type)
        result = EvictDeviceNextNodeResult()
        node_id, made_progress, tracker_delta, device_frees, host_frees = (
            self._tree.evict_device_next_node(ct, tracker.get(component_type, 0))
        )
        result.node_id = int(node_id) if node_id is not None else None
        result.made_progress = made_progress
        for c, n in tracker_delta:
            result.tracker[ComponentType(c)] += n
        for c, chunks in device_frees:
            result.device_frees[ComponentType(c)].extend(
                self._tensors(chunks, self.device)
            )
        for c, chunks in host_frees:
            result.host_frees[ComponentType(c)].extend(
                self._tensors(chunks, torch.device("cpu"))
            )
        return result

    def evict_device_leaf(
        self, node_id: NodeId, is_write_back: bool
    ) -> EvictDeviceLeafResult:
        result = EvictDeviceLeafResult()
        tracker, device_frees, host_frees, _dropped, backup_kv = (
            self._tree.evict_device_leaf(node_id, is_write_back)
        )
        for ct, n in tracker:
            result.tracker[ComponentType(ct)] += n
        for ct, chunks in device_frees:
            result.device_frees[ComponentType(ct)].extend(
                self._tensors(chunks, self.device)
            )
        for ct, chunks in host_frees:
            result.host_frees[ComponentType(ct)].extend(
                self._tensors(chunks, torch.device("cpu"))
            )
        if backup_kv is not None:
            result.backup_kv = BackupKV(node_ids=[int(n) for n in backup_kv])
        return result

    def drop_subtree_no_host(self, node_id: NodeId) -> DropSubtreeNoHostResult:
        result = DropSubtreeNoHostResult(is_dropped=False)
        tracker, device_frees, host_frees, is_dropped, _backup = (
            self._tree.drop_subtree_no_host(node_id)
        )
        result.is_dropped = is_dropped
        for ct, n in tracker:
            result.tracker[ComponentType(ct)] += n
        for ct, chunks in device_frees:
            result.device_frees[ComponentType(ct)].extend(
                self._tensors(chunks, self.device)
            )
        for ct, chunks in host_frees:
            result.host_frees[ComponentType(ct)].extend(
                self._tensors(chunks, torch.device("cpu"))
            )
        return result

    def demote(self, node_id: NodeId) -> DemoteResult:
        result = DemoteResult()
        tracker, device_frees, host_frees, _dropped, _backup = self._tree.demote(node_id)
        for ct, n in tracker:
            result.tracker[ComponentType(ct)] += n
        for ct, chunks in device_frees:
            result.device_frees[ComponentType(ct)].extend(
                self._tensors(chunks, self.device)
            )
        for ct, chunks in host_frees:
            result.host_frees[ComponentType(ct)].extend(
                self._tensors(chunks, torch.device("cpu"))
            )
        return result

    def evict_device_end(self, component_type: ComponentType) -> None:
        self._tree.evict_device_end(int(component_type))

    # ---- sizes / values ----

    def evictable_size(self) -> int:
        return self._tree.evictable_size()

    def protected_size(self) -> int:
        return self._tree.protected_size()

    def component_evictable_size(self, component_type: ComponentType) -> int:
        return self._tree.component_evictable_size(int(component_type))

    def component_protected_size(self, component_type: ComponentType) -> int:
        return self._tree.component_protected_size(int(component_type))

    def full_evictable_size(self) -> int:
        return self.component_evictable_size(ComponentType.FULL)

    def full_protected_size(self) -> int:
        return self.component_protected_size(ComponentType.FULL)

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size(ComponentType.SWA)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size(ComponentType.MAMBA)

    def swa_protected_size(self) -> int:
        return self.component_protected_size(ComponentType.SWA)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size(ComponentType.MAMBA)

    def total_size(self) -> tuple[int, int]:
        """(full_tokens, aux_tokens) summed across the whole tree."""
        return self._tree.total_size()

    def all_values_flatten(self) -> torch.Tensor:
        return self._dev_tensor(self._tree.all_values_flatten())

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._dev_tensor(self._tree.all_mamba_values_flatten())

    def walk_for_kv_canary(
        self, unlocked_only: bool, swa_resident_only: bool
    ) -> RadixCacheWalkResult:
        slots, positions, prev_slots = self._tree.walk_for_kv_canary(
            unlocked_only, swa_resident_only
        )
        return RadixCacheWalkResult(
            slot_indices=torch.tensor(list(slots), dtype=torch.int64),
            positions=torch.tensor(list(positions), dtype=torch.int64),
            prev_slot_indices=torch.tensor(list(prev_slots), dtype=torch.int64),
        )

    # ---- match ----

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        ns = self._ns_for(key.extra_key, key.cache_salt)
        (
            device_indices,
            last_device,
            last_host,
            best,
            host_hit,
            swa_host_hit,
            mamba_host_hit,
            mamba_branch,
            full_hit,
            actions,
        ) = self._tree.match_prefix(ns, list(key.raw_token_ids()))
        if device_indices:
            device_indices_t = self._dev_tensor(device_indices)
        else:
            device_indices_t = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=device_indices_t,
            last_device_node=int(last_device),
            last_host_node=int(last_host),
            best_match_node=int(best),
            host_hit_length=host_hit,
            swa_host_hit_length=swa_host_hit,
            mamba_host_hit_length=mamba_host_hit,
            mamba_branching_seqlen=mamba_branch,
            full_kv_hit_length=full_hit,
            cache_actions=self._actions(actions),
        )

    @property
    def empty_match_result(self) -> MatchResult:
        return self._empty_match_result

    def is_full_device_evicted(self, node_id: NodeId) -> bool:
        return self._tree.is_full_device_evicted(node_id)

    def collect_full_device_indices(
        self, from_node_id: NodeId, until_node_id: NodeId
    ) -> torch.Tensor:
        return self._dev_tensor(
            self._tree.collect_full_device_indices(from_node_id, until_node_id)
        )

    # ---- stepped insert ----

    def begin_insert(self, params: InsertParams) -> InsertStepResult:
        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if params.c128_value is not None:
            raise NotImplementedError(
                "the Rust unified tree core does not support C128 sidecar pages"
            )
        if value is None:
            value = list(key.raw_token_ids()[: len(key)])
        else:
            value = [int(x) for x in value[: len(key)]]
        priority = params.priority or 0
        mamba_value = (
            [int(x) for x in params.mamba_value]
            if params.mamba_value is not None
            else None
        )
        ns = self._ns_for(key.extra_key, key.cache_salt)
        if len(key) == 0:
            return InsertStepResult(
                actions=[],
                result=InsertResult(
                    prefix_len=0,
                    mamba_exist=True,
                    last_device_node=self._tree.root_id(),
                ),
            )
        step = self._tree.begin_insert(
            ns,
            list(key.raw_token_ids()),
            value,
            params.prev_prefix_len,
            params.chunked,
            priority,
            params.swa_evicted_seqlen,
            mamba_value,
        )
        return self._step_result(step)

    def _step_result(self, step) -> InsertStepResult:
        actions, result = step
        if result is None:
            return InsertStepResult(actions=self._actions(actions), result=None)
        return InsertStepResult(
            actions=self._actions(actions),
            result=InsertResult(
                prefix_len=result[0],
                total_len=result[1],
                last_device_node=int(result[2]),
                mamba_exist=result[3],
                inserted_host_node=result[4],
                host_insert_dropped=result[5],
                cache_actions=self._actions(result[7]),
            ),
        )

    def resume_insert(self) -> InsertStepResult:
        return self._step_result(self._tree.resume_insert())

    def has_ongoing_insert(self) -> bool:
        return self._tree.has_ongoing_insert()

    def end_insert(self) -> list:
        return self._actions(self._tree.end_insert())

    # ---- host eviction / mamba cap ----

    def drive_host_eviction(
        self, component_type: ComponentType, num_tokens: int
    ) -> DriveHostEvictionResult:
        result = DriveHostEvictionResult()
        tracker, device_frees, host_frees, _dropped, _backup = self._tree.drive_host_eviction(
            int(component_type), num_tokens
        )
        for ct, n in tracker:
            result.tracker[ComponentType(ct)] += n
        for ct, chunks in device_frees:
            result.device_frees[ComponentType(ct)].extend(
                self._tensors(chunks, self.device)
            )
        for ct, chunks in host_frees:
            result.host_frees[ComponentType(ct)].extend(
                self._tensors(chunks, torch.device("cpu"))
            )
        return result

    def evict_excess_path_states(
        self,
        tail_node_id: NodeId,
        device_frees: dict[ComponentType, list],
        host_frees: dict[ComponentType, list],
    ) -> None:
        tracker, d_frees, h_frees, _dropped, _backup = self._tree.evict_excess_path_states(
            tail_node_id
        )
        for ct, chunks in d_frees:
            device_frees.setdefault(ComponentType(ct), []).extend(
                self._tensors(chunks, self.device)
            )
        for ct, chunks in h_frees:
            host_frees.setdefault(ComponentType(ct), []).extend(
                self._tensors(chunks, torch.device("cpu"))
            )

    # ---- HiCache ----

    def set_hicache_enabled(self) -> None:
        self._hicache = True
        self._tree.set_hicache_enabled()

    def insert_host(
        self,
        node_id: NodeId,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        ns = self._ns_for(key.extra_key, key.cache_salt)
        (
            prefix,
            total,
            last_device,
            mamba_exist,
            inserted_host,
            dropped,
            _created,
            cache_actions,
        ) = self._tree.insert_host(
            node_id,
            ns,
            list(key.raw_token_ids()),
            [int(x) for x in host_value],
            list(hash_value),
        )
        return InsertResult(
            prefix_len=prefix,
            total_len=total,
            last_device_node=None if last_device == 0 else int(last_device),
            mamba_exist=mamba_exist,
            inserted_host_node=inserted_host,
            host_insert_dropped=dropped,
            cache_actions=self._actions(cache_actions),
        )

    def build_backup_spec(
        self, node_id: NodeId
    ) -> tuple[torch.Tensor, dict[ComponentType, list[PoolTransfer]]]:
        device_value, raw_xfers = self._tree.build_backup_spec(node_id)
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for ct, xfers in raw_xfers:
            comp_xfers[ComponentType(ct)] = [self._transfer(x) for x in xfers]
        return self._dev_tensor(device_value), comp_xfers

    def build_storage_backup_spec(
        self, node_id: NodeId, pass_prefix_keys: bool
    ) -> Optional[StorageBackupSpec]:
        spec = self._tree.build_storage_backup_spec(node_id, pass_prefix_keys)
        if spec is None:
            return None
        host_value, token_ids, hash_value, prefix_keys, raw_xfers = spec
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for ct, xfers in raw_xfers:
            comp_xfers[ComponentType(ct)] = [self._transfer(x) for x in xfers]
        return StorageBackupSpec(
            host_value=self._cpu_tensor(host_value),
            token_ids=array("q", token_ids),
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            comp_xfers=comp_xfers,
        )

    def build_hicache_transfers(
        self,
        component_type: ComponentType,
        node_id: NodeId,
        phase: CacheTransferPhase,
        *,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Build a component's HiCache transfers for the given node and phase."""
        if component_type is ComponentType.FULL:
            if phase == CacheTransferPhase.LOAD_BACK:
                spec = self._tree.build_load_back_spec(node_id, None)
                if spec is not None:
                    return [self._transfer(spec[0])]
            return None
        xfer = self._tree.build_comp_transfer(
            int(component_type),
            node_id,
            _PHASE[phase],
            None if host_indices is None else [int(x) for x in host_indices],
            None,
        )
        if xfer is None:
            return None
        return [self._transfer(xfer)]

    def build_load_back_spec(
        self, node_id: NodeId, req: Optional["Req"] = None
    ) -> tuple[PoolTransfer, dict[ComponentType, list[PoolTransfer]]]:
        mamba_pool_idx = (
            [int(x) for x in req.mamba_pool_idx]
            if req is not None and req.mamba_pool_idx is not None
            else None
        )
        spec = self._tree.build_load_back_spec(node_id, mamba_pool_idx)
        if spec is None:
            # Conflict (a would-be loaded node is pinned by another anchor):
            # the empty spec makes the caller back off and recompute.
            empty_kv = PoolTransfer(
                name=PoolName.KV,
                host_indices=torch.empty((0,), dtype=torch.int64, device="cpu"),
                nodes_to_load=[],
            )
            return empty_kv, {}
        kv, raw_xfers = spec
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for ct, xfers in raw_xfers:
            comp_xfers[ComponentType(ct)] = [self._transfer(x) for x in xfers]
        return self._transfer(kv), comp_xfers

    def prefetch_anchor_info(
        self, node_id: NodeId
    ) -> tuple[Optional[str], Optional[str]]:
        """The anchor node's key extra_key and cache_salt."""
        if node_id == self._tree.root_id():
            return None, None
        ns = self._tree.node_ns(node_id)
        if not ns or ns >= len(self._ns_pairs):
            return None, None
        extra_key, cache_salt = self._ns_pairs[ns]
        return extra_key, cache_salt

    def commit_hicache_transfers(
        self,
        node_id: NodeId,
        phase: CacheTransferPhase,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
        *,
        cache_actions: list,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional["PoolTransferResult"] = None,
    ) -> None:
        """Commit each component's HiCache transfers onto the node.

        Only the PREFETCH phase reaches the tree core through this entry point
        (backups go through commit_backup, load-backs through commit_load_back).
        """
        if phase != CacheTransferPhase.PREFETCH:
            return
        for ct, xfers in comp_xfers.items():
            if not xfers:
                continue
            host = xfers[0].host_indices
            host_list = None if host is None else [int(x) for x in host]
            if ct is ComponentType.SWA:
                loaded_pages = (
                    pool_storage_result.extra_pool_hit_pages.get(PoolName.SWA, 0)
                    if pool_storage_result is not None
                    else 0
                )
                actions = self._tree.commit_swa_prefetch(
                    node_id,
                    host_list or [],
                    loaded_pages,
                    self._insert_result_tuple(insert_result),
                )
                cache_actions.extend(self._actions(actions))
            elif ct is ComponentType.MAMBA:
                loaded = (
                    pool_storage_result is not None
                    and pool_storage_result.extra_pool_hit_pages.get(
                        PoolName.MAMBA, 0
                    )
                    >= 1
                )
                actions, mamba_exist, inserted_host = self._tree.commit_mamba_prefetch(
                    host_list, loaded, self._insert_result_tuple(insert_result)
                )
                if insert_result is not None:
                    insert_result.mamba_exist = mamba_exist
                cache_actions.extend(self._actions(actions))

    def commit_backup(
        self,
        node_id: NodeId,
        host_indices: torch.Tensor,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> None:
        """Commit a successful backup to the node."""
        raw_xfers = self._comp_xfers_to_raw(comp_xfers)
        self._tree.commit_backup(
            node_id,
            [int(x) for x in host_indices] if host_indices is not None else [],
            raw_xfers,
        )

    @staticmethod
    def _comp_xfers_to_raw(
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list:
        out = []
        for ct, xfers in comp_xfers.items():
            raw = []
            for x in xfers:
                raw.append(
                    (
                        _POOL_ID[x.pool],
                        None
                        if x.device_indices is None
                        else [int(v) for v in x.device_indices],
                        None
                        if x.host_indices is None
                        else [int(v) for v in x.host_indices],
                        x.keys,
                        [int(n) for n in (x.nodes_to_load or [])],
                        0
                        if x.hit_policy == PoolHitPolicy.ALL_PAGES
                        else 1,
                    )
                )
            out.append((int(ct), raw))
        return out

    def commit_load_back(
        self,
        node_id: NodeId,
        device_indices: torch.Tensor,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list:
        """Commit a successful H->D load-back onto the node; returns any cache actions."""
        raw_xfers = self._comp_xfers_to_raw(comp_xfers)
        nodes_to_load = [int(n) for n in (kv_xfer.nodes_to_load or [])]
        actions = self._tree.commit_load_back(
            node_id,
            [int(x) for x in device_indices] if device_indices is not None else None,
            nodes_to_load,
            raw_xfers,
        )
        return self._actions(actions)

    def finish_load_back(self, anchor_node_id: NodeId) -> None:
        self._tree.finish_load_back(anchor_node_id)

    def mark_write_through_pending(self, node_id: NodeId) -> None:
        self._tree.mark_write_through_pending(node_id)

    def finish_write_through(self, node_ids: list[NodeId], ack_id: int) -> None:
        self._tree.finish_write_through([int(n) for n in node_ids], ack_id)

    # ---- component values ----

    def set_component_device_value(
        self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
    ) -> None:
        self._tree.set_component_device_value(
            node_id, int(component_type), [int(x) for x in value]
        )

    def get_component_device_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        value = self._tree.get_component_device_value(node_id, int(component_type))
        if value is None:
            return None
        return self._dev_tensor(value)

    def component_has_host_value_only(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        return self._tree.component_has_host_value_only(
            node_id, int(component_type)
        )

    # ---- others ----

    def sanity_check(
        self,
        ongoing_write_through: list[tuple[int, NodeId]],
        ongoing_load_back: list[tuple[int, NodeId]],
    ) -> None:
        errors = self._tree.sanity_check(
            [(int(i), int(n)) for i, n in ongoing_write_through],
            [(int(i), int(n)) for i, n in ongoing_load_back],
        )
        if errors:
            raise AssertionError("\n".join(errors))

    def pretty_print(self) -> None:
        """Print the tree structure for debugging."""
        root = self._tree.root_id()
        for dump in self._tree.dump_nodes():
            (
                node_id,
                key,
                _last,
                _creation,
                _hits,
                _prio,
                full,
                full_host,
                swa,
                swa_host,
                mamba,
                mamba_host,
                _locks,
                _host_locks,
                _swa_uuid,
                _swa_host_uuid,
                _wt,
                _lb,
                in_dev,
                in_host,
                _dup,
            ) = dump
            indent = "  " * (0 if node_id == root else self._depth(node_id))
            marks = ("D" if in_dev else "-") + ("H" if in_host else "-")
            logger.info(
                "%s%s%s full=%s swa=%s mamba=%s",
                indent,
                marks,
                list(key),
                None if full is None else len(full),
                None if swa is None else len(swa),
                None if mamba is None else len(mamba),
            )

    def _depth(self, node_id: NodeId) -> int:
        depth = 0
        cur = node_id
        root = self._tree.root_id()
        while cur != root and cur is not None:
            parent = self._tree.node_parent(cur)
            if parent is None:
                break
            cur = parent
            depth += 1
        return depth


def _rust_tree_core_factory(params, components) -> UnifiedTreeCoreRust:
    return UnifiedTreeCoreRust(params, components)
