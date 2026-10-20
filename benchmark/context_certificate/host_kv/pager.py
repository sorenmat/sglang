"""CPU-testable core of the host-KV tier: page ownership and admission.

Pure logic, no torch: the GPU integration points (pinned buffers, async
copies, events) are declared as injectable callables so tests run with
fakes and the live window fills in real CUDA plumbing. See
HOST_KV_DESIGN.md for the design this implements.
"""

from collections import OrderedDict
from enum import Enum


class PageState(Enum):
    HOST = 1
    IN_FLIGHT = 2
    GPU_RESIDENT = 3


class TransferKind(Enum):
    DEMAND = 1
    PREFETCH = 2


class HostTierPager:
    """Owns the page-state machine and the GPU-page cache of one rank.

    ``page_bytes`` is bytes per page across all layers (1.57 MiB for the
    64-token page of the target model); ``gpu_slots`` is the number of GPU
    page frames the cache may use.
    """

    def __init__(self, gpu_slots, page_bytes=64 * 24_576, transfer=None,
                 wait=None):
        if gpu_slots < 1:
            raise ValueError("gpu_slots must be positive")
        self.gpu_slots = gpu_slots
        self.page_bytes = page_bytes
        # Injected CUDA plumbing: transfer(page) -> handle; wait(handle).
        self._transfer = transfer or (lambda page: None)
        self._wait = wait or (lambda handle: None)
        self.state = {}                 # page_id -> PageState
        self.handles = {}               # page_id -> in-flight handle
        self.lru = OrderedDict()        # page_id -> None, GPU_RESIDENT only
        self.transferred_bytes = 0
        self.transfers = {"demand": 0, "prefetch": 0}

    # -- ownership ----------------------------------------------------

    def touch(self, page_id):
        """Mark a GPU-resident page used this pass (pins against eviction)."""
        if self.state.get(page_id) is not PageState.GPU_RESIDENT:
            raise ValueError(f"touch({page_id}) on non-resident page")
        self.lru.move_to_end(page_id)

    def ensure_resident(self, page_id):
        """Demand path: block until `page_id` is GPU-resident."""
        state = self.state.get(page_id, PageState.HOST)
        if state is PageState.GPU_RESIDENT:
            self.touch(page_id)
            return False
        if state is PageState.IN_FLIGHT:
            self._wait(self.handles[page_id])
        else:
            handle = self._start(page_id, TransferKind.DEMAND)
            self._wait(handle)
        self._finish(page_id)
        return True

    def _start(self, page_id, kind):
        self._evict_if_full()
        self.state[page_id] = PageState.IN_FLIGHT
        self.handles[page_id] = self._transfer(page_id)
        self.transferred_bytes += self.page_bytes
        self.transfers["demand" if kind is TransferKind.DEMAND else "prefetch"] += 1
        return self.handles[page_id]

    def _finish(self, page_id):
        self.state[page_id] = PageState.GPU_RESIDENT
        self.lru[page_id] = None
        self.handles.pop(page_id, None)

    def _evict_if_full(self):
        while len(self.lru) >= self.gpu_slots:
            victim, _ = self.lru.popitem(last=False)  # LRU, never pinned
            self.state[victim] = PageState.HOST

    def evict(self, page_id):
        if self.state.get(page_id) is not PageState.GPU_RESIDENT:
            raise ValueError(f"evict({page_id}) on non-resident page")
        del self.lru[page_id]
        self.state[page_id] = PageState.HOST


class AdmissionController:
    """Per-pass link budget: demand always admits, prefetch fits the rest."""

    def __init__(self, bytes_per_pass, prefetch_overfetch_cap=0.35):
        if bytes_per_pass <= 0:
            raise ValueError("bytes_per_pass must be positive")
        self.bytes_per_pass = bytes_per_pass
        self.overfetch_cap = prefetch_overfetch_cap
        self.reset()

    def reset(self):
        self.demand_bytes = 0
        self.prefetch_bytes = 0

    def admit(self, page_bytes, kind):
        if kind is TransferKind.DEMAND:
            self.demand_bytes += page_bytes
            return True
        cap = self.bytes_per_pass * self.overfetch_cap
        if self.prefetch_bytes + page_bytes > cap:
            return False
        # Prefetch may never eat into the demand budget either.
        if self.demand_bytes + self.prefetch_bytes + page_bytes > self.bytes_per_pass:
            return False
        self.prefetch_bytes += page_bytes
        return True

    @property
    def used_bytes(self):
        return self.demand_bytes + self.prefetch_bytes
