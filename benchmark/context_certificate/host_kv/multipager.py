"""P2.3 multi-request page mapper: the brain of the shrunk-pool config.

One GPU card, one TP1 server, up to N requests sharing a SMALL pool of
GPU page frames (the pool shrink is where the memory savings come from).
This module owns the mapping and the PLANS; the runtime integration
performs the copies and req_to_token rewrites the plans describe.

Model (see HOST_KV_DESIGN.md, P2.3 section):

- Every request owns logical pages (its context // page_size); each page's
  content lives on the host always, and in at most one GPU frame.
- ``PageMapper`` maps (request, logical_page) <-> frame, evicts LRU among
  unpinned frames when the budget is exhausted, and emits copy plans:
  ``Eviction`` (frame -> host, then the frame is reusable) and the
  allocation results the caller must fulfill with host->frame copies.
- Pinned pages (this pass's selection set) are never eviction victims;
  eviction of a dirty frame is the caller's copy, tracked via ``dirty``.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum


class Residence(Enum):
    HOST = 0
    GPU = 1


@dataclass(frozen=True)
class Eviction:
    request: object
    logical_page: int
    frame: int
    dirty: bool


@dataclass
class RequestPages:
    frames: dict = field(default_factory=dict)          # page -> frame
    host_pages: set = field(default_factory=set)         # pages resident on host
    dirty: set = field(default_factory=set)              # frames with unmirrored writes
    order: "OrderedDict[int, None]" = field(default_factory=OrderedDict)  # LRU of pages


class PageMapper:
    def __init__(self, gpu_frames, page_tokens=64):
        if gpu_frames < 1:
            raise ValueError("gpu_frames must be positive")
        self.gpu_frames = gpu_frames
        self.page_tokens = page_tokens
        self.free_frames = list(range(gpu_frames))
        self.frame_owner = {}                             # frame -> (request, page)
        self.requests = {}                                # request id -> RequestPages
        self.pinned = set()                               # (request, page)
        self.touch_seq = {}                               # (request, page) -> global recency
        self.clock = 0
        self.stats = {"evictions": 0, "dirty_evictions": 0, "allocations": 0,
                      "stages": 0, "faults": 0}

    # -- lifecycle ----------------------------------------------------

    def open_request(self, request):
        if request in self.requests:
            raise ValueError(f"request {request!r} already open")
        self.requests[request] = RequestPages()

    def close_request(self, request):
        pages = self.requests.pop(request)
        for page, frame in pages.frames.items():
            self.frame_owner.pop(frame, None)
            self.free_frames.append(frame)
        self.pinned = {key for key in self.pinned if key[0] != request}

    # -- mapping ------------------------------------------------------

    def residence(self, request, page):
        pages = self.requests[request]
        if page in pages.frames:
            return Residence.GPU
        return Residence.HOST

    def _touch(self, request, page):
        self.clock += 1
        self.touch_seq[(request, page)] = self.clock
        pages = self.requests[request]
        pages.order[page] = None
        pages.order.move_to_end(page)

    # -- allocation (prefill streaming) --------------------------------

    def alloc(self, request, page, dirty=True):
        """Make room for one page of `request`; returns (frame, evictions).

        The caller must: perform each Eviction's frame->host copy (if
        dirty), then write/point the new page's data at `frame` and record
        `dirty` in the request's dirty set when the write is not mirrored.
        """
        if request not in self.requests:
            self.open_request(request)
        pages = self.requests[request]
        if page in pages.frames:                          # already resident
            self._touch(request, page)
            return pages.frames[page], []
        evictions = []
        while not self.free_frames:
            if not any(p.order for p in self.requests.values()):
                raise RuntimeError("no evictable pages; all frames pinned")
            evictions.append(self._evict_one())
        frame = self.free_frames.pop()
        self.frame_owner[frame] = (request, page)
        pages.frames[page] = frame
        if dirty:
            pages.dirty.add(page)
        pages.host_pages.discard(page)
        self._touch(request, page)
        self.stats["allocations"] += 1
        return frame, evictions

    def _evict_one(self) -> Eviction:
        # GLOBAL least-recently-used victim across all requests; iterating
        # requests in dict order would thrash whichever comes first.
        candidates = []
        for request, pages in self.requests.items():
            for page in pages.order:
                if (request, page) in self.pinned:
                    continue
                candidates.append((self.touch_seq.get((request, page), 0), request, page))
                break                                     # each request's LRU head only
        if not candidates:
            raise RuntimeError("all frames pinned; cannot evict")
        _, request, page = min(candidates)
        pages = self.requests[request]
        frame = pages.frames[page]
        dirty = page in pages.dirty
        self.touch_seq.pop((request, page), None)
        del pages.frames[page]
        pages.dirty.discard(page)
        pages.order.pop(page, None)
        pages.host_pages.add(page)
        self.frame_owner.pop(frame, None)
        # The frame frees immediately: the caller performs all Eviction
        # copies BEFORE writing the new page at the frame.
        self.free_frames.append(frame)
        self.stats["evictions"] += 1
        self.stats["dirty_evictions"] += dirty
        return Eviction(request, page, frame, dirty)

    # -- demand paging (decode gather) ---------------------------------

    def ensure(self, request, pages_needed) -> list:
        """Plan the staging of `pages_needed` (selection set of one pass).

        Returns a list of (page, frame, evictions) like alloc(); pages
        already GPU-resident are touched only. The caller performs the
        copies and rewrites req_to_token rows for the returned pages.
        """
        plan = []
        for page in pages_needed:
            if self.residence(request, page) is Residence.GPU:
                self._touch(request, page)
                continue
            frame, evictions = self.alloc(request, page, dirty=False)
            self.stats["stages"] += 1
            self.stats["faults"] += 1
            plan.append((page, frame, evictions))
        return plan

    def pin(self, request, pages):
        """Pin this pass's selection set (protects them from eviction)."""
        for page in pages:
            if self.residence(request, page) is not Residence.GPU:
                raise ValueError(f"pin of non-resident page {page}")
            self.pinned.add((request, page))

    def unpin_all(self):
        self.pinned.clear()

    def mark_clean(self, request, page):
        """The frame's content now matches the host copy (write-through
        completed, or a staged copy is pristine by construction)."""
        self.requests[request].dirty.discard(page)

    def frame_of(self, request, page):
        return self.requests[request].frames[page]

    # -- accounting ----------------------------------------------------

    def used_frames(self):
        return self.gpu_frames - len(self.free_frames)

    def snapshot(self):
        return {
            "gpu_frames": self.gpu_frames,
            "used": self.used_frames(),
            "requests": len(self.requests),
            "resident_pages": sum(len(p.frames) for p in self.requests.values()),
            **self.stats,
        }
