"""P2.3 runtime glue: turns PageMapper plans into slot-table operations.

CPU-testable against torch tensors; the overlay performs the same
operations on the live runtime tensors. Design contract (HOST_KV_DESIGN.md
P2.3): the shrunk pool's page ids ARE the mapper's frames; the allocator's
free_pages is fronted by the mapper; req_to_token rows are rewritten on
every stage (victim page's tokens point at the reserved slot 0 after
eviction, and at the new frame after staging).
"""

import torch

from multipager import PageMapper, Residence

PAGE_TOKENS = 64
RESERVED_SLOT = 0  # the pools' padding slot: inert dump target


def logical_page_of(position: int) -> int:
    return position // PAGE_TOKENS


def page_token_span(page: int):
    """Token positions [start, end) of one logical page."""
    return page * PAGE_TOKENS, (page + 1) * PAGE_TOKENS


def new_slots_for(page: int, frame: int) -> torch.Tensor:
    """req_to_token values that map a page's tokens onto `frame`."""
    return frame * PAGE_TOKENS + torch.arange(PAGE_TOKENS, dtype=torch.int64)


class PassDriver:
    """One target-verify pass over the shrunk pool.

    begin(selections) computes the complete plan for the pass:
      stage_copies   [(page, frame)]        host -> GPU copies to perform
      evictions      [Eviction]             GPU -> host copies to perform FIRST
      table_rewrites [(row, start, slots)]  req_to_token scatter ops
      pin set        pages protected until end_pass
    """

    def __init__(self, mapper: PageMapper, req_to_token: torch.Tensor):
        self.mapper = mapper
        self.req_to_token = req_to_token  # [num_requests, max_context]

    def begin(self, selections):
        """selections: {request_row: (request_id, [logical_positions])}"""
        plan = {"stage_copies": [], "evictions": [], "table_rewrites": []}
        pinned = []
        for row, (request, positions) in selections.items():
            pages = sorted({logical_page_of(p) for p in positions})
            resident = [page for page in pages
                        if self.mapper.residence(request, page) is Residence.GPU]
            self.mapper.pin(request, resident)
            pinned.extend((request, page) for page in resident)
            for page, frame, evictions in self.mapper.ensure(request, pages):
                plan["stage_copies"].append((page, frame))
                plan["evictions"].extend(evictions)
                start, _ = page_token_span(page)
                plan["table_rewrites"].append(
                    (row, start, new_slots_for(page, frame)))
        plan["pinned"] = pinned
        return plan

    def apply_rewrites(self, plan):
        for row, start, slots in plan["table_rewrites"]:
            self.req_to_token[row, start:start + PAGE_TOKENS] = slots.to(
                self.req_to_token.device)

    def end_pass(self):
        self.mapper.unpin_all()


class EvictionRewriter:
    """Rewrites req_to_token for evicted pages (tokens -> reserved slot).

    Called by the allocator wrap when the mapper evicts a victim: the
    victim's tokens must not resolve to the recycled frame until they are
    staged again.
    """

    def __init__(self, req_to_token: torch.Tensor, row_of=None):
        self.req_to_token = req_to_token
        self.row_of = row_of or (lambda request: request)  # request id == row

    def on_eviction(self, eviction):
        row = self.row_of(eviction.request)
        start, end = page_token_span(eviction.logical_page)
        self.req_to_token[row, start:end] = RESERVED_SLOT


class PrefillDriver:
    """Streaming prefill bookkeeping: allocation + write-through marking.

    After each prefill chunk's writes complete, `chunk_written` records
    which pages became dirty; `mirror_done` marks them clean. The overlay
    performs the actual GPU->host copies between the two calls.
    """

    def __init__(self, mapper: PageMapper):
        self.mapper = mapper
        self.pending_mirror = []

    def chunk_written(self, request, start_pos, num_tokens):
        """Pages touched by one chunk of writes; returns the page list so
        the caller can mirror exactly those spans."""
        pages = []
        for pos in range(start_pos, start_pos + num_tokens):
            page = logical_page_of(pos)
            if page not in pages:
                pages.append(page)
        self.pending_mirror.extend((request, page) for page in pages)
        return pages

    def mirror_done(self, request, pages):
        for page in pages:
            self.mapper.mark_clean(request, page)
