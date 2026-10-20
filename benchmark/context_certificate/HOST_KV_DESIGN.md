# P2.1 design — host-tier KV for QSATokenToKVPool (pinned runtime 4ccff141)

Grounded in the pinned source (`python/sglang/srt/mem_cache/qsa_kv_pool.py`,
`memory_pool.py`) and the validated Phase-1 capacity model
([capacity-model.json](artifacts/phase1-20260930/capacity-model.json)).
Numbers are measured or model-derived and labeled as such.

## What moves, what stays

The pool already separates two caches with very different access patterns:

| Cache | Size (32×256k, measured model) | Access pattern (measured) | Tier |
| --- | --- | --- | --- |
| Full KV (12 attn layers, BF16) | 192 GiB | gathered per selected block, ~4.8 GB/pass union, 80% hot | **host tier, staged on demand** |
| Compressed index K (12 layers, BF16, 128-dim) | ~1.5 GiB | scanned every step, ~1.2 ms/step, flat in batch | **GPU-resident, never moves** |
| Pre-compression ring + RoPE coords | per-request, small | write-mostly | GPU-resident |
| GDN/mamba states | ~1.8 GiB | fixed-size per request | GPU-resident |

Decision rule: anything read by a per-step scan stays on-GPU (the scan is
1.2 ms and must not pay the link); anything read by *selection* (a sparse,
predictable gather) is host-tier eligible. The full KV is the only cache
that is both large and gathered — it is the tier's entire payload.

## Page model (reuses the existing arithmetic)

- Page size 64 tokens (the runtime's `page_size=64`), = 16 compressed
  groups. **1.57 MB per page** across the 12 layers (64 × 24.6 KB). At the
  measured 56.9 GB/s link one page stages in ~28 µs.
- The existing invariant `compressed_slot = full_slot // 4` requires pages
  to stay contiguous units in BOTH tiers: a host page stages into a
  reserved GPU page slot, never split. Page-granular radix sharing is
  thereby preserved unchanged.
- GPU-resident page set = a *cache* of host pages: identity is
  `(request_token_position // page_size)`; the radix tree keeps its
  sharing semantics because copies are exact.

## Page state machine (the new ownership bookkeeping)

```
HOST  --stage_in-->  IN_FLIGHT  -->  GPU_RESIDENT
GPU_RESIDENT --evict-->  HOST            (write-through: no dirty state)
```

- **Write-through**: prefill/append writes go to the GPU page AND are
  copied to the host page before the page is marked complete (prefill is
  not latency-critical; this removes writeback entirely).
- **Eviction**: GPU pages evict LRU among pages not touched in the current
  pass (the selection set pins pages).
- **Stall semantics**: if attention gathers a page still IN_FLIGHT, the
  layer waits on the transfer event. Correctness gate: outputs bit-identical
  to the all-GPU run (temperature-0 token stream equality).

## Transfer admission

Per-pass byte budget from the Phase-1 model: miss traffic ~484 MB/pass
(8.5 ms serialized at 56.9 GB/s). The controller:

1. **Demand** transfers (pages selected by the current layer) always admit.
2. **Prefetch** transfers (pages predicted from an earlier layer's
   selections; measured coverage 0.65, overfetch 0.35) admit only into the
   remaining budget — overfetch is capped so it can never starve demand.
3. Budget is refilled per verify pass; unused bytes do not carry over.

Phase 2 ships (1) and (2) with prediction disabled (pure demand paging);
Phase 3 enables the predictor. The controller is pure logic, CPU-tested.

## Interfaces to the pinned runtime (integration points, filled in P2.2+)

- Extend `QSATokenToKVPool`: allocate the full-KV buffers as
  `min(pool, hot_pages)` GPU pages + a pinned host pool of the full size;
  `get_kv_size_bytes` accounting unchanged for the GPU part.
- The gather path (`qwen_sparse_attn_backend`'s paged MQA attention)
  resolves slots through a page table that now returns GPU slots for
  staged pages; the host tier stages *before* the gather, per layer.
- `cudaMemcpyAsync` page staging on a side stream with events; PLE pinned
  backend (`ple_offload_backend='pinned'`) already proves pinned-host
  access from kernels on this runtime.
- EP2: pages are per-rank private (attention is data-parallel per request
  under EP2 MoE); no cross-rank page traffic. The EP2 NCCL init failure is
  fixed first (Phase-2 item zero) — shard 42.7 GB/GPU measured.

## Memory budget (Phase-1 model, per GPU)

GPU: 42.7 weights + ~2.0 hot KV + 1.5 index + 1.8 GDN + 4 workspace ≈ 53 GiB
of 94.2. Host: 192 KV + 6 index backup + 95.4 PLE + 16.5 draft ≈ 310 GiB —
needs the ≥352 GB-RAM machine. Staged-page GPU cache size is a tunable
(start: hot-5120-tokens-per-request ≈ 2 GiB).

## Test plan

- CPU unit tests (now): state-machine invariants, admission arithmetic,
  LRU eviction, page/compressed alignment, demand-over-prefetch priority.
- P2.2 gate: one 256k request, partially host-resident, token stream
  bit-identical to the all-GPU run; step time ≤ 1.3× resident.
- P2.3 gate: N concurrent, regression bounded by the Phase-1 prediction.
- P2.4 gate: 32×256k on the end-state box vs the 37.5 ms/pass budget;
  pipelined 29.8 ms model-derived target.

## Non-goals / explicit risks

- No dtype changes anywhere (bit-identical outputs).
- No multi-node, no disaggregation interplay, no radix-tree changes.
- Unmeasured: PCIe P2P for EP dispatch (assumed 25 GB/s, sensitivity 15);
  clean N=16 anchor; real route distributions. All listed for window 1.


## P2.3 — the shrunk-pool configuration (design, 2026-10-02)

v0 proved the mechanism with a full-size pool and in-place eviction. P2.3
is where the memory savings become real: the GPU full-KV pool SHRINKS to a
frame budget (start: ~1,300 frames = 2 GiB ≈ 83k tokens per card), pages
become load-bearing, and eviction demotes real data to the host mirror.

### The brain: [host_kv/multipager.py](host_kv/multipager.py) (CPU-tested)

`PageMapper` owns `(request, logical_page) <-> frame` for up to 16
requests sharing one card's frames:

- **Streaming prefill:** a 262k-token prefill against 1,300 frames
  allocates page-by-page; each `alloc` returns the frame plus the
  `Eviction` plan (frame→host copy) the caller must run first. A full
  prefill writes every page once and leaves only the tail resident —
  the write-through contract is `mark_clean` after mirroring.
- **Demand staging:** `ensure(request, selection_pages)` plans the
  host→frame copies for this pass's gather; resident pages are touched
  only. The caller rewrites `req_to_token` rows for staged pages.
- **Global LRU with pinning:** the victim is the least-recently-used page
  across ALL requests (per-request iteration would thrash the first
  request in the table — caught by a unit test); the current pass's
  selection set is pinned and can never be a victim.
- **No frame aliasing, ever:** one page of one request per frame; frames
  free at eviction-plan time with an explicit ordering contract.

### Runtime integration points (upgrade of the v0 overlay)

1. **Allocator wrap:** the pool's paged allocator hands out frames via
   `PageMapper.alloc`; `close_request` on finish/abort; free-list owned
   by the mapper, not the allocator.
2. **Write path:** `set_kv_buffer` per layer writes the frame AND (lazily,
   at eviction) the host copy — dirty tracking until `mark_clean`.
3. **Gather hook (v0's `forward_extend` hook, upgraded):** per pass,
   selection set → `pin` current + `ensure` missing → perform copies →
   rewrite `req_to_token` rows for staged pages → unpin-all after the
   pass. The runtime's `token_slot_table` is rebuilt from `req_to_token`
   per batch, so staged pages appear at their frames naturally.
4. **Compressed index K stays GPU-resident** (unchanged from the design:
   it is scanned every step, ~1.5 GiB).

### Gates

- G1 (one request): 262k prefill + decode on the shrunk pool; outputs
  semantically coherent; every staged page byte-verified (v0's check).
- G2 (16 requests): concurrency with the shared frame budget; eviction
  storm bounded (admission from the P2.1 controller bounds prefetch;
  demand always admits); per-pass link bytes within the Phase-1 budget
  (~484 MB/pass serialized worst case).
- G3 (two servers, one endpoint): 32 users, measured tok/s per request
  vs the 80 target; model-derived expectation 30.3 ms/pass per card.
