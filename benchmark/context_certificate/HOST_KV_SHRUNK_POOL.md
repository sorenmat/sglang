# HKV-4 design — the genuinely shrunk pool (2026-10-05)

Source-grounded in the pinned commit 4ccff141. Every claim below names its
file. This note replaces the "confiscation" sketch in HOST_KV_DESIGN.md's
P2.3 history (two failed attempts documented there).

## The complete accounting identity

`SchedulerInvariantChecker._check_pool_invariant`
(invariant_checker.py:65) enforces, checked during busy operation
(scheduler.py:1756 → `self_check_during_busy`):

```
available + evictable + protected + session_held + uncached == total
```

with, for our config (hybrid-SSM, radix disabled → ChunkCache):

| Term | Source | We control it? |
| --- | --- | --- |
| `available` | `allocator.available_size()` = `(len(free_pages)+len(release_pages)) × 64` (allocator/base.py:60) | directly (free_pages is a tensor we already discover) |
| `evictable` | tree cache (`0` for ChunkCache) | no |
| `protected` | **tree-cache accessor** `full_protected_size()` (invariant_checker.py:87-99) | **yes — wrap it** (monkey-patch at arm; we already reach `runner.scheduler.tree_cache`) |
| `session_held` | pool_stats_observer (0 for us) | no |
| `uncached` | Σ per running req: `kv_allocated_len − cache_protected_len` (invariant_checker.py:222-256) | **only via `req.kv.kv_allocated_len` mutation** |
| `total` | `allocator.size` (static) | no |

## What this settles

1. **Why confiscation failed** (window: "pool memory leak detected"):
   pages removed from free_pages appear in NO bucket at idle → identity
   fails. **Fix: register the confiscated token count by wrapping
   `tree_cache.protected_size`/`full_protected_size` to add
   `reserve_tokens`.** Then idle: available + protected == total ✓; busy:
   available + uncached + protected == total ✓.

2. **Why naive retirement-freeing fails twice:**
   - Finish path frees TABLE CONTENT — `cache_finished_req` frees
     `req_to_token[row, cache_protected_len : kv_len]` (chunk_cache.py:79-87;
     mamba variant uses `free_segment`, mamba_radix_cache.py:547+). Cleared
     columns (0) free page 0 once per finished request → duplicate page 0
     across requests → allocation corruption (`free()` dedups within a
     call only, paged.py:261-268).
   - `uncached` uses the runtime's `kv_allocated_len`, NOT the table.
     Freeing pages at retirement raises `available` without lowering
     `uncached` → identity over-totals → leak flag.

## The mechanism (v1)

**Arm time** (single overlay change set):
- Confiscate: `free_pages = free_pages[:KEEP_PAGES].clone()`, hold the
  rest as our reserve tensor.
- Wrap the tree cache's protected accessors: `+ reserve_tokens`.
- Wrap `allocator.free/free_segment`: drop any page id == the SINK page
  (see below) before the original call.

**Runtime (retirement = real page return):**
- Retire: re-mirror (done), then `allocator.free(original slots)` —
  pages re-enter circulation for the request's own later chunks and for
  other requests.
- Clear the table columns to the **SINK page's slots** (a reserve page,
  never allocatable), NOT to 0: the finish path then frees the sink once
  per request (in-call `unique` dedups the 64 sink slots to one page),
  and our free-wrapper filters it — net effect: finish frees exactly the
  still-live pages.
- **Decrement the request's uncached**: locate the `Req` via
  `runner.scheduler` (running batch lookup by `req_pool_idx`) and reduce
  `req.kv.kv_allocated_len` by 64 per retired page. This is the one
  runtime-state mutation the design requires; it is also the piece to
  validate first on GPU with the invariant checker as the oracle.

**Staging:** allocate from OUR reserve tensor (not the runtime
allocator), copy mirror→frame, rewrite table columns to the frame slots.
`kv_allocated_len` grows by 64 per staged page — matching the runtime's
own convention — and shrinks again if the page is re-retired. The
invariant stays balanced by construction.

## GPU memory actually freed

The KV pool tensors are allocated at pool size at startup — freeing
PAGES does not shrink the tensor. Real VRAM savings need the pool
LAUNCHED small: `--max-total-tokens = KEEP_PAGES×64 + margin` (e.g.
24,576 tokens), with admission passing long inputs because the tier
answers admission via its own reserve... but admission reads
`max_total_tokens` (the 400 we hit at 96k pool). Therefore the small
launch must ALSO raise the server's admission limit. Options:
`--allow-auto-truncate` is wrong; the right one is overriding
`max_req_input_len`/`context_len` accounting at arm (they derive from
max_total_tokens; patch on the tokenizer-manager side is a second
process — NOT reachable from the backend hook). Conclusion: **small-pool
launch requires either an upstream flag or a scheduler-side env the
tokenizer honors**; until then the measurable claim is "tier-held
fraction" (pages logically ours vs runtime's), with the VRAM delta
demonstrated at G3 using the 520k→24k pool shrink where admission still
passes (total_tokens stays large; only free_pages shrink + the POOL
TENSOR size is what we cannot shrink without the small launch).

## Validation plan (CPU first)

- CPU simulation of the identity under: confiscate, retire-free,
  finish-free (sink filtering), stage-from-reserve — assert the identity
  holds at every step and no duplicate pages ever appear in free_pages.
- GPU oracle: the runtime's own `self_check_during_busy` — if it stays
  silent through a full 167k request on a confiscated pool, the
  accounting is right by the runtime's own standard.
