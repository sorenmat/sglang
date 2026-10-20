# HKV-12 final validation — 2026-10-20, instance-20261020-hkv12b (us-central1-c)

**The between-replay tier works with decode CUDA graphs: 179.2 tok/s
single-request steady decode with the tier LIVE (vs 38.1 eager+tier, 219
graphs+dormant-tier - an ~18% tier cost at N=1), full-length completions,
verify_mismatches=0, coverage 99.99% (345 misses / 39,756 ticks / 3
volleys). True-160k validated on eager (159,907 tokens, forced 512-decode,
46.8s). The strict idle checker passes with ZERO leak lines.**

## What it took (all committed)

1. Capture-mode guard was necessary but the deployed bundle carried a
   STALE wrapper - the recon's success came from a separately-shipped
   guarded copy; refreshed the bundle (hosttier.py with the guard).
2. The scheduler tick's stash resolution, proven by boot-time BFS:
   model_worker._draft_worker.draft_extend_attn_backend.full_attn_backend
   (model_worker IS EAGLEWorkerV2 under spec; tp_worker is the plain
   target worker; the earlier candidate paths all miss).
3. The tick must EXEMPT the current selection from re-retire: selections
   live below the horizon by construction; without the exemption the
   same tick retires what it restores (116k coverage misses, zeroed
   replays).
4. Reserve sizing is the concurrency lever: 615 frames starved instantly
   (1.57M starved restores); KEEP_PAGES=1600 (budget 102,400 - admits
   101k inputs with 781 tokens slack) grows the reserve to 1,525 frames
   and starvation went to zero at N=1 (but returns at N=8: 8 rows x
   ~350-page working sets exceed 1,525 - follow-up ticket).

## Full-stack ladder (101k contexts, 1024 forced-length, graphs+tick)

- N=1: steady 179.2 tok/s/request
- N=4: wall 67.1s, all full-length (interval sampling running=1-dominated
  from the admission cascade)
- N=8: wall 113.9s, all full-length
- Session totals: br_ticks 39,756, coverage_misses 345 (0.87%/tick the
  other way: 99.99% clean), verify 0, prefetch 100% hits.

## HKV-13 closed

160k eager run with SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1 (armed):
ZERO pool-leak lines at idle; counters: clamp_skips=0, dedup_dropped=0,
ff_dropped_reserve=0 across 4,887 tier frees and 162,814 finish-path
slot frees. The floor-clamp + restore-cycling fixes closed the leak.

## HKV-16 closed

True-160k: input 159,907 tokens (volley 296000 at 4.996 chars/token),
admitted under KEEP_PAGES=2510 (budget 160,640), forced 512-token decode,
wall 46.8s, coverage 11 misses / 24,569 stagings (99.96%), verify 0.
256k remains beyond max_total_tokens=200k (pool resize or residency-based
admission needed - the HKV-11 line of work).

## Follow-ups (new tickets)

- HKV-17: reserve sizing for N-way overlap (measure per-row working-set
  width; 8 rows starve a 1,525-frame reserve).
- HKV-18: residual coverage misses (0.005-0.009/tick): lag-tolerance
  (wider prefetch radius) or hybrid first-step eager verify.
