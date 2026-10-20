# HKV-6 measurement — 2026-10-19 (GCP instance-20261019-065736, 2x RTX PRO 6000)

**Result: per-request steady decode 50.4 / 50.7 tok/s (two servers) at
~101k-token contexts with the host tier active and zero verify
mismatches — 7.9x the G3 rate (6.4 tok/s) and at the measured no-tier
eager ceiling (44-57). The restore overhead is eliminated; the 80 tok/s
target now requires decode CUDA graphs, not tier work.**

## What was measured

32-user volley (16+16 across two TP1 servers, corpus prompts at 101,435
tokens, 256 output tokens): 32/32 OK, wall 334 s. Steady decode
intervals (excluding first-interval logging artifacts): 50.4 tok/s avg
(GPU0, 16 intervals) / 50.7 (GPU1, 12). GPU0 staged 9,114 page restores
with verify_mismatches=0; prefetch issued 2,723 / hit 2,135 (78%).

Single-request A/B at 159,367 tokens (512 forced-length decode):
- no-prefetch (batched restores only): steady 47.1/47.0/44.3 tok/s
- prefetch on: steady 51.7/42.1/42.0/32.6 tok/s
- all-resident (retirement accidentally disabled): 44-57 tok/s
Batching alone recovers the ceiling at N=1; prefetch pays off at
concurrency (78% hit rate in the 32-user run).

## What fixed the 6.4

1. One stream sync per gather (was per page): the dominant cost.
2. Neighbor prefetch into reserve frames with a table-rewrite-only fast
   path (opt-in SGLANG_HOSTTIER_PREFETCH=1).
3. Re-retire returns frames to the DYNAMIC reserve (the static
   tier_pages snapshot missed grown-reserve frames; _dedup_free then
   dropped them entirely -> 5-page/320-token idle leak -> invariant
   checker kill after the 4-user volley).
4. Wrapper pool guard: draft-backend forwards (single-layer pool) no
   longer reach the tier hooks.

## What did NOT work: decode-onset detection

Every mode/shape heuristic misfires in this build because the MTP draft
and target-verify produce decode-shaped batches during chunked prefill
(12-call streaks of 1-6 token writes at ctx 4-14 and 3719). Four rules
tried and rejected: delta<PAGE, mode+same-ctx, mode+growth,
mode+streak-of-8. Shipped with FREEZE_STREAK (env) set to disable
freezing entirely: retirement runs during decode, restores re-retire,
but with one sync per gather the ping-pong costs ~nothing. The
retire-freeze remains future work; the likely correct signal is the
request's input length from a scheduler-side hook.

## Config

KEEP_PAGES=2510 (budget 160,640 tokens, reserve 615 pages),
NO_RADIX=1, --disable-radix-cache, HOT_PAGES=64, ASYNC=1,
VERIFY_SAMPLE=8, PREFETCH=1, FREEZE_STREAK=999999 (freeze off),
MTP 3/1/4 (accept ~2.4-3.0).

## Standing conclusion

At the eager (no-CUDA-graph) ceiling the tier is now free; the path to
80 tok/s is decode CUDA graphs (hooks must move outside the captured
region — between-replay table rewrites are graph-compatible) and/or
multi-request restore/compute overlap.
