# HKV-6 deployment and measurement recipe

Tier changes (this commit), all in `host_kv/hosttier23m.py`, validated by
`test_hosttier23m_shrunk.py::ModeFreezeAndPrefetch` (CPU fixtures):

1. **Freeze by forward_mode** — `after_write` freezes every row of a
   decode/target-verify batch. The delta<PAGE heuristic never fired in
   the dev-qwen38-next-local build (G3: zero freezes, 5,038 re-staged
   pages, 6.4 tok/s decode). This is the ping-pong killer.
2. **One sync per gather** — restores issue all copies on the tier
   stream and synchronize once; the previous loop synced per page.
3. **Neighbor prefetch (opt-in)** — `SGLANG_HOSTTIER_PREFETCH=1` stages
   the ±`PREFETCH_RADIUS` (default 1) neighbors of each step's
   selections into reserve frames asynchronously, leaving the table
   untouched; a later gather that selects such a page only rewrites its
   table columns (sync-before-publish guarantees the copy completed).
   Bounded by `PREFETCH_FLOOR` (default 128) reserve pages kept free
   for synchronous restores. Counters: `prefetch_issued`,
   `prefetch_hits`.

## Launcher deltas vs the G3 scripts (diag-gate.sh / g3b-only.sh)

    -e SGLANG_HOSTTIER_KEEP_PAGES=2510 \   # budget 160,640 tokens
    -e SGLANG_HOSTTIER_NO_RADIX=1 \
    -e SGLANG_HOSTTIER_PREFETCH=1 \
    ... sglang serve ... --disable-radix-cache ... (rest unchanged)

The overlay to mount is the rebuilt `artifacts/hosttier-overlay-*/`
(this build carries the G3 five-fix tier plus HKV-6).

## Protocol on a 2x RTX PRO 6000 box

1. **Baseline**: single `volley.py 1 295000` with `PREFETCH=0`
   (env override at launch). Record: wall, decode gen-throughput lines,
   `staged_pages`, absence of "retire frozen by mode".
2. **HKV-6 on**: relaunch with `PREFETCH=1`, same volley. Expect
   "retire frozen by mode at ctx=..." lines, `staged_pages` far below
   baseline, `prefetch_hits > 0`, higher steady decode tok/s.
3. **Concurrency ladder**: 4 x volley, then the 32-user sender
   (16+16 across the two servers, `G3_CTX_TOKENS=160000` — prompts land
   at ~101k tokens; true-160k needs ~251k but leaves only ~500 tokens
   of admission slack).
4. Gates: `verify_mismatches == 0`, no scheduler exceptions, full pool
   recovery between requests (`watch#` lines: free_pages back to 2,510).

Success target for this iteration: per-request steady decode >= 30
tok/s (from 6.4). The 80 tok/s target additionally needs decode CUDA
graphs (currently disabled for the tier) and/or multi-request overlap
of restores with compute.
