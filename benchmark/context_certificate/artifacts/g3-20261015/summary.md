# G3 window — 2026-10-15 (ecuser@213.171.186.59, 2× RTX PRO 6000, 235 GB)

Status: **G3 did not run: the tier overlay wedges the scheduler on this
launch configuration — isolated by A/B to our code, not the host.** GPUs
released (0 MiB). All other queue items landed (see below).

## What the window achieved first

- **HKV-9 model fold-in (item 1, done before the runs)**: the measured
  512-expert curve replaced the analytic union in phase1_model.py
  (commit 0a77b71b1e). Validation tightened to 1.025; 32×256k prediction
  now 28.6 ms pipelined / 37.1 serialized vs the 37.5 ms budget.
- **N=16 profile (item 2, partial win)**: plain `num_steps` profiling
  FIXED the trace flush (three prior windows of silent failures) — a
  trace file landed and was captured+renamed by the HKV-8 detection.
  The 600-step default window I set then landed in a 58-event idle
  stretch (arm timing still needs one line: arm after decode begins).

## The G3 blocker, precisely

Both tier servers (identical config: TP1 each, SHRUNK=1 KEEP_PAGES=1500,
perf-mode flags, max-running 16, mamba 32) wedge after warmup:

- `sglang::scheduler` pegs at **100% CPU** (`RLl`, 30+ min of CPU time),
  GPU 0%, no batches ever scheduled for real requests.
- HTTP GETs (`/v1/models`) keep working; POSTs connect (ESTABLISHED
  through docker-proxy) and hang forever — from inside the container too.
- Reproduces across two clean container restarts.
- **A/B: a plain server (identical flags, no overlay) on the same box,
  same GPU, immediately after — tiny POST 1.8 s, 30k-token POST 4.7 s.**

Leading suspect: the arm-time `allocator.size` shrink (to 96k tokens)
under max-running 16 — the scheduler's pool-budget/eviction loop likely
spins trying to satisfy a budget it can never reach once the radix cache
holds entries. NOTE: the HKV-4 gate passed the identical shrink at
max-running 1 on the .53 box with the oracle silent; the difference is
this launch's concurrency/cache regime. Next session's first move:
attach py-spy (or SIGQUIT) to the spinning scheduler for the exact
stack, then fix the budget interaction — one focused cycle.

Also documented: my 262k-target sender initially caused 32× clean 400
rejections (admission: 216,635 > 199,994) — the sizing note "32 users
needs the 352 GB box" stands; 32×~138k fits THIS box's 235 GB.

Artifacts: sender logs, g3b-docker.log (full), launch script.


## Follow-up session (same day): the wedge DEBUNKED, G3 armed, host lost

py-spy (after relaxing ptrace_scope on the host) settled everything:

- The 100%-CPU scheduler is **normal idle busy-poll** (`on_idle →
  get_pool_stats`), present in plain servers too — it was never the bug.
- The tier config **passed every probe on this very box**: tiny POST
  1.9 s, 30k-token POST 7.4 s, an 8x oversized-rejection storm followed
  by a 0.2 s canary. The earlier "wedge" was host networking (the same
  provider family's boxes showed multi-minute SSH outages all session).
- The original 32x400s were MY sender bug: a 3.5 chars/token estimate
  vs the real ~2.7, making every '167k-target' request ~216k tokens —
  correctly rejected against the 199,994 admission limit. The sender is
  now calibrated (2.7) and port-corrected (30030/30011) locally.

The final armed state: diag (GPU 0, port 30030) and g3b (GPU 1, port
30011), both tier-enabled and verified answering — with the calibrated
32-user sender ready. The host then entered a 30+ minute total outage
before the run could fire, and stayed down. **G3 remains one command
away on a stable box**: `G3_CTX_TOKENS=160000 python3 g3-send.py`.


## G3 VERDICT (2026-10-15, ecuser@213.171.186.59, 2x RTX PRO 6000)

**Run: 32/32 requests completed, bitwise-correct, wall 316 s. Decode rate
~6.4 tok/s per request steady (transient peaks 56-59) vs the 80 tok/s
target. Verdict: correctness PASS, performance FAIL by ~12.5x, with the
bottleneck measured and specific.**

Setup: two TP1 servers (KEEP_PAGES=2510 -> shrunk budget 160,640 tokens,
reserve 615 pages; --disable-radix-cache; tier P23M with ASYNC copies,
VERIFY_SAMPLE=8, HOT_PAGES=64), sender 16+16, prompts 101,435 tokens
each (corpus text measures 4.24 chars/token, not the 2.7 assumed), 256
output tokens, MTP 3/1/4 (accept len 2.8-3.0).

### The five defects fixed to get here (tier/config)

1. KEEP_PAGES semantics: allocator.size = KEEP_PAGES*PAGE is the KEPT
   budget, not the confiscation. 1500 -> 96k budget: every >96k request
   queued forever at the FCFS head (the original "wedge"; the scheduler's
   own comment at the max_new_tokens clamp warns of exactly this).
   2510 -> 160,640 admits 101k-160k requests.
2. Radix evictable claims vs tier confiscation diverge -> prefill adder
   over-committed (usage 1.12) -> retract storm -> new_token_ratio 0 ->
   catatonia. --disable-radix-cache collapses the accounting to plain
   free.
3. returned_tokens never reset without radix reconciliation -> available
   5,376 of 160,640 after the first finish -> permanent FCFS block.
4. Finish re-frees retired pages -> double-listed frames -> invariant
   checker kills the scheduler (available 168,768 > total 160,640).
   Fixed by deduping frees against live free_pages and the grown
   reserve (re-retired frames join the reserve dynamically).
5. Recycled rows kept stale ctx_written (159,373 + 106,490 = 265,863,
   past the 262,144 table width) -> CUDA device-side assert mid-prefill.
   Fixed with a total-ctx staleness check against extend_seq_lens.

Validation after the fixes: 3 sequential + 4 concurrent 159k requests
with full pool recovery between requests (free_pages 2510 / avail
160,640), then the 32-user volley clean end to end.

### The performance result and its cause

- Requests serialize through the shrunk pool: each prefill fills to
  ~99%, retirement streams it back, the next request prefills. Decode
  batches ran with #running-req: 1 throughout.
- Per-request decode: ~6.4 tok/s steady at 101k ctx. The tier staged
  5,038 (GPU0) / 3,047 (GPU1) page restores across the volley: every
  step's top-k selections reach past the 64-page hot horizon into
  retired pages; each restore is K+V copies x 12 layers, and the
  retire-freeze never fires in this build (the decode hook signature
  differs), so restored pages re-retire next step: ping-pong.
- Eager decode (cuda graphs off for the tier) plus MTP accept ~3 puts
  the no-tier ceiling near 25-30 tok/s; the restore traffic takes the
  measured rate to 6.4.

### What the 80 tok/s target needs (HKV-6, designed not built)

- Prefetch: restore the next step's likely selections asynchronously
  instead of synchronously per step (the phase-1 model's 28.6 ms
  pipelined vs 37.1 ms serialized prediction is exactly this).
- The decode-path retire-freeze actually firing (hook signature for
  this build), or a decode working set larger than the selection
  horizon.
- True-160k contexts need G3_CTX_TOKENS ~= 251k (4.24 chars/token); the
  budget admits them (160,000 <= 160,575) with ~500 tokens of slack, so
  keep max_tokens small.

Artifacts: /home/ecuser/work/g3-results-live.json on the host, docker
logs on both containers, tier stats (staged/verify counters) in
/tmp/stats-*.json on the host. Tier fixes committed as 44f8680a2c.
