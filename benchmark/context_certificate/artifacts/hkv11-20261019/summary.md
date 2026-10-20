# HKV-11 session — 2026-10-19, instance-20261019-hkv11

**Simultaneous decode ACHIEVED for the first time (#running-req: 2,
zero coverage misses under real overlap, pool at ~160.5k available), by
fixing the tier bug that had silently disabled retirement during decode
all along. The remaining blocker is now precisely measured: live
per-step retire/restore machinery costs ~2 tok/s (vs ~50 when retirement
is idle) because decode selections roam the entire context every step.**

## The chain of findings

1. The admission-diagnosis overlay (schedule_policy.py logging refusals)
   showed the 2nd request needs total=101,619 while rem_total sat at
   71-99k during request 1: the active request held ~89k tokens resident.
2. Tier chunk logs showed row state at ctx=3,879/retired=0 DURING decode
   of a 101k request: the row state was being wiped every step.
3. Root cause A: my G3-era staleness reset validated against
   extend_seq_lens, which THIS build fills with the batch's own delta
   (8192 per prefill chunk, 4-6 per verify step) on single-row paths -
   so prefill reset the row every OTHER chunk (chunk #1, ctx=8192
   repeating) and decode reset it every step. Fixed by using
   forward_batch.seq_lens (the per-row running total; also fixes the
   G3-era "totals" comment which was only true for multi-row batches).
   mode=1 (decode) is reported even for 8192-token prefill chunks -
   mode-based logic is dead in this build.
4. Root cause B (the speed): with retirement live, every decode step
   restores selected pages (12 layers x K+V H2D) AND re-retires them
   with a full D2H re-mirror - the mirror copy is pure waste for
   just-restored pages (restored-set skip added, one redundant sync per
   fast-path publish remains). Even so: ~2 tok/s. The ping-pong cost
   HKV-6 measured as "free" was free only because retirement was
   accidentally dead during decode in those runs.

## Measured states (2x101k volley, min_tokens=400)

- Retirement dead during decode (the old bug): 50 tok/s steady, but
  resident ~89k -> no concurrent admission, requests serialize.
- Retirement live (fixed): resident ~small, avail ~160.5k, req 2 admits
  mid-decode, #running-req: 2 observed, 0 coverage misses - but decode
  ~2 tok/s from per-step Python retire/restore churn (~4,200 staged
  pages per request earlier; selections genuinely roam the full
  context).

## What the fix needs (next session, HKV-11 proper)

The per-step cycle must leave the Python hook path entirely: bulk
per-STEP (not per-page) retire+restore batches, single stream sync per
step in each direction, ultimately scheduler-side state where the whole
working set flips once at decode onset (the freeze HKV-14 describes -
with residency-aware admission from HKV-11, the frozen working set can
grow without blocking others). The 'stop' finish at 266 tokens (below
min_tokens 400) in slow runs is unresolved - check min_tokens mapping
vs retraction aborts.

HKV-13 progress: floor-clamp + categorized counters landed (leak 5 -> 3
pages, no dedup drops, no reserve hits at finish); the tail needs
scheduler-side state.

Commits: 738dc051ca (overlay+counters), this session's tier fixes
(seq_lens totals, restored-set mirror skip, staleness guard).
