# HKV-17/18 — 2026-10-20, instance-20261020-hkv17 (asia-south2-c after zone sweep)

**HKV-18 CLOSED: the residual lag misses were entirely recent-band - the
hot horizon was one MTP-iteration's drift too narrow. HOT_PAGES=64 ->
256 (16k tokens never retired) eliminated them: miss_recent = 0 across
N=1, 4 and 8 (was 34 at N=1, 276 cumulative before), verify 0,
full-length completions throughout, steady decode 144.9 tok/s at N=1.**

**HKV-17 CLOSED with the measured formula and a structural finding.**
Per-row decode working set: 27.7 pages at N=1, 46-50 mixed 1/4/8-way
(~1.8-3.2k tokens); inter-tick drift ~0-0.8 pages (selections are
nearly static - consistent with the model's own MTP index sharing).
Decode-side reserve need is tiny: N_max x 60 pages suffices. The real
pressure is PREFILL restore bursts: each eager chunk-gather selects
~440 pages (the chunk attends over the whole prefix), and concurrent
prefill+decode drains the reserve - restore_starved 831k-1.1M under
N=4/8 vs 0 at N=1. Starved prefill restores surface as the deep-band
coverage misses (511 cumulative / 176k ticks ~= 0.3%; outputs remain
full-length and coherent; verify_mismatches=0).

Formula: reserve >= 500 x N_prefill_concurrent + 60 x N_decode +
margin. Structural constraint: at 101k contexts the admission budget
(KEEP_PAGES x 64 >= input + max_new + page) caps the reserve at ~1,525
pages - already the deployed value - so eliminating prefill-burst
starvation at N>=4 requires either residency-based admission (input no
longer pool-reserved; the HKV-11 line) or quality-tolerant prefill
(deep selections read zeroed pages, quantified at 0.3%). Both are
documented; neither blocks current operation.

## Session mechanics

Spot capacity was dry in all familiar zones; a sweep of every
g4-standard-96 zone found asia-south2-c (one europe-west2-b allocation
was reclaimed mid-provisioning and deleted). The docker bind-mount
directory trap bit hosttier23m.py once (silently no-op'ing the tier -
caught by zero tier log lines); the pre-launch `file` check is now
routine. Stats survive SIGKILLed stops via periodic watchdog dumps
(every 20s).
