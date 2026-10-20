# HKV-12 reconnaissance — 2026-10-19, instance-20261019-hkv12

**Three decisive measurements: (1) eager + tier aggregate ceiling is
~82 tok/s per GPU regardless of overlap width; (2) decode CUDA graphs
capture SUCCEEDS with the tier mounted (capture-mode guard) and lift the
ceiling to 475-730 tok/s aggregate with per-request 60-219 tok/s - the
80/request target is feasible WITH graphs; (3) the tier cannot run
inside replays (Python is skipped), so HKV-12's build is between-replay
scheduling hooks with lag-one-step selection-driven restores.**

## Steady-state overlap ladder (101k contexts, 1024 forced-length tokens)

Eager + live tier (per GPU):
- running=1: 38.1 tok/s/request
- running=2: 72.6 aggregate (36.3/req)
- running=4: 82 aggregate (~20/req)
- running=5: 83 aggregate (16.6/req)
Aggregate saturates ~82: the GPU is step-time-bound under eager decode
with the tier's per-step hooks; more overlap divides the same ceiling.

Graphs + dormant-during-decode tier (capture guard; decode replays run
no Python, so restores/retirement pause during decode - speed recon
only, not correctness):
- running=1: 219.5 tok/s (5.8x eager+tier)
- running=3: 470 aggregate (157/req)
- running=4: 475 aggregate (119/req)
- running=7: 725-730 aggregate (~104/req)
- 16-way volley: wall 211.3s for 16x1024 tokens (~78 tok/s/request
  average INCLUDING the admission cascade), aggregates 486-730 in
  mixed windows.

## Feasibility arithmetic

32 users x 80 tok/s = 2,560 tok/s per GPU. Eager+tier ceiling ~82
(31x short). Graphs ceiling measured 730 at running=7 and still
climbing with N; per-request holds >100 up to 7-way and ~60-100 at
8-16-way mixed. Graphs are NECESSARY and largely sufficient on raw
speed; the remaining engineering is making the tier ride along.

## The HKV-12 build definition (next session)

1. Capture-mode guard already lands (commit ba06032e0c): tier stands
   down during capture; replays skip Python by nature.
2. Between-replay tier: a scheduler-step-level hook (scheduler.py
   overlay) that, after each replay, reads the step's topk selection
   buffer (the graph's output tensors are GPU-readable between
   replays), restores those pages for the NEXT step, and re-retires.
   Selections drift slowly - the prefetch hit rate measured 100% under
   load, so lag-one-step restores should cover; first-step and abrupt
   shifts need the audit (also moved scheduler-side) to quantify.
3. Retirement during prefill stays on the current eager path unchanged.

Config note: launch-graphs.sh (bundle) drops --disable-cuda-graph while
keeping prefill graphs disabled; capture took ~10s for draft
decode/extend graphs at bs<=16 with no errors.
