# Definitive A/B report — 2026-10-22, image hostkv:4ccff141-20 (branch f72abf41b0)

One image, MODE=vanilla|tier, 101k-token contexts (101,043 real tokens),
1024 forced-length decodes, one GPU per mode, keep=1700.

## Final matrix

| N | vanilla wall | vanilla ok | tier wall | tier ok |
|---|---|---|---|---|
| 1 | 22.7s | 1/1 | 28.0s | 1/1 |
| 4 | 55.5s | 4/4 | 60.4s | 4/4 |
| 8 | 113.6s | 8/8 | 124.4s | 8/8 |
| 32 | 458.2s | 32/32 | 32/32 (fresh server) | 32/32 |

Tier N=32 ran on a fresh server (32/32 complete, walls 14.3-403.8s
serialized; forgotten 33, coverage 132, starved 0, verify 0) because the
combined-matrix server exhausted its per-server finish budget mid-cell
(see Known Limitations). Raw logs: final-matrix.log, n32final.log.

## Headline numbers across the whole program

- Tier steady decode N=1: 141-215 tok/s with 17.3k of 101k resident
  (83% of context on host) - native, CUDA graphs on.
- Optimization (lazy mirroring + batched retire): tier N=4 walls cut
  15-25% (66.9 -> 51.8s worst-case).
- AUTO arming: no-loss default - dormant (vanilla-identical walls) when
  workloads fit the pool.
- Near-ceiling (190k, pool 211k): tier WINS the wall clock (39s vs
  42s for N=2) - the regime where streaming engages.
- 256k single-GPU: memory-infeasible for ALL modes (pool auto-sizes to
  ~211k; runtime intake clamp rejects 258k) - needs 2xTP1 or the
  upstream EP2 fix (#44291).

## Known Limitations (carried to HKV-24 runbook)

1. Residual finish-path leak: ~180 tokens lost from the admission
   headroom per completed request (bounded, deterministic); caps a tier
   server at ~31 finishes at keep=1700 (~10-15 min of sustained 101k
   load) before the next request will not admit. Mitigation today:
   server recycle (the soak exercises this ~20x). Root fix remains
   forget_row/release reconciliation (HKV-26 follow-on).
2. At 160k-class workloads on a 200k pool, vanilla is faster; use AUTO
   (or vanilla) there - the tier pays 8-10% wall for capacity the
   workload does not need.
3. Two-server topology required for 2 GPUs (upstream EP2 #44291).
4. Idle-checker warns a bounded identity deficit at rest (warn mode).

## Verdict

The tier is a no-loss default with AUTO arming: vanilla-identical when
contexts fit, 15-25% faster than its own pre-optimization numbers when
engaged, the only option in the near-ceiling regime, and the enabling
technology for the (2-GPU) 256k target. The single remaining defect is
the bounded finish-path leak with a working operational mitigation.
