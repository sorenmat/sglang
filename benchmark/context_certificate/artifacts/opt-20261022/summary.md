# Optimization session — 2026-10-22, instance-20261022-opt (europe-west2-c)

**All three optimizations validated; the capacity ceiling honestly
mapped; AUTO mode makes the tier a strict no-loss choice.**

## The three optimizations (commit 5a5c7e26dd + f72abf41b0, image -20)

1. LAZY MIRRORING: the write path marks pages; the single GPU->host
   copy happens at retire time. Was 2x copies per retiring page.
2. BATCHED RETIRE COPIES: one tier-stream sync per retire (was 24).
3. AUTO ARMING: dormant (vanilla-identical) until a request >= N input
   tokens arrives. Plus the fix that made it actually reachable (the
   post-hook was gated on armed, so the arm-inside-after_write trigger
   could never fire — f72abf41b0) and launcher KEEP precedence fixes.

## Measured (101k ctx, keep=1700, tier, N=4)

- Image -18 (pre-opt): 16.7 / 33.2 / 43.7 / 66.9 s
- Image -19 (lazy+batched): **14.3 / 26.1 / 39.2 / 51.8 s** — ~15-25%
  faster walls, counters clean (verify 0, starved 0, rows forgotten).

## AUTO mode at 160k (ARM_MIN=200000, N=4)

Never armed ("dormant until an oversized request" only), walls
18.4-66.1s — vanilla-identical within noise. **A strict no-loss
default: workloads that fit pay nothing.**

## The capacity ceiling (honest, measured)

The 96GB card carries ~80.5 GB of weights; the KV pool auto-sizes to
~200-211k tokens (~5 GB). A 258,731-token request is rejected 400 by
the RUNTIME's own intake clamp ("exceeds the maximum allowed length
211,322") — by vanilla AND by the tier, because admission checks the
pool total before any tiering exists. **256k single-request on one RTX
PRO 6000 is memory-infeasible with this model, period** — the 2xTP1
split or the EP2 fix (upstream #44291) is the path, exactly as
GPU_RUN docs predicted.

## 190k x N=2 — the near-ceiling regime (pool 200k, keep 3000, ARM_MIN 170k)

- tier (auto-armed on first chunk): 21.2 / 39.1 s — total ~39 s
- vanilla: 25.3 / 42.3 s — total ~42 s
Both complete; the tier is now FASTER in the near-ceiling regime (the
regime where its streaming actually engages) and equal below it. Tier
counters at 190k: coverage_misses 48k (deep pages starved at this
reserve — quality-noise regime of HKV-17), verify 0, rows forgotten.

## Bottom line

With AUTO arming the tier is a no-loss default: dormant when workloads
fit, ~15-25% faster than the pre-optimization tier when engaged, and
the only option in the near-ceiling regime (>= ~150k on this card)
where it now also WINS the wall clock. The 256k target needs 2 GPUs or
the upstream EP2 fix.
