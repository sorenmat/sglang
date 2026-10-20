# A/B benchmark: original sglang vs host-kv tier

One image, two modes (`launch-hostkv.sh MODE=vanilla|tier`). The harness
runs the ladder on a single GPU per mode, interleaved A/B/A/B with
cool-downs, and emits `results.json` + a markdown report.

## Ordering (noise floor FIRST)

1. `vanilla` vs `vanilla` on identical config — the nondeterminism
   baseline (the base server differs from itself at temp=0; output
   equality is NOT a gate).
2. `tier` vs `vanilla` interleaved: for each cell (N, ctx), A then B
   then A (A=vanilla, B=tier), cool-down 30s between runs.
3. Quality gate: forced-continuation logprob comparison (mean KL +
   top-1 flip rate) tier vs vanilla over fixed prompts; acceptance =
   within the step-1 noise floor.
4. Regression: `vanilla` (fork, flag off) vs the vanilla pinned image
   on a subset — proves the fork is a no-op when disabled.

## Metrics

- Per-request steady decode: decode-only windows binned by
  #running-req from the server's decode log lines (no prefill
  contamination), plus client-side tok/s.
- TTFT, wall, p50/p99 per cell.
- Tier counters: stats JSON (verify_mismatches, coverage_misses,
  restore_starved, prefetch hit rate, ws/drift) + /metrics scrape.
