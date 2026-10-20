# Host-KV productionization plan — native fork + A/B benchmark

Goal: turn the overlay experiments into a **native, flag-gated feature of
an sglang fork branch**, CI-tested, and benchmarkable as *original sglang
vs ours* from one build (flag off vs on) plus a vanilla-image reference
run proving flag-off == vanilla. Two exit states:

- **P1 (benchmark-ready)**: fork branch builds one Docker image; A/B
  harness produces a machine-readable report; quality gate has a defined
  noise floor.
- **P2 (production bar)**: P1 + soak passing + runbook + explicit
  known-limitations list. Upstream PRs are a follow-up program, not in
  scope here.

## Grounding facts (measured, this repo)

- The pinned runtime is commit `4ccff141db` (the only lineage with
  Qwen3.8-Flash-Next / QSA support). The working tree's `python/` is a
  DIFFERENT sglang lineage (2,279 files differ; no `qsa/`). The fork
  branch MUST be cut at `4ccff141db`; no rebase to newer upstream until
  P1 lands (model-support risk).
- Everything proven so far (correctness, 179 tok/s full-stack,
  between-replay tick, hot-horizon 256) lives in overlay files:
  `host_kv/hosttier23m.py` (tier), `hosttier.py` (wrapper), a
  scheduler.py replacement, and a schedule_policy.py diagnosis overlay,
  bind-mounted into image `lmsysorg/sglang:dev-qwen38-next-local`.
- 18 CPU tests (fake fixtures) cover tier internals; zero tests run
  against the real sglang module graph.
- Known quality noise: ~0.3% of decode ticks read zeroed deep-context
  pages under concurrent prefill (outputs coherent; verify_mismatches=0).
  The base server is nondeterministic at temp=0, so output-equality is
  NOT a gate; distribution-level comparison is.

## Phase 0 — branch and cleanup (half day)

- `git checkout -b host-kv/main 4ccff141db` in the sglang clone; bring
  over `benchmark/context_certificate/` (kept out of the build) from the
  current lineage.
- Push branch to the `optimized` remote. Record the lineage decision in
  the plan (above) — future sessions must not rebase casually.

## Phase 1 — de-overlay: tier as native code (2-3 days)

1. **Module**: `hosttier23m.py` →
   `python/sglang/srt/layers/attention/qsa/host_kv.py`. Drop the legacy
   P23/P23 modes (P23M semantics only); real imports; the G/R globals
   become a `HostKVTier` instance owned by the backend.
2. **Config**: new server-args group
   (`--host-kv-tier`, `-keep-pages`, `-hot-pages`, `-prefetch-radius`,
   `-reserve-floor`, `-verify-sample`) in the arg_groups framework,
   plumbed via a dataclass through runtime context. Env vars die.
3. **Backend**: delete the probe_install monkey-patch; direct guarded
   calls at the three QSA forward entrypoints (prefill before/after,
   capture-guard inline — `is_current_stream_capturing` stays).
4. **Scheduler**: the between-replay tick becomes a native hook in
   `event_loop_overlap` calling a `TierManager.tick()` the scheduler
   owns; the proven stash path
   (`model_worker._draft_worker.draft_extend_attn_backend
   .full_attn_backend`) goes in `qsa/glue.py` as
   `resolve_selection_stash()` with the fallback chain.
5. **Allocator accounting**: shrunk-pool confiscation, available-size
   wrap, filtered frees and dedup become a proper allocator mode (or
   thin subclass); all counters (dedup/ff/clamp/coverage/starved/...)
   exposed through --enable-metrics.
6. The overlay builder stays, re-labeled "legacy experiment
   reproducer" only.

## Phase 2 — tests and CI (1-1.5 days)

- Port the 18 CPU fixtures to the native path (in-tree, following the
  repo's test layout, runnable via plain pytest against real imports).
- New: scheduler-tick unit test (fake eagle worker + stash buffers),
  allocator identity test, server-args plumbing test.
- CI (GitHub Actions on the fork): CPU docker job — pytest suite +
  import smoke (`python -c "import sglang.srt..."`) + the repo's own
  relevant test subset (attention/qsa). Nightly GPU smoke optional.
- Gate: green CI on `host-kv/main`.

## Phase 3 — build and deploy (half day)

- `docker/Dockerfile.hostkv` building sglang from the branch (reuse the
  dev image's build steps); publish one tag, e.g.
  `hostkv:4ccff141-n`.
- One launch script with MODE=vanilla|tier (tier = flags on) — same
  image both modes. GPU_HOST_POLICY.md discipline unchanged.

## Phase 4 — A/B benchmark harness (1.5-2 days)

`benchmark/context_certificate/ab_bench/`:

- **Workloads**: existing corpus (workload.jsonl); ladder = {N: 1, 4, 8,
  16, 32} × {ctx: 101k, 160k} × 1024 forced-length decode.
- **Protocol**: per-GPU server (two-server topology documented as the
  EP2 workaround, upstream #44291); explicit warm-priming (the H200
  lesson); interleaved A/B/A/B ordering to cancel thermal/drift;
  cool-downs; single-GPU pinning; fixed seeds.
- **Metrics**: per-request steady decode (decode-only windows binned by
  running-req — no prefill contamination), TTFT, wall, p50/p99, tier
  counters + /metrics scrape. Output: results.json + markdown report.
- **Noise floor FIRST**: vanilla-vs-vanilla runs on identical config to
  establish the nondeterminism baseline that any tier-on comparison must
  sit within.
- **Quality gate**: forced-continuation logprob comparison (mean KL +
  top-1 flip rate) tier-on vs tier-off over fixed prompts; acceptance =
  within the measured noise floor. This replaces output-equality, which
  the base server itself fails.
- **Regression check**: flag-off vs vanilla pinned image on a subset —
  proves the fork is a no-op when disabled.

## Phase 5 — execute matrix and report (1 GPU day)

- Full ladder on a 2×RTX PRO 6000 spot box (zone-sweep tooling exists).
- Acceptance: tier-on per-request ≥80 tok/s at N ≤ 8; aggregate scaling
  curve documented (the 32×80 aggregate question gets its first honest
  data); verify_mismatches=0; coverage within documented rates; quality
  within noise floor.

## Phase 6 — soak and ops (1 GPU day + 24h calendar)

- ≥4h continuous mixed load (target 24h if budget allows): counters
  flat (no leak drift), zero scheduler exceptions, throughput stable in
  the last hour vs first.
- Failure drills: kill -9 a server mid-run (recovery), spot-preemption
  restart.
- Runbook: launch/monitor/teardown, alert thresholds on counters.
- Known-limitations doc: two-server topology (EP2 bug), 256k pool cap,
  0.3% deep-miss noise under prefill concurrency, reserve budget
  tension at N ≥ 4 concurrent prefills.

## Rough timeline

~7-9 working days CPU + 2 GPU days + soak calendar. Critical path:
Phase 1 → 2 → 3 → 4 → 5; Phase 6 parallelizes after 4.

## Risks

| Risk | Mitigation |
|---|---|
| Upstream divergence while we integrate | Hard-pin at 4ccff141db; no rebase until P1 |
| Graph-capture interplay regresses | Capture-guard unit test + graphs-on arm in every matrix cell |
| Quality gate flaky (base nondeterminism) | Noise-floor baseline before any tier-on claim; distribution metrics, not token equality |
| Spot capacity dry | Zone-sweep automation exists; budget a second region |
| Reserve starvation at N≥4 prefills | Documented (HKV-17); matrix quantifies aggregate impact; residency-based admission is future work |
