# Host-KV architecture plan — 32×256k×80 tok/s on 2× RTX PRO 6000

Goal: 32 concurrent 256k-token contexts at 80 tokens/s per request
(2,560 tok/s aggregate) on two RTX PRO 6000 cards plus a host-RAM tier,
with model quality unchanged. Engineering effort is accepted; every phase
still ends in a measured gate so the gain stays a number.

The gain being purchased: the on-GPU ceiling on the 2-card system is
~14–16 concurrent 256k requests (weights ~39 GiB/card, ~24.6 KB KV/token,
~94 GiB usable/card). The host tier lifts the ceiling to 32 — a 2–2.3×
concurrency gain on identical silicon — and requires ~310 GB of host RAM
(KV 192 GiB + pinned PLE ~96 GiB + index ~6 GiB + draft state ~16.5 GiB),
i.e. a 384 GB-RAM machine for the end state.

## Established ground truth (2026-09-29/30, all artifacts local)

- **Phase-0 gates (measured 2026-09-30, GCP 2× RTX PRO 6000,
  [summary](artifacts/gcp-20260930/phase0-summary.json)):**
  M0.1 **PASS** — decode step time is context-flat (14.37 → 14.32 ms/step
  from 30k to 167k tokens; only the index scan grows, 0.58 → 0.82 ms).
  M0.4 **DONE** — pinned host↔device bandwidth 56.9 GB/s per GPU (2.8× the
  budget's 20 GB/s link assumption). M0.2 **PARTIAL** — 16 concurrent
  requests run in a 520k pool, but the clean profile window needs one rerun.
  M0.3 **BLOCKED** with two plan-relevant findings: TP2 is structurally
  impossible on this checkpoint (NVFP4 block divisibility vs the 640→320
  sharding), and EP2 shards weights correctly (~42.7 GB/GPU) but dies at
  NCCL init on this runtime. **The 2-GPU design must be modeled as EP2, and
  the EP2 init failure is the first engineering item of Phase 2.**
- 80 tok/s per request is already met at 30k contexts: ~121 tok/s per
  request at N=8, ~968 tok/s aggregate on ONE card, TP1
  ([summary](artifacts/rtx1-20260929/summary.json)).
- Index screening is dead by measurement (4.1% share, selection-bound);
  the index stage stays ~1 ms/step at every batch size.
- Real expert-route correlation is far friendlier than modeled: GEMM grows
  1.51× from batch 1→8 where the uniform-iid model predicted 5.9×.
- Cross-layer selection overlap (layer 3→7): coverage 0.65, overfetch 0.35
  — selection-driven prefetch is viable and parked until Phase 3.
- The architecture is built for this: attention is budgeted (~2,048 tokens
  regardless of context), GDN state is fixed-size; only the index scan
  (measured small), storage, and link traffic grow with context.
- The existing capacity model is unreliable on its dominant term and must
  be rewritten (Phase 1) before any design is trusted.

## Phase 0 — Sizing measurements (2–3 single/dual-GPU windows, no code)

| ID | Measurement | Method | Gate |
| --- | --- | --- | --- |
| M0.1 | Context-flatness at 256k | One 256k request, TP1, profile decode window; contrast with the 8×30k point; optional capture run at 256k to verify index structure at 65,536 keys | 256k step time ≤ ~1.5× the same-batch 30k step time |
| M0.2 | Clean steady-state large batch | N=16 (and 12), all requests resident BEFORE profiling (submit, wait for #running-req, then arm); explains the confounded 57.5 ms/step reading | step time at N=16 ≤ ~2× N=8, else investigate scheduler/GDN effects before host-KV work |
| M0.3 | TP2 baseline (never measured) | 2-card TP2 server; step time at N=1/8/16 × 30k and 4×256k | TP2 N=8×30k step ≤ ~35 ms ⇒ 2,560 tok/s aggregate plausible on 2 cards |
| M0.4 | Link bandwidth of the box | Stream H2D on both cards concurrently (pinned memory) | establishes the link constant Phase 1 needs |

Failure of M0.1/M0.3 gates ⇒ stop and re-scope (e.g. on-GPU 14–16×256k
target with no host tier) before any engineering.

## Phase 1 — Capacity model rewrite (CPU, days)

**STATUS (2026-09-30): first cut DONE and gate PASSED.**
[phase1_model.py](phase1_model.py) rebuilds the ledger on measured
constants only; its validation gate requires reproducing the measured TP1
step times within 1.3× and achieves **1.043× worst-case**
([capacity-model.json](artifacts/phase1-20260930/capacity-model.json)).
Prediction for the end-state (2× RTX PRO 6000, EP2, 32×256k, host tier,
S=4): **29.8 ms/pass pipelined vs the 37.5 ms budget (20% margin);
38.3 ms serialized (2% over)** — the link term (8.5 ms/pass for
selected-KV misses at the measured 56.9 GB/s) is exactly what pipelining
or prefetch (coverage 0.65 measured) recovers. Memory fits both tiers:
53.3 GiB/GPU of 94.2, host 309.9 GiB (fits the 354 GB GCP-class box).

Open items that firm this up (all cheap): the clean N=16 measurement (one
rerun), real route distributions instead of the analytic union, a measured
PCIe P2P number (currently assumed 25 GB/s with a 15 GB/s sensitivity),
and end-to-end EP2 (NCCL init fix). The original to-dos below are
superseded by the measured-constants model.

- Replace the uniform-iid expert union with measured route data: enable
  `--enable-return-routed-experts` on a diagnostic run, collect real route
  distributions from the workload corpus, fit union-vs-batch growth.
- Rebuild the per-pass byte model on three tiers (GPU resident / host /
  link) with measured constants: 1,491 GB/s card bandwidth (measured),
  M0.4's link number, measured step times from Phase 0.
- Produce the layout design: page size (runtime page 64 today), KV
  residency split, index placement (original-on-GPU won the S=4 economics;
  recheck with the rewritten model), GDN states on GPU (~3.4 GiB total).
- Gate: model-predicted pass floors within ~1.3× of measured TP2 steps,
  and a predicted 32×256k×80 floor under the 37.5 ms/pass budget
  (80 tok/s ÷ ~3 tokens/pass at S=4).

## Phase 2 — Host-KV prototype (the build, staged with gates)

**STATUS (2026-09-30): P2.1 DONE.** The design is written against the
pinned runtime's real interfaces
([HOST_KV_DESIGN.md](HOST_KV_DESIGN.md)): the full-KV allocator's existing
page-64 arithmetic is reused unchanged, the compressed index K stays
GPU-resident (per-step scan), and the tier's entire payload is the 192 GiB
full-KV cache staged at 1.57 MB page granularity over the measured
56.9 GB/s link. The CPU-testable core — page state machine (HOST /
IN_FLIGHT / GPU_RESIDENT with write-through and LRU eviction) and the
per-pass admission controller (demand always admits, prefetch capped at
35% overfetch and never eating the demand budget) — is implemented and
tested in [host_kv/pager.py](host_kv/pager.py) with injected CUDA
plumbing; 46 tests green. Next: P2.2 integration (extend
`QSATokenToKVPool`, hook the paged gather, real `cudaMemcpyAsync`
staging) on the first GPU window, after the EP2/NCCL fix.

Each stage must preserve numerics exactly (temperature-0 token streams
identical to a fully-resident run on the same request) and never report a
throughput number without the repo's warm-priming discipline.

- **P2.1 Design doc + page ownership.** Host pinned pool, per-request page
  tables, eviction policy, transfer admission. Decide what is host-resident
  (cold KV, original index if the model says so) vs GPU-resident (weights,
  GDN states, hot KV).
- **P2.2 Smallest viable slice — CLOSED 2026-10-01 (window 5).** Every
  component hardware-proven: mapping-based layer discovery, pinned mirror,
  full-pool snapshot, evict-in-place (2,046 pages), demand-stage +
  byte-verify (arm-time self-check, verified=true on hardware), coherent
  decode with poisoned pages. Gate evidence:
  [window 5](artifacts/p22-20261001c/summary.md). The memory-level byte
  verify replaces token-stream identity (cross-restart bit-repro is
  unattainable on this runtime — measured).
- **EP2 resolved as a non-gate (2026-10-02).** Broken identically on the
  pinned build AND mainline dev-cu13 (IPC handshake → graph-capture
  illegal access → Triton illegal access; NCCL fine throughout) — a clean
  upstream bug with a two-build reproduction
  ([summary](artifacts/ep2-20261002/summary.md)). The 32-user target is
  REFRAMED as two independent TP1 servers, 16 users per card, each with
  its own host tier: 30.3 ms/pass modeled vs the 37.5 budget (19% margin),
  avoiding the entire broken distributed surface. EP2 demotes to a future
  throughput optimization (30% margin when fixed).
- **P2.3 G1 PASSED (2026-10-04):** the host tier served a full 167,765-token
  request — 2,112 pages (~135k tokens) retired to host at peak, byte-verified
  demand restore, coherent completion, exact accounting
  ([summary](artifacts/g1-20261004/summary.md)). The enabling fix:
  content-key dedupe of per-layer after_write calls (the ~1.56× inflation
  that read past the request end). G2 next: 16 concurrent requests on the
  multi-request mapper; then G3, two servers / 32 users.
  **Window 1 (2026-09-30):** overlay builds, installs, and loads; the
  staging hook is correctly placed but never fires — the live verify
  attention path does not pass through `_forward_paged_attention` for
  this hybrid+MTP config, and the entrypoint hunt is the next window's
  first task ([summary](artifacts/p22-20260930/summary.md)). Two gate
  requirements were learned the hard way: pin `--random-seed` +
  `--enable-deterministic-inference` for any A/B identity test, and CUDA
  graphs must be off for Python-level hooks.
- **P2.3 Concurrency + admission.** Scale to N concurrent requests with
  LRU hot tier; measure achieved link bandwidth and step-time regression
  vs resident-only at equal N. Gate: regression bounded by the Phase 1
  prediction.
- **P2.4 Full target on the end-state box** (2× RTX PRO 6000, ≥384 GB RAM):
  32×256k demand-paged baseline. Gate: sustained per-request rate and the
  gap to 80 tok/s quantified. This number — not a hope — is what Phase 3
  optimizes.

## Phase 3 — Selection-driven prefetch + link admission (optimization)

- Extend the (validated, reusable) capture hook: cross-layer coverage for
  more layer pairs (3→11, 3→23 decay curve) and more requests; recompute
  the prefetchable fraction of link traffic with Phase 1's model.
- Implement: layer-L top-512 selections issue H2D prefetches for layer-L+4
  selected-KV misses, bounded by spare link budget (admission controller).
- Gate: ≥20% step-time improvement over the P2.4 demand-paging baseline at
  32×256k; otherwise keep demand paging and record the negative result.

## Phase 4 — Validation and reporting

32 independent 256k contexts sustained at target; quality checks
(deterministic comparisons plus standard evals); honest write-up including
every negative result. Original bars that remain unproven anywhere in this
plan until this phase: model-wide quality, 32-request concurrency at 256k,
and the 80 tok/s target itself.

## Explicitly not doing

- Index screening (dead by measurement).
- Index-stage micro-optimization (~1 ms/step; cannot matter).
- Deeper speculation (S=12) upstream redesign — parked as a known lever
  (0.45× expert bytes/token) if Phase 2 misses its gate narrowly.

## Operational notes

- Spot-instance discipline: extract every result live; all launchers,
  senders, parsers, and tests live in [the proof kit](proof_kit/) and this
  directory; 27-test suite must stay green.
- Instance classes: Phase 0 M0.1/M0.2 and Phases 1–2 early stages need one
  96 GB card + ~176 GB RAM; M0.3, P2.3 need both cards; P2.4 needs the
  384 GB-RAM machine.
- The 1,950-kernel-per-step launch count is a standing risk against the
  37.5 ms budget at scale; CUDA graphs are already on, and if it binds,
  the remedy is upstream fusion, not this plan's components.
