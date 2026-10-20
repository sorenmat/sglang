# Proving or killing the screening kernel's economics

**RESULT (2026-09-29, measured on 2× RTX PRO 6000 spot instance): the
screening kernel proposal is DROPPED by its own decision rule.** C1 measured
the index stage at 4.12% of decode-window GPU kernel time (below the 10%
threshold), C2 measured selection — not bandwidth — dominating the index
stage, and the unfused screened path lost 8.3×. Details:
[results directory](../artifacts/rtx2x-20260929/).

| Claim | Threshold | Measured | Outcome |
| --- | --- | --- | --- |
| C1 index-stage share | ≥ 10% to proceed | **4.12%** (50.5 ms of 1227 ms, 64 verify steps, 0.79 ms/step) | fail |
| C2 floor vs BF16 scan | floor ≥ 10% faster AND measured ≤ 0.8× | floor 0.12 ms vs 2.74 ms, but selection is 84% of baseline; measured screened 8.31× slower | fail |
| C2 exactness on GPU | zero pick mismatches | **0 mismatches** across 16 banks × 65,536 keys | pass (evidence, not proof) |

The scorer kernel (`kernel_kernel`, TileLang's default JIT name) was
identified by launch count: exactly 13/step (12 target QSA layers + 1 draft
layer), matching the three named QSA kernels. The dominant cost was MoE GEMM
at 62.5% of kernel time — consistent with the budget model's claim that
expert reads, not the index, govern this workload. The instance was
reclaimed before the raw 12 MB trace was copied; result artifacts were
reconstructed from the session's tool output and are labeled as such.

The capture/replay infrastructure remains validated and reusable; the next
open engineering question is the host-link and expert-read budget, not the
index.

---

The capture experiment settled the correctness question (native top-512
survives certified 4/6/8-bit screening on real activations). What remains is
whether a production screening kernel is WORTH building. Three claims, two
measurable with the kit in this directory:

- **C0 — arithmetic (DONE, no GPU needed).** Measured refine-union fractions
  plugged into the runtime-compatible S=4 budget. Verdict:
  [screening-economics.json](../artifacts/h200-20260929/screening-economics.json).
  Screening saves 32–36% of index bytes = about 1 ms/pass of a 37.5 ms
  budget (2.8%), and in its proposed compact-on-GPU + host-refinement
  placement it nearly doubles host-link load (48 → 84–97 ms at 20 GB/s),
  which is the binding resource. As a throughput intervention it is NOT
  justified by the byte model; only C1 can rescue it.
- **C1 — in-loop share (MEASURED: 4.12%).** Kernel-profile the
  real serving loop and measure the index stage's share of verify-step GPU
  time. If the index stage is actually far larger than ~8% of step time
  (e.g. top-k/selection compute dominating), the byte model is wrong and the
  verdict reopens. If it is near the byte model's ~8% or smaller, the kernel
  is dead: drop it.
- **C2 — wall-time bracket (MEASURED: unfused screened 8.31× slower; floor
  0.12 ms vs baseline 2.74 ms; selection dominates).** Time a full
  BF16 scan vs the screened path at target scale (16 banks × 65,536 keys,
  real key distributions). Bracketed: `screened_measured` (decode included,
  upper bound) and `screened_floor` (byte model at measured streaming
  bandwidth, lower bound). If the FLOOR does not beat the BF16 scan by ≥10%,
  no kernel can save enough; if the MEASURED path wins by ≥20%, screening is
  clearly viable; between them requires a fused-kernel prototype to decide.
- **C3 — GPU rounding (NOT settled by this kit).** The certified interval
  math is proved for the float64 reference only. C2's pick-mismatch count is
  evidence about CUDA arithmetic escapes, but a production design needs a
  rounding-aware radius proof (tensor-core accumulation) before any claim of
  exactness on GPU.

## C1: profile the serving loop (H200 or one RTX PRO 6000)

Relaunch the pinned runtime production-like: same image/checkpoint/config as
[the capture run](../GPU_RUN_20260929.md), but with capture disabled
(unset `SGLANG_QSA_CAPTURE_DIR`), FlashInfer autotuning ENABLED, CUDA graphs
ENABLED, and `SGLANG_TORCH_PROFILER_DIR=/profiles` exported. Readiness via
`/v1/models` (the `/health` endpoint runs inference). Kernel caches from the
capture run remain valid if the host kept `/home/ecuser/qsa-capture-20260929`.

Send the same 32k request, and once decode is steady start a window:

```bash
curl -s http://127.0.0.1:PORT/start_profile -H 'content-type: application/json' \
  -d '{"output_dir": "/profiles", "num_steps": 40, "activities": ["GPU"]}'
```

(`num_steps` auto-stops; fields verified against the pinned revision's
`ProfileReq`.) Then parse, confirming clusters against the top-kernels table:

```bash
python index_share.py /profiles/*.json --include-memcpy
```

Cluster patterns cover the pinned runtime's kernel names (`qsa_fast_topk`,
`qsa_index_(q|k)_*` JIT prep, TileLang scorer). If a TileLang JIT name does
not match, fix `CLUSTERS` — never accept a share with an unexplained cluster.

## C2: scan-vs-screen microbenchmark (any CUDA torch, no model)

```bash
python torch_screen.py bench CAPTURES/qsa-*.npz --banks 16 --bank-keys 65536 --bits 8
```

The screened path is validated CPU-side first (`torch_screen.py check`, must
print `ok` for all captures — picks and candidate unions reproduce the NumPy
probe exactly at 4/6/8 bits, and picks match the native selections as sets).

## Decision rule

Proceed to a fused production kernel only if C1's index-stage share exceeds
10% of step GPU time AND C2's floor beats the BF16 scan by ≥10% with zero
pick mismatches. Otherwise drop or redesign the screening proposal; the
byte-level case is already too thin at S=4.

## Next measurement window (de-risking selection-driven prefetch)

Everything below is scripted and spot-loss-resilient (extract results live).
One single-GPU instance window of about two hours covers both.

**Prerequisite prep (done, all tests green):** multi-layer capture needs no
overlay change (`SGLANG_QSA_CAPTURE_LAYERS` accepts a comma-separated list);
[cross_layer_overlap.py](cross_layer_overlap.py) analyzes two same-request
layer captures; [batch_profile_send.py](batch_profile_send.py) submits N
distinct ~32k corpus prompts concurrently with a mid-decode profile window.

1. **Cross-layer selection overlap** (~40 min). Install the capture overlay
   exactly as in [GPU_RUN_20260929.md](../GPU_RUN_20260929.md), pick a second
   QSA layer ID from the checkpoint's `config.json` on the instance, launch
   with `SGLANG_QSA_CAPTURE_LAYERS=3,<other>` and `MAX_FILES=12`, send one
   32k request, then:
   `python cross_layer_overlap.py EARLY.npz LATE.npz --output overlap.json`
   The number that matters: `per_union_mean.coverage_of_b_by_a` — the
   prefetch hit rate early-layer selections would give later layers. Above
   ~0.5 the prefetch design has legs; near zero kills it cheaply.
2. **Batch-scaling profile** (~60 min). Reuse `c1-launch.sh` with the context
   pool raised for concurrency (`--context-length` ≥ N × 40,000 and
   `--max-running-requests N`), then for N in 1, 4, 8, 16:
   `python batch_profile_send.py --workload /root/proof/workload.jsonl --requests N`
   Parse each DECODE trace with `index_share.py --assume-scorer kernel_kernel`
   and normalize per step by fast_topk launches ÷ 13. The number that
   matters: GEMM ms/step growth versus N against the uniform-iid union
   assumption in the budget (470 of 512 experts at batch 32). Slower growth
   means the 80 tok/s target has more slack than modeled.
3. Ship `workload.jsonl` to the instance alongside the kit files.

Card choice: the RTX PRO 6000 is preferred (target hardware, native FP4
path), but the overlap measurement is a model property and runs on any
proven single card, including the H200 configuration.

## Limits

H200 numbers proxy the two-RTX PRO 6000 target imperfectly (Marlin vs native
FP4 path, 4.8 vs ~1.8 TB/s). Nothing here measures model quality, 32-request
concurrency, or 80 tokens/s. Host state at kit-creation time: the H200
(213.171.182.156) went offline 2026-09-29 ~08:10 UTC after the capture
session ended; the 2× RTX PRO 6000 spot instance (root@86.38.238.229) was
reclaimed the same day after both measurements completed; all artifacts are
local.

