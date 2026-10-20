# HKV-3 — tier performance design (2026-10-14)

The G2/Gate runs measured ~3.7x native wall time in diagnostic mode
(95.2 s vs 25.7 s for the 167k G1 workload; 8.5-42 s for the 25k HKV-4
runs vs sub-second native decode). The overhead decomposes into four
costs, addressed in phases:

## Cost decomposition

| Cost | Share (est.) | Phase 1 (this change) |
| --- | --- | --- |
| (a) synchronous `.to("cpu")` mirror copies with implicit device syncs | large | **fixed**: side CUDA stream + `non_blocking=True`; the runtime's compute stream never waits unless we sync |
| (b) `torch.cuda.synchronize()` per restored page | large | **fixed**: one `_tier_stream().synchronize()` per gather, not per page |
| (c) per-page full 12-layer byte verify | medium | **gated**: `SGLANG_HOSTTIER_VERIFY_SAMPLE=N` verifies the first restore plus every Nth; `VERIFY=0` disables (production default once G3 passes) |
| (d) graphs disabled (Python hooks bypass under replay) | the remainder | **Phase 2 fork** — see below |

## Phase 1 implementation (committed)

`hosttier23m.py`: `SGLANG_HOSTTIER_ASYNC=1` enables the side stream and
async copies for both the retire-time re-mirror and the restore; verify
sampling via `VERIFY_SAMPLE`. All default-off — the diagnostic modes that
passed G2/HKV-4 are bit-identical when the flags are unset (97 tests
green includes every prior gate scenario).

## Phase 2 fork: CUDA graphs

Decode graphs replay without invoking Python-level hooks, so a graphed
step would read retired (cleared) table columns as zeros and skip
staging entirely. Three options:

1. **Eager decode, async everything** (phase 1 as-is): keeps the hooks
   authoritative; per-step cost stays ~launch-bound. Measured before
   deciding.
2. **Stage-before-capture**: at graph capture time, pre-restore every
   page the captured batch can touch; retired columns never appear in a
   graph. Requires knowing the selection set at capture — the capture
   batches are synthetic, so this reduces to "no retired pages during
   capture" — viable if retirement pauses across capture (a warmup
   barrier), at the cost of bursty staging.
3. **CUDA-graph-aware hook** (upstream-class work): register a
   pre-replay callback that stages from the graph's planned indices.
   Correct long-term; largest lift.

Decision rule: measure phase 1 first. If eager+async lands under ~1.5x
native on the 25k workload, option 1 carries to G3 and the fork stays
open. If launch-bound overhead dominates (per-step Python + ~50 kernel
launches), option 2 is the next increment, not option 3.

## What must NOT be optimized away

The byte-verify (even sampled) and the G-gate scenario tests. Any perf
change that cannot show G2/HKV-4-equivalent behavior with the flags
defaulted off is rejected; perf modes are additive, not replacements.
