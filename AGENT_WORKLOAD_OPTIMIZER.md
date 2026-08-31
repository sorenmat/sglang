# Agent-workload optimizer branch (Qwen3.8 + RTX PRO 6000)

Five changes targeting coding-agent traffic (bursty multi-K tool-return
prefills alongside steady decode) on a single-GPU Qwen3.8-27B NVFP4 +
FP8-KV + MTP 3/1/4 server. All on this branch; each commit is
independently revertable.

| # | Change | Flag / env | Default |
|---|--------|-----------|---------|
| 1 | Auto-size the GDN/Mamba state pool from target concurrency | `--mamba-full-memory-ratio auto` | off (0.9) |
| 2 | Decode-pressure-aware chunked prefill | `--enable-adaptive-prefill` (+ `--decode-latency-budget-ms`, `--adaptive-prefill-min-chunk-tokens`) | off |
| 3 | DFlash2 accept_len double-buffering + sentinel isolation test | -- (bug fix) | always on |
| 4 | SM120 GDN: FlashInfer prefill auto-default, opt-in verify | `SGLANG_GDN_FLASHINFER_VERIFY_SM120=1` | prefill auto / verify Triton |
| 5 | Incremental NVFP4 KV dequant mirror | `SGLANG_NVFP4_DQ_MIRROR_FRACTION=0..1` | off (0.0) |

## Measured results (RTX PRO 6000 Blackwell 96GB, Qwen3.8-27B-NVFP4,
## FP8 KV, MTP 3/1/4, agent-workload harness)

### `--mamba-full-memory-ratio auto` (item 1) -- e2e, c x ctx sweep

| cell | baseline (0.9) | auto | delta |
|---|---|---|---|
| c=24, ctx=8K | 330.9 tok/s | 594.9 tok/s | **+80%** |
| c=24, ctx=32K | 80.4 tok/s | 198.1 tok/s | **+146%** |
| c=16, ctx=32K | 218.2 tok/s | 252.4 tok/s | +16% |
| c=16, ctx=8K | 549.7 tok/s | 577.9 tok/s | +5% |
| c<=8 | -- | -- | +-2% (pool not binding) |

TTFT p99 at c=24/32K: 31.1s -> 2.0s. The fixed 0.9 ratio over-reserves
state memory (fp32 GDN state + MTP intermediates) and under-provisions
the KV pool; auto sizes exactly for the target concurrency.

### FlashInfer GDN on SM120 (item 4) -- kernel microbenchmark

- Prefill (8192 tokens, Qwen3.8 geometry): **FlashInfer 0.495 ms/layer
  vs Triton 1.252 -- 2.53x** (23.8 vs 60.1 ms across 48 layers/chunk).
- Decode: FlashInfer 1.28-1.33x faster (1.01 vs 1.32 ms/step x48).
- Target-verify (opt-in `SGLANG_GDN_FLASHINFER_VERIFY_SM120=1`):
  numerically correct vs Triton (maxdiff <= 6e-5); 1.27-1.42x faster at
  batch <= 8, parity at 16, 1.03x slower at 24.
- The auto-default (prefill) fired in the real server log.

### Incremental NVFP4 dequant mirror (item 5) -- kernel microbenchmark

| context | full-prefix dequant | mirror | speedup |
|---|---|---|---|
| 32K | 9.3 ms/cycle | 2.0 ms | 4.8x |
| 128K | 36.1 ms/cycle | 6.7 ms | 5.4x |

The residual is the FP8 row gather (memory-bound copy); the dequant
compute itself drops to O(newly written tokens).

### `--enable-adaptive-prefill` (item 2) -- e2e

After tuning (backlog-aware floor, budget-covering-floor invariant,
admission-saturated bypass), measured against prefill-first + auto:

| cell | throughput vs auto | ITL p99 |
|---|---|---|
| c=16, ctx=32K | -2% | 1876ms vs 1149ms |
| c=8-24, ctx=8K | -16% | 400-1100ms vs 220-1100ms |
| c=24, ctx=32K (overload) | -54% | 2448ms vs 3491ms |

vs *baseline* (default memory + prefill-first) the worst-case decode
stall drops from 8.5s to 2.4s. This is a tunable tail-latency knob, not
a throughput win: enable it when interactive smoothness matters, and
tune `--adaptive-prefill-min-chunk-tokens` (the budget lower-bounds one
floor-sized chunk at a conservative 2000 tok/s -- a budget below that is
unsatisfiable and would time-slice prefill/decode 50/50).

## bf16 SSM state + FlashInfer GDN decode/verify (measured 2026-08-31,
## acceptance protocol: 65k input / 60k shared prefix / 1024 out / EAGLE 3/4/1,
## retractions 0 and accept-length ~3.1 in every arm)

| arm | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|
| recipe (fp32 state, Triton GDN) | 21.3 | 46.7 | 74.5 | 118.2 |
| bf16 state (Triton GDN) | 18.1 | 36.3 | **84.3** | **131.9** |
| bf16 + FlashInfer decode+verify | 12.9 | 53.7 | 71.0 | 114.4 |
| bf16 + FlashInfer verify only | 12.9 | 42.6 | **86.6** | 121.5 |

(output tok/s aggregate)

Findings:
- EAGLE acceptance is UNCHANGED with bf16 state (3.12 vs 3.09 of 4) -- the
  bf16-state slowdowns below are kernel-side, not accuracy-side.
- bf16 state frees ~19 GB -> +9.5% KV tokens (855K -> 937K): wins +12-13%
  at c>=4, loses 15-22% at c=1-2 (Triton's bf16-state path is slower per
  step; the KV headroom only pays off when requests compete for cache).
- FlashInfer GDN decode costs ~40% SOLO: it forfeits the packed Triton
  replaySSM decode fast path (dispatcher reports packed_decode=False).
  From c=2 up it is competitive; verify-only is the best c=4 arm (+16%).
- Verdict: not a 30% lever at the deployment's c<=2 operating point. The
  context-scaling cost lives in the 16 full-attention layers x EAGLE
  verify over 65k+ KV (already FlashInfer fp8); GDN-side switches cannot
  move solo decode by 30%.

Realistic 30% paths: (a) NVFP4 KV cache -- halves full-attn verify bytes
at long context (upstream spec-compat PRs pending; the incremental
dequant mirror in this branch is the follow-on), (b) cap interactive
context near 64k (product guidance; the solo curve is 142 -> 11 tok/s
from 17k -> 232k), (c) adaptive speculation to cut verify passes when
acceptance decays at long context, (d) for sustained c>=4 traffic:
`--mamba-ssm-dtype bfloat16 --linear-attn-verify-backend flashinfer` +
`SGLANG_GDN_FLASHINFER_VERIFY_SM120=1` is a free +13-16%.

## Uniform FP8 -> uniform NVFP4 KV (one pool, no tiering) -- measured
## 2026-08-31, acceptance protocol WITHOUT EAGLE (KV4+spec verify is not
## wired upstream yet); trtllm_mha native-FP4 decode + flashinfer
## dequant-workspace prefill (the only arg-valid pairing on SM120)

Capacity: **max_total_num_tokens 855,673 -> 1,513,728 = 1.77x** prefix
capacity at identical mem-fraction.

Quality (greedy, temperature 0, 15-prompt battery incl. three 60k-context
factual-recall questions):
- 0/15 exact token match vs the FP8 reference; first divergence typically
  at token 30-165 of 300-950. Coherent output throughout -- this is KV
  quantization jitter, not corruption.
- **All three long-context facts answered correctly on NVFP4 KV**
  ("fifteen seconds", "the router", "15") at 60k context.

Speed (16 prompts x 512 out, c = 1/4/8 aggregate tok/s):

| arm | c=1 | c=4 | c=8 |
|---|---|---|---|
| fp8 (flashinfer decode) | 34.9 | 116.6 | 235.1 |
| nvfp4 (trtllm_mha decode) | 31.2 | 145.5 | 190.6 |
| nvfp4 + mirror 0.25 | 31.3 | 131.7 | 190.8 |

ITL p99 is consistently higher on the FP4 decode path (78/94/110ms vs
49/93/106ms): the native-FP4 decode kernel gives back more than the
halved bandwidth saves at low batch. The bandwidth win is real in bytes
but only cashes in under EAGLE verify (4 full-context reads per cycle),
which needs the upstream spec-compat wiring.

Mirror: performance-neutral without spec (nothing re-reads prefixes per
cycle); its 5.4x/cycle value applies once verify exists. Fixed an
integration bug found at fraction=1.0 (23 GB mirror OOM): the pool
sizing path (DefaultPoolConfigurator._compute_cell_size) never saw the
mirror term -- now charged there as a linear per-token cost.

Verdict: NVFP4 KV today = a 1.77x CAPACITY play at -11..-19% speed
(c=1/c=8) through the only available decode kernel. The 30% speed case
needs it combined with EAGLE verify (upstream PRs) where the bandwidth
halving multiplies.

## 1. `--mamba-full-memory-ratio auto`

A fixed split of free VRAM between the linear-attention state pool and
the KV pool strands memory on one side or silently caps concurrency on
the other (each running request holds 3-5 state slots, more under
speculative decoding). `auto` sizes the state pool for exactly the
target concurrency (`--max-running-requests`, or derived from the model
context length and KV cell size when unset), reserves the spec-decode
intermediates, and gives KV the remainder (capped at 85% of rest).
Exact integer sizing; the derived numeric ratio is written back to the
config leaf for `/server_info` readbacks. See
`test/registered/unit/mem_cache/test_mamba_full_memory_ratio_auto.py`.

## 2. `--enable-adaptive-prefill`

Running decode requests get a wall-clock latency budget. Prefill chunks
shrink so their projected cost (from a measured prefill-throughput
EWMA) fits the remaining budget, and once the budget is exceeded the
scheduler yields the iteration to decode before admitting new prefill.
TARGET_VERIFY counts as the decode step, so MTP/DFlash2 are covered.
Disabled under DP attention and PP dynamic chunking. See
`test/registered/unit/scheduler/test_adaptive_prefill_scheduling.py`.

## 3. DFlash2 concurrency fix

The accept computation's five output buffers rotate on a two-slot
schedule because overlap scheduling can launch step N+1's kernel before
step N's consumer reads the results -- but `accept_len` shared storage
across both slots, so a step could read the previous step's accept
lengths (corrupted commit lengths, garbled outputs at concurrency).
Now double-buffered like its siblings.
`test/registered/spec/dflash/test_dflash_sentinel_isolation.py` is the
deterministic reproduction harness (N concurrent requests with unique
sentinels; shared-prefix, radix-reuse, and cancellation-churn variants).

## 4. SM120 GDN backends

flashinfer ships a dedicated SM120 chunked-prefill GDN kernel and the
SGLang wrapper already handles its fp32-initial-state quirk, so the
FlashInfer prefill auto-default now covers SM120 (CUDA 13+, chunk
<= 8192, head dims 128, either state dtype). FlashInfer target-verify
(spec decoding) stays Triton-by-default on SM120 and is opt-in via
`SGLANG_GDN_FLASHINFER_VERIFY_SM120=1`. Pick per-phase winners on your
box with:

    python benchmark/bench_linear_attention/bench_gdn_backends_all_phases.py

## 5. Incremental NVFP4 dequant mirror

Quantized-KV pools dequantize the whole cached prefix on every use
(every chunked-prefill chunk per layer; every speculative-verify
cycle). KV slot content is write-once until the slot is reused, so a
slot-keyed FP8 mirror with per-(layer, slot) write epochs only
re-dequantizes newly written slots: O(new tokens) instead of
O(context). Opt-in; bytes are charged to the KV budget. Measure with:

    python benchmark/nvfp4/bench_nvfp4_dequant_mirror.py

## End-to-end harness

`benchmark/agent_workload/bench_agent_workload.py` launches the server
per feature configuration and sweeps concurrency x context length with
agent-shaped sessions (long context -> decode burst -> tool return ->
decode ...), reporting output tok/s, TTFT and ITL p50/p99, e.g.:

    python benchmark/agent_workload/bench_agent_workload.py \
        --model <qwen3.8-27b-nvfp4> --draft-model <mtp-draft> \
        --mtp 3/1/4 --kv-cache-dtype fp8_e4m3 \
        --concurrencies 1 4 8 16 24 --contexts 8192 32768 65536 131072 \
        --runs baseline auto adaptive

## Tests (CPU, no GPU needed)

    pytest test/registered/unit/mem_cache/test_mamba_full_memory_ratio_auto.py
    pytest test/registered/unit/scheduler/test_adaptive_prefill_scheduling.py
    pytest test/registered/unit/spec/test_dflash_accept_buffer_rotation.py
    pytest test/registered/unit/layers/attention/test_gdn_sm120_gating.py
    pytest test/registered/unit/layers/quantization/test_nvfp4_dequant_mirror.py

GPU tests / benchmarks (run on the RTX PRO 6000 box): the sentinel
isolation test, both microbenchmarks, and the end-to-end harness above.
