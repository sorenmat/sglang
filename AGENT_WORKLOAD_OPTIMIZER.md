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
