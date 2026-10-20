# Local benchmark evidence audit

2026-09-25. Read-only inspection of `/home/smo/code/llm-performance` raw
results and benchmark code. No live host access, new inference runs, or changes
to that repository. These are existing baseline results, not results of the
proposed certificate.

## Measured anchor points

All three points below used 64 requests, one repeat, and reported zero failures.
Context lengths are actual API token usage, not the requested input-length flag.

| Stack | GPUs | Client concurrency | Mean input tokens | Aggregate output tok/s | Approx. median decode tok/s/request | p95 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGLang NVFP4 TP2, driver 580, agentic | 2 | 16 | 20,547 | 576.3 | 42.5 | 27.03 ms |
| vLLM FP8 TP2, synthetic | 2 | 16 | 3,770 | 557.0 | 47.0 | 34.05 ms |
| SGLang NVFP4 cookbook TP1, synthetic | 1 | 8 | 3,770 | 451.0 | 75.4 | 19.95 ms |

Raw sources:

- [SGLang TP2 point](/home/smo/code/llm-performance/results/serve-20260924-152038-flashnext-nvfp4-rtxpro-tp2-16k-agentic-580/point-c16-r1.json)
- [vLLM TP2 point](/home/smo/code/llm-performance/results/serve-20260924-184742-flashnext-fp8-vllm-tp2-4k/point-c16-r1.json)
- [SGLang TP1 point](/home/smo/code/llm-performance/results/serve-20260924-172333-flashnext-nvfp4-rtxpro-cookbook-mtp-4k/point-c8-r1.json)

Aggregate output throughput includes prefill and run drain. Approximate median
decode rate above is `1000 / summary.tpot_ms.p50`; request TPOT is
`(E2E - TTFT) / (completion_tokens - 1)`. The existing `per_stream_tps` field
includes TTFT and is therefore different. A client concurrency limit does not
prove that all requests decode simultaneously for the measurement interval.
Request summaries lack the timestamps needed to reconstruct that overlap.
See [stats.py](/home/smo/code/llm-performance/llmperf/stats.py) and
[client.py](/home/smo/code/llm-performance/llmperf/client.py).

The desired 2,560 aggregate tok/s is 4.44 times the best whole-run aggregate
above, with twice the concurrency and much longer contexts. This ratio is a
scale comparison, not a measured kernel-speedup requirement. The different
checkpoints, kernels, inputs, output lengths and cache conditions prevent a
framework ranking from this table.

## Capacity and independence

The [ledger](/home/smo/code/llm-performance/results/index.md) records a
1,801,152-token SGLang TP2 pool and a 1,132,178-token vLLM TP2 pool: approximately
6.87 and 4.32 full 262,144-token context equivalents respectively. The target
requires 8,388,608 context tokens before additional generation space. Neither
pool demonstrates the required capacity.

The recorded dual-card host has RTX PRO 6000 Blackwell **Server Edition** GPUs
and 235 GB RAM. The proposal's 309.87 GiB static original-precision host archive
does not fit that reported RAM budget. Its 512 GiB host provision and workstation
bandwidth assumption must not be silently applied to this box. Changing cache
placement or baseline precision requires a new explicit capacity ledger.

The agentic sweep has 50 unique trajectory-derived prompts repeated to fill
64 requests. Successive concurrency points reuse the corpus without a cache
flush. `shared_prefix_len=0` therefore does not establish independence. vLLM's
agentic c2/c4 samples report roughly 15%/14% prefix hits. Agentic prompts are
also carved and padded with repeated transcript content. See
[prompts.py](/home/smo/code/llm-performance/llmperf/prompts.py) and
[bench_serve.py](/home/smo/code/llm-performance/bench_serve.py).

## Acceptance and quality

The ledger's favorable 3.1–4.0 acceptance excerpt is not the whole distribution.
H200 NVFP4 n50 sampled acceptance-length gauges average 2.994, or 3.036 when
restricted to samples with running requests. RTX TP2 driver580 n10 averages
2.876, or 2.853 with running requests. These are periodic gauges, not
verification-round-weighted accepted lengths. Idle samples can preserve stale
values; the long H200 run also spans an outage and server counters do not
reconcile to that run's trajectory usage totals.

The checked samples satisfy `accept_length = 1 + 3 * accept_rate`: the
configuration emits at most four tokens per verification, including the target
or bonus token. These observations do not establish acceptance with 12 target
positions or seven emitted tokens per pass.

Sources: `metrics.csv` in
[H200 n50](/home/smo/code/llm-performance/results/swebench-20260923-161724-flashnext-nvfp4-n50-defaults),
[RTX TP2 n10](/home/smo/code/llm-performance/results/swebench-20260924-153927-flashnext-nvfp4-rtxpro-tp2-n10-580),
and [sampler.py](/home/smo/code/llm-performance/llmperf/sampler.py).

The H200 n50 evaluation resolved 48/50 tasks over 1,933 calls. It supplies a
useful coding baseline, not paired evidence of preserved BF16 quality or
32-by-256k performance. Its longest successful API input was 167,766 tokens;
the associated astropy-13398 task was not resolved. RTX TP2 driver580 n10's
maximum observed input was 94,064 tokens. No full 256k or dual-RTX c32 target
validation was found in the inspected FlashNext results.

## Consequence for the proposal

Earlier conversational estimates of 35–65 tok/s/request on one card and
70–130 on two cards are **conditional bandwidth scenarios, not empirical
forecasts**. Existing measurements do not justify an expected rate for
32 independent full 256k contexts. Index compression alone cannot be credited
with closing the throughput gap: the model already uses sparse attention and
the proposal still depends on efficient expert execution, drafting, state
management and transfers.

Reuse the existing trajectories and evaluation harness. First freeze one
checkpoint and workload, establish capacity and decode behavior as context and
concurrency increase, and profile MoE, QSA/indexing, recurrent state, drafting
and communication. Record per-round accepted lengths and routing unions.
Then capture original QSA queries, keys and selected blocks for offline
certificate replay. No such activation captures were found in the results.

SGLang remains a practical first integration target for this repository's
research code. The existing vLLM TP2 deployment is a valuable baseline. These
measurements establish neither framework as intrinsically faster.
