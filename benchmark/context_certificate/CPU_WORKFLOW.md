# GPU-free experiment kit

Everything here runs locally, without a serving endpoint, GPU, model download
or network access. Run the commands from `/home/smo/code/sglang-perf`.
NumPy and CPU PyTorch are already installed in `.venv`; the workload and budget
tools use only the standard library. Large generated inputs and captures stay in ignored
`artifacts/`. The source benchmark repository is read-only.

## Reconstruct the workload

```bash
python benchmark/context_certificate/workload.py \
  --source /home/smo/code/llm-performance/results/swebench-20260923-161724-flashnext-nvfp4-n50-defaults \
  --output benchmark/context_certificate/artifacts/workload.jsonl \
  --tool-schema-source /home/smo/code/llm-performance/.venv/lib/python3.14/site-packages/minisweagent/models/utils/actions_toolcall.py
```

This selected 50 distinct tasks with historical input lengths of 9,501 to
167,766 tokens (median 34,355.5). Counts reaching 32k/64k/128k/256k are
28/2/1/0, using powers of two. The corpus cannot supply 32 natural full-256k
requests. It is a real coding workload for initial profiling, not a substitute
for the target workload or a long-context quality evaluation.

Each JSONL record contains `request` (candidate API payload without a model
name), original usage, task identity, source and request hashes, and invocation
metadata. Select the target model explicitly. The response supplying original
usage is excluded from its own request. There is no padding or truncation.
Only one longest recorded successful call is retained per task; identical
canonical message histories are deduplicated.

The tool schema is explicitly inferred from the supplied local client source,
using AST literal extraction without executing it. Original server defaults,
tokenizer revision, chat template and exact HTTP serialization are unknown.
Original token counts must be rechecked on the target server. All 50 requests
share their first system message; disable prefix caching when testing
independent contexts. Distinct tasks can still share repositories and code.

## Replay the selector on CPU

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmark/context_certificate/replay.py \
  fixture benchmark/context_certificate/artifacts/synthetic-qsa.npz
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmark/context_certificate/replay.py \
  run benchmark/context_certificate/artifacts/synthetic-qsa.npz \
  --positions 1 4 8 12 \
  --output benchmark/context_certificate/artifacts/replay-synthetic.json
```

The fixture has 4,096 synthetic BF16-valued keys, four query heads of dimension
128, and 12 consecutive logical token positions. Complete index blocks become
visible every four tokens. It contains correlated synthetic queries, not model
activations. Prefix replay tests 24 combinations of position count, precision
and bound method. The existing `probe.py` additionally checks 30 large synthetic
cases with 65,536 keys and 48 adversarial/randomized configurations.

Replay distinguishes CPU certificate agreement, native selected-set agreement,
native output order, native winners excluded by screening, and native logits
escaping the intervals. A mismatch exits with status 2 and records diagnostics.
Input validation rejects unsupported scorers, bad dtypes, future keys,
duplicate/padded IDs, decreasing causal counts and empty experiment grids.
Native score ties are not permission to select different blocks: their
attention values may differ.

The score is `sum_h ReLU(q_h dot k) / sqrt(128)`. The bounds currently use
float64 and a numerical guard. A successful capture replay is empirical
evidence on those captured arrays, not a production GPU rounding proof.
The comparison is against **one shared BF16 index scan**, already amortized
over verification positions. Reported bytes do not include all inference work.

`codec.py` implements actual CPU 4/6/8-bit packing and unpacking, with FP32
scale and outward-rounded FP32 residual radius. A 128-dimensional key occupies
72/104/136 bytes respectively, including both sidecars, versus 256 bytes for
the original BF16 key. Independent formula/round-trip tests include zeros,
negative extremes, row padding and corrupt data. A full 65,536-key six-bit
index occupies 6,815,744 array bytes. Python objects, temporary unpacked
arrays and allocator overhead are excluded. Replay decodes to float64;
no GPU unpacking kernel or GPU traffic reduction has been measured.

## Capture contract for the later GPU experiment

One `.npz` file holds one request, one QSA layer and one chain verification
pass (or one decode position). Load with `allow_pickle=False`. Required arrays:

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `metadata` | Scalar Unicode JSON | Fields below |
| `keys` | `[N,128]` float32 | Exact BF16 values exported after finalized four-token compression, in logical block order |
| `queries` | `[S,4,128]` float32 | Exact BF16 inputs to the native index scorer, including native transforms/casts |
| `query_positions` | `[S]` int64 | Consecutive zero-based logical token positions, not RoPE coordinates |
| `valid_counts` | `[S]` integer | `(query_positions + 1) // 4`; final count equals N |
| `native_selected_ids` | `[S,K]` integer | Native logical compressed-block IDs, preserving order |
| `native_scores` | `[S,N]` float32 for model captures | Actual native logits; invalid columns ignored |

Required metadata:

```json
{
  "schema_version": 1,
  "source_kind": "model_capture",
  "model_id": "PIN_THE_CHECKPOINT",
  "model_revision": "PIN_THE_REVISION",
  "runtime_revision": "PIN_THE_BUILD_AND_KERNELS",
  "request_id": "CORPUS_REQUEST_HASH",
  "verification_id": "ROUND_ID",
  "layer_id": 0,
  "forward_mode": "target_verify",
  "compress_ratio": 4,
  "score_semantics": "sum_relu_dot_div_sqrt_dim",
  "top_k": 512,
  "key_dtype": "bfloat16",
  "query_dtype": "bfloat16",
  "keys_stage": "finalized_compressed_index_keys",
  "queries_stage": "native_index_score_input",
  "shared_key_bank_verified": true
}
```

Verify that every position sees the same content at each visible logical key
before setting `shared_key_bank_verified`. Physical page IDs and expanded
token IDs are not logical block IDs. If rows have different visible key
contents, export separate one-position captures; do not invent a shared bank.
Each one-position file ends its bank at that position's valid count. Initial
model captures require native FP32 scores as well as top-512 IDs. Native IDs
must also have top-k score rank; malformed captures or a native selector
disagreement with its own logits fail separately from certificate comparison.
This version deliberately excludes short contexts with fewer than K blocks,
tree speculation, and the unfinished block's up-to-three tail KV entries.

The inspected local source is Git commit
`4ccff141dbe992794f9da6c3aa23535b4f72000d` (`origin/qwen4-main-squashed`),
not the current checkout. Inspect it without switching branches:

```bash
git show 4ccff141dbe992794f9da6c3aa23535b4f72000d:python/sglang/srt/layers/attention/qsa/qsa_indexer.py
```

The capture boundary is `select_decode_tokens`, after compression/cache
updates and scoring, before `expand_qsa_block_indices`. Gather keys through
the logical page table. Paged queries may have eight heads: check that the
last four are zero before stripping them. The source scorer uses BF16 inputs,
FP32 tensor-core accumulation and fast math; preserve its actual logits.
Native top-k uses unspecified atomic tie order, and its coarse candidate bin
has a 4,096-entry cap. Check native selection rank instead of assuming it is
exact for every possible input distribution.

**The inspected backend permits only chain verification with at most four
positions.** Its pending index-key ring aliases positions modulo four.
S=8/12/16 requires redesign of pending state and write planning; changing a
launch flag is insufficient. The synthetic 12-position fixture and budget
remain research scenarios, not supported launch configurations.

Start with bounded eager captures, with CUDA graphs, FlashInfer autotuning
and server warmup disabled (see the runbook). Python hooks
inside graph construction do not run on every graph replay, and host copies
inside graph capture may fail. Capture timing is diagnostic, not performance
timing. At full context, one four-position file uses about 33 MiB before file
compression; limit requests, layers and rounds explicitly. Capture a few
representative cases first and transfer them for offline replay. A working
runtime capture hook has now been prepared as a source-pinned overlay and
tested on CPU, but has not been installed or GPU-validated. The
[first-capture runbook](CAPTURE_RUN.md) covers compatibility checks, bounded
capture settings and the decision criteria for real-array replay.

For throughput analysis, also record accepted prefix length per verification
round (including the output/bonus convention), actual overlapping request
lifetimes, expert routes, cache hits/misses and transfer bytes. Periodic
acceptance gauges alone cannot recover those distributions.

## Recompute capacity and traffic assumptions

```bash
python benchmark/context_certificate/budget.py \
  --gpu-count 2 --host-gb 235 --positions 4 --emitted-per-pass 3 \
  --refinement-fraction .1 --kv-union-multiplier 3 --kv-hit-rate .8
```

The example's acceptance, union sizes, cache hits and bandwidth are explicit
assumptions. It flags the original BF16 host archive's capacity failure on a
235-decimal-GB host. `--host-gib` is available when the measured host capacity
is in binary units. Do not confuse the two. The 309.87 GiB static archive
requires additional loading, process and prefill headroom.

The calculator reports ideal TP1/TP2 memory partitioning, bytes per pass and
per emitted token, required bandwidth, and optimistic traffic latency floors.
TP replication/imbalance, launch dependencies, compute, communication and
unmodeled work can increase costs. The shared host link does not double when
adding a GPU. An infeasible hot-cache hit rate or index refinement fraction is
flagged. Passing the arithmetic is necessary but insufficient for feasibility.
There is deliberately no throughput forecast.

The actual two-GPU deployment constraint is now approximately 235–250 decimal
GB of host RAM. `budget.py` retains the full host archive and correctly flags
that layout as too large. A separate
[partitioned layout calculation](constrained-host-layout.json) charges each
original KV page to one authoritative tier: 64 GiB on GPUs and 144 GiB on the
host. With the existing checkpoint's FP8 PLE and a 20 GiB host reserve, host
usage is about 211.5 GiB. GPU usage includes original indexes, weights, states,
workspace and an additional transfer reserve. Eight weight/replication/host
scenarios fit arithmetically. This requires a new ownership-aware memory path;
it is not implemented by a calculator flag or by the current serving stack.
Page migration adds bidirectional transfer work, and cache reuse is unmeasured.

Repeat a scenario with `--index-placement original-on-gpu` to compare a
resident BF16 index against the certificate. This charges the original target
and draft index storage, removes coarse/refinement traffic and index fetches,
and keeps the host archive unchanged. Refinement fraction is explicitly marked
inapplicable. Also vary `--hot-tokens` to account for the different KV-cache
capacity. More index compression is not automatically a faster system.

The saved TP2/S4/three-emitted-token resident-index example, with 8,192 hot
tokens, a three-position-size KV union and 80% entry hits, still requires
48.42 ms of shared-link payload time at 20 GB/s. The target permits 37.5 ms.
Those assumptions already fail the bandwidth test before adding omitted work;
this is a falsification of that particular scenario, not a throughput forecast.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s benchmark/context_certificate -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmark/context_certificate/probe.py
```

Tests cover request/response separation, tool linkage, deduplication, causal
masking, native mismatch detection, malformed captures, memory anchors,
packed codecs, traffic units, resident-index controls and unsupported runtime
windows. Capture tests additionally cover unchanged native selections, request
mapping, paged BF16 gathering, persistent quotas, graph/rank guards and native
acceptance count semantics. Actual model quality, GPU performance and full
32-by-256k residency remain unmeasured.
