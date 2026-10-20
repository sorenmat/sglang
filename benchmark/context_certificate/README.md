# Research proposal: exact QSA indexing with speculative memory planning

Date: 2026-09-25. **Status: design and synthetic CPU proof-of-concept, not a
working inference backend or a demonstrated throughput/quality result.**

The subsequent [local evidence audit](local-evidence-audit.md) anchors this
proposal to raw RTX benchmarks. Existing dual-card runs reach about 42–47
median decode tokens/s per request at concurrency 16 and much shorter inputs.
Earlier conversational 35–65 (one card) and 70–130 (two cards) estimates are
conditional bandwidth scenarios, not empirical forecasts for 32 full contexts.

The [GPU-free experiment kit](CPU_WORKFLOW.md) now provides a reconstructed
50-task workload, capture replay, causal/native-parity checks and a configurable
capacity/traffic calculator. Its source audit also found that the inspected
QSA backend limits target verification to four positions. Longer speculative
windows in this proposal require runtime state/write-plan changes.

The [first-capture runbook](CAPTURE_RUN.md) now accompanies an opt-in runtime
capture helper and source-pinned patch builder. Both have CPU tests; the
overlay has not been deployed or GPU-validated. The next experiment is a
bounded real-model capture to test selector agreement, refinement fraction
and selected-block reuse before implementing GPU kernels.

For the two-GPU variant, the current hardware constraint is approximately
235–250 decimal GB of host RAM. The earlier 512 GiB full-host-archive provision
does not fit that constraint. The [partitioned layout calculation](constrained-host-layout.json)
instead keeps 64 GiB of original target/draft KV exclusively on GPUs and
144 GiB on the host, with the checkpoint's existing FP8 PLE table unchanged.
All original index keys stay on GPUs. Including explicit reserves, this uses
about 211.5 GiB of host RAM and 84.3–89.5 GiB per GPU across the modeled
weight/index-replication cases. This is a conditional allocation, not a
working backend: GPU-resident KV cannot retain permanent host backing copies,
and page migration needs safe ownership transfer and bidirectional I/O.

The target remains one RTX PRO 6000 Blackwell, 32 independent contexts of
262,144 tokens, and **80 accepted output tokens/s for each request**. This
proposal does not replace that target with aggregate throughput, shorter
prompts, shared prefixes, or a smaller model. It assumes substantial host RAM
is allowed alongside the single GPU. Initial prompt processing is a separate
cost; it is not hidden in a claim of fast time to first token.

## Correct the model before designing the solution

The earlier discussion substituted Qwen3-Next-80B-A3B for the user's
Qwen3.8-Next. The likely intended model is **Qwen3.8-Flash-Next**, whose
[official configuration][config] and [model card][card] specify:

- 125B main parameters, 6B activated, plus 51B n-gram embeddings and 4B MTP.
- 48 layers: 36 Gated DeltaNet layers and 12 Qwen Sparse Attention (QSA) layers.
- QSA has 24 attention query heads, two KV heads, and head dimension 256.
- Its indexer has four query heads, one key head, and dimension 128.
- Four token index keys become one compressed index key. The indexer selects
  512 microblocks (2,048 tokens), plus up to three unfinished-block tokens.

Thus **the earlier 8.2 TB/s dense-FP8-KV read estimate is not this model's
decode baseline**. QSA already avoids reading most attention KV. It still
retains the full cache, scans its smaller index, and reads many expert weights.

The [SGLang implementation report][sglang] explicitly distinguishes the
compressed index keys from the original attention KV. Its current PLE host
offload and IndexShare draft-selection reuse are existing techniques, not
inventions in this proposal.

## Proposed contribution

**Quantize the searchable index, prove which candidates cannot win, and
retrieve original keys and values for all remaining uncertainty. Share that
work across a speculative verification batch.**

The potentially new portion is the combination of:

1. A joint error bound specific to QSA's four ReLU score heads, used to avoid
   full-precision *index score production*, rather than approximate attention.
2. Union refinement of uncertain index records over target-verification
   positions, followed by original-precision KV retrieval with separate masks
   for each position.
3. Speculative length and cache-admission decisions based on the combined
   expert-weight union, index-refinement union, and KV-miss union per accepted
   token. Predictions affect scheduling and retention, never target routes or
   selected attention membership.

This is a **narrow research hypothesis**. A literature search cannot establish
global novelty, and each broad building block has prior art, listed below.

### The new bound to test

Let `k_j` be the original BF16 compressed index key and `c_j` a compact
reconstruction with an outward-rounded radius `r_j >= ||k_j - c_j||_2`.
For the actual QSA score (scale included), verified in the
[pinned Transformers implementation][qsa-source]:

```
s_j = sum(h=1..4, ReLU(q_h dot k_j)) / sqrt(128)
```

For every subset A of the four heads, define `p_A = sum(h in A, q_h)`.
There are only 16 subsets, including the empty set. Then, exactly in real
arithmetic:

```
sum_h ReLU(q_h dot k_j) = max_A(p_A dot k_j)

L_j = max_A[p_A dot c_j - ||p_A|| r_j] / sqrt(128)
U_j = max_A[p_A dot c_j + ||p_A|| r_j] / sqrt(128)

L_j <= s_j <= U_j
```

All subset dot products come from subset sums of the four already-computed
head dot products. The 16 query norms can be obtained from a 4x4 query Gram
matrix. This exploits the fact that all heads share the **same key error**;
bounding four unrelated errors separately is often unnecessarily loose.
Use the intersection with independent-head intervals whenever it is tighter.

For each query position, let `tau` be its 512th-largest lower bound. Discard
only records with `U_j < tau`. At least 512 true scores are at least `tau`, so
no discarded record can be a winner. Refine the candidates from their original
keys and select the exact top 512. Equality remains a candidate. The union of
candidate records across verification positions is safe to fetch together;
membership remains independent for every position.

This is a **selection theorem**, not a claim that approximate attention errors
cannot accumulate through a network. Approximate index values never enter
the target model's final attention calculation.

The production certificate must include rounding errors from the coarse
scorer, original scorer, dequantization, reductions, and interval operations.
The CPU probe uses float64 and a generous numerical guard; it is not a
machine-checked GPU floating-point proof. It uses deterministic logical-ID tie
breaking. If the existing native top-k has different or unstable tie behavior,
reproduce that policy or fall back to its full original scorer at ambiguous
boundaries. Preserve attention ordering and accumulation semantics too.

### Storage and execution

1. During exact prefill, retain all original attention KV and original
   compressed index keys in pinned host memory. Never discard context tokens.
   Create compact copies only after the original four-token index block is
   finalized. Keep the incomplete block exact.
2. Keep compact index keys and error metadata on the GPU. Start with 8-bit
   signed quantization. Six-bit CPU packing is implemented in `codec.py`;
   an efficient CUDA format/kernel remains unimplemented. The 4-bit screen must earn its cost.
   The capacity ledger holds **one compact representation**, not simultaneous
   4/6/8-bit replicas. Changing codec must charge re-encoding/transfer costs.
3. Draft several positions using a small/quantized or otherwise improved
   drafter. An unchanged target model verifies every proposed position.
4. At each QSA layer, stream the compact index once across the verification
   positions, compute intervals, and fetch the union of uncertain original
   index keys. Compute their original scores.
5. Fetch missing original KV only for the exact selected microblocks. Union
   transfers across positions, but apply each position's own causal mask and
   selected set. Fetching a superset is allowed; attending to it is not.
6. Retain pages according to expected use in accepted prefixes and cache
   pressure. A miss or bad prediction causes an exact demand fetch; it never
   silently drops a required page. Use counters to check whether retention
   helps relative to LRU.
7. Commit only accepted recurrent updates and KV. Retracted proposals cannot
   pollute the next iteration's index, n-gram history, or GDN state.

The target queries for later layers are not available at the start of a
verification pass. Refinement is therefore **layer-local**, not a magical
lookahead of all future exact selections. Forecasts can initiate prefetch, but
their misses and synchronization remain in measured latency.

For GDN, retain the committed state, evaluate one layer's speculative sequence
using a scratch state, save the small recurrence inputs/gates, and discard
that layer's scratch. After acceptance, replay only accepted updates into the
committed states. This avoids retaining S full-model state snapshots. It is an
extension of existing replay ideas, not claimed as a new recurrence algorithm.

## Capacity budget

All following binary sizes are GiB; bandwidth uses decimal GB/s. Exact usable
GPU capacity must be measured (the local cookbook notes roughly 94 GiB under
its driver). A label saying 96 GB is not permission to allocate 96 GiB.

| Item, all 32 requests | Size or condition |
| --- | ---: |
| Full original BF16 attention KV, host | 192 GiB |
| Full original BF16 compressed index, host | 6 GiB |
| Original BF16 PLE table, host | 95.37 GiB |
| One-layer QSA MTP original KV + index, host, if separately retained | 16.5 GiB |
| Host subtotal before process/prefill/loading buffers | 309.87 GiB |
| Int8 index on GPU, FP32 scale + FP32 radius per key | 3.1875 GiB |
| BF16 hot KV, 5,120 positions per request per QSA layer | 3.75 GiB |
| Additional one-layer QSA MTP compact index + hot KV | 0.578125 GiB |
| Committed FP32 GDN state | 3.375 GiB |
| One GDN layer's batch scratch state | 0.09375 GiB |
| Saved recurrence inputs, convolution histories and metadata | reserve 0.7 GiB; measure |
| Main + draft weights | working allocation target <=78 GiB; unverified |
| Kernel, logits, selection, graph and transfer workspace | working allowance 4 GiB; unverified |
| Conditional GPU total | about 93.68 GiB |

**512 GiB host RAM is the design provision**; at least 384 GiB would be needed
to offer meaningful space above the static archive. This is one GPU, not a
96 GB self-contained machine. A deployment using an existing FP8 PLE table
needs less host memory but has a different quality baseline.

That is the original one-GPU/full-host-archive candidate. For the user's
two-GPU host with approximately 235–250 GB RAM, use the partitioned candidate
above; do not treat 512 GiB as available on that host. The existing `budget.py`
models full host backing and therefore correctly rejects it there. It does
not yet model authoritative page ownership split across host and GPUs.

The weight target is an engineering gate, not a measured allocation: local
recipe comments describe about 78 GiB remaining weights after PLE offload.
More specific older-export runtime notes in
`docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx:811` report **81.8 GiB
with speculation**, including loader-held memory. Substituting that figure
would make this budget about **97.48 GiB**, above the reported 94.2 GiB
process-usable capacity. A newer export has a smaller draft but no isolated
allocation in this session establishes the 78 GiB gate.
Runtime duplication, quantization padding, draft loading and graphs may exceed
it. Routed NVFP4 experts alone are 63.28125 GiB with one scale byte per 16
parameters. If the actual allocation does not fit, explicitly reduce and
validate the **drafter's** footprint, share immutable embedding weights, or
reject the memory design. Do not silently quantize the target further.

The probe's int8 metadata costs **8 bytes per index key**. Claiming 3.09375 GiB
would require only four metadata bytes, plus a sound reduced-precision
rounding design; it is not what the current probe models.

Naive S=8 GDN state snapshots would cost 27 GiB. The above memory budget
requires deferred commitment/replay. The 0.7 GiB reserve includes roughly
0.432 GiB of FP32 replay vectors/gates at S=12 and 0.066 GiB of target
convolution history; neither may be overwritten by rejected proposals.
It also requires bounded/tiled score
scratch: materializing lower and upper tensors for every layer, sequence,
position and key simultaneously can consume multiple GiB.

## Speed budget: expert reuse is essential

The workstation card's [published bandwidth][nvidia] is 1,792 GB/s. The
existing dual-card benchmark host records Server Edition GPUs; that host needs
its own measured bandwidth budget. The routed expert
parameters alone number:

```
P = 48 layers * 512 experts * 3 matrices * 2560 * 640 = 120,795,955,200
```

As an explicit scenario, assume each token chooses 10 distinct experts
uniformly and independently across tokens. Across 32*S target positions:

```
fraction_of_experts_read = 1 - (1 - 10/512) ** (32*S)
expert_bytes_per_pass = P * 0.5625 * fraction_of_experts_read
```

Actual routing is neither necessarily uniform nor independent. The scenario
also assumes one read per used weight in a pass; real GEMM tiling can reread
weights. Measure both effects, do not present this as a hardware theorem.

| Target positions S | Emitted/request/pass at conditional acceptance 0.90 | Routed weight bandwidth at 80 tok/s/request |
| ---: | ---: | ---: |
| 1 | 1.000 | 2.544 TB/s |
| 4 | 3.439 | 1.454 TB/s |
| 8 | 5.695 | 0.948 TB/s |
| 12 | 7.176 | 0.757 TB/s |
| 16 | 8.147 | 0.667 TB/s |

Emitted tokens include the correction/bonus token. For S target positions,
`A(S)=sum(i=0..S-1, p**i)` under a constant conditional acceptance p. This
assumption is illustrative; calculate empirical accepted-length distributions
per request and workload. At p=0.70, A(S) saturates below 3.34 even with a very
long draft. **Increasing draft length cannot fix a weak drafter.**

At S=12 and A≈7.176, there are about 11.15 verification passes/s. A necessary
illustrative GPU-traffic target is <=1.20 TB/s total (about 67% of peak):

- Routed experts consume about 0.757 TB/s in the scenario above.
- One int8 index scan/pass consumes about 0.038 TB/s before index refinement.
- Reading all selected BF16 KV separately for all 12 verifier positions,
  with no overlap savings, consumes about 0.216 TB/s.
- That leaves only about **0.189 TB/s** for remaining target weights, draft
  work, states, refinement, activations, metadata and transfer writes.

This is deliberately a restrictive budget. KV unions could create more
headroom, but cannot be assumed identical across positions. Native sparse
attention, fused index scoring, grouped MoE GEMMs and state replay must each
reach useful efficiency. FLOP throughput and sequential kernel latency can
still dominate even if this byte ledger fits. A rough target-only parameter
FLOP estimate is `2*6B*(32*80)*(S/A)`, about 51 TFLOP/s for S=12: this excludes
some attention, indexing, recurrence and draft work, and is not a measured
compute result or a comparison to sparse marketing TOPS.

For PCIe, the original index refinements and cold selected KV share the link.
Set an initial **<=20 GB/s total host-to-device payload** design gate, then
replace it with the measured bandwidth and latency budget on the actual host.
For illustration, a 10% original-index union fetch at S=12 is about 7.18 GB/s.
Only about 12.8 GB/s remains for KV. With the verifier KV union equal to
three times one position's selected set (6,153 positions), an 80% hot-cache
hit rate gives about 10.79 GB/s of KV transfers. The **5,120-position cache**
can hold the required 4,923 resident positions. A 4,096-position cache cannot:
its maximum pass-entry hit rate for that unique union is only 66.6%, making
total index + KV transfer exceed 25 GB/s. PLE lookups
are small (2,560 BF16 values/token), but random host access and scheduling
latency still count. Page granularity and fragmentation can inflate payload.

This example shows why **joint** acceptance, union-size, and miss-rate traces
matter. Averages from separate favorable workloads cannot be combined into a
claim that the target has been reached.

The controller minimizes measured time per accepted token using separate
GPU and PCIe costs:

```
time_per_output(S) = [max(overlappable_gpu_time, overlappable_io_time)
                      + nonoverlapped_compute_and_io_latency] / A(S)
```

Its inputs include routed-weight union, index-refinement union, KV-miss union,
draft cost, recurrence replay and kernel overhead. It must never pretend
unoverlappable fetches are free by placing everything under max().

## What is actually checked here

`probe.py` is an executable CPU reference. `probe-results.json` records its
deterministic synthetic results. It exercises 65,536 compressed index keys,
128 dimensions, four heads and top-512 selection; it also tests ties, zeros,
duplicate keys, all-selected cases and extreme rescalings.

It checks interval containment and exact selected logical IDs against a
full-precision NumPy scorer. Packed byte counts include scales and radii.
They are **modeled reads**, not measured GPU traffic: the CPU codec stores
packed arrays but the scorer uses decoded floating-point arrays. Synthetic
random keys are not captured Qwen activations.

The initial independent-head bound rejected 4-bit indexing: it retained
roughly 97–100% of keys and read more bytes than a full BF16 scan. The 8-bit
version retained 1.6% for one synthetic query and an 18.3% union for 12
independent queries, costing about 55–71% of one shared BF16 index scan.
This motivated the tighter joint-head bound above. All **30 large synthetic
cases and 48 adversarial/randomized configurations passed**. The final modeled
byte ratios to one already-shared BF16 index scan are:

| Synthetic queries | 6-bit independent bounds | 6-bit joint bounds | 8-bit joint bounds |
| --- | ---: | ---: | ---: |
| One query | 0.498 | 0.443 | 0.543 |
| Eight with small drift | 0.502 | 0.445 | 0.544 |
| Eight independent | 0.996 | 0.677 | 0.623 |
| Twelve independent | 1.115 | 0.775 | 0.666 |
| All scores tied | 1.406 | 1.406 | 1.531 |

Joint bounds materially improve six-bit economics in these examples. Eight-bit
still wins for large diverse query unions. These are not measured speedups:
bound computation and bit unpacking may cost more than the reads they save.
The complete results are in `probe-results.json`; do not treat them as model
quality evidence.

If intervals become uninformative, exact refinement can require the whole
index. Falling back *after* screening still pays the screen cost. A controller
can bypass screening on subsequent passes with bad observed economics; it
cannot retroactively erase bytes already read. No worst-case latency guarantee
is claimed for adversarial contexts.

Reproduce without installing a new dependency:

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmark/context_certificate/probe.py \
  > benchmark/context_certificate/probe-results.json
```

## Prior art and the novelty boundary

| Work | Established contribution | Difference being investigated here |
| --- | --- | --- |
| [QSA / SGLang][sglang] | Native microblock sparsity, PLE offload, IndexShare draft reuse | Exact target-index precision certificates, not a new sparse model |
| [HiSparse][hisparse] | Full host KV archive, small GPU cache, exact demand fetch | Joint verifier index refinement and transactional miss planning |
| [SpecSA][specsa] | Union loading of verifier attention blocks, per-query masks | Add certified index-key refinement and CPU transfer admission |
| [Guess-Verify-Refine][gvr] and [HPC-Ops][hpc] | Exact top-k over already-produced scores | Avoid original-precision score production itself |
| [HISA][hisa] and [MISA][misa] | Hierarchical approximate index filtering | Prove no native QSA winner was excluded |
| [Runtime-Certified Quantized Attention][rcqa] | Quantized KV, interval checks and staged bounded-error fallback | Quantize only index navigation; preserve native selected KV exactly |
| [EcoSpec][ecospec] and [DraftExpert][draftexpert] | Acceptance-aware expert-cost drafting/prefetch | Include certified index-refinement and KV-miss unions in the controller |

The 16-subset identity is elementary mathematics, not a claim to have invented
a new identity. The research contribution would be making its **shared-error
certificate economical in a QSA index codec and speculative serving path**.
An unresolved [ProofKV prior-art lead][proofkv] requires full-text inspection
before any publication-level novelty claim; the retrieved page was blocked.

## Falsifiable validation plan

1. **Freeze reference and quality baseline.** Pin checkpoint, quantization,
   tokenizer/template, sampler, RoPE, kernels and native top-k semantics.
   Compare the new memory path against the same unmodified QSA target first.
   Compare that target's NVFP4 quality separately against the original BF16
   model. Matching a quantized target is not proof that quantization was free.
2. **Capture real traces before implementing a serving backend.** At 32
   independent long contexts record QSA queries, compressed original keys,
   margins/ties, selected IDs, route unions, cache reuse, and accepted draft
   prefixes over hundreds of steps. Test prose, code, long documents, and
   deliberate topic switches. Include original index/metadata creation cost.
3. **Reject uneconomic certificates.** Replay 4/6/8-bit codecs, joint versus
   independent bounds, S=1/4/8/12/16. Require zero missed native winners and
   conservative roundoff coverage. Compare total bytes and measured kernel
   time against an already-batched original-BF16 indexer, including sidecars,
   union refinement and transfers. Do not compare to S redundant full scans.
   Also compare equal-capacity systems with the entire original BF16 index
   resident, making their memory tradeoffs explicit. The compact int8 index
   saves 2.8125 GiB of target index storage, but at S=12 the gross index-scan
   bandwidth saving is only about 34 GB/s before refinement. Its main value
   may be enabling a larger KV working set; it must still beat the latency
   cost of fetching original index candidates over PCIe. If loader/draft
   cleanup makes the resident original index fit and faster, reject this
   invention for that configuration rather than presenting it as a win.
4. **Prove the joint memory budget.** Measure allocated, reserved and peak
   VRAM, all 32 states, draft caches, rollback/replay, causal partial blocks,
   host pinned memory, NUMA placement, CUDA graphs and transient prefill.
   Test cancellation, slot reuse, retraction, and mixed acceptance lengths.
   No swap/thrashing or hidden second accelerator is allowed.
5. **Drafter gate.** Measure S/A and routed bytes per accepted token on the
   same long-context workloads. The local evidence audit finds sampled
   acceptance-length gauges around 2.8–3.0 on several coding runs, with the
   measured configuration capped at four outputs including the bonus token.
   These are not per-round distributions and do not support A≈7 at S=12.
   The inspected QSA backend at commit
   `4ccff141dbe992794f9da6c3aa23535b4f72000d` rejects windows larger than four
   because its pending index-key ring would alias. Redesign and validate that
   state/write plan before attempting longer verification windows; the
   existing runtime cannot obtain S=12 merely from a launch option.
   Train/improve the
   drafter if necessary, leaving target verification unchanged; abandon this
   performance path if the measured cost cannot fit. Exact rejection sampling
   is required for stochastic decoding, including its correction distribution.
6. **Numerical and task quality.** Test selector and KV parity against the
   reference, token/logprob parity where kernels are deterministic, and the
   stochastic sampler separately. Then run paired RULER-style retrieval,
   multiple needles, exact copy, counting/aggregation, far-apart dependencies,
   instruction precedence, long code tasks and representative user tasks.
   Pre-register an acceptable noninferiority margin with the user; a proposed
   gate is lower 95% paired confidence bound above -0.5 percentage points per
   major task family, no failures on deterministic preservation fixtures,
   and no unexplained long-context regressions. A small smoke set or favorable
   mean across unrelated tasks cannot establish this.
7. **End-to-end target.** Keep 32 independent requests active through the
   measurement interval, verify all full input lengths server-side, and count
   only accepted outputs. Require >=2,560 output tokens/s aggregate AND
   `(n_output-1)/(t_last-t_first) >=80` for every request over sustained runs.
   Report p50/p95/p99 TPOT, actual streaming burst gaps, per-stream fairness,
   failures, TTFT and cold prefill throughput. Admission serialization must
   not masquerade as concurrency. Test several independent seeds/workload
   batches and a sustained run long enough to expose cache churn.
8. **Context boundary.** Native 262,144 is input plus output. For literal
   262,144-token prompts followed by output, use a separately validated
   extended-position configuration identically for both reference and
   candidate. Do not label a 252K prompt with 4K reserved output as a literal
   256K-input test or silently truncate history.

Local evidence in `/home/smo/code/llm-performance/results/index.md:128` reports
the existing one-card MTP setup at about 451 aggregate tok/s for eight short
requests and a 58.9K-token GPU pool. It does **not** demonstrate this proposal.
Its reported per-stream TPS includes TTFT, so it is not interchangeable with
steady-state decode TPS. Existing MultiMask work in this repository targets
the 27B model and has documented quality/performance failures; it is not a
passing baseline for QSA Flash-Next.

See [the raw-result audit](local-evidence-audit.md) for additional dual-card
measurements, actual prompt lengths, cache-reuse confounds, quality limits and
the recorded host's 235 GB RAM constraint. The proposed original-precision
host archive is not a drop-in fit for that machine.

No live target GPU was available to this session: local `nvidia-smi` was absent
and read-only probes of known hosts did not establish access. No runtime was
changed. Therefore the following remain unproven: real-key certificate
economics, rounded GPU correctness, actual memory fit, high draft acceptance,
end-to-end quality, and the requested 32x256Kx80 throughput.

[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json
[qsa-source]: https://raw.githubusercontent.com/huggingface/transformers/f324707307757d9c0b8dac1c4462eceff911fa2f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
[card]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[sglang]: https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/
[nvidia]: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
[hisparse]: https://arxiv.org/html/2608.07009v1
[specsa]: https://arxiv.org/html/2605.19893
[gvr]: https://arxiv.org/html/2604.22312
[hpc]: https://arxiv.org/html/2609.08450v1
[hisa]: https://arxiv.org/html/2603.28458v3
[misa]: https://arxiv.org/html/2605.07363v1
[rcqa]: https://arxiv.org/html/2605.20868v1
[ecospec]: https://arxiv.org/html/2607.12696v1
[draftexpert]: https://arxiv.org/html/2607.24434
[proofkv]: https://openreview.net/forum?id=3Zmk1MTztm
