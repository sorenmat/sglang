# Plan: Migrate SGLang's CPU Control Plane to Rust

Status: in progress (phases 0–2 code complete + §4.2 replay backbone code complete + M2/1b SWA, M2/1c Mamba, M2/1d HiRadix, M2/1e unified multi-pool tree ports + M5 stream payload build + M6 spec accept-run bookkeeping on branch `rust-scheduler`; host-gated steps remain) · Owner: Soren · Target: Qwen3.8 (NVFP4) on RTX PRO 6000, 16 concurrent coding agents
Last updated: 2026-08-29

## 0. Goal and non-goals

**Goal.** Move the CPU-bound "front half" of SGLang serving — request queues,
radix/prefix cache, scheduling policy & admission, batch construction,
per-iteration result bookkeeping, streaming output assembly — into Rust,
while leaving model execution (CUDA/Triton/CuTe/FlashInfer kernels, Python
model code, speculative-decoding GPU paths) completely unchanged.

**Non-goals.**

- No GPU kernel work. No model code changes.
- No rewrite of the HTTP front: `rust/sglang-server` (the embedded Rust
  TokenizerManager + DetokenizerManager + OpenAI API, `SGLANG_RUST_SERVER`)
  already does that job. This plan builds on it.
- The paged KV allocator stays in Python/torch for the first phases — it
  manipulates GPU tensors (`free_pages` is a device tensor, `alloc_extend` /
  `alloc_decode` are CUDA kernels, `python/sglang/srt/mem_cache/allocator/paged.py`).
  Rust *plans* allocations; Python *executes* them.

**Why this pays** (recap of the analysis): on a fast NVFP4 model doing small
speculative-decode steps, per-iteration CPU bookkeeping (queue re-sorting,
radix traversals, lock-ref parent walks, stop-string scans, ~40-list batch
payload construction, detokenize passes) is a growing fraction of iteration
time — and sometimes it fails to hide behind GPU work at all. The
coding-agent traffic shape (tokenize → schedule → ~200 tokens → tool call →
4K tool result → radix lookup → schedule → …) touches the CPU far more often
per useful token than a long-batch benchmark.

## 1. Current state of this fork

### 1.1 What is already Rust

| Component | Where | Status |
|---|---|---|
| Embedded Rust HTTP + TokenizerManager + DetokenizerManager + OpenAI streaming | `rust/sglang-server` (~20.5k LOC), wired via `python/sglang/srt/managers/rust_server.py` (`RustServer`, gated by `SGLANG_RUST_SERVER`) | In-wheel, env-gated, no in-crate test suite yet |
| In-process Rust gRPC server | `rust/sglang-grpc` (~3k LOC) + `proto/sglang/runtime/v1/sglang.proto` | In-wheel, env-gated |
| Multimodal preprocessing (Qwen-VL family) | `rust/sglang-mm` (~2.9k LOC) | Shipped, has CI test suite |
| Standalone Rust router/gateway | `sgl-model-gateway/` (~94k LOC), mature CI/release pipeline | Production router, out of scope here |
| Slim KV-aware router (experimental) | `experimental/sgl-router/` (~35k LOC) | Out of scope here |
| Grammar FSM + sampling regex | `rust/sglang-server/src/utils/fsm.rs`, `src/message/sampling.rs` | In Rust already |

### 1.2 What remains Python and is CPU-hot (the migration surface)

All paths under `python/sglang/srt/`:

| Area | File (~LOC) | Hot functions |
|---|---|---|
| Scheduler event loop | `managers/scheduler.py` (5,479) | `event_loop_normal` / `event_loop_overlap` (1797/1832), `get_next_batch_to_run` (3194), `get_new_batch_prefill` (3363), `update_running_batch` (3717), `process_input_requests` (1955) |
| Batch & request state | `managers/schedule_batch.py` (3,560) | `Req` (837), `Req.init_next_round_input` (1342), `Req.update_finish_state` (1673), `ScheduleBatch.prepare_for_extend` (2442), `check_decode_mem` (2918), `retract_decode` (2925), `prepare_for_decode` (3181), `filter_batch` (3273), `copy` (3428, overlap ring) |
| Scheduling policy + admission | `managers/schedule_policy.py` (1,507) | `SchedulePolicy.calc_priority` (237, re-sorts waiting queue **every iteration**; LPM = full-tree `match_prefix` per queued req; DFS_WEIGHT = full recursive tree walk), `PrefillAdder.add_one_req` (1208, per-req budget math + `inc_lock_ref`), `add_chunked_req` (1004) |
| Radix cache (base) | `mem_cache/radix_cache.py` (863) | `match_prefix` / `_match_prefix_helper` (377/679, Python dict-tree traversal + `torch.cat` over per-node value tensors), `insert` (437), `evict` (593, heapify of all evictable leaves), `inc_lock_ref` / `dec_lock_ref` (623/638, O(depth) parent walk per request), `cache_finished_req` (459), `cache_unfinished_req` (516) |
| Radix variants | `mem_cache/unified_radix_cache.py` (3,043), `hiradix_cache.py` (2,028), `swa_radix_cache.py` (1,456), `mamba_radix_cache.py` (1,427) | same method set, more state (host tier, SWA dual counters, Mamba/GDN slots) |
| Native-tree prior art | `mem_cache/radix_cache_cpp.py` (275) + `mem_cache/cpp_radix_tree/` (C++ tree, `RadixCacheCpp`) | experimental; host tier `NotImplementedError`; no `cache_salt` |
| Memory pools | `mem_cache/memory_pool.py` (5,172) | `ReqToTokenPool.alloc/free` (292/340), `MambaPool` (377, ~800 LOC of GDN/SSM slot lifecycle), `HybridReqToTokenPool.alloc` (1357, ping-pong state slots) |
| Allocator | `mem_cache/allocator/paged.py` (346) + `swa.py`, `mamba.py`, `hisparse.py` | GPU-tensor free list, CUDA alloc kernels — stays in Python (planned in §7) |
| Per-iteration result processing | `managers/scheduler_components/batch_result_processor.py` (1,367) | `process_batch_result_decode` (869): per-req `output_ids.extend`, `update_finish_state(new_accept_len)` (stop-string window scans), grammar FSM advance, spec accept-run materialization |
| Streaming output assembly | `managers/scheduler_components/output_streamer.py` (774) + `managers/rust_server.py` `push_generation` (570) | builds ~40 parallel per-req lists into `BatchTokenIDOutput` every step; the Rust-server egress path still flattens columns in **Python** on the CUDA-launch thread |
| Detokenizer | `managers/detokenizer_manager.py` (560) | `_decode_batch_token_id_output` (301): per-req list slicing, two `batch_decode` passes per step, `trim_matched_stop` |
| Overlap ring | `managers/overlap_utils.py` (576) | `FutureMap` (248): pool-indexed GPU/CPU double-buffer for seq lens |
| IPC structs | `managers/io_struct.py` (2,491) | msgspec msgpack structs; `BatchTokenIDOutput` / `BatchStrOutput` carry ~40 per-req list columns |
| Speculative decoding CPU bookkeeping | `speculative/` (~17.6k LOC total) | `eagle_worker_v2.py` `on_verify_complete_cpu` (1356), `frozen_kv_mtp_*` (MTP), `dflash_info*.py`, `spec_utils.py` `move_accept_tokens_to_target_kvcache` (704) |

### 1.3 The scheduler loop today (what we are replacing)

```
recv_requests (zmq or Rust ring)          request_receiver.py:76
process_input_requests                    scheduler.py:1955
  └─ handle_generate_request → Req, tree_cache.match_prefix + inc_lock_ref,
     enqueue waiting_queue
plan = get_next_batch_to_run              scheduler.py:3194
  ├─ get_new_batch_prefill → policy.calc_priority (re-sort), PrefillAdder loop,
  │                          chunked-prefill continuation                      scheduler.py:3363
  └─ else update_running_batch → filter_batch, check_decode_mem (→ evict),
     retract_decode (sort + recheck), prepare_for_decode (alloc)             scheduler.py:3717
run_batch(batch)                          scheduler.py:3861   ← GPU, unchanged
process_batch_result(batch, result)       scheduler.py:4205
  └─ per-req finish checks, spec accept commit, radix cache_unfinished/finished,
     output_streamer.stream_output → msgpack → zmq (or Rust egress ring)
```

In `SGLANG_RUST_SERVER` mode the ingress/egress hops are already in-process
rings (`RustServer.drain` / `push_generation`, `rust_server.py:485/570`) and
the scheduler parks on the ring when idle (`RustServerIdleSleeper`,
`idle_sleeper.py:45`). **Everything between `recv_requests` and
`stream_output` above is still Python and runs on the GIL.** That is the
surface of this plan.

## 2. Target architecture

```
        RUST (in-process, PyO3, no GIL for planning)
 ┌──────────────────────────────────────────────────────┐
 │ rust/sglang-server (existing)                        │
 │   HTTP/OpenAI · tokenizer · detokenizer · streaming  │
 │ NEW: rust/sglang-scheduler (planner + core)          │
 │   waiting queue · running queue · admission          │
 │   schedule policy · chunked-prefill · retract        │
 │   GDN/SSM state-slot planning · per-iter bookkeeping │
 │ NEW: rust/sglang-radix (rlib, optional pyo3)        │
 │   radix tree core (match/insert/evict/lock_ref)     │
 │   variants: SWA, Mamba/GDN, HiRadix host tier        │
 └──────────────────────┬───────────────────────────────┘
                        │ BatchPlan (compact, value types)
                        ▼
        PYTHON (thin driver, unchanged GPU work)
 ┌──────────────────────────────────────────────────────┘
 │ Scheduler loop:  plan = core.step(ingress, gpu_state, prev_result)
 │                  apply plan → ScheduleBatch (few torch ops)
 │                  run_batch() → forward / sample (unchanged)
 │ PagedTokenToKVPoolAllocator, ReqToTokenPool, MambaPool
 │ CUDA/Triton/CuTe/FlashInfer, NVFP4, MTP/DFlash kernels
 └──────────────────────────────────────────────────────┘
```

Key properties:

1. **The boundary is a compact value type, not a Python object.** Rust owns
   `Req` state, queues, the radix tree, and budgets. Python receives a
   `BatchPlan` (rids, pool indices, lens, forward mode, alloc descriptor,
   retract list, cache ops) and does a small number of torch ops to launch
   forward. No Python-level tree/queue work remains on the hot path.
2. **Planning must stay sync-free.** The planner reads only CPU mirrors:
   CPU seq lens (`FutureMap.resolve_seq_lens_cpu`), `len(allocator.free_pages)`
   (known without a device sync — it's a tensor shape), the new-token-ratio
   tracker. This is what keeps the plan inside the overlap window and makes
   the "supposedly overlapped CPU phase serializes with GPU" failure mode
   structurally impossible.
3. **TP/DP correctness by determinism.** Every TP rank runs its own scheduler
   process with its own state copy; requests are broadcast at ingress
   (`request_receiver.py:153`). A deterministic Rust core fed the same ingress
   on each rank produces identical plans, so no cross-rank plan broadcast is
   needed — the existing request broadcast is sufficient.
4. **Rollback at every stage** via env flags (below); Python implementations
   stay in the tree until the Rust path has run the full e2e test suite green.

### Flag design

`SGLANG_RUST_SCHEDULER=<off|radix|planner|core|stream>` (additive stages; each
stage implies the previous):

- `radix` — base `RadixCache` replaced by the Rust tree (Python facade).
- `planner` — `next_batch`-style planning in Rust; Python loop still owns
  queues/results (stateless snapshot per iteration — the clean A/B stage).
- `core` — queues, budgets, result bookkeeping move into a persistent
  `SchedulerCore` pyclass; the Python loop becomes `plan = core.step(...)`.
- `stream` — per-iteration payload build + finish checks run in Rust
  (completing what `push_generation` currently does in Python).

(Implies `SGLANG_RUST_SERVER=1`; the detokenizer/streaming must be in Rust
for `stream` to close the loop.)

## 3. New crates

```
rust/
  sglang-radix        # new, rlib (+ optional pyo3 feature); no deps beyond thiserror/bytes
  sglang-scheduler    # new, rlib + pyo3 cdylib; depends on sglang-radix
  sglang-server       # existing; links sglang-scheduler for the `stream` stage
  sglang-mm, sglang-grpc  # untouched
```

- Both new crates join the `rust/` workspace, are exposed through
  `python/setup.py` rust-ext discovery (`[tool.sglang] rust-extensions` in
  `python/pyproject_other.toml` + `SGLANG_BUILD_RUST_EXTS`), and are lazy-loaded
  via `sglang.srt.rust_extensions.load_rust_extension` like the others.
- `sglang-radix` is deliberately dependency-free and testable without a GPU or
  torch; the tree's "values" are `i64` KV-index runs (arena-allocated,
  contiguous on match) so `match_prefix` returns a buffer the Python side can
  wrap into a torch tensor **zero-copy** — eliminating today's
  `torch.cat` over per-node tensors (`radix_cache.py` `_match_prefix_helper`).
- GIL rule: every PyO3 entry point does its work under `py.allow_threads()`;
  only the final "apply" step touches torch (which needs the GIL). Planning
  itself holds no GIL, so it can overlap Python-side CUDA launches.

## 4. Phase 0 — Baselines, instrumentation, trace capture (1–2 weeks)

Nothing ships. This phase produces the measurement floor every later gate
compares against.

### 4.1 Python microbench harness (`benchmark/scheduler/`)

Extend the existing `bench_token_storage.py` pattern into a reusable suite.
Each bench: realistic shapes, no GPU (CPU torch only), wall-clock p50/p99 over
many iterations, run on the *unmodified* codebase first to record baselines
into `benchmark/scheduler/baselines/*.json` (commit the JSON; CI nightly
re-checks against it).

| ID | Bench | Shapes |
|---|---|---|
| M1 | `RadixCache.match_prefix` | tree 1k/10k/100k/1M tokens; key len 1k/8k/32k; depth 1/16/128; ±EAGLE bigram view |
| M2 | `RadixCache.insert` | same tree sizes, fresh + high-overlap keys |
| M3 | `RadixCache.evict(n)` | evict 1k/10k/100k tokens from 100k-token tree; LRU + LFU |
| M4 | `inc/dec_lock_ref` walk | depth 1/32/256, 1k ops |
| M5 | `calc_priority` (fcfs / lpm / dfs-weight) | waiting 16/64/128/256 against live trees |
| M6 | `PrefillAdder.add_one_req` admission loop | batch of 16/64/256 candidates, chunked on/off, SWA/mamba budget on/off |
| M7 | `filter_batch` + `retract_decode` | running 64/256, memory pressure synthetic |
| M8 | `update_finish_state` stop-string scans | accepted runs 1/4/16, 0/4 stop strings, grammar on/off |
| M9 | `stream_output` payload build (`BatchTokenIDOutput`) | B=1/16/64/256, logprobs off/on |
| M10 | Detokenizer step (`_decode_batch_token_id_output`) | B=1/16/64/256, surrogate + read passes |
| M11 | End-to-end scheduler iteration (plan+result, no GPU forward) | replay of one captured coding-agent iteration |

Reference targets from upstream profiling (already-optimized Python): 128-req
filter ≈ 3.9 µs, finish-state processing ≈ 6.7 µs. Rust gates are set against
*these* numbers, plus p99.

### 4.2 Trace capture + replay

**Status (branch `rust-scheduler`):** capture is live at stage `core`
(`SGLANG_TRACE_SCHEDULER` JSONL: `cfg` header + ingress / plan / apply_result /
drop lines; raw token ids by default, `SGLANG_TRACE_SCHEDULER_TOKENS=hash` for
fingerprint-only captures). The replay side is implemented in
`python/sglang/test/scripted_runtime/replay.py`:

- `replay_core` — torch-free offline replayer: rebuilds a fresh
  `SchedulerCore` from the `cfg` header and re-drives the recorded op
  sequence, diffing every recomputed plan against the capture field-for-field
  (hard = decision disagreement, soft = tolerated drift; a hard divergence
  raises `ReplayError` instead of panicking inside the engine).
- `live_replay_script` — the target-host feeder: a `ScriptedContext` generator
  that submits the recorded ingresses into a live scheduler and mocks
  `run_batch` with the recorded sampled tokens, so the attached drivers write
  a second trace that `diff_sessions` can compare line-for-line (one script
  yield == one iteration == one recorded plan line).
- Tests: `test/registered/rust/test_rust_trace_replay.py` (CPU CI: lossless
  round-trip of a synthetic session, diff detection, schema validation) and
  `test/registered/scripted_runtime/test_rust_scheduler_live_replay.py`
  (CUDA CI: capture → engine+core reset → live replay → hard-diff assertion;
  needs the built `.so` via `cargo build -p sglang-scheduler --features
  python`).

**M2/1b (SWA dual-counter tree) is code-complete on this branch:**
`rust/sglang-radix/src/swa.rs` ports the `SWARadixCache` tree semantics
(dual full/SWA lock refs, `swa_tombstone` + uuid-based SWA-lock boundary,
intrusive dual LRU lists, window-validated match with the Python
list-slice-of-runs truncation quirk, dual-budget evict with cascade
tombstone-leaf deletion, and the insert tombstone-recovery branches
including locked-full recover). The allocator does not cross the boundary:
`free` / `free_full` / `free_swa` come back as value-run lists on each
result (`FreeOps` + `SWARecover`). Exposed as the `SWARadixTree` pyclass in
`sglang-scheduler` pybind; verified by 16 Rust unit tests (torch-free) and
`test/registered/rust/test_rust_swa_radix_parity.py` (CPU CI, differential
against the unmodified Python `SWARadixCache` driven through a recording
fake `SWATokenToKVPoolAllocator`). The Python-side `SWARadixCacheRust`
facade + default-flip wiring is M7.

**M2/1c (Mamba hybrid tree) is code-complete on this branch:**
`rust/sglang-radix/src/mamba.rs` ports the `MambaRadixCache` tree
semantics: the full/mamba lock-ref pair (full lock = node→root exclusive
walk, mamba lock = node only, invariant `full_lock_ref >= mamba_lock_ref`),
the dual LRU model (FULL list holds every non-root node including mamba
tombstones, MAMBA list holds only live-state nodes), `match_prefix` with
the run-count best-node check on the current node, the chunk-aligned
`mamba_branching_seqlen`, insert with `prev_prefix_len` partial frees
(carrying `start_pos`) and mamba-tombstone revival, and the two-phase
`evict` (full phase leaf-only with cascade counting; mamba phase with its
own fresh budget where internal nodes tombstone and leaf deletes free
full KV without counting it). The shared intrusive LRU was factored out
of `swa.rs` into `lru.rs` (used by both trees). Allocator calls come
back as `MambaFreeOps` (`free_segment(run, start_pos)` + mamba frees in
call order); int8-ckpt/active-pool routing and the deferred COW stay
caller-side. Exposed as the `MambaRadixTree` pyclass in `sglang-scheduler`
pybind; verified by 10 Rust unit tests (torch-free), 4 criterion benches
(match / insert / dual-budget evict / lock walk over the 256-agent shape
and a depth-256 chain), and
`test/registered/rust/test_rust_mamba_radix_parity.py` (CPU CI,
differential against the unmodified Python `MambaRadixCache` driven
through a recording fake `TokenToKVPoolAllocator` + mamba allocator,
with `mamba_cache_chunk_size` pinned to 64). The Python-side
`MambaRadixCacheRust` facade + default-flip wiring is M7.

**M2/1d (HiRadix host tier) is code-complete on this branch:**
`rust/sglang-radix/src/hiradix.rs` ports the `HiRadixCache` tree
semantics: device + host values as independent `Option`s (device
eviction DEMOTES a backed-up leaf to host-only or deletes a regular
leaf; host eviction DELETES the node), the contiguous-prefix backup
invariant, the write-through hit-count threshold (the tree reports
`backup_needed` instead of firing the DMA), the two-phase
`init_load_back` / `finish_load_back` / `abort_load_back` with the
temporary ancestor lock + permanent last-node chain lock,
`evict_host` with root/skip/parent-promotion rules, `insert_host`
(host-only nodes, split slicing of `host_value`), and the
write-back facade primitives the deprecated `_evict_write_back` loop
is decomposed into (`detach_backuped`, `drop_subtree_no_host`,
`promote_parent`, `evictable_leaves_ordered`). Heaps are rebuilt per
call from the `evictable_leaves` / `evictable_host_leaves` sets with
stale-entry filtering at pop time, exactly like Python's
`heapq` snapshot; the shared `EvictionPolicy`/`Prio` ordering was
factored out of `policy.rs` so all four trees use one definition.
Exposed as the `HiRadixTree` pyclass in `sglang-scheduler` pybind
(constructor takes `page_size, is_eagle, write_policy,
eviction_policy, write_through_threshold, load_back_threshold`);
verified by 15 Rust unit tests (torch-free), 5 criterion benches
(match / insert / evict / evict_host / lock walk over the 256-agent
shape and a depth-256 host chain), and
`test/registered/rust/test_rust_hiradix_parity.py` (CPU CI,
differential against the unmodified Python `HiRadixCache`
constructed without its `__init__` — `__new__` + the attributes the
tree ops touch + the real `reset()` — driven through a recording
fake controller whose `write` / `load` hand out deterministic arange
runs that the test feeds back into the Rust tree via
`begin_backup` / `finish_load_back`). The Python-side
`HiRadixCacheRust` facade + default-flip wiring is M7.

**M2/1e (unified multi-pool tree) is code-complete on this branch:**
`rust/sglang-radix/src/unified/` ports the `UnifiedTreeCore` tree
semantics (the multi-pool `unified_radix_cache.py` engine): one node
arena carrying a device + host value list per component (FULL / SWA /
Mamba / C128), per-component intrusive LRU lists (device + host layer)
with the FULL last-access-time strategy, per-namespace child keys
(`extra_key` / `cache_salt` as the `ns` u32 under the single root), the
stepped resumable insert walk (`begin_insert` / `resume_insert` /
`end_insert` with the split / unevict / overlap-claim barriers that emit
`FreeDeviceKV` / `ReplaceWriteThroughOnNodeSplit` / Mamba-excess
actions), the multi-validator match with device/host boundary tracking
and host-hit lengths, the inc/dec lock-ref walks (FULL node→root, SWA
uuid-bounded, Mamba node-only, skip-id replay), the stepwise
device-eviction walks (leaf heap for FULL, LRU cursor for SWA/Mamba)
with demote vs delete vs deferred write-back backup
(`evict_device_leaf` → `BackupKV` → `commit_backup` + `demote_node`),
`drop_subtree_no_host`, the host-tier `insert_host` +
`drive_host_eviction` (write_back duplicate-host reclaim + rolling
digest), the write-through / load-back pending marks and their commits,
the KV placement event log, `sanity_check`, and node dumps. No torch
crosses the boundary: component values are plain pool-index lists and
allocator calls come back as action / value-run lists the controller
drains. Exposed as the `UnifiedRadixTree` pyclass in
`sglang-scheduler` pybind (module constants `CT_*` + `PHASE_*`) and
wrapped as `UnifiedTreeCoreRust` behind the
`UnifiedTreeCoreInterface` (`python/sglang/srt/mem_cache/unified_cache/
tree_core_rust.py`, registered as the `rust` tree-core backend via
`tree_core_registry.py` — env
`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`); the Python components
keep only their facade-level hooks under the Rust backend, the
tree-level FULL/SWA/Mamba hooks run inside the engine. Verified by 56
library + 16 integration Rust tests (torch-free), 15 criterion benches
(`unified_bench`: stepped insert, full / split / partial match,
write-through + write-back full drains and partials, insert_host +
drive_host_eviction, lock walks, KV-canary walk, clone, over the
256-agent shape), and
`test/registered/rust/test_rust_unified_radix_parity.py` (CPU CI,
differential against the unmodified Python `UnifiedTreeCore` with the
real `FullComponent`, FULL-only, driven by one shared op script:
match / insert / split / lock / full-drain / un-evict / namespaces in
the main script, and the write-back backup-demote round trip +
insert_host + host-pressure reclaim in a second script; node identity
by root path, freed values compared as multisets, full drains only —
same LRU tie-break caveat as the other parity tests). The facade
default-flip + upstream RFC PRs are M7.

Host-gated remainder: record the two canonical target-hardware sessions
(below) and the Python-side M1–M11 baselines for the A/B numbers.

- **Capture.** New env `SGLANG_TRACE_SCHEDULER=<path>`: per iteration, dump
  one JSONL line — ingress reqs (rids, token-id hashes + lens, sampling params),
  queue snapshots (rids, lens, priorities), plan (forward mode, per-req lens,
  alloc counts, retract rids), cache ops (evict n / insert lens / unlock rids),
  result (accepted lens, finished rids + reasons). Token ids are hashed or
  stored verbatim (configurable) to keep files small; captures are debug
  artifacts, not shipped.
- **Replay.** `sglang.test.scripted_runtime` already has
  `ScriptedSchedulerHook` + `ScriptedTokenizerRecvProxy`
  (imported by `request_receiver.py:42`): extend it to *feed recorded ingress
  sequences* into a live scheduler with a mocked `run_batch` that returns
  recorded GPU results. The same recorded session then drives both the Python
  and the Rust paths; outputs (plans, cache state hashes, stream payloads) are
  diffed field-by-field. This is the correctness backbone of every phase.
- **Record two canonical sessions** on the target hardware (Qwen3.8 NVFP4,
  RTX PRO 6000): (a) 16 coding agents, multi-turn with ~4K tool results and
  heavy prefix reuse; (b) a high-rate short-turn session. Commit the traces
  (anonymized) to `test/scripted_runtime/traces/`.

### 4.3 End-to-end baselines

- `python/sglang/bench_serving.py` at c1 / c4 / c16, plus the multi-turn
  coding-agent-shaped workload (reuse `benchmark/` multi-turn tooling; if
  absent, add `benchmark/scheduler/bench_coding_agents.py`: multi-turn
  conversations, tool-call rounds, 2–8K tool-result injections, shared
  system prompts to force radix reuse).
- Record: throughput (tok/s), TTFT p50/p99, ITL p50/p99/p999, e2e p99,
  scheduler-thread CPU%, and a `torch.profiler` profile isolating
  `scheduler.*` spans (the nvtx method decorators,
  `scheduler.py` `@scheduler_nvtx_method`, already name them).
- Store in `benchmark/scheduler/baselines/e2e_*.json`. These are the A/B
  numbers of record; every phase re-runs the identical script.

**Phase 0 exit gate:** baselines JSON committed; trace capture verified lossless
(replaying a trace through unmodified Python reproduces the original plans);
M1–M11 baselines recorded.

## 5. Phase 1 — Radix tree core in Rust (`sglang-radix`)

The single most bounded, highest-leverage component: pure data structure,
no torch dependency, and the C++ tree already proves the abstraction.

### 5.1 Scope

- Port `RadixCache` base semantics (`mem_cache/radix_cache.py`):
  `RadixKey` (page alignment, `extra_key`, `cache_salt`, EAGLE bigram view),
  `match_prefix`, `insert` (with duplicate-free semantics),
  `cache_finished_req` / `cache_unfinished_req` accounting,
  `evict` (LRU/LFU/FIFO/MRU/FILO from `mem_cache/evict_policy.py`),
  `inc/dec_lock_ref`, `evictable_size` / `protected_size` / `total_size`.
- Node values: `i64` KV-index runs in an arena; `match_prefix` returns one
  contiguous buffer (or a single fused allocation) + the terminal node handle;
  `last_access_time` is a monotonic counter (not wall clock, for determinism).
- `TreeNode` handle semantics match the Python/C++ `last_node` API so
  `req.last_node` bookkeeping is unchanged at call sites.
- Python facade `RadixCacheRust(BasePrefixCache)` in
  `mem_cache/radix_cache_rust.py` — same `MatchPrefixParams`/`InsertParams`/
  `EvictParams` dataclasses as the base class, so `PrefillAdder` and
  `Req.init_next_round_input` work unmodified.
- **Decision:** `RadixCacheCpp` becomes the *oracle* for parity tests, not a
  parallel path; the Rust tree replaces it (its write-through host-tier stubs
  fold into the Phase 1d HiRadix port).
- Variants, in order (each a separate sub-PR with its own gate):
  - 1a. base MHA tree (page_size 1 and 64)
  - 1b. SWA dual-counter tree (`swa_radix_cache.py`)
  - 1c. Mamba/GDN hybrid tree (`mamba_radix_cache.py` — includes the
    GDN/SSM state-slot accounting that rides on the tree)
  - 1d. HiRadix host tier (`hiradix_cache.py`, write-through policies,
    `cache_controller` handoff — the C++ tree's `ongoing_write_through` set
    semantics)
  - 1e. unified multi-pool tree (`unified_radix_cache.py`, last)

### 5.2 Correctness

- Property tests: random insert/match/evict/lock sequences → invariants
  (prefix-closure, lock-ref ⇒ no-evict, evict budget exactness, value
  continuity), plus differential testing: same op sequence against
  Python `RadixCache` and `RadixCacheCpp`, assert identical
  `match_prefix` lengths, node handles' token ranges, and eviction sets.
- Trace parity (§4.2): replayed session's cache-op log must match
  op-for-op, and tree-state hashes (per-node token range + lock count) must
  match at every iteration.

### 5.3 Microbench gates (criterion, `sglang-radix/benches/`)

M1–M4 Rust versions, identical shapes to §4.1. Gate:
**match_prefix / insert ≥ 10× the Python baseline p50 at 100k-token tree;
evict(10k) ≥ 5×; lock_ref walk ≥ 10×; zero trace-parity diff; e2e A/B neutral
or better at c16** (first visible e2e signal: LPM re-scoring in M5 stops
paying Python tree cost, though the policy itself is still Python in this
phase).

## 6. Phase 2 — Rust planner: `next_batch → BatchPlan` (the clean A/B)

Stateless-or-shallow state: Rust replans from compact snapshots each
iteration; Python still owns the actual queues and result application.
This is exactly the prototype shape in the working analysis:

```rust
pub fn next_batch(
    running: &[ReqSnapshot],
    waiting: &[ReqSnapshot],
    radix: &RadixQuery,        // or &Rc<RefCell<RustRadixTree>> in 1b+
    gpu_state: &GpuState,      // cpu seq lens, free_pages.len(), new_token_ratio, budgets
    policy: &PolicyConfig,     // schedule_policy flag, chunked_prefill_size, max_running...,
                               // i.e. the ~15 server_args flags of §1.2
) -> BatchPlan
```

### 6.1 What moves

- `get_new_batch_prefill` / `_get_new_batch_prefill_raw` (scheduler.py:3363/3390):
  policy re-sort (`calc_priority` — with LPM querying the Rust tree directly,
  DFS_WEIGHT as an iterative tree walk), `PrefillAdder` admission loop
  (budget gates: `rem_total_tokens` / `rem_swa_tokens` / mamba-gap, page
  alignment, max_new clamping), chunked-prefill continuation.
- `update_running_batch` planning (3717): `filter_batch`, `check_decode_mem`
  (→ evict plan), `retract_decode` (retraction order + recheck loop).
- `prepare_for_extend` / `prepare_for_decode` *planning* (lens, pool indices,
  alloc descriptors) — the actual tensor construction stays in Python.

### 6.2 `BatchPlan` (compact, value-type, msgpack-serializable for traces)

```rust
pub struct BatchPlan {
    pub forward_mode: ForwardMode,          // Extend | Decode | Mix | Idle
    pub reqs: Vec<PlanReq>,                 // rid index, req_pool_idx, prefix_len,
                                            // extend_len/seq_len, sampling slot, priority
    pub extend_tokens: i64,                 // total prefill tokens this batch
    pub alloc: AllocDescriptor,             // pages for extend + decode + GDN/SSM
                                            // slot requests (pool idx, count)
    pub retract: Vec<RetractOp>,            // rid, drop_len
    pub cache_ops: Vec<CacheOp>,            // Evict(n) | Insert(rid,len) | Unlock(rid)
    pub spec: Option<SpecPlan>,             // Phase 5 placeholder: verify/draft shapes
}
```

Python `apply_plan(plan)` builds `ScheduleBatch` with a handful of torch ops
(gather seq-lens, allocator `alloc_extend`/`alloc_decode` with the planned
`num_new_pages`, `req_to_token` writes). Target: **< 100 Python-level
operations per iteration** in the apply step.

### 6.3 Determinism contract

Greedy A/B parity requires Rust plans == Python plans for identical inputs:
int-only budget math (all quantities are token/page counts — no floats),
stable tie-breaking (documented order: priority, then arrival seq, then rid
hash), same policy-degradation rules (e.g. LPM→FCFS when waiting > 128,
`schedule_policy.py:_determine_active_policy`). Traces (§4.2) are the
regression test: **plan-for-plan match on the canonical sessions.**

### 6.4 Gates

- M5–M7 Rust versions: planner total ≤ 50 µs at waiting 128 / running 256
  (Python baseline to be measured in Phase 0; the upstream micro-opts imply
  Python is in the low-µs regime after filtering/finish-state fixes, so the
  Rust target is "≤ ½ of Python p50, and no p99 spikes from allocation churn").
- Plan-for-plan trace parity on both canonical sessions (greedy).
- e2e A/B: Qwen3.8 NVFP4 c1/c4/c16. **This is the decision point the user
  specified:** if throughput is flat at c16 and ITL p99 is flat, the GPU is
  the bottleneck for this workload — stop here, ship the Rust radix (Phase 1)
  for the cache-hit path, and close the project. If throughput improves ≥ 2%
  at c16 (or ITL p99 improves ≥ 10%), proceed to Phase 3.

## 7. Phase 3 — `SchedulerCore`: persistent state in Rust

Turns the Phase 2 snapshot into owned state, removing the per-iteration
snapshot/apply overhead and moving all per-iteration bookkeeping off the GIL:

- `SchedulerCore` pyclass owns: waiting/running queues (`Vec<Req>`), budgets,
  new-token-ratio tracker, the Rust radix tree (from Phase 1), GDN/SSM
  state-slot allocator bookkeeping (`MambaPool`/`HybridReqToTokenPool` slot
  lifecycle planning — the *planning* of ping-pong slots; the pool tensors
  stay in Python), and the retract/reuse accounting.
- Python loop becomes:

  ```python
  while True:
      ingress = self.request_receiver.recv_requests()      # unchanged (Rust ring)
      result  = self.run_batch_and_wait(plan_from_last_iter)  # unchanged GPU path
      plan    = self.core.step(ingress, self.gpu_state(), last_plan, result_snapshot)
      batch   = apply_plan(plan)                            # thin torch glue
      ...
  ```

- Result bookkeeping (`batch_result_processor.py`): `update_finish_state`
  token-based checks, accepted-run commit (incl. MTP/DFlash accept-length
  materialization), `cache_unfinished_req`/`cache_finished_req` calls — all
  into `core.step`.
- Overlap mode: `core.step` runs while the previous forward is in flight
  (it only reads CPU mirrors — §2.2), so the overlap window is *genuinely*
  overlapped; the WAR-barrier / `is_disable_overlap_for_batch` logic moves
  into the core and must be replicated exactly (it disables overlap for
  consecutive prefills and grammar sync — `scheduler.py:1906`).

**Gates:** M8 (finish checks) and M11 (full iteration) Rust ≤ ½ Python p50;
trace parity incl. retract + GDN slot allocations; e2e c16 throughput ≥ the
Phase-2 number + further improvement or ITL p999 ≥ 10% better; overlap-mode
on with no correctness regressions (the scripted-replay overlap test).

## 8. Phase 4 — Per-iteration output pipeline in Rust (`stream` flag)

Today, even with `SGLANG_RUST_SERVER`, the Python CUDA-launch thread builds
the egress payload: `rust_server.py:push_generation` flattens `output_ids`,
computes `tok_lens`, packs ~40 columns (and up to 7 logprob/hidden families —
the code comments record a measured 0.37 ms → 7.90 ms GIL-held regression
when that flatten ran unconditionally). Move:

- `output_streamer.py` payload construction → Rust: the core already has
  rids, lens, finish reasons, logprob column shapes; it emits the
  columnar frame directly into the egress ring (header + raw int32/f32
  buffer), matching the existing `BatchHeader`/`for_each_chunk` wire
  contract in `sglang-server`.
- String-based stop checks: the detokenizer (Rust, in `sglang-server`) owns
  the decoded text; move `_check_str_based_finish` /
  `_locate_str_based_finished_len` next to it, feeding decisions back into
  `core.step` instead of detokenizer→streamer→processor ping-pong.
- Detokenizer step microbench M10 already exists as a Rust subsystem; this
  phase just closes the handoff.

**Gates:** M9 Rust ≤ ½ Python p50 at B=256; trace parity on stream payloads
(byte-identical egress frames on the canonical sessions); ITL p99 improved at
c16 (this is where the tail-latency win is expected to show up first).

**M5 (stream payload build + string-stop decisions) is code-complete on this
branch:** `rust/sglang-server/src/stream.rs` builds the whole egress frame in
Rust — the msgpack `BatchHeader` (4 core columns, +12 shape columns when any
of the 7 logprob/hidden families is active), the ragged/hidden flattens, the
f32/i32 LE data buffers, and the `[BATCH][len][header][data…]` framing,
byte-compatible with the `for_each_chunk` decoder; `stream_py.rs` exposes
`build_generation_frame` + `Server.push_generation_frame` (the GIL-heavy
flatten the Python `push_generation` used to do) and moves the string-stop
decisions next to the detokenizer (`tokenizer_manager/stop_check.rs`:
`stop_match_tail_len`, `stop_prefix_match`, `locate_str_stop_len`,
`check_str_stop` — the stop-string / stop-regex tail + trim-length logic of
`Req._check_str_based_finish` / `_locate_str_stop_finished_len`).
`rust_server.push_generation` takes the Rust path under the existing staged
flag `SGLANG_RUST_SCHEDULER=…|stream` (implied by `SGLANG_RUST_SERVER`),
shipping the collected columns (`rids`, finish reasons, `prompt_tokens`,
`tok_lens`, flat ids, the 7 optional families) to `push_generation_frame`.
Verified by 264 `sglang-server` unit tests (frame round-trips through the
real `for_each_chunk` decoder — core-4, all-families + hidden rows,
inactive-family empty-column spelling, NaN sentinel, finish-reason maps;
stop-check decisions over the hand-checkable char tokenizer), the M9
criterion benches (`stream_bench`: B=1/16/64/256 × logprobs off/on —
~15 µs / ~645 µs at B=256 in release), and
`test/registered/rust/test_rust_stream_parity.py` (CPU CI, differential:
byte-identical frame header + data vs the unmodified Python packing,
string-stop decisions vs the unmodified `Req` methods, and the M9 gate
Rust p50 ≤ ½ Python p50 at B=256), with a dedicated `sglang-server-unit`
job in `pr-test-rust-exts.yml`. The per-request collection (rids / lens /
finish reasons off the `Req`s) still runs in Python — the core owning that
state is the Phase-4 remainder; the canonical-session trace parity + ITL p99
gates are host-gated.

**M6 status (code complete on `rust-scheduler`):** the CPU accept-run
resolution of `_resolve_spec_v2_tokens` is ported to `sglang-scheduler`
(`src/spec.rs`: `resolve_spec_runs` — stride-padded slice + grammar-retained
substitution + retracted/finished gate + the four result fields, and
`SpecCounters` — the `Req` spec counters + the two growable histograms).
The grammar FSM (per-req xgrammar object) stays in Python:
`advance_grammar_fsm` runs first and its `grammar_retained_tokens` feed
Rust as an input column; the torch-side KV move and the adaptive
controller feed also stay Python. `_resolve_spec_v2_tokens` takes the Rust
path under `SGLANG_RUST_SCHEDULER≥core` (Python still settles the per-req
counters it reads off the `Req`s for the egress payload, and records
`resolved_spec_*` on the result so the core driver folds the same rows into
the core's per-req `SpecCounters` via the new `ResultRow.spec` metadata —
`apply_result` is the only bookkeeping path). The synthetic replay session
now carries a spec-v2 row (iter 3, rA) so the lossless-replay backbone
round-trips the metadata. Verified by 38 `sglang-scheduler` unit tests
(resolve edge cases + counter/histogram growth + the core spec-row path),
the M6 criterion benches (`spec_bench`: B=1/16/64/256 × MTP/EAGLE stride ×
grammar/block-cap), and
`test/registered/rust/test_rust_spec_parity.py` (CPU CI, differential:
accepted runs byte-parity vs the Python commit contract, counters vs the
production `Req` histogram methods where torch is available, and the
live-core spec-row path). The accepted-run trace gate on a real MTP/DFlash
session + the e2e spec-workload throughput gate remain host-gated.

## 9. Phase 5 — Speculative-decoding bookkeeping (MTP/DFlash)

Only after Phases 3–4 land and the target workload runs MTP/DFlash.

- `on_verify_complete_cpu` (eagle_worker_v2.py:1356),
  `_resolve_spec_v2_tokens` / `_accept_grammar_tokens`
  (batch_result_processor.py:691/768), `move_accept_tokens_to_target_kvcache`
  (spec_utils.py:704): accept-run resolution and commit into `core.step`
  (`SpecPlan` fills out).
- Grammar FSM advance is already Rust (`sglang-server/utils/fsm.rs`);
  `advance_grammar_fsm` mid-iteration call (`scheduler.py:_advance_pending_grammar`)
  routes through the existing FSM module instead of Python.
- Per-iteration `BatchTokenIDOutput` spec fields (`spec_verify_ct`,
  `spec_num_correct_drafts`, acceptance histograms) become core-emitted
  columns (Phase 4 wire).

**Gates:** accepted-run byte parity vs Python on spec traces (record one
MTP + one DFlash session in Phase 0.5 if the target model ships both);
e2e spec-workload throughput ≥ Python at c16.

## 10. Phase 6 (optional) — Allocator & pools in Rust

Defer until Phases 1–4 show allocator/free-list maintenance is still on the
critical path (measure: `torch.unique(free_index // page_size)` in
`PagedTokenToKVPoolAllocator.free`, `merge_and_sort_free`, GDN slot ping-pong
bookkeeping). Options if so:

- Page-level bookkeeping in Rust + a thin CUDA kernel for the actual
  index writes (replacing `alloc_extend_kernel`/`alloc_decode_kernel`), or
- Keep allocator in Python but batch frees through the Rust core's
  `free_group_begin/end` already planned by `CacheOp`s.

Not scheduled; decision is data-driven from Phase 3 profiling.

## 11. Microbenchmark & A/B discipline (applies to every phase)

1. **Two-tier measurement, always both:**
   - *Micro* (M1–M11): criterion benches in `sglang-radix/benches` and
     `sglang-scheduler/benches`, parameterized over the §4.1 shapes; Python
     baselines live in `benchmark/scheduler/` and are re-run in the same CI
     job so drift is visible.
   - *Macro* (E1–E3): `bench_serving` c1/c4/c16 + the coding-agent
     multi-turn workload, identical flags, 3 runs, median; recorded to
     `benchmark/scheduler/results/<phase>_<date>.json`.
2. **Every PR touching a hot path carries:** (a) the diff vs the phase's
   baseline JSON in the PR description, (b) trace-parity pass on the
   canonical sessions, (c) clippy + unit tests. No PR merges a hot-path
   change on "it should be faster" without numbers.
3. **Nightly benchmark CI** (extend `pr-benchmark-rust.yml`): `cargo bench`
   with criterion, 10% regression threshold on tracked benches; nightly
   re-run of Python M-benches against the committed baselines to catch
   *Python-side* drift too (important: an upstream Python micro-optimization
   can silently close a Rust lead — the 25.8 µs→3.9 µs filter example).
4. **Profiling cadence:** per phase, one `torch.profiler` + nvtx
   (`@scheduler_nvtx_method` spans) session at c16 to re-locate the
   scheduler's remaining CPU cost; the phase plan is updated from the
   measured hotspot, not from the original list.
5. **Optimization micro-targets** (candidates to benchmark explicitly,
   recorded in `benchmark/scheduler/` as they are touched):
   - arena vs per-node allocation for tree values; slab size for `PlanReq`
   - iterative vs recursive DFS_WEIGHT walk; LPM candidate pruning
   - `Vec` pre-sizing of plan columns from `max_running_requests`
   - radix `match_prefix` single-pass with deferred node-split
   - payload column packing: one `concat` vs per-family copy
   - GDN slot free-list: LIFO vs LRU, with ping-pong constraint

## 12. Testing & correctness strategy

- **Unit:** each Rust module; property tests for the tree (§5.2); fuzz
  `core.step` with random ingress + result sequences (assert invariants:
  no double-free of pages, lock counts ≥ 0, plan allocs ≤ free pages,
  deterministic replay).
- **Parity (the backbone):** scripted-replay trace diffing (§4.2) — plans,
  cache-op logs, egress frames — on the canonical sessions, run in CI
  (small session) and nightly (full sessions).
- **Existing suite:** the full `test/srt` pytest suite must pass with
  `SGLANG_RUST_SCHEDULER` at each flag stage before the stage's default
  flips; CI matrix adds one GPU job per stage (`radix`, `core`), reusing
  `_pr-test-rust-ext-build.yml` for the extension build cache.
- **Determinism:** greedy-decode e2e outputs byte-identical between
  Python and Rust paths for the same seed (this is also the A/B
  correctness check for any throughput claim).

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GIL/CUDA-stream affinity: Rust planning on a foreign thread breaks stream or torch C-API assumptions | Core does **no** CUDA work; torch is only touched in `apply_plan` on the scheduler thread; planning uses CPU mirrors only (§2.2). Fuzz + scripted-replay overlap tests. |
| Overlap-ring semantics (`batch.copy()`, WAR barrier, one-behind result queue) subtly broken | Phase 3 gate includes an explicit overlap-mode replay test; `is_disable_overlap_for_batch` rules replicated and diffed trace-by-trace. |
| Floating drift between Python and Rust decision logic (tie-breaks, degradation rules) | Determinism contract (§6.3); plan-for-plan trace parity is a CI gate, not a spot check. |
| HiRadix host tier / DP attention / PD-disagg long tail | Staged out: base tree + single-node first; DP-attention plan replication is by-determinism (same broadcast at ingress); disagg control paths untouched until Phase 3+ shows they're hot. |
| C++ tree divergence (two native trees) | `RadixCacheCpp` frozen as oracle in this branch; Rust tree supersedes it; removal tracked as a cleanup item, not a phase. |
| Upstream merge friction (this is a fork) | Each phase is independently valuable and upstream-shaped: §1 notes the RFC direction (Rust front-half + Rust radix core). Contribution order: `sglang-radix` (pure lib, easiest review) → planner as opt-in flag → core. Trace/parity tooling is shareable as-is. |
| Measurement noise on e2e claims | 3-run medians, fixed GPU clock policy, pinned request traces, baselines committed; a "gain" only counts if it survives re-run on a fresh day. |

## 14. Milestones & decision gates

| # | Milestone | Exit gate |
|---|---|---|
| M0 | Phase 0 complete | Baselines + traces committed; replay lossless |
| M1 | `sglang-radix` base tree live behind `SGLANG_RUST_SCHEDULER=radix` | §5.3 gates; `RadixCacheCpp` demoted to oracle |
| M2 | Variants 1b–1e (SWA, GDN, HiRadix, unified) | each sub-PR: M-bench gate + trace parity |
| M3 | Planner A/B (`planner` flag) | §6.4: **go/no-go decision point** — flat at c16 → stop, ship radix; improving → continue |
| M4 | `SchedulerCore` (`core` flag) | §7 gates; overlap replay green |
| M5 | Output pipeline in Rust (`stream` flag) | §8 gates; ITL p99 visible at c16 |
| M6 | Spec bookkeeping | §9 gates |
| M7 | Default flip + upstream contribution | full `test/srt` green at `core|stream`; RFC PRs for `sglang-radix` + planner |
| — | Phase 6 | data-driven from M4 profiling; not scheduled |

**Overall success:** at c16 on Qwen3.8-NVFP4/RTX PRO 6000, ≥ 5% throughput
or ≥ 10% ITL p99 vs Phase-0 baseline, with zero trace-parity diffs and the
full pytest matrix green — the 5–15% system-level estimate from the working
analysis, verified rather than assumed.

## 15. Immediate next actions

Progress on branch `rust-scheduler` (all code lives there; nothing is
default-on, so the unmodified Python path is unchanged until a flag is set):

Done (this effort):
- [x] `rust/sglang-radix` crate: base radix tree with the Python
      `RadixCache` semantics (match/insert/evict/lock-ref, LRU, page-floor,
      bigram/EAGLE view). Golden + property tests, `radix_bench` criterion
      bench, clippy-clean. CI job `sglang-radix-unit` in
      `pr-test-rust-exts.yml`.
- [x] `rust/sglang-scheduler` crate: pure decision engine
      (`planner`/`adder`/`ntr`/`policy`/`core` = the persistent
      `SchedulerCore` that owns queues + tree + NTR tracker). 30 unit tests,
      `planner_bench` criterion bench, clippy-clean. CI job
      `sglang-scheduler-unit` (rlib tests + pyo3 cdylib build + `smoke_test.py`).
- [x] PyO3 bindings (`sglang-scheduler` `python` feature → the
      `_scheduler` extension: `RadixTree` facade incl. `match_prefix_meta`
      fast path, shadow `plan_next_batch`, `SchedulerCore`). `smoke_test.py`
      + `driver_test.py` (21 checks) exercise the `.so` directly.
- [x] Python integration, all env-gated off by default:
      `python/sglang/srt/mem_cache/rust_radix.py` (dual-write
      `RustRadixShadow` facade, resync on divergence) and
      `python/sglang/srt/managers/rust_scheduler.py` (shadow planner +
      `SchedulerCore` driver + trace capture), wired into
      `scheduler.py`. Staged flag `SGLANG_RUST_SCHEDULER=off|radix|planner|core|stream`;
      core cutover further gated by `SGLANG_RUST_CORE_APPLY` (default off =
      bookkeeping/trace only, no double-free).
- [x] Phase 0 capture: `SGLANG_TRACE_SCHEDULER` JSONL (ingress/plan/result/
      cache-op lines) in the driver; `TraceRecorder` is crash-safe (flushes
      per line).
- [x] `benchmark/scheduler/` harness (`bench_common.py`) + Rust M1–M7/M11
      through the PyO3 boundary (`bench_rust_scheduler.py`, `baselines/rust.json`
      recorded) + Python M1–M4/M5–M7 baseline scripts (`bench_py_radix.py`,
      `bench_py_planner.py` — run on the target machine, torch) + README.
- [x] Differential parity test `test/registered/rust/test_rust_radix_parity.py`
      (Rust `RadixTree` vs Python `RadixCache`, same op sequence); new crate
      added to the checked-in-crate discovery test.
- [x] M5 (plan §8): stream frame builder + string-stop decisions in
      `sglang-server` (`stream.rs` / `stream_py.rs` /
      `tokenizer_manager/stop_check.rs`), `push_generation_frame` pybind +
      `rust_server.push_generation` Rust path under `SGLANG_RUST_SCHEDULER=stream`,
      `stream_bench` (M9) criterion benches, `test_rust_stream_parity.py`
      (byte-parity frame + stop decisions + M9 p50 gate), `sglang-server-unit`
      CI job. 264 `sglang-server` unit tests.
- [x] M6 (plan §9): spec-v2 accept-run bookkeeping in `sglang-scheduler`
      (`spec.rs`: `resolve_spec_runs` + `SpecCounters`), `ResultRow.spec`
      metadata folded into `core.apply_result`, `resolve_spec_runs` /
      `SpecCounters` / `SchedulerCore.spec_counters` pybind, the Rust branch of
      `_resolve_spec_v2_tokens` under `SGLANG_RUST_SCHEDULER≥core`, spec rows in
      the trace schema + lossless replay (synthetic session now carries a spec
      row), `spec_bench` criterion benches, `test_rust_spec_parity.py`. 38
      `sglang-scheduler` unit tests.
- [x] M7 RFC content: `rust/rfc/0001-sglang-radix.md` +
      `rust/rfc/0002-sglang-scheduler.md` (upstream contribution drafts —
      the PRs themselves open on the target host, contribution order
      `sglang-radix` → planner per §13).

Remaining (needs the target GPU host, in plan order):
1. Record the two canonical e2e sessions (coding-agent multi-turn at c16;
   short-turn high-rate) with `SGLANG_TRACE_SCHEDULER` on, and the M1–M11
   Python baselines via `bench_py_radix.py` / `bench_py_planner.py`.
2. Extend `sglang.test.scripted_runtime` to feed recorded ingress sequences
   through a mocked `run_batch` so a trace replays into the live scheduler
   and the Python vs Rust plans diff field-for-field (the correctness
   backbone of every later gate).
3. Run the M3-style e2e A/B on the Qwen3.8 27B target: flat at c16 → stop;
   ≥2% throughput → continue to `core`/`stream` cutover.
4. M7 default flip: run the full `test/srt` matrix at `core` (and `stream`)
   on the target host; when green, move the `SGLANG_RUST_SCHEDULER` default
   in `python/sglang/srt/environ.py` from `"off"` to the validated stage
   (one line + the docstring; the staged flag makes it reversible).
5. M7 upstream contribution: open the RFC PRs from `rust/rfc/` (0001
   `sglang-radix` first, then 0002 `sglang-scheduler`), in plan order.
