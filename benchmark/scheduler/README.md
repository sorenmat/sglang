# Scheduler microbenchmark suite (plan.md §4.1)

Measurement floor for the Rust scheduler migration. Every bench reports
per-shape p50/p95/p99 wall-clock in microseconds over many iterations and
can append to `baselines/*.json` (`--record NAME`) or check for a >10%
p50 regression against a committed baseline (`--compare NAME`).

| File | Measures | Needs |
|---|---|---|
| `bench_common.py` | shared harness (percentiles, record/compare) | – |
| `bench_rust_scheduler.py` | Rust M1–M4 + M5–M7 + M11 **through the PyO3 boundary** (the cost the Python driver actually pays per call) | `cargo build -p sglang-scheduler --features python` in `rust/` |
| `bench_py_radix.py` | Python M1–M4 baselines against the unmodified `RadixCache` (simulated pools) | torch (CPU-only ok) + full sglang import → target machine |
| `bench_py_planner.py` | Python M5–M7 baselines (`calc_priority`, `PrefillAdder.add_one_req`, `ScheduleBatch.filter_batch`) | same |

## Running

```bash
# Rust side (any machine with the .so built):
cd rust && cargo build -p sglang-scheduler --features python && cd -
python3 benchmark/scheduler/bench_rust_scheduler.py --iters 300

# Python side (target machine with torch):
python3 benchmark/scheduler/bench_py_radix.py   --record py_radix
python3 benchmark/scheduler/bench_py_planner.py --record py_planner
```

Re-check after changes (CI nightly does the same against committed
baselines, plan.md §11.3):

```bash
python3 benchmark/scheduler/bench_rust_scheduler.py --compare rust
```

## Baselines

`baselines/rust.json` — recorded Rust M1–M7/M11 numbers (debug build,
single host; a *relative-regression* floor, not an absolute perf claim —
release-build numbers on the target hardware are the perf record).

`baselines/py_radix.json`, `baselines/py_planner.json` — recorded on the
target machine (Qwen3.8 27B / RTX PRO 6000 host) by the two Python
scripts above; they are the denominator of the Phase-1 gate.

## Shapes and gates

| ID | Shape | Rust gate (plan.md) |
|---|---|---|
| M1 | match, tree 1k/10k/100k, key = full 8-leaf branch; `match` (full KV list) + `match_meta` (shadow fast path) | `match_prefix`/`insert` ≥ 10× Python baseline p50 at 100k tree |
| M2 | insert fresh 128-token keys into 10k/100k tree | ditto |
| M3 | evict 1k/10k from a freshly built 100k tree (fresh tree per sample) | – |
| M4 | inc/dec_lock_ref walk, chain depth 1/32/256 | – |
| M5 | plan: fcfs/lpm × waiting 16/64/128/256 | planner total ≤ 50 µs at waiting 128 / running 256 |
| M6 | admit: waiting 16/64/256, chunked on/off | ditto |
| M7 | decode plan: running 64/256, idle/memory pressure | ditto |
| M11 | full core loop: plan → apply_result over `last_batch`, 32 reqs | trace parity (§4.2) |

M8–M10 (finish-state, stream payload, detokenizer) belong to later phases
(`stream` flag, §8); their Rust versions are benched when implemented.
`retract_decode`'s memory-pressure loop is bound to the live allocator
pools and is baseline-recorded via the §4.2 trace replay / §4.3 e2e A/B
on the target hardware instead of a microbench.

## Discipline (plan.md §11)

1. Baseline JSONs are committed; a "gain" only counts if it survives a
   re-run on a fresh day (3-run medians for e2e claims).
2. `--compare` fails the CI job on any >10% p50 regression.
3. Trace parity (`SGLANG_TRACE_SCHEDULER` + scripted replay) is the
   correctness backbone; these benches are the *performance* floor.
