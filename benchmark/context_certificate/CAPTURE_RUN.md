# First real-model capture

The next decision is whether exact index screening remains useful on real QSA
activations. The local capture overlay is prepared and CPU-tested. It has not
been installed, imported in a serving container, or exercised on a GPU.
Collecting these arrays is a diagnostic experiment, not a throughput benchmark.

## Prepare locally

From `/home/smo/code/sglang-perf`:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s benchmark/context_certificate -p 'test_*.py' -v
python benchmark/context_certificate/prepare_capture_patch.py \
  --output benchmark/context_certificate/artifacts/capture-overlay-20260928
```

The builder reads local Git commit
`4ccff141dbe992794f9da6c3aa23535b4f72000d`; it does not switch branches or
modify the checkout. It creates two patched source files, a standalone helper,
`capture.patch`, and a manifest with hashes. It refuses an existing output
directory. The generated patch and full files are alternative installation
forms; do not apply both.

Before installation, run the read-only compatibility check against the
**unmodified source root of the actual diagnostic runtime**, or an exact local
copy of those files:

```bash
python benchmark/context_certificate/prepare_capture_patch.py \
  --check-source-root /path/to/unmodified/runtime/source
```

This checks the patched files and the source dependencies that define scoring,
page mapping, native selection and model roles. A mismatch requires explicitly
porting and reviewing the hook; do not overwrite an unknown runtime with these
files. Matching Python files do not establish matching installed kernels:
record the container/build identity and kernel versions as well.

## Bounded GPU experiment

Use an isolated diagnostic runtime and its known-working model launch
configuration. Start with **one real request around 32k tokens**, one confirmed
QSA target layer, and the existing supported chain-verification window of at
most four positions. Keep model weights, sampling and native selection
unchanged. Disable CUDA graphs and simulated acceptance. Also pass
`--disable-flashinfer-autotune --skip-server-warmup`: the pinned runtime's
autotuning uses synthetic verification batches without real request IDs,
which this hook rejects. This overlay rejects graph capture and positive
`SGLANG_SIMULATE_ACC_LEN` values.

Set these environment variables before the runtime starts. The identity values
below are placeholders to replace with the exact checkpoint and build:

```bash
export SGLANG_QSA_CAPTURE_DIR=/captures/first-real-run
export SGLANG_QSA_CAPTURE_MODEL_ID=PIN_CHECKPOINT_ID
export SGLANG_QSA_CAPTURE_MODEL_REVISION=PIN_CHECKPOINT_REVISION
export SGLANG_QSA_CAPTURE_RUNTIME_REVISION=PIN_CONTAINER_AND_KERNEL_BUILD
export SGLANG_QSA_CAPTURE_MAX_FILES=6
export SGLANG_QSA_CAPTURE_MAX_BYTES=268435456
export SGLANG_QSA_CAPTURE_MAX_KEYS=65536
export SGLANG_QSA_CAPTURE_MAX_EVENTS=32
export SGLANG_QSA_CAPTURE_LAYERS=3
export SGLANG_SIMULATE_ACC_LEN=0
```

Layer 3 is an example: confirm the target checkpoint's QSA layer IDs first.
Mount a writable output directory at `SGLANG_QSA_CAPTURE_DIR`. Only global
rank zero writes. Leaving that variable unset disables capture. File, byte
and acceptance-event quotas persist across restarts in the same directory;
use a fresh directory for a new run. The byte quota is conservative and may
stop before the file-count quota. One full-context four-position capture is
approximately 33 MiB. Capture introduces synchronization and additional
temporary allocations; keep it separate from performance measurements.

The hook captures target decode/verification only, after index-cache updates
and before selected blocks expand into token IDs. It checks request mapping,
shared physical page prefixes, BF16 query/key values, zero padded query heads,
native FP32 scores and native selected IDs. It never replaces the native
selector. Short contexts below 512 compressed blocks are skipped. Unsupported
tensor layouts or inconsistent native selections fail explicitly.

Alongside the NPZ files, `acceptance.jsonl` records accepted drafts and
`accepted_output_slots_including_bonus`. These are **not actual emitted-token
counts**: grammar, EOS and cancellation can shorten output. Request and batch
metadata support inspection, but no exact join between a QSA capture call and
an acceptance event is established. Expert routes, KV hits/misses, transfer
bytes and actual output timing still need separate instrumentation.

## Replay and decide

Copy captures back to the ignored local `artifacts/` directory, then run:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmark/context_certificate/replay.py \
  run benchmark/context_certificate/artifacts/first-real-run/qsa-CAPTURE_ID.npz \
  --output benchmark/context_certificate/artifacts/first-real-run/replay-CAPTURE_ID.json
```

The default tests 4/6/8-bit bounds over all positions present in that file.
Require native selected-set agreement, no excluded native winners and native
scores contained within the intervals. Investigate every disagreement,
including ties. Passing is empirical evidence for those samples, not a GPU
rounding proof or an end-to-end quality result.

Measure how many original keys require refinement and how much the selected
KV blocks overlap across positions. Compare total screening/refinement work
against one shared resident BF16 index scan. If bounds are too loose or the
union too large, revise or drop that part of the proposal before writing a
production GPU kernel. Increase coverage to more requests, layers and longer
contexts only after the first capture passes.

Then benchmark the promising kernel separately with capture disabled. A later
memory-manager prototype must validate the approximately 235–250 GB host
layout, page ownership and transfer traffic. Full output-quality evaluation
and 32 independent 256k-context throughput tests remain required; none of the
CPU results establishes 80 tokens/s per request.
