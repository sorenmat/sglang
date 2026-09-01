# GPU validation runbook (M7 gate)

Everything needed to run the remaining `test/srt` matrix on an authorized
RTX PRO 6000 (96 GB) host, mechanically. All parity/CPU work is already
green; this is the serving-stage validation left before the default flip.

## 0. Host prerequisites

- NVIDIA RTX PRO 6000 Blackwell (96 GB), docker + nvidia container toolkit.
- ~50 GB free disk (cargo target + small model downloads).
- The `sorenmat/sglang:latest` image (has `/opt/sglang` python 3.12 + torch).
- **Stop any GPU-serving container first** — the matrix needs the whole GPU;
  tests OOM above ~90 GB occupied (verified: kernel tests fail with CUDA
  OOM when 500 MB is left).

## 1. One-time setup

```bash
rsync -a --delete --exclude=.git --exclude='rust/target' \
  --exclude='__pycache__' --exclude='*.pyc' \
  <checkout>/ ecuser@<host>:~/sglang-rust/

ssh ecuser@<host> 'sudo mkdir -p /root/sglang-cargo && sudo docker run -d \
  --name sglang-rust-test --gpus all --shm-size 32g --ipc=host \
  -m 70g --memory-swap 102g \
  -v /home/ecuser/sglang-rust:/sgl-workspace/sglang \
  -v /home/ecuser/.cache/huggingface:/root/.cache/huggingface \
  -v /root/sglang-cargo:/root/.cache/sglang-cargo \
  -w /sgl-workspace/sglang sorenmat/sglang:latest sleep infinity'

# toolchain + extensions (~8 min cold; the cargo volume caches rebuilds)
ssh ecuser@<host> 'sudo docker exec sglang-rust-test bash -lc "
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1;
  /opt/sglang/bin/pip install -q setuptools setuptools-rust setuptools-scm wheel;
  source /root/.cargo/env; cd /sgl-workspace/sglang/python &&
  CARGO_TARGET_DIR=/root/.cache/sglang-cargo/target SGLANG_BUILD_RUST_EXTS=all \
  /opt/sglang/bin/python setup.py build_rust --inplace"'
```

Gotchas learned the hard way:
- Mount `/root/.cache/sglang` too if the image lacks the flashinfer autotune
  cache, or pass `--disable-flashinfer-autotune` on manual server launches —
  a cold autotune JIT-compiles with several parallel `cicc` processes and
  OOM-kills an 88 GB host.
- Keep the `-m 70g` container cap: it contains any runaway allocation.

## 2. 27 B serving smoke (per stage)

```bash
ssh ecuser@<host> 'sudo docker exec -d -e SGLANG_RUST_SCHEDULER=<stage> \
  -e HF_HUB_OFFLINE=1 sglang-rust-test bash -lc "
  cd /sgl-workspace/sglang && python3 -m sglang.launch_server \
    --trust-remote-code --model-path RadixArk/Qwen3.8-27B-NVFP4 \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
    --attention-backend flashinfer --disable-flashinfer-autotune \
    --chunked-prefill-size 2048 --max-running-requests 16 \
    --max-mamba-cache-size 24 --mamba-full-memory-ratio 1.08 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --mamba-ssm-dtype float32 --speculative-algorithm EAGLE \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec \
    --host 127.0.0.1 --port 30100 > /sgl-workspace/sglang/.smoke-logs/smoke.log 2>&1"'
# health + shared-prefix generations; grep the log for
# "rust scheduler: planner shadow enabled" / "core bookkeeping enabled" /
# "rust server cores=" and for "Scheduler hit an exception".
```

`<stage>` ∈ radix | planner | core | stream (`stream` also serves through
the Rust HTTP front end). DONE on 2026-09-01 for all four stages —
planner/core/stream healthy with correct generations; radix's tree
dual-write cannot engage on `UnifiedRadixCache` (see plan.md gaps).

## 3. Test matrix (the M7 gate)

```bash
ssh ecuser@<host> 'sudo docker exec -d sglang-rust-test bash -c "
  cd /sgl-workspace/sglang && export SGLANG_RUST_SCHEDULER=core &&
  for s in base-a-test-1-gpu-small base-b-test-1-gpu-small \
           base-b-test-1-gpu-large base-b-kernel-unit-test-1-gpu-large \
           base-b-kernel-benchmark-test-1-gpu-large extra-a-test-1-gpu-small \
           extra-a-test-1-gpu-large nightly-test-1-gpu-large; do
    python3 test/run_suite.py --hw cuda --suite \$s \
      >> /sgl-workspace/sglang/.smoke-logs/matrix-core.log 2>&1;
  done"'
# then the same loop with SGLANG_RUST_SCHEDULER=stream → matrix-stream.log
```

- ~864 min estimated for the full set at one stage; partitions via
  `--auto-partition-id i --auto-partition-size n`.
- 25 of 582 files need gated `meta-llama/*` — either drop an HF token at
  `/home/ecuser/.cache/huggingface/token` (root-only) or accept them as
  skipped; everything else uses public small models.
- Fix + rerun until the summaries are green, then repeat at `stream`.

## 4. Default flip (last)

`python/sglang/srt/environ.py` `SGLANG_RUST_SCHEDULER` default `"off"` →
the validated stage (one line + its docstring), on `rust-scheduler`.
