#!/bin/bash
# Single-image launcher: MODE=vanilla|tier (HKV-21).
# Usage: launch-hostkv.sh GPU PORT NAME MODE [KEEP_PAGES] [CTX_160K]
set -euo pipefail
GPU="${1:-0}"; PORT="${2:-30030}"; NAME="${3:-diag}"; MODE="${4:-tier}"
KEEP="${5:-1600}"; CTX160K="${6:-0}"
# Pool geometry overrides (256k probe: MAX_TOTAL_TOKENS=300000 KEEP=4100)
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-200000}"
CTX_LEN="${CTX_LEN:-262144}"
# AUTO mode: identical to tier but dormant until a request of at least
# ARM_MIN tokens arrives (vanilla-identical for workloads that fit).
ARM_MIN="${ARM_MIN:-0}"
W=/home/runner
IMAGE="${IMAGE:-hostkv:4ccff141-1}"
sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$W/work/$NAME"

TIER_FLAGS=()
KEEP="${KEEP:-1600}"
if [ "$MODE" = "tier" ] || [ "$MODE" = "auto" ]; then
  if [ "$CTX160K" = "1" ] && [ "$KEEP" = "1600" ]; then KEEP=2510; fi
  TIER_FLAGS=(--host-kv-tier
              --host-kv-keep-pages "$KEEP"
              --host-kv-hot-pages 256
              --host-kv-prefetch
              --host-kv-prefetch-radius 1
              --host-kv-reserve-floor 128
              --host-kv-verify-sample 8
              --host-kv-stats-path /root/phase0/host-kv-stats.json)
  if [ "$MODE" = "auto" ]; then
    TIER_FLAGS+=(--host-kv-arm-min-input-tokens "$ARM_MIN")
  fi
fi

sudo -n docker run -d --name "$NAME" --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPU" --init --ipc=host \
  -p "127.0.0.1:$PORT:30000" --ulimit memlock=-1 --shm-size 100g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false \
  -v "$W/hf-cache/flash-next-nvfp4":/root/phase0/weights:ro \
  -v "$W/work/$NAME":/root/phase0 \
  "$IMAGE" sglang serve \
  --model-path /root/phase0/weights --served-model-name flash-next-nvfp4 \
  --json-model-override-args '{"language_model_only":true}' \
  --moe-runner-backend flashinfer_cutlass --ple-offload-embedding \
  --max-mamba-cache-size 32 --mamba-ssm-dtype bfloat16 --chunked-prefill-size 8192 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --cuda-graph-backend-prefill disabled \
  --context-length "$CTX_LEN" --max-total-tokens "$MAX_TOTAL_TOKENS" --max-running-requests 16 \
  --disable-radix-cache --schedule-conservativeness 1.5 \
  --mem-fraction-static 0.95 --enable-metrics \
  ${TIER_FLAGS[@]+"${TIER_FLAGS[@]}"} \
  --host 0.0.0.0 --port 30000 >/dev/null
echo "$NAME launched (gpu=$GPU port=$PORT mode=$MODE keep=$KEEP)"
