#!/bin/bash
# P2.2 correctness gate: one 167k request, twice.
#   run A: stock pinned runtime                    -> response-a.json
#   run B: hosttier overlay, 90% of pages evicted  -> response-b.json + hosttier stats
# Verdict: token streams identical  =>  demand staging is bit-exact.
set -euo pipefail
W=/root/work
IMAGE=lmsysorg/sglang:dev-qwen38-next-local
ARGS=(--model-path /root/phase0/weights --served-model-name flash-next-nvfp4
  --json-model-override-args '{"language_model_only":true}'
  --moe-runner-backend flashinfer_cutlass --ple-offload-embedding
  --max-mamba-cache-size 8 --mamba-ssm-dtype bfloat16 --chunked-prefill-size 8192
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --cuda-graph-backend-prefill disabled --disable-cuda-graph --enable-deterministic-inference
  --enable-metrics --random-seed 20260930 --host 0.0.0.0 --port 30000
  --context-length 262144 --max-total-tokens 200000 --max-running-requests 1
  --mem-fraction-static 0.95)

run_container () {  # $1 name, $2 extra env (ENABLE), $3.. extra mounts
  local name=$1; shift
  local enable=$1; shift
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    --ipc=host -p 127.0.0.1:30000:30000 --ulimit memlock=-1 --shm-size 100g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e SGLANG_HOSTTIER_ENABLE="$enable" -e SGLANG_HOSTTIER_EVICT_FRACTION=0.9 \
    -e SGLANG_HOSTTIER_LOG=/root/phase0/hosttier-stats.json \
    -v "$W/hf-cache/flash-next-nvfp4":/root/phase0/weights:ro \
    -v "$W/job":/root/phase0 -v "$W/assets":/assets:ro "$@" \
    "$IMAGE" sglang serve "${ARGS[@]}"
}

echo "=== run A: baseline (no overlay)"
if [ ! -f /root/work/job/response-a.json ]; then
  run_container p22-a 0
  python3 /root/work/assets/p22_send.py a
  docker rm -f p22-a >/dev/null
else echo "run A already complete, skipping"; fi

echo "=== run B: hosttier overlay, 90% evicted"
OVER=/root/work/hosttier-overlay/python
run_container p22-b 1 \
  -v "$OVER/sglang/srt/layers/attention/qwen_sparse_attn_backend.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro \
  -v "$OVER/sglang/srt/layers/attention/qsa/hosttier.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/hosttier.py:ro
python3 /root/work/assets/p22_send.py b
docker logs p22-b > /root/work/job/p22-b-docker.log 2>&1 || true
docker rm -f p22-b >/dev/null

echo "=== compare"
python3 /root/work/assets/p22_compare.py
