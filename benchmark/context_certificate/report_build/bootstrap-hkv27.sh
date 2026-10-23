#!/bin/bash
# HKV-27 bootstrap: weights + images + repo on the g4 box.
set -uo pipefail
W=/home/runner
LOG=$W/bootstrap-hkv27.log
exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap start $(date -u)"
command -v docker >/dev/null || { echo "docker missing"; exit 1; }
sudo -n systemctl start docker 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

mkdir -p $W/hf-cache/flash-next-nvfp4
if [ ! -f $W/hf-cache/flash-next-nvfp4/.complete ]; then
  pip install -q -U "huggingface_hub[cli]" hf_transfer 2>&1 | tail -1
  export HF_HUB_ENABLE_HF_TRANSFER=1
  hf download nvidia/Qwen3.8-Flash-Next-FP4-Local \
    --local-dir $W/hf-cache/flash-next-nvfp4 \
    && touch $W/hf-cache/flash-next-nvfp4/.complete \
    || { echo "WEIGHTS FAILED"; exit 1; }
fi
echo "weights: $(du -sh $W/hf-cache/flash-next-nvfp4 2>/dev/null | cut -f1)"

sudo -n docker pull lmsysorg/sglang:dev-qwen38-next-local 2>&1 | tail -1
cd $W/sglang-perf-repo || { echo "repo missing"; exit 1; }
sudo -n docker build -q -f docker/Dockerfile.hostkv -t hostkv:4ccff141-21 . \
  2>&1 | tail -1
echo "=== bootstrap done $(date -u)"
