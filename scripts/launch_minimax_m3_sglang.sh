#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/llm_models/MiniMax-M3-NVFP4}
IMAGE=${IMAGE:-minimax-m3-sglang:dev-cu13-minimax-m3-patched}
CONTAINER=${CONTAINER:-minimax-m3-sglang}
PORT=${PORT:-8000}

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Missing $MODEL_DIR/config.json" >&2
  exit 1
fi

python3 "$(dirname "$0")/patch_model_config.py" "$MODEL_DIR/config.json"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" "$(cd "$(dirname "$0")/.." && pwd)"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER" \
  --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /mnt/llm_models:/mnt/llm_models:ro \
  "$IMAGE" \
    --model-path "$MODEL_DIR" \
    --tokenizer-path "$MODEL_DIR" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name minimax-m3 \
    --tp-size 4 \
    --context-length 450000 \
    --max-total-tokens 450000 \
    --kv-cache-dtype bfloat16 \
    --quantization modelopt_fp4 \
    --reasoning-parser minimax-m3 \
    --tool-call-parser minimax-m3 \
    --moe-runner-backend flashinfer_cutlass \
    --cuda-graph-backend-decode full \
    --cuda-graph-backend-prefill disabled \
    --max-running-requests 4 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 16384 \
    --mem-fraction-static 0.95

