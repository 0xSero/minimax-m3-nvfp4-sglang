# MiniMax-M3 NVFP4 SGLang Docker

Reproducible Docker launcher for the working MiniMax-M3 deployment validated on 4x NVIDIA RTX PRO 6000 Blackwell.

This repo does not contain model weights. It expects the NVFP4 checkpoint at:

```text
/mnt/llm_models/MiniMax-M3-NVFP4
```

## Verified Live Settings

```text
base image tag used live: lmsysorg/sglang:dev-cu13-minimax-m3
base image digest pinned here: lmsysorg/sglang@sha256:a081e28f8d6368de246ae66459efbc6f94ea20ffa15fe3f3ce6b0820e9b58eea
live image id: sha256:7271b6b9feb96bf8f6705ed5e9760919857bd19b01f47fd23ecffbf2a44c6517
served model id: minimax-m3
model display/name: MiniMax-M3
port: 8000
tensor parallel: 4
context length: 450000
max total tokens: 450000
KV cache dtype: bfloat16
quantization: modelopt_fp4
reasoning parser: minimax-m3
tool call parser: minimax-m3
MoE backend: flashinfer_cutlass
decode CUDA graph: full
prefill CUDA graph: disabled
max running requests: 4
chunked prefill size: 4096
max prefill tokens: 16384
static memory fraction: 0.95
```

The prefill CUDA graph backend is disabled because the valid SGLang prefill graph backends crashed on this model with an `idx_q` argument mismatch in `sglang::unified_attention_with_output`. Decode CUDA graph remains enabled.

## Quick Start

```bash
git clone https://github.com/0xSero/minimax-m3-nvfp4-sglang.git
cd minimax-m3-nvfp4-sglang
make run
make verify
```

Equivalent explicit command:

```bash
MODEL_DIR=/mnt/llm_models/MiniMax-M3-NVFP4 \
PORT=8000 \
IMAGE=minimax-m3-sglang:dev-cu13-minimax-m3-patched \
./scripts/launch_minimax_m3_sglang.sh
```

## What The Launcher Patches

The live deployment needed two compatibility fixes:

1. The model `config.json` is normalized before launch:
   - `minimax_m3_sparse` layer labels become `sparse`.
   - `vision_config.rope_theta` is filled from `vision_config.rope_parameters.rope_theta` when missing.
   - `sparse_attention_config` is derived from `text_config.layer_types` and MiniMax index-head fields.

2. The SGLang MiniMax loader is patched so HF checkpoints that include a root `model.` prefix are normalized before weight dispatch. This prevents text attention q/k/v tensors from entering the VIT QKV merge path.

The Dockerfile bakes the loader patch into the image. The launcher applies the config patch to the mounted model directory, then starts the container with the model mount read-only, matching the working deployment.

## Verify

```bash
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
./scripts/verify_openai.sh
```

Expected model list:

```json
{
  "object": "list",
  "data": [
    {
      "id": "minimax-m3",
      "object": "model",
      "owned_by": "sglang",
      "root": "minimax-m3",
      "parent": null,
      "max_model_len": 450000
    }
  ]
}
```

## Notes

- Do not add `--disable-cuda-graphs`.
- Do not add `--enforce-eager`.
- Keep power caps external to this repo. The validated host used 275W caps on all four GPUs.
