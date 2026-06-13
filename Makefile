IMAGE ?= minimax-m3-sglang:dev-cu13-minimax-m3-patched
MODEL_DIR ?= /mnt/llm_models/MiniMax-M3-NVFP4
PORT ?= 8000

.PHONY: build run verify stop

build:
	docker build -t $(IMAGE) .

run: build
	IMAGE=$(IMAGE) MODEL_DIR=$(MODEL_DIR) PORT=$(PORT) ./scripts/launch_minimax_m3_sglang.sh

verify:
	PORT=$(PORT) ./scripts/verify_openai.sh

stop:
	docker rm -f minimax-m3-sglang >/dev/null 2>&1 || true

