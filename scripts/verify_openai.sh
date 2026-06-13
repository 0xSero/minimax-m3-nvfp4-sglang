#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8000}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}

curl -fsS "$BASE_URL/v1/models" | python3 -m json.tool

curl -fsS \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3","messages":[{"role":"user","content":"Say ready in one word."}],"max_tokens":8,"temperature":0}' \
  "$BASE_URL/v1/chat/completions" | python3 -m json.tool

