#!/usr/bin/env bash

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"

curl --fail-with-body --silent --show-error \
    "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"Paged KV cache stores keys and values. Paged KV cache stores keys and values. Paged KV cache stores keys and\",
        \"max_tokens\": 32,
        \"temperature\": 0
    }"
printf '\n'
