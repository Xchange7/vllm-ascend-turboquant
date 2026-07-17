#!/usr/bin/env bash

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant_4bit_nc}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
NETWORK_IFNAME="${NETWORK_IFNAME:-eth0}"

if [[ ! -d "/sys/class/net/${NETWORK_IFNAME}" ]]; then
    printf 'Network interface does not exist: %s\n' "${NETWORK_IFNAME}" >&2
    exit 2
fi

export VLLM_USE_V2_MODEL_RUNNER=0
export GLOO_SOCKET_IFNAME="${NETWORK_IFNAME}"
export TP_SOCKET_IFNAME="${NETWORK_IFNAME}"
export HCCL_SOCKET_IFNAME="${NETWORK_IFNAME}"

EAGER_ARGS=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    EAGER_ARGS+=(--enforce-eager)
fi

GRAPH_ARGS=()
if [[ "${ENFORCE_EAGER}" != "1" ]]; then
    if [[ -z "${COMPILATION_CONFIG}" ]]; then
        COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    fi
    GRAPH_ARGS+=(--compilation-config "${COMPILATION_CONFIG}")
fi

SPECULATIVE_ARGS=()
if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
    SPECULATIVE_ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --block-size 128 \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    "${EAGER_ARGS[@]}" \
    "${GRAPH_ARGS[@]}" \
    "${SPECULATIVE_ARGS[@]}" \
    --trust-remote-code \
    "$@"
