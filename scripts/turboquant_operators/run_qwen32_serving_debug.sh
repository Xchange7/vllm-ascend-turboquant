#!/usr/bin/env bash

# Low-concurrency end-to-end Qwen3-32B decode comparison for either 910B4/A2
# or 910_93/A3. This intentionally avoids the long concurrency matrix.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL="${MODEL:-/run/test_llm/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-4}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-}"
NETWORK_IFNAME="${NETWORK_IFNAME:-eth0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
INPUT_TOKENS="${INPUT_TOKENS:-7168}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
MEASURE_REQUESTS="${MEASURE_REQUESTS:-2}"
PORT="${PORT:-18100}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/qwen32_serving_debug_${TIMESTAMP}}"

if [[ ! "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'TP_SIZE must be a positive integer, got %s.\n' "${TP_SIZE}" >&2
    exit 2
fi
if [[ -z "${VISIBLE_DEVICES}" ]]; then
    for ((device_index = 0; device_index < TP_SIZE; device_index++)); do
        if [[ -n "${VISIBLE_DEVICES}" ]]; then
            VISIBLE_DEVICES+=,
        fi
        VISIBLE_DEVICES+="${device_index}"
    done
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
    printf 'Model config is not readable: %s/config.json\n' "${MODEL}" >&2
    exit 2
fi
if ! find "${MODEL}" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) \
    -print -quit | grep -q .; then
    printf 'No local model weight shard was found in %s.\n' "${MODEL}" >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export NETWORK_IFNAME
export VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION="ascend_fused"

printf 'Qwen3-32B serving debug: soc=%s devices=%s tp=%s input=%s output=%s concurrency=1\n' \
    "${SOC_VERSION}" "${VISIBLE_DEVICES}" "${TP_SIZE}" "${INPUT_TOKENS}" "${OUTPUT_TOKENS}"

exec env -u ASCEND_LAUNCH_BLOCKING \
    MODEL="${MODEL}" \
    PORT="${PORT}" \
    TP_SIZE="${TP_SIZE}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    INPUT_TOKENS="${INPUT_TOKENS}" \
    OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
    CONCURRENCY=1 \
    CONCURRENCY_LEVELS=1 \
    WARMUP_REQUESTS="${WARMUP_REQUESTS}" \
    MEASURE_REQUESTS="${MEASURE_REQUESTS}" \
    MAX_NUM_SEQS=1 \
    ENFORCE_EAGER=1 \
    NATIVE_CACHE_DTYPE=auto \
    TQ_CACHE_DTYPE=turboquant_4bit_nc \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${TQ_SCRIPT_ROOT}/performance/run_serving_benchmark.sh"
