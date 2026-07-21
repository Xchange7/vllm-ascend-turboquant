#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/operators_profile_${TIMESTAMP}}"
CACHE_DTYPE="${CACHE_DTYPE:-turboquant_4bit_nc}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"
NUM_QUERY_HEADS="${NUM_QUERY_HEADS:-16}"
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
HEAD_DIM="${HEAD_DIM:-128}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-5}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/run.log") 2>&1

finalize() {
    local status=$?
    trap - EXIT
    archive_output "${OUTPUT_DIR}" || status=1
    exit "${status}"
}
trap finalize EXIT

cd "${REPO_ROOT}"
collect_environment "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${TQ_PROFILE_SCRIPT}" \
    --operation all \
    --native-baseline \
    --cache-dtype "${CACHE_DTYPE}" \
    --activation-dtype "${ACTIVATION_DTYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --store-tokens "${SEQUENCE_LENGTH}" \
    --num-query-heads "${NUM_QUERY_HEADS}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --block-size 128 \
    --adaptive-splits \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --profile-iterations "${PROFILE_ITERATIONS}" \
    --device "${DEVICE}" \
    --trace-dir "${OUTPUT_DIR}/profile" \
    | tee "${OUTPUT_DIR}/profile.log"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --result-root "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_DIR}/summary"
