#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/operators_smoke_${TIMESTAMP}}"
CACHE_DTYPES="${CACHE_DTYPES:-turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-float16}"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"

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

printf '\n== Operator correctness ==\n'
for cache_dtype in ${CACHE_DTYPES}; do
    case_dir="${OUTPUT_DIR}/accuracy_${cache_dtype}_${ACTIVATION_DTYPE}"
    mkdir -p "${case_dir}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/operator_accuracy.py" \
        --cache-dtype "${cache_dtype}" \
        --activation-dtype "${ACTIVATION_DTYPE}" \
        --device "${DEVICE}" \
        --output "${case_dir}/accuracy.json" \
        | tee "${case_dir}/accuracy.log"
done

printf '\n== Short latency benchmark ==\n'
profile_dir="${OUTPUT_DIR}/profile_b2_s512"
"${PYTHON_BIN}" "${TQ_PROFILE_SCRIPT}" \
    --operation all \
    --native-baseline \
    --cache-dtype turboquant_4bit_nc \
    --activation-dtype "${ACTIVATION_DTYPE}" \
    --batch-size 2 \
    --sequence-length 512 \
    --store-tokens 512 \
    --num-query-heads 16 \
    --num-kv-heads 2 \
    --head-dim 128 \
    --block-size 128 \
    --adaptive-splits \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --profile-iterations 0 \
    --device "${DEVICE}" \
    --trace-dir "${profile_dir}" \
    --no-trace \
    | tee "${OUTPUT_DIR}/profile_b2_s512.log"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --result-root "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_DIR}/summary"
