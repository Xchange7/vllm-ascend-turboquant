#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/ascend_fused_${TIMESTAMP}}"
DEVICE="${DEVICE:-0}"
CACHE_DTYPE="${CACHE_DTYPE:-turboquant_4bit_nc}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
NUM_QUERY_HEADS="${NUM_QUERY_HEADS:-16}"
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
HEAD_DIM="${HEAD_DIM:-128}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-2048 16384}"
BATCH_SIZES="${BATCH_SIZES:-1 4 16}"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
COLLECT_TRACE="${COLLECT_TRACE:-0}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/validation.log") 2>&1

finalize() {
    local status=$?
    trap - EXIT
    tar -czf "${OUTPUT_DIR}.tar.gz" \
        -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")" || status=1
    printf 'Ascend fused validation output: %s\n' "${OUTPUT_DIR}"
    printf 'Ascend fused validation archive: %s.tar.gz\n' "${OUTPUT_DIR}"
    exit "${status}"
}
trap finalize EXIT

cd "${REPO_ROOT}"
printf 'Revision: '
git rev-parse HEAD
python3 "${TQ_COMMON_DIR}/check_environment.py"
python3 - <<'PY'
from vllm_ascend.ops.turboquant import has_turboquant_paged_dequant

if not has_turboquant_paged_dequant():
    raise RuntimeError(
        "npu_turboquant_paged_dequant is missing; source CANN and rerun `pip install -e .`."
    )
print("Ascend TurboQuant fused operator is registered.")
PY

printf '\n== Fused correctness ==\n'
pytest -sv tests/ut/ops/test_turboquant_triton.py -k "ascend_fused"

TRACE_ARGS=(--no-trace)
if [[ "${COLLECT_TRACE}" == "1" ]]; then
    TRACE_ARGS=()
fi

printf '\n== Fused performance matrix ==\n'
for sequence_length in ${SEQUENCE_LENGTHS}; do
    for batch_size in ${BATCH_SIZES}; do
        case_dir="${OUTPUT_DIR}/s${sequence_length}_b${batch_size}"
        mkdir -p "${case_dir}"
        python3 "${TQ_PERFORMANCE_DIR}/profile_kernels.py" \
            --operation all \
            --cache-dtype "${CACHE_DTYPE}" \
            --activation-dtype "${ACTIVATION_DTYPE}" \
            --batch-size "${batch_size}" \
            --sequence-length "${sequence_length}" \
            --store-tokens "${sequence_length}" \
            --num-query-heads "${NUM_QUERY_HEADS}" \
            --num-kv-heads "${NUM_KV_HEADS}" \
            --head-dim "${HEAD_DIM}" \
            --block-size 128 \
            --adaptive-splits \
            --warmup "${WARMUP}" \
            --iterations "${ITERATIONS}" \
            --profile-iterations 3 \
            --device "${DEVICE}" \
            --trace-dir "${case_dir}" \
            "${TRACE_ARGS[@]}" \
            | tee "${case_dir}/profile.log"
    done
done

printf '\nFor the end-to-end eager comparison, run:\n'
printf 'VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=ascend_fused ENFORCE_EAGER=1 '
printf 'bash scripts/turboquant_triton/performance/run_serving_benchmark.sh\n'
