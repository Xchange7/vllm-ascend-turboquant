#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/operators_matrix_${TIMESTAMP}}"
CACHE_DTYPES="${CACHE_DTYPES:-turboquant_4bit_nc}"
ACTIVATION_DTYPES="${ACTIVATION_DTYPES:-bfloat16}"
BATCH_SIZES="${BATCH_SIZES:-1 4 16}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-512 2048 16384}"
NUM_QUERY_HEADS="${NUM_QUERY_HEADS:-16}"
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
HEAD_DIM="${HEAD_DIM:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"
RUN_ACCURACY="${RUN_ACCURACY:-1}"

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
printf 'case\tstatus\n' > "${OUTPUT_DIR}/results.tsv"
failures=0

if [[ "${RUN_ACCURACY}" == "1" ]]; then
    printf '\n== Correctness matrix ==\n'
    for cache_dtype in ${CACHE_DTYPES}; do
        for activation_dtype in ${ACTIVATION_DTYPES}; do
            case_name="accuracy_${cache_dtype}_${activation_dtype}"
            case_dir="${OUTPUT_DIR}/${case_name}"
            mkdir -p "${case_dir}"
            if "${PYTHON_BIN}" "${SCRIPT_DIR}/operator_accuracy.py" \
                --cache-dtype "${cache_dtype}" \
                --activation-dtype "${activation_dtype}" \
                --num-query-heads "${NUM_QUERY_HEADS}" \
                --num-kv-heads "${NUM_KV_HEADS}" \
                --head-dim "${HEAD_DIM}" \
                --block-size "${BLOCK_SIZE}" \
                --device "${DEVICE}" \
                --output "${case_dir}/accuracy.json" \
                2>&1 | tee "${case_dir}/accuracy.log"; then
                printf '%s\tPASS\n' "${case_name}" >> "${OUTPUT_DIR}/results.tsv"
            else
                printf '%s\tFAIL\n' "${case_name}" >> "${OUTPUT_DIR}/results.tsv"
                failures=$((failures + 1))
            fi
        done
    done
fi

printf '\n== Performance matrix ==\n'
for cache_dtype in ${CACHE_DTYPES}; do
    for activation_dtype in ${ACTIVATION_DTYPES}; do
        for batch_size in ${BATCH_SIZES}; do
            for sequence_length in ${SEQUENCE_LENGTHS}; do
                case_name="perf_${cache_dtype}_${activation_dtype}_b${batch_size}_s${sequence_length}"
                case_dir="${OUTPUT_DIR}/${case_name}"
                mkdir -p "${case_dir}"
                printf '\n-- %s --\n' "${case_name}"
                if "${PYTHON_BIN}" "${TQ_PROFILE_SCRIPT}" \
                    --operation all \
                    --native-baseline \
                    --cache-dtype "${cache_dtype}" \
                    --activation-dtype "${activation_dtype}" \
                    --batch-size "${batch_size}" \
                    --sequence-length "${sequence_length}" \
                    --store-tokens "${sequence_length}" \
                    --num-query-heads "${NUM_QUERY_HEADS}" \
                    --num-kv-heads "${NUM_KV_HEADS}" \
                    --head-dim "${HEAD_DIM}" \
                    --block-size "${BLOCK_SIZE}" \
                    --adaptive-splits \
                    --warmup "${WARMUP}" \
                    --iterations "${ITERATIONS}" \
                    --profile-iterations 0 \
                    --device "${DEVICE}" \
                    --trace-dir "${case_dir}" \
                    --no-trace \
                    2>&1 | tee "${case_dir}/profile.log"; then
                    printf '%s\tPASS\n' "${case_name}" >> "${OUTPUT_DIR}/results.tsv"
                else
                    printf '%s\tFAIL\n' "${case_name}" >> "${OUTPUT_DIR}/results.tsv"
                    failures=$((failures + 1))
                fi
            done
        done
    done
done

if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --result-root "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_DIR}/summary"; then
    failures=$((failures + 1))
fi

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info > "${OUTPUT_DIR}/npu_smi_after.txt" 2>&1 || true
fi

if ((failures > 0)); then
    printf '%d operator cases failed.\n' "${failures}"
    exit 1
fi
