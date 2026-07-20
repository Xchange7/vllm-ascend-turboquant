#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/concurrency_splits_${TIMESTAMP}}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
NUM_KV_SPLITS_LIST="${NUM_KV_SPLITS_LIST:-1 2 4 8 16 32}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16384}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-10}"
DEVICE="${DEVICE:-0}"

mkdir -p "${OUTPUT_DIR}/cases"
RESULTS_FILE="${OUTPUT_DIR}/results.tsv"
printf 'mode\tbatch_size\tconfigured_splits\tstatus\tlog\n' >"${RESULTS_FILE}"
FAILURES=0

finalize() {
    local status=$?
    trap - EXIT INT TERM
    tar -czf "${OUTPUT_DIR}.tar.gz" \
        -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")" || status=1
    printf 'Concurrency profile output: %s\n' "${OUTPUT_DIR}"
    printf 'Concurrency profile archive: %s.tar.gz\n' "${OUTPUT_DIR}"
    exit "${status}"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_case() {
    local mode="$1"
    local batch_size="$2"
    local num_splits="$3"
    local adaptive="$4"
    local name="${mode}_b${batch_size}_k${num_splits}"
    local case_dir="${OUTPUT_DIR}/cases/${name}"
    local log_file="${OUTPUT_DIR}/${name}.log"
    local adaptive_args=()
    local status

    if [[ "${adaptive}" == "1" ]]; then
        adaptive_args+=(--adaptive-splits)
    fi
    mkdir -p "${case_dir}"
    set +e
    python3 "${SCRIPT_DIR}/profile_kernels.py" \
        --operation decode \
        --native-baseline \
        --cache-dtype turboquant_4bit_nc \
        --activation-dtype "${ACTIVATION_DTYPE}" \
        --batch-size "${batch_size}" \
        --sequence-length "${SEQUENCE_LENGTH}" \
        --num-query-heads 16 \
        --num-kv-heads 2 \
        --head-dim 128 \
        --num-kv-splits "${num_splits}" \
        --decode-implementation auto \
        --grouped-block-kv 16 \
        --device "${DEVICE}" \
        --warmup "${WARMUP}" \
        --iterations "${ITERATIONS}" \
        --no-trace \
        --trace-dir "${case_dir}" \
        "${adaptive_args[@]}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ ${status} -eq 0 ]]; then
        printf '%s\t%s\t%s\tPASS\t%s\n' \
            "${mode}" "${batch_size}" "${num_splits}" "$(basename "${log_file}")" \
            >>"${RESULTS_FILE}"
    else
        printf '%s\t%s\t%s\tFAIL(%s)\t%s\n' \
            "${mode}" "${batch_size}" "${num_splits}" "${status}" "$(basename "${log_file}")" \
            >>"${RESULTS_FILE}"
        FAILURES=$((FAILURES + 1))
    fi
}

cd "${REPO_ROOT}"
python3 "${TQ_COMMON_DIR}/check_environment.py" | tee "${OUTPUT_DIR}/environment.log"

for batch_size in ${BATCH_SIZES}; do
    run_case adaptive "${batch_size}" 32 1
    for num_splits in ${NUM_KV_SPLITS_LIST}; do
        run_case fixed "${batch_size}" "${num_splits}" 0
    done
done

set +e
python3 "${SCRIPT_DIR}/summarize_profiles.py" \
    --profile-root "${OUTPUT_DIR}/cases" \
    --output-prefix "${OUTPUT_DIR}/summary" \
    | tee "${OUTPUT_DIR}/summary.log"
SUMMARY_STATUS=${PIPESTATUS[0]}
set -e
if [[ ${SUMMARY_STATUS} -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
fi

if [[ ${FAILURES} -ne 0 ]]; then
    printf 'Concurrency profiling completed with %s failure(s).\n' "${FAILURES}" >&2
    exit 1
fi
