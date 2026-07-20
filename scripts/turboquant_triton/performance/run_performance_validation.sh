#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/performance_${TIMESTAMP}}"
CACHE_DTYPES="${CACHE_DTYPES:-turboquant_4bit_nc}"
ACTIVATION_DTYPES="${ACTIVATION_DTYPES:-float16 bfloat16}"
BATCH_SIZES="${BATCH_SIZES:-1 4}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-1024 4096 8192}"
NUM_KV_SPLITS_LIST="${NUM_KV_SPLITS_LIST:-8 16}"
NUM_QUERY_HEADS="${NUM_QUERY_HEADS:-32}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
HEAD_DIM="${HEAD_DIM:-128}"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
DEVICE="${DEVICE:-0}"
RUN_COMPONENT_PROFILE="${RUN_COMPONENT_PROFILE:-1}"
COLLECT_TRACE="${COLLECT_TRACE:-0}"

mkdir -p "${OUTPUT_DIR}/cases"
RESULTS_FILE="${OUTPUT_DIR}/results.tsv"
printf 'case\tstatus\texit_code\tlog\n' >"${RESULTS_FILE}"
FAILURES=0

finalize() {
    local status=$?
    trap - EXIT INT TERM
    if ! tar -czf "${OUTPUT_DIR}.tar.gz" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"; then
        status=1
    fi
    printf 'Performance output: %s\n' "${OUTPUT_DIR}"
    printf 'Performance archive: %s.tar.gz\n' "${OUTPUT_DIR}"
    exit "${status}"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_case() {
    local name="$1"
    shift
    local case_dir="${OUTPUT_DIR}/cases/${name}"
    local log_file="${OUTPUT_DIR}/${name}.log"
    local status

    mkdir -p "${case_dir}"
    set +e
    python3 "${SCRIPT_DIR}/profile_kernels.py" \
        --operation decode \
        --native-baseline \
        --num-query-heads "${NUM_QUERY_HEADS}" \
        --num-kv-heads "${NUM_KV_HEADS}" \
        --head-dim "${HEAD_DIM}" \
        --device "${DEVICE}" \
        --warmup "${WARMUP}" \
        --iterations "${ITERATIONS}" \
        --no-trace \
        --trace-dir "${case_dir}" \
        "$@" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
        printf '%s\tPASS\t0\t%s\n' "${name}" "$(basename "${log_file}")" >>"${RESULTS_FILE}"
    else
        printf '%s\tFAIL\t%s\t%s\n' "${name}" "${status}" "$(basename "${log_file}")" >>"${RESULTS_FILE}"
        FAILURES=$((FAILURES + 1))
    fi
}

cd "${REPO_ROOT}"
python3 "${TQ_COMMON_DIR}/check_environment.py" | tee "${OUTPUT_DIR}/environment.log"

for cache_dtype in ${CACHE_DTYPES}; do
    for activation_dtype in ${ACTIVATION_DTYPES}; do
        for batch_size in ${BATCH_SIZES}; do
            for sequence_length in ${SEQUENCE_LENGTHS}; do
                for num_splits in ${NUM_KV_SPLITS_LIST}; do
                    name="${cache_dtype}_${activation_dtype}_b${batch_size}_s${sequence_length}_k${num_splits}"
                    run_case "${name}" \
                        --cache-dtype "${cache_dtype}" \
                        --activation-dtype "${activation_dtype}" \
                        --batch-size "${batch_size}" \
                        --sequence-length "${sequence_length}" \
                        --num-kv-splits "${num_splits}"
                done
            done
        done
    done
done

set +e
python3 "${SCRIPT_DIR}/summarize_profiles.py" \
    --profile-root "${OUTPUT_DIR}/cases" \
    --output-prefix "${OUTPUT_DIR}/performance_summary" \
    | tee "${OUTPUT_DIR}/summary.log"
SUMMARY_STATUS=${PIPESTATUS[0]}
set -e
if [[ ${SUMMARY_STATUS} -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    printf 'summary\tFAIL\t%s\tsummary.log\n' "${SUMMARY_STATUS}" >>"${RESULTS_FILE}"
else
    printf 'summary\tPASS\t0\tsummary.log\n' >>"${RESULTS_FILE}"
fi

if [[ "${RUN_COMPONENT_PROFILE}" == "1" ]]; then
    COMPONENT_ARGS=(
        --operation all
        --native-baseline
        --cache-dtype turboquant_4bit_nc
        --activation-dtype float16
        --batch-size 4
        --sequence-length 4096
        --num-query-heads "${NUM_QUERY_HEADS}"
        --num-kv-heads "${NUM_KV_HEADS}"
        --head-dim "${HEAD_DIM}"
        --num-kv-splits 8
        --device "${DEVICE}"
        --warmup "${WARMUP}"
        --iterations "${ITERATIONS}"
        --trace-dir "${OUTPUT_DIR}/component_profile"
    )
    if [[ "${COLLECT_TRACE}" != "1" ]]; then
        COMPONENT_ARGS+=(--no-trace)
    fi
    set +e
    python3 "${SCRIPT_DIR}/profile_kernels.py" "${COMPONENT_ARGS[@]}" \
        2>&1 | tee "${OUTPUT_DIR}/component_profile.log"
    status=${PIPESTATUS[0]}
    set -e
    if [[ ${status} -ne 0 ]]; then
        FAILURES=$((FAILURES + 1))
        printf 'component_profile\tFAIL\t%s\tcomponent_profile.log\n' "${status}" >>"${RESULTS_FILE}"
    else
        printf 'component_profile\tPASS\t0\tcomponent_profile.log\n' >>"${RESULTS_FILE}"
    fi
fi

printf 'Performance summary: %s\n' "${OUTPUT_DIR}/performance_summary.md"

if [[ ${FAILURES} -ne 0 ]]; then
    printf 'Performance validation completed with %s failed case(s).\n' "${FAILURES}"
    exit 1
fi
