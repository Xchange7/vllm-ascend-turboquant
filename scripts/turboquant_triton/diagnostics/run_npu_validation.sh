#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/logs/turboquant/npu_validation_${TIMESTAMP}}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"
RUN_ACCURACY="${RUN_ACCURACY:-1}"
RUN_PERFORMANCE="${RUN_PERFORMANCE:-1}"
RESULTS_FILE="${RUN_ROOT}/results.tsv"
COMBINED_LOG="${RUN_ROOT}/combined.log"
FAILURES=0

mkdir -p "${RUN_ROOT}"
printf 'stage\tstatus\texit_code\tlog\n' >"${RESULTS_FILE}"
: >"${COMBINED_LOG}"

run_stage() {
    local stage="$1"
    shift
    local log_file="${RUN_ROOT}/${stage}.log"
    local status

    printf '\n===== %s =====\n' "${stage}" | tee -a "${COMBINED_LOG}"
    set +e
    "$@" 2>&1 | tee "${log_file}" | tee -a "${COMBINED_LOG}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ ${status} -eq 0 ]]; then
        printf '%s\tPASS\t0\t%s\n' "${stage}" "$(basename "${log_file}")" >>"${RESULTS_FILE}"
    else
        printf '%s\tFAIL\t%s\t%s\n' "${stage}" "${status}" "$(basename "${log_file}")" >>"${RESULTS_FILE}"
        FAILURES=$((FAILURES + 1))
    fi
}

skip_stage() {
    printf '%s\tSKIP\t0\tdisabled by environment\n' "$1" >>"${RESULTS_FILE}"
}

cd "${REPO_ROOT}"

if [[ "${RUN_CORRECTNESS}" == "1" ]]; then
    run_stage correctness env \
        LOG_ROOT="${RUN_ROOT}/correctness" \
        RUN_PROFILE=0 \
        bash "${TQ_DIAGNOSTICS_DIR}/run_diagnostic_suite.sh"
    if [[ -n "${MODEL:-}" ]]; then
        run_stage graph_e2e env \
            OUTPUT_DIR="${RUN_ROOT}/graph_e2e" \
            BASE_LABEL=turboquant_eager \
            BASE_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}" \
            BASE_ENFORCE_EAGER=1 \
            TEST_LABEL=turboquant_aclgraph \
            TEST_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}" \
            TEST_ENFORCE_EAGER=0 \
            bash "${TQ_CORRECTNESS_DIR}/run_accuracy_comparison.sh"
    else
        printf 'graph_e2e\tSKIP\t0\tMODEL is unset\n' >>"${RESULTS_FILE}"
    fi
else
    skip_stage correctness
    skip_stage graph_e2e
fi

if [[ "${RUN_ACCURACY}" == "1" ]]; then
    if [[ -z "${MODEL:-}" ]]; then
        printf 'MODEL must be set when RUN_ACCURACY=1.\n' | tee -a "${COMBINED_LOG}"
        printf 'accuracy\tFAIL\t2\tMODEL is unset\n' >>"${RESULTS_FILE}"
        FAILURES=$((FAILURES + 1))
    else
        run_stage accuracy env \
            OUTPUT_DIR="${RUN_ROOT}/accuracy" \
            bash "${TQ_CORRECTNESS_DIR}/run_accuracy_comparison.sh"
    fi
else
    skip_stage accuracy
fi

if [[ "${RUN_PERFORMANCE}" == "1" ]]; then
    run_stage performance env \
        OUTPUT_DIR="${RUN_ROOT}/performance" \
        bash "${TQ_PERFORMANCE_DIR}/run_performance_validation.sh"
else
    skip_stage performance
fi

{
    printf 'TurboQuant NPU validation\n'
    printf 'Generated: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    printf 'Run root: %s\n' "${RUN_ROOT}"
    printf 'Failed stages: %s\n\n' "${FAILURES}"
    column -t -s $'\t' "${RESULTS_FILE}" 2>/dev/null || cat "${RESULTS_FILE}"
} >"${RUN_ROOT}/summary.txt"
cat "${RUN_ROOT}/summary.txt"

tar -czf "${RUN_ROOT}.tar.gz" -C "$(dirname "${RUN_ROOT}")" "$(basename "${RUN_ROOT}")"
printf 'Validation archive: %s.tar.gz\n' "${RUN_ROOT}"

if [[ ${FAILURES} -ne 0 ]]; then
    exit 1
fi
