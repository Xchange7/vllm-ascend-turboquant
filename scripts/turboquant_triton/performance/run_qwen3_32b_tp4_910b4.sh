#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

MODEL="${MODEL:?Set MODEL to the local Qwen3-32B model directory}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"
NETWORK_IFNAME="${NETWORK_IFNAME:-eth0}"
PORT="${PORT:-18104}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
INPUT_TOKENS="${INPUT_TOKENS:-12288}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-16}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-16}"
MEASURE_REQUESTS="${MEASURE_REQUESTS:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"
NPU_POLL_INTERVAL="${NPU_POLL_INTERVAL:-10}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}"
NATIVE_CACHE_DTYPE="${NATIVE_CACHE_DTYPE:-auto}"
RUN_OPERATOR_PROFILE="${RUN_OPERATOR_PROFILE:-1}"
RUN_SERVING="${RUN_SERVING:-1}"
OPERATOR_WARMUP="${OPERATOR_WARMUP:-30}"
OPERATOR_ITERATIONS="${OPERATOR_ITERATIONS:-200}"
COLLECT_OPERATOR_TRACE="${COLLECT_OPERATOR_TRACE:-0}"
SOURCE_ENVIRONMENT="${SOURCE_ENVIRONMENT:-1}"
HCCL_IF_IP_OVERRIDE="${HCCL_IF_IP_OVERRIDE:-}"
DRY_RUN="${DRY_RUN:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/qwen3_32b_tp4_910b4_${TIMESTAMP}}"

readonly TP_SIZE=4

NPU_MONITOR_PID=""
FAILURES=0
FINALIZED=0
RESULTS_FILE="${OUTPUT_DIR}/results.tsv"
CONFIG_FILE="${OUTPUT_DIR}/configuration.env"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

source_runtime_environment() {
    if [[ "${SOURCE_ENVIRONMENT}" != "1" ]]; then
        return
    fi

    local cann_env="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
    local atb_env="${ATB_ENV_SCRIPT:-/usr/local/Ascend/nnal/atb/set_env.sh}"
    local custom_env="${CUSTOM_OP_ENV_SCRIPT:-${REPO_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash}"
    if [[ -f "${cann_env}" ]]; then
        # CANN setup scripts are not guaranteed to be nounset-clean.
        set +u
        # shellcheck disable=SC1090
        source "${cann_env}" >/dev/null 2>&1
        set -u
    fi
    if [[ -f "${atb_env}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${atb_env}" >/dev/null 2>&1
        set -u
    fi
    if [[ -f "${custom_env}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${custom_env}" >/dev/null 2>&1
        set -u
    fi
}

require_non_negative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s must be a non-negative integer; got %s.\n' \
            "${name}" "${value}" >&2
        return 2
    fi
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    require_non_negative_integer "${name}" "${value}"
    if ((value == 0)); then
        printf '%s must be positive.\n' "${name}" >&2
        return 2
    fi
}

validate_configuration() {
    local setting
    local value
    local concurrency
    local device
    local max_concurrency=0
    local seen_concurrencies=" "
    local seen_devices=" "
    local concurrency_values=()
    local device_values=()

    for setting in \
        MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS INPUT_TOKENS OUTPUT_TOKENS MEASURE_REQUESTS \
        OPERATOR_WARMUP OPERATOR_ITERATIONS REQUEST_TIMEOUT \
        SERVER_TIMEOUT NPU_POLL_INTERVAL PORT; do
        value="${!setting}"
        require_positive_integer "${setting}" "${value}"
    done
    require_non_negative_integer WARMUP_REQUESTS "${WARMUP_REQUESTS}"

    for setting in \
        RUN_OPERATOR_PROFILE RUN_SERVING COLLECT_OPERATOR_TRACE \
        SOURCE_ENVIRONMENT DRY_RUN ENFORCE_EAGER ENABLE_CHUNKED_PREFILL; do
        value="${!setting}"
        if [[ "${value}" != "0" && "${value}" != "1" ]]; then
            printf '%s must be 0 or 1; got %s.\n' "${setting}" "${value}" >&2
            return 2
        fi
    done

    IFS=',' read -r -a device_values <<<"${VISIBLE_DEVICES}"
    if ((${#device_values[@]} != TP_SIZE)); then
        printf 'VISIBLE_DEVICES must contain exactly %s devices; got %s.\n' \
            "${TP_SIZE}" "${VISIBLE_DEVICES}" >&2
        return 2
    fi
    for device in "${device_values[@]}"; do
        if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
            printf 'Invalid NPU device id: %s\n' "${device}" >&2
            return 2
        fi
        if [[ "${seen_devices}" == *" ${device} "* ]]; then
            printf 'Duplicate NPU device id: %s\n' "${device}" >&2
            return 2
        fi
        seen_devices+="${device} "
    done

    read -r -a concurrency_values <<<"${CONCURRENCY_LEVELS}"
    if ((${#concurrency_values[@]} == 0)); then
        printf 'CONCURRENCY_LEVELS must contain at least one positive integer.\n' >&2
        return 2
    fi
    for concurrency in "${concurrency_values[@]}"; do
        if [[ ! "${concurrency}" =~ ^[1-9][0-9]*$ ]]; then
            printf 'Invalid concurrency: %s\n' "${concurrency}" >&2
            return 2
        fi
        if [[ "${seen_concurrencies}" == *" ${concurrency} "* ]]; then
            printf 'Duplicate concurrency: %s\n' "${concurrency}" >&2
            return 2
        fi
        seen_concurrencies+="${concurrency} "
        if ((concurrency > max_concurrency)); then
            max_concurrency="${concurrency}"
        fi
    done

    if [[ -z "${MAX_NUM_SEQS}" ]]; then
        MAX_NUM_SEQS="${max_concurrency}"
    fi
    require_positive_integer MAX_NUM_SEQS "${MAX_NUM_SEQS}"
    if ((MAX_NUM_SEQS < max_concurrency)); then
        printf 'MAX_NUM_SEQS=%s is smaller than maximum concurrency %s.\n' \
            "${MAX_NUM_SEQS}" "${max_concurrency}" >&2
        return 2
    fi
    if ((MEASURE_REQUESTS < max_concurrency)); then
        printf 'MEASURE_REQUESTS=%s is smaller than maximum concurrency %s.\n' \
            "${MEASURE_REQUESTS}" "${max_concurrency}" >&2
        return 2
    fi
    if ((WARMUP_REQUESTS != 0 && WARMUP_REQUESTS < max_concurrency)); then
        printf 'WARMUP_REQUESTS must be zero or at least one full concurrency wave (%s).\n' \
            "${max_concurrency}" >&2
        return 2
    fi
    if ((INPUT_TOKENS + OUTPUT_TOKENS > MAX_MODEL_LEN)); then
        printf 'Token budget exceeds MAX_MODEL_LEN: %s + %s > %s.\n' \
            "${INPUT_TOKENS}" "${OUTPUT_TOKENS}" "${MAX_MODEL_LEN}" >&2
        return 2
    fi

    if [[ "${DRY_RUN}" != "1" ]]; then
        if [[ ! -d "${MODEL}" ]]; then
            printf 'MODEL must be a local model directory for a reproducible benchmark: %s\n' \
                "${MODEL}" >&2
            return 2
        fi
        if [[ ! -d "/sys/class/net/${NETWORK_IFNAME}" ]]; then
            printf 'Network interface does not exist: %s\n' "${NETWORK_IFNAME}" >&2
            return 2
        fi
        for command in "${PYTHON_BIN}" curl git vllm; do
            if ! command -v "${command}" >/dev/null 2>&1; then
                printf 'Required command is unavailable: %s\n' "${command}" >&2
                return 2
            fi
        done
    fi
}

validate_910b4_devices() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN=1: skipping physical 910B4 validation.\n'
        return
    fi

    ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}" "${PYTHON_BIN}" - "${TP_SIZE}" <<'PY'
import re
import sys

import torch
import torch_npu  # noqa: F401

expected_count = int(sys.argv[1])
device_count = torch.npu.device_count()
if device_count < expected_count:
    raise SystemExit(
        f"TP4 requires four visible NPUs, but torch reports {device_count}"
    )

names = [torch.npu.get_device_name(index) for index in range(expected_count)]
normalized = [re.sub(r"[\s_-]+", "", name).lower() for name in names]
unsupported = [
    name for name, value in zip(names, normalized) if not value.startswith("ascend910b4")
]
if unsupported:
    raise SystemExit(
        "This benchmark is restricted to Ascend 910B4; detected: "
        + ", ".join(names)
    )
print("Validated TP4 devices: " + ", ".join(names))
PY
}

write_configuration() {
    mkdir -p "${OUTPUT_DIR}"
    {
        printf 'MODEL=%q\n' "${MODEL}"
        printf 'VISIBLE_DEVICES=%q\n' "${VISIBLE_DEVICES}"
        printf 'TP_SIZE=%q\n' "${TP_SIZE}"
        printf 'NETWORK_IFNAME=%q\n' "${NETWORK_IFNAME}"
        printf 'PORT=%q\n' "${PORT}"
        printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN}"
        printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "${MAX_NUM_BATCHED_TOKENS}"
        printf 'INPUT_TOKENS=%q\n' "${INPUT_TOKENS}"
        printf 'OUTPUT_TOKENS=%q\n' "${OUTPUT_TOKENS}"
        printf 'CONCURRENCY_LEVELS=%q\n' "${CONCURRENCY_LEVELS}"
        printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS}"
        printf 'WARMUP_REQUESTS=%q\n' "${WARMUP_REQUESTS}"
        printf 'MEASURE_REQUESTS=%q\n' "${MEASURE_REQUESTS}"
        printf 'GPU_MEMORY_UTILIZATION=%q\n' "${GPU_MEMORY_UTILIZATION}"
        printf 'ENFORCE_EAGER=%q\n' "${ENFORCE_EAGER}"
        printf 'ENABLE_CHUNKED_PREFILL=%q\n' "${ENABLE_CHUNKED_PREFILL}"
        printf 'HCCL_IF_IP=%q\n' "${HCCL_IF_IP-<unset>}"
        printf 'TQ_CACHE_DTYPE=%q\n' "${TQ_CACHE_DTYPE}"
        printf 'NATIVE_CACHE_DTYPE=%q\n' "${NATIVE_CACHE_DTYPE}"
        printf 'RUN_OPERATOR_PROFILE=%q\n' "${RUN_OPERATOR_PROFILE}"
        printf 'RUN_SERVING=%q\n' "${RUN_SERVING}"
        printf 'OPERATOR_WARMUP=%q\n' "${OPERATOR_WARMUP}"
        printf 'OPERATOR_ITERATIONS=%q\n' "${OPERATOR_ITERATIONS}"
        printf 'COLLECT_OPERATOR_TRACE=%q\n' "${COLLECT_OPERATOR_TRACE}"
        printf 'GIT_COMMIT=%q\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'GIT_BRANCH=%q\n' "$(git -C "${REPO_ROOT}" branch --show-current)"
        printf 'GENERATED_AT=%q\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    } >"${CONFIG_FILE}"
}

start_npu_monitor() {
    local stage="$1"
    local output="${OUTPUT_DIR}/${stage}_npu.log"
    if [[ "${DRY_RUN}" == "1" ]] || ! command -v npu-smi >/dev/null 2>&1; then
        return
    fi
    (
        while true; do
            printf '\n===== %s stage=%s =====\n' \
                "$(date --iso-8601=seconds 2>/dev/null || date)" "${stage}"
            if command -v timeout >/dev/null 2>&1; then
                timeout 20s npu-smi info || true
            else
                npu-smi info || true
            fi
            sleep "${NPU_POLL_INTERVAL}"
        done
    ) >"${output}" 2>&1 &
    NPU_MONITOR_PID=$!
}

stop_npu_monitor() {
    if [[ -n "${NPU_MONITOR_PID}" ]] && kill -0 "${NPU_MONITOR_PID}" 2>/dev/null; then
        kill "${NPU_MONITOR_PID}" 2>/dev/null || true
        wait "${NPU_MONITOR_PID}" 2>/dev/null || true
    fi
    NPU_MONITOR_PID=""
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_stage() {
    local stage="$1"
    local description="$2"
    shift 2
    local stage_log="${OUTPUT_DIR}/${stage}.log"
    local status

    printf '\n===== %s: %s =====\n' "${stage}" "${description}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command "$@"
        printf '%s\tDRY_RUN\t0\t%s\n' "${stage}" "$(basename "${stage_log}")" \
            >>"${RESULTS_FILE}"
        return
    fi

    start_npu_monitor "${stage}"
    set +e
    "$@" 2>&1 | tee "${stage_log}"
    status=${PIPESTATUS[0]}
    set -e
    stop_npu_monitor
    if ((status == 0)); then
        printf '%s\tPASS\t0\t%s\n' "${stage}" "$(basename "${stage_log}")" \
            >>"${RESULTS_FILE}"
    else
        printf '%s\tFAIL\t%s\t%s\n' \
            "${stage}" "${status}" "$(basename "${stage_log}")" >>"${RESULTS_FILE}"
        FAILURES=$((FAILURES + 1))
    fi
}

write_summary() {
    {
        printf 'Qwen3-32B TurboQuant TP4 / Ascend 910B4 performance benchmark\n'
        printf 'Model: %s\n' "${MODEL}"
        printf 'Devices: %s\n' "${VISIBLE_DEVICES}"
        printf 'Concurrency levels: %s\n' "${CONCURRENCY_LEVELS}"
        printf 'Workload: input=%s output=%s max_model_len=%s\n' \
            "${INPUT_TOKENS}" "${OUTPUT_TOKENS}" "${MAX_MODEL_LEN}"
        printf 'Scheduler: max_num_batched_tokens=%s max_num_seqs=%s chunked_prefill=%s\n' \
            "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" "${ENABLE_CHUNKED_PREFILL}"
        printf 'Failures: %s\n\n' "${FAILURES}"
        column -t -s $'\t' "${RESULTS_FILE}" 2>/dev/null || cat "${RESULTS_FILE}"
        printf '\nOperator reports: %s/operator\n' "${OUTPUT_DIR}"
        printf 'Serving reports: %s/serving\n' "${OUTPUT_DIR}"
        printf 'Configuration: %s\n' "${CONFIG_FILE}"
    } >"${SUMMARY_FILE}"
    cat "${SUMMARY_FILE}"
}

finalize() {
    local status=$?
    if [[ "${FINALIZED}" == "1" ]]; then
        exit "${status}"
    fi
    FINALIZED=1
    trap - EXIT INT TERM
    stop_npu_monitor
    if [[ -f "${RESULTS_FILE}" ]]; then
        write_summary
    fi
    if [[ "${DRY_RUN}" != "1" && -d "${OUTPUT_DIR}" ]]; then
        if ! tar -czf "${OUTPUT_DIR}.tar.gz" \
            -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"; then
            status=1
        fi
        printf 'Benchmark archive: %s.tar.gz\n' "${OUTPUT_DIR}"
    fi
    if ((status == 0 && FAILURES != 0)); then
        status=1
    fi
    exit "${status}"
}

run_operator_profile() {
    env \
        ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES%%,*}" \
        OUTPUT_DIR="${OUTPUT_DIR}/operator" \
        WARMUP="${OPERATOR_WARMUP}" \
        ITERATIONS="${OPERATOR_ITERATIONS}" \
        COLLECT_TRACE="${COLLECT_OPERATOR_TRACE}" \
        bash "${SCRIPT_DIR}/profile_qwen3_32b_tp4.sh"
}

run_serving_benchmark() {
    env \
        ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
        VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
        MODEL="${MODEL}" \
        TP_SIZE="${TP_SIZE}" \
        NETWORK_IFNAME="${NETWORK_IFNAME}" \
        PORT="${PORT}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        INPUT_TOKENS="${INPUT_TOKENS}" \
        OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
        CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS}" \
        WARMUP_REQUESTS="${WARMUP_REQUESTS}" \
        MEASURE_REQUESTS="${MEASURE_REQUESTS}" \
        REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
        SERVER_TIMEOUT="${SERVER_TIMEOUT}" \
        ENFORCE_EAGER="${ENFORCE_EAGER}" \
        ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL}" \
        NATIVE_CACHE_DTYPE="${NATIVE_CACHE_DTYPE}" \
        TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=ascend_fused \
        OUTPUT_DIR="${OUTPUT_DIR}/serving" \
        bash "${SCRIPT_DIR}/run_serving_benchmark.sh"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source_runtime_environment
validate_configuration
if [[ -n "${HCCL_IF_IP_OVERRIDE}" ]]; then
    export HCCL_IF_IP="${HCCL_IF_IP_OVERRIDE}"
else
    # Do not inherit an address that conflicts with NETWORK_IFNAME. A
    # single-node TP4 launch only needs interface-based selection.
    unset HCCL_IF_IP
fi
validate_910b4_devices
write_configuration
printf 'stage\tstatus\texit_code\tlog\n' >"${RESULTS_FILE}"

if [[ "${RUN_OPERATOR_PROFILE}" == "1" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_stage operator \
            'TP4 per-rank attention microbenchmark' \
            env ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES%%,*}" \
            OUTPUT_DIR="${OUTPUT_DIR}/operator" \
            WARMUP="${OPERATOR_WARMUP}" \
            ITERATIONS="${OPERATOR_ITERATIONS}" \
            COLLECT_TRACE="${COLLECT_OPERATOR_TRACE}" \
            bash "${SCRIPT_DIR}/profile_qwen3_32b_tp4.sh"
    else
        run_stage operator \
            'TP4 per-rank attention microbenchmark' \
            run_operator_profile
    fi
fi

if [[ "${RUN_SERVING}" == "1" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_stage serving \
            'Actual Qwen3-32B TP4 native/TurboQuant serving A/B' \
            env ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
            MODEL="${MODEL}" TP_SIZE="${TP_SIZE}" \
            CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS}" \
            INPUT_TOKENS="${INPUT_TOKENS}" OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
            MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
            ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL}" \
            WARMUP_REQUESTS="${WARMUP_REQUESTS}" \
            MEASURE_REQUESTS="${MEASURE_REQUESTS}" \
            OUTPUT_DIR="${OUTPUT_DIR}/serving" \
            bash "${SCRIPT_DIR}/run_serving_benchmark.sh"
    else
        run_stage serving \
            'Actual Qwen3-32B TP4 native/TurboQuant serving A/B' \
            run_serving_benchmark
    fi
fi

if ((FAILURES != 0)); then
    exit 1
fi
