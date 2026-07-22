#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:?Set MODEL to the local model path}"
PORT="${PORT:-18001}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
CONCURRENCY="${CONCURRENCY:-1}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-${CONCURRENCY}}"
read -r -a CONCURRENCY_VALUES <<<"${CONCURRENCY_LEVELS}"
if ((${#CONCURRENCY_VALUES[@]} == 0)); then
    printf 'CONCURRENCY_LEVELS must contain at least one positive integer.\n' >&2
    exit 2
fi
MAX_CONCURRENCY=0
SEEN_CONCURRENCIES=" "
for concurrency_value in "${CONCURRENCY_VALUES[@]}"; do
    if [[ ! "${concurrency_value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Invalid concurrency value: %s\n' "${concurrency_value}" >&2
        exit 2
    fi
    if [[ "${SEEN_CONCURRENCIES}" == *" ${concurrency_value} "* ]]; then
        printf 'Duplicate concurrency value: %s\n' "${concurrency_value}" >&2
        exit 2
    fi
    SEEN_CONCURRENCIES+="${concurrency_value} "
    if ((concurrency_value > MAX_CONCURRENCY)); then
        MAX_CONCURRENCY="${concurrency_value}"
    fi
done
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${MAX_CONCURRENCY}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INPUT_TOKENS="${INPUT_TOKENS:-1024}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
MEASURE_REQUESTS="${MEASURE_REQUESTS:-10}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
NATIVE_CACHE_DTYPE="${NATIVE_CACHE_DTYPE:-auto}"
TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}"
MAX_TTFT_RATIO="${MAX_TTFT_RATIO:-}"
MAX_TPOT_RATIO="${MAX_TPOT_RATIO:-}"
MIN_THROUGHPUT_RATIO="${MIN_THROUGHPUT_RATIO:-}"
MIN_KV_CAPACITY_RATIO="${MIN_KV_CAPACITY_RATIO:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/serving_${TIMESTAMP}}"

for positive_setting in MAX_NUM_SEQS MEASURE_REQUESTS; do
    value="${!positive_setting}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer; got %s.\n' "${positive_setting}" "${value}" >&2
        exit 2
    fi
done
if [[ ! "${WARMUP_REQUESTS}" =~ ^[0-9]+$ ]]; then
    printf 'WARMUP_REQUESTS must be a non-negative integer; got %s.\n' "${WARMUP_REQUESTS}" >&2
    exit 2
fi
if ((MAX_NUM_SEQS < MAX_CONCURRENCY)); then
    printf 'MAX_NUM_SEQS=%s is smaller than maximum concurrency %s.\n' \
        "${MAX_NUM_SEQS}" "${MAX_CONCURRENCY}" >&2
    exit 2
fi
if ((MEASURE_REQUESTS < MAX_CONCURRENCY)); then
    printf 'MEASURE_REQUESTS=%s is smaller than maximum concurrency %s.\n' \
        "${MEASURE_REQUESTS}" "${MAX_CONCURRENCY}" >&2
    exit 2
fi

SERVER_PID=""
mkdir -p "${OUTPUT_DIR}"

port_is_listening() {
    python3 -c '
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1)
    sys.exit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
' "${PORT}"
}

stop_server() {
    if [[ -z "${SERVER_PID}" ]]; then
        return
    fi
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 60); do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -KILL "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""

    for _ in $(seq 1 60); do
        if ! port_is_listening; then
            return
        fi
        sleep 1
    done
    printf 'Port %s is still listening after server shutdown.\n' "${PORT}" >&2
    return 1
}

finalize() {
    local status=$?
    trap - EXIT INT TERM
    if ! stop_server; then
        status=1
    fi
    if ! tar -czf "${OUTPUT_DIR}.tar.gz" \
        -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"; then
        status=1
    fi
    printf 'Benchmark output: %s\n' "${OUTPUT_DIR}"
    printf 'Benchmark archive: %s.tar.gz\n' "${OUTPUT_DIR}"
    exit "${status}"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
    local log_file="$1"
    local deadline=$((SECONDS + SERVER_TIMEOUT))
    local next_report=$((SECONDS + 30))
    while ((SECONDS < deadline)); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            printf 'Server exited before becoming healthy. Last log lines:\n' >&2
            tail -n 100 "${log_file}" >&2 || true
            return 1
        fi
        if curl --noproxy '*' --fail --silent \
            --connect-timeout 2 --max-time 5 \
            "http://127.0.0.1:${PORT}/health" >/dev/null; then
            printf 'Server is healthy on port %s; starting requests.\n' "${PORT}"
            return 0
        fi
        if ((SECONDS >= next_report)); then
            printf 'Still waiting for server health on port %s; latest log: ' \
                "${PORT}"
            tail -n 1 "${log_file}" || true
            next_report=$((SECONDS + 30))
        fi
        sleep 5
    done
    printf 'Server health check timed out after %ss. Last log lines:\n' \
        "${SERVER_TIMEOUT}" >&2
    tail -n 100 "${log_file}" >&2 || true
    return 1
}

start_server() {
    local label="$1"
    local cache_dtype="$2"
    local log_file="${OUTPUT_DIR}/${label}_server.log"

    printf 'Starting %s server (cache dtype: %s).\n' "${label}" "${cache_dtype}"
    MODEL="${MODEL}" \
        PORT="${PORT}" \
        TP_SIZE="${TP_SIZE}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        KV_CACHE_DTYPE="${cache_dtype}" \
        ENFORCE_EAGER="${ENFORCE_EAGER}" \
        bash "${TQ_COMMON_DIR}/serve_qwen3_32b.sh" \
            --no-enable-prefix-caching >"${log_file}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${log_file}"
}

run_client() {
    local label="$1"
    local cache_dtype="$2"
    local concurrency="$3"
    local artifact_label="${label}"
    if ((${#CONCURRENCY_VALUES[@]} > 1)); then
        artifact_label="${label}_c${concurrency}"
    fi
    python3 "${SCRIPT_DIR}/serving_benchmark.py" run \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "${MODEL}" \
        --label "${artifact_label}" \
        --cache-dtype "${cache_dtype}" \
        --input-tokens "${INPUT_TOKENS}" \
        --output-tokens "${OUTPUT_TOKENS}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --warmup-requests "${WARMUP_REQUESTS}" \
        --requests "${MEASURE_REQUESTS}" \
        --concurrency "${concurrency}" \
        --timeout "${REQUEST_TIMEOUT}" \
        --output "${OUTPUT_DIR}/${artifact_label}.json" \
        | tee "${OUTPUT_DIR}/${artifact_label}_client.log"
}

artifact_label() {
    local label="$1"
    local concurrency="$2"
    if ((${#CONCURRENCY_VALUES[@]} > 1)); then
        printf '%s_c%s' "${label}" "${concurrency}"
    else
        printf '%s' "${label}"
    fi
}

cd "${REPO_ROOT}"
if port_is_listening; then
    printf 'Port %s is already in use; refusing to benchmark another process.\n' \
        "${PORT}" >&2
    exit 2
fi
python3 "${TQ_COMMON_DIR}/check_environment.py" | tee "${OUTPUT_DIR}/environment.log"

start_server native "${NATIVE_CACHE_DTYPE}"
for concurrency in "${CONCURRENCY_VALUES[@]}"; do
    run_client native "${NATIVE_CACHE_DTYPE}" "${concurrency}"
done
stop_server

start_server turboquant "${TQ_CACHE_DTYPE}"
for concurrency in "${CONCURRENCY_VALUES[@]}"; do
    run_client turboquant "${TQ_CACHE_DTYPE}" "${concurrency}"
done
stop_server

COMPARISON_FAILURES=0
for concurrency in "${CONCURRENCY_VALUES[@]}"; do
    native_label="$(artifact_label native "${concurrency}")"
    turboquant_label="$(artifact_label turboquant "${concurrency}")"
    comparison_suffix=""
    if ((${#CONCURRENCY_VALUES[@]} > 1)); then
        comparison_suffix="_c${concurrency}"
    fi
    comparison_args=(
        compare
        --native "${OUTPUT_DIR}/${native_label}.json"
        --turboquant "${OUTPUT_DIR}/${turboquant_label}.json"
        --native-log "${OUTPUT_DIR}/native_server.log"
        --turboquant-log "${OUTPUT_DIR}/turboquant_server.log"
        --output "${OUTPUT_DIR}/comparison${comparison_suffix}.json"
        --summary-markdown "${OUTPUT_DIR}/summary${comparison_suffix}.md"
    )
    [[ -n "${MAX_TTFT_RATIO}" ]] && comparison_args+=(--max-ttft-ratio "${MAX_TTFT_RATIO}")
    [[ -n "${MAX_TPOT_RATIO}" ]] && comparison_args+=(--max-tpot-ratio "${MAX_TPOT_RATIO}")
    [[ -n "${MIN_THROUGHPUT_RATIO}" ]] && comparison_args+=(--min-throughput-ratio "${MIN_THROUGHPUT_RATIO}")
    [[ -n "${MIN_KV_CAPACITY_RATIO}" ]] && comparison_args+=(--min-kv-capacity-ratio "${MIN_KV_CAPACITY_RATIO}")
    set +e
    python3 "${SCRIPT_DIR}/serving_benchmark.py" "${comparison_args[@]}" \
        | tee "${OUTPUT_DIR}/comparison${comparison_suffix}.log"
    comparison_status=${PIPESTATUS[0]}
    set -e
    if ((comparison_status != 0)); then
        COMPARISON_FAILURES=$((COMPARISON_FAILURES + 1))
    fi
done

if ((${#CONCURRENCY_VALUES[@]} > 1)); then
    {
        printf '# TurboQuant Serving Matrix\n\n'
        for concurrency in "${CONCURRENCY_VALUES[@]}"; do
            printf '## Concurrency %s\n\n' "${concurrency}"
            summary_path="${OUTPUT_DIR}/summary_c${concurrency}.md"
            if [[ -f "${summary_path}" ]]; then
                cat "${summary_path}"
            else
                printf 'Comparison did not produce a summary; inspect `comparison_c%s.log`.\n' \
                    "${concurrency}"
            fi
            printf '\n\n'
        done
    } >"${OUTPUT_DIR}/summary.md"
fi

if ((COMPARISON_FAILURES > 0)); then
    printf '%s serving comparison(s) failed performance gates.\n' "${COMPARISON_FAILURES}" >&2
    exit 1
fi
