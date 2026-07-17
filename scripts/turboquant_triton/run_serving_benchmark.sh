#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:?Set MODEL to the local model path}"
PORT="${PORT:-18001}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/serving_${TIMESTAMP}}"

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
    while ((SECONDS < deadline)); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            printf 'Server exited before becoming healthy. Last log lines:\n' >&2
            tail -n 100 "${log_file}" >&2 || true
            return 1
        fi
        if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
            return 0
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
        bash "${SCRIPT_DIR}/serve_qwen3_32b.sh" \
            --no-enable-prefix-caching >"${log_file}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${log_file}"
}

run_client() {
    local label="$1"
    local cache_dtype="$2"
    python3 "${SCRIPT_DIR}/serving_benchmark.py" run \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "${MODEL}" \
        --label "${label}" \
        --cache-dtype "${cache_dtype}" \
        --input-tokens "${INPUT_TOKENS}" \
        --output-tokens "${OUTPUT_TOKENS}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --warmup-requests "${WARMUP_REQUESTS}" \
        --requests "${MEASURE_REQUESTS}" \
        --timeout "${REQUEST_TIMEOUT}" \
        --output "${OUTPUT_DIR}/${label}.json" \
        | tee "${OUTPUT_DIR}/${label}_client.log"
}

cd "${REPO_ROOT}"
if port_is_listening; then
    printf 'Port %s is already in use; refusing to benchmark another process.\n' \
        "${PORT}" >&2
    exit 2
fi
python3 "${SCRIPT_DIR}/check_environment.py" | tee "${OUTPUT_DIR}/environment.log"

start_server native "${NATIVE_CACHE_DTYPE}"
run_client native "${NATIVE_CACHE_DTYPE}"
stop_server

start_server turboquant "${TQ_CACHE_DTYPE}"
run_client turboquant "${TQ_CACHE_DTYPE}"
stop_server

python3 "${SCRIPT_DIR}/serving_benchmark.py" compare \
    --native "${OUTPUT_DIR}/native.json" \
    --turboquant "${OUTPUT_DIR}/turboquant.json" \
    --native-log "${OUTPUT_DIR}/native_server.log" \
    --turboquant-log "${OUTPUT_DIR}/turboquant_server.log" \
    --output "${OUTPUT_DIR}/comparison.json" \
    --summary-markdown "${OUTPUT_DIR}/summary.md" \
    | tee "${OUTPUT_DIR}/comparison.log"
