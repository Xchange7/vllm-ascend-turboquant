#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-/run/test_llm/Qwen3-0.6B-hf}"
PORT="${PORT:-18000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant_4bit_nc}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-600}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
GLOO_PROBE_TIMEOUT="${GLOO_PROBE_TIMEOUT:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/model_startup_${TIMESTAMP}}"
SERVER_LOG="${OUTPUT_DIR}/turboquant_server.txt"
DIAGNOSTIC_LOG="${OUTPUT_DIR}/startup_diagnostics.txt"
SERVER_PID=""

mkdir -p "${OUTPUT_DIR}"

# This diagnostic targets one host. Loopback removes hostname and container
# route discovery from vLLM's TCP rendezvous and Gloo CPU groups.
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

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
    if [[ -z "${SERVER_PID}" ]] || ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        return
    fi
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
}

finalize() {
    local status=$?
    trap - EXIT INT TERM
    stop_server
    tar -czf "${OUTPUT_DIR}.tar.gz" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")" || status=1
    printf 'Startup diagnostics: %s\n' "${DIAGNOSTIC_LOG}"
    printf 'Server log: %s\n' "${SERVER_LOG}"
    printf 'Archive: %s.tar.gz\n' "${OUTPUT_DIR}"
    exit "${status}"
}

snapshot() {
    {
        printf '\n===== %s =====\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
        printf 'server_pid=%s port=%s\n' "${SERVER_PID}" "${PORT}"
        ps -o pid,ppid,stat,wchan:24,etime,pcpu,pmem,cmd -p "${SERVER_PID}" || true
        if command -v pstree >/dev/null 2>&1; then
            pstree -ap "${SERVER_PID}" || true
        else
            pgrep -af 'vllm|EngineCore|spawn_main|resource_tracker' || true
        fi
        printf '\nLatest server lines:\n'
        tail -n 20 "${SERVER_LOG}" || true
        if command -v npu-smi >/dev/null 2>&1; then
            printf '\nNPU state:\n'
            timeout 15s npu-smi info || true
        fi
    } | tee -a "${DIAGNOSTIC_LOG}"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${REPO_ROOT}"

if port_is_listening; then
    printf 'Port %s is already in use. Set PORT to an unused port.\n' "${PORT}"
    exit 2
fi
if [[ ! -r "${MODEL}/config.json" ]]; then
    printf 'Model config is not readable: %s/config.json\n' "${MODEL}"
    exit 2
fi

{
    printf 'Model: %s\n' "${MODEL}"
    printf 'Port: %s\n' "${PORT}"
    printf 'TP size: %s\n' "${TP_SIZE}"
    printf 'KV cache dtype: %s\n' "${KV_CACHE_DTYPE}"
    printf 'Visible devices: %s\n' "${ASCEND_RT_VISIBLE_DEVICES-<unset>}"
    printf 'vLLM host IP: %s\n' "${VLLM_HOST_IP}"
    printf 'Gloo interface: %s\n' "${GLOO_SOCKET_IFNAME}"
    git rev-parse HEAD
} | tee "${DIAGNOSTIC_LOG}"

python3 "${SCRIPT_DIR}/check_environment.py" | tee -a "${DIAGNOSTIC_LOG}"
timeout "${GLOO_PROBE_TIMEOUT}s" python3 -c '
import datetime
import socket
import time

import torch.distributed as dist

with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]

started = time.perf_counter()
group_timeout = datetime.timedelta(seconds=20)
dist.init_process_group(
    "gloo",
    init_method=f"tcp://127.0.0.1:{port}",
    rank=0,
    world_size=1,
    timeout=group_timeout,
)
group = dist.new_group([0], backend="gloo", timeout=group_timeout)
dist.barrier(group=group)
dist.destroy_process_group(group)
dist.destroy_process_group()
print(f"Single-rank Gloo probe passed in {time.perf_counter() - started:.3f}s")
' | tee -a "${DIAGNOSTIC_LOG}"
timeout 60s python3 -c '
import sys
from transformers import AutoConfig

config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=True)
print(f"Model config loaded: architecture={config.architectures} model_type={config.model_type}")
' "${MODEL}" | tee -a "${DIAGNOSTIC_LOG}"

printf 'Starting TurboQuant service; full output: %s\n' "${SERVER_LOG}"
env -u ASCEND_LAUNCH_BLOCKING \
    PYTHONUNBUFFERED=1 \
    MODEL="${MODEL}" \
    PORT="${PORT}" \
    TP_SIZE="${TP_SIZE}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
    ENFORCE_EAGER=1 \
    bash "${SCRIPT_DIR}/serve_qwen3_32b.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_TIMEOUT))
while ((SECONDS < deadline)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        printf 'vLLM exited before becoming healthy.\n'
        snapshot
        wait "${SERVER_PID}"
        exit $?
    fi
    if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
        printf 'vLLM is healthy; sending one TurboQuant completion.\n'
        MODEL="${MODEL}" PORT="${PORT}" bash "${SCRIPT_DIR}/smoke_request.sh" | tee "${OUTPUT_DIR}/response.txt"
        exit 0
    fi
    snapshot
    sleep "${POLL_INTERVAL}"
done

printf 'vLLM did not become healthy within %ss.\n' "${SERVER_TIMEOUT}"
snapshot
exit 1
