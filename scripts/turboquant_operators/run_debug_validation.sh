#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Short, single-NPU operator validation for the per-rank Qwen3-0.6B/TP2
# attention shape. It intentionally avoids model serving and high concurrency.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/operators_debug_${TIMESTAMP}}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-17 9}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
STORE_TOKENS="${STORE_TOKENS:-32}"
WARMUP="${WARMUP:-2}"
ITERATIONS="${ITERATIONS:-5}"
RUN_TRACE="${RUN_TRACE:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
CASE_LABEL="${CASE_LABEL:-Qwen3-0.6B/TP2}"
NUM_KV_SPLITS="${NUM_KV_SPLITS:-8}"

# Qwen3-0.6B has Hq=16/Hkv=8. TP2 leaves Hq=8/Hkv=4 per rank.
NUM_QUERY_HEADS="${NUM_QUERY_HEADS:-8}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
HEAD_DIM="${HEAD_DIM:-128}"

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

printf '\n== Real ACLNN launch ==\n'
ASCEND_LAUNCH_BLOCKING=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/operator_runtime_probe.py" \
    --device "${DEVICE}" \
    --activation-dtype "${ACTIVATION_DTYPE}" \
    --head-dim "${HEAD_DIM}" \
    --output "${OUTPUT_DIR}/runtime_probe.json" \
    | tee "${OUTPUT_DIR}/runtime_probe.log"

printf '\n== Short %s correctness ==\n' "${CASE_LABEL}"
# shellcheck disable=SC2086
ASCEND_LAUNCH_BLOCKING=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/operator_accuracy.py" \
    --cache-dtype turboquant_4bit_nc \
    --activation-dtype "${ACTIVATION_DTYPE}" \
    --sequence-lengths ${SEQUENCE_LENGTHS} \
    --num-query-heads "${NUM_QUERY_HEADS}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --num-kv-splits 2 \
    --decode-implementation auto \
    --device "${DEVICE}" \
    --output "${OUTPUT_DIR}/accuracy.json" \
    | tee "${OUTPUT_DIR}/accuracy.log"

printf '\n== Short asynchronous latency benchmark ==\n'
trace_args=(--no-trace --profile-iterations 0)
if [[ "${RUN_TRACE}" == "1" ]]; then
    trace_args=(--profile-iterations 1)
fi
"${PYTHON_BIN}" "${TQ_PROFILE_SCRIPT}" \
    --operation all \
    --native-baseline \
    --native-backend fia \
    --cache-dtype turboquant_4bit_nc \
    --activation-dtype "${ACTIVATION_DTYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --store-tokens "${STORE_TOKENS}" \
    --num-query-heads "${NUM_QUERY_HEADS}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --block-size 128 \
    --num-kv-splits "${NUM_KV_SPLITS}" \
    --adaptive-splits \
    --decode-implementation auto \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --device "${DEVICE}" \
    --trace-dir "${OUTPUT_DIR}/profile" \
    "${trace_args[@]}" \
    | tee "${OUTPUT_DIR}/profile.log"

if [[ "${RUN_UNIT_TESTS}" == "1" ]]; then
    printf '\n== Focused unit tests ==\n'
    "${PYTHON_BIN}" -m pytest tests/ut/test_turboquant_kv_cache.py -q \
        | tee "${OUTPUT_DIR}/pytest_kv_cache.log"
    "${PYTHON_BIN}" -m pytest tests/ut/ops/test_turboquant_triton.py \
        -k 'not aclgraph' -q \
        | tee "${OUTPUT_DIR}/pytest_triton.log"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --result-root "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_DIR}/summary"
