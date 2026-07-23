#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# B1 long-context operator diagnostic for Qwen3-32B. The benchmark is
# single-NPU but uses the exact per-rank attention shape for TP2 or TP4.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TP_SIZE="${TP_SIZE:-2}"

case "${TP_SIZE}" in
    2)
        NUM_QUERY_HEADS="${NUM_QUERY_HEADS:-32}"
        NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
        ;;
    4)
        NUM_QUERY_HEADS="${NUM_QUERY_HEADS:-16}"
        NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
        ;;
    *)
        printf 'Qwen3-32B decode debug supports TP_SIZE=2 or 4, got %s.\n' "${TP_SIZE}" >&2
        exit 2
        ;;
esac

export NUM_QUERY_HEADS NUM_KV_HEADS
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16384}"
export STORE_TOKENS="${STORE_TOKENS:-16}"
export SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-17 9}"
export WARMUP="${WARMUP:-1}"
export ITERATIONS="${ITERATIONS:-2}"
export RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-0}"
export RUN_TRACE="${RUN_TRACE:-1}"
export CASE_LABEL="Qwen3-32B/TP${TP_SIZE}"
export NUM_KV_SPLITS="${NUM_KV_SPLITS:-32}"
export VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION="${VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION:-ascend_fused}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../../logs/turboquant/qwen32_tp${TP_SIZE}_decode_debug_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_debug_validation.sh"
