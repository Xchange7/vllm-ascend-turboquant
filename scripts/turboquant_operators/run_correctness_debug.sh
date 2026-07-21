#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/operators_correctness_${TIMESTAMP}}"
FAILURES=0

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/run.log") 2>&1

finalize() {
    local status=$?
    trap - EXIT
    archive_output "${OUTPUT_DIR}" || status=1
    exit "${status}"
}
trap finalize EXIT

run_case() {
    local name=$1
    local num_kv_heads=$2
    local num_kv_splits=$3
    local implementation=$4
    local case_dir="${OUTPUT_DIR}/${name}"
    mkdir -p "${case_dir}"
    printf '\n== %s ==\n' "${name}"
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/operator_accuracy.py" \
        --cache-dtype turboquant_4bit_nc \
        --activation-dtype bfloat16 \
        --sequence-lengths 1 17 129 \
        --num-query-heads 16 \
        --num-kv-heads "${num_kv_heads}" \
        --head-dim 128 \
        --num-kv-splits "${num_kv_splits}" \
        --decode-implementation "${implementation}" \
        --device "${DEVICE}" \
        --output "${case_dir}/accuracy.json" \
        | tee "${case_dir}/accuracy.log"; then
        FAILURES=$((FAILURES + 1))
    fi
}

cd "${REPO_ROOT}"
collect_environment "${OUTPUT_DIR}"

# Hkv=2 is the previously passing Qwen3-32B/TP4-style control. Hkv=8 is the
# Qwen3-0.6B shape that exercises adjacent packed heads and reference GQA decode.
run_case control_hkv2_auto_split1 2 1 auto
run_case control_hkv2_auto_split4 2 4 auto
run_case qwen3_0_6b_hkv8_auto_split1 8 1 auto
run_case qwen3_0_6b_hkv8_auto_split4 8 4 auto
run_case qwen3_0_6b_hkv8_reference_split1 8 1 reference

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --result-root "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_DIR}/summary"

if ((FAILURES > 0)); then
    printf '%d correctness case(s) failed. See summary.md and accuracy.json.\n' "${FAILURES}"
    exit 1
fi
