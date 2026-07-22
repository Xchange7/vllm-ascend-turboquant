#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL="${MODEL:-/run/test_llm/Qwen3-32B}"
FULL_RUNNER="${TQ_SCRIPT_ROOT}/diagnostics/run_full_910b4_validation.sh"

if (($# > 0)); then
    printf 'This script accepts configuration through environment variables, not positional arguments.\n' >&2
    exit 2
fi

if [[ ! -f "${MODEL}/config.json" ]]; then
    printf 'Model config is not readable: %s/config.json\n' "${MODEL}" >&2
    printf 'Set MODEL to a local Qwen3 model directory.\n' >&2
    exit 2
fi

exec env \
    MODEL="${MODEL}" \
    SOC_VERSION="${SOC_VERSION}" \
    bash "${FULL_RUNNER}"
