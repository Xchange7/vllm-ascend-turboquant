#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

cd "${REPO_ROOT}"
python3 "${TQ_COMMON_DIR}/check_environment.py"
pytest -sv tests/ut/test_turboquant_kv_cache.py
pytest -sv tests/ut/ops/test_turboquant_triton.py
