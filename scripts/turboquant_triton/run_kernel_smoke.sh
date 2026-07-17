#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
python3 scripts/turboquant_triton/check_environment.py
pytest -sv tests/ut/test_turboquant_kv_cache.py
pytest -sv tests/ut/ops/test_turboquant_triton.py
