#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

cd "${REPO_ROOT}"
python3 "${TQ_COMMON_DIR}/check_environment.py"
exec python3 "${TQ_PERFORMANCE_DIR}/profile_kernels.py" "$@"
