#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
python3 scripts/turboquant_triton/check_environment.py
pytest -sv \
    tests/ut/test_turboquant_kv_cache.py::test_turboquant_graph_capture_metadata_uses_device_sequence_lengths \
    tests/ut/test_turboquant_kv_cache.py::test_turboquant_uniform_multi_token_decode_reuses_request_workspace \
    tests/ut/test_turboquant_kv_cache.py::test_turboquant_uniform_decode_clamps_padded_request_lengths \
    tests/ut/test_turboquant_kv_cache.py::test_turboquant_nonuniform_decode_uses_per_request_causal_lengths \
    tests/ut/ops/test_turboquant_triton.py::test_turboquant_store_and_decode_aclgraph_replay_matches_eager
