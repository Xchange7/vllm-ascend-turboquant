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
# This file is a part of the vllm-ascend project.

import re
from pathlib import Path


def _tiling_key(
    key_bits: int,
    value_bits: int,
    norm_correction: bool,
    head_dim: int,
) -> int:
    return key_bits * 100_000 + value_bits * 10_000 + int(norm_correction) * 1_000 + head_dim


def test_turboquant_kernel_declares_every_tiling_key_as_a_literal():
    repo_root = Path(__file__).parents[3]
    kernel_path = repo_root / "csrc/attention/turbo_quant_paged_dequant/op_kernel" / "turbo_quant_paged_dequant.cpp"
    kernel_source = kernel_path.read_text(encoding="utf-8")
    declared_keys = {
        int(match)
        for match in re.findall(
            r"\bTILING_KEY_IS\((\d+)\)",
            kernel_source,
        )
    }
    expected_keys = {
        _tiling_key(key_bits, value_bits, norm_correction, head_dim)
        for key_bits in (3, 4)
        for value_bits in (3, 4)
        for norm_correction in (False, True)
        for head_dim in (32, 64, 128, 256)
    }

    assert declared_keys == expected_keys
    assert 441128 in declared_keys
