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

from __future__ import annotations

import torch


def has_turboquant_paged_dequant() -> bool:
    """Return whether the installed extension contains the AscendC kernel."""
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        return False
    return hasattr(torch.ops._C_ascend, "npu_turboquant_paged_dequant") and hasattr(
        torch.ops._C_ascend, "npu_turboquant_paged_dequant_out"
    )


def turboquant_paged_dequant(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    centroids: torch.Tensor,
    *,
    max_seq_len: int,
    key_bits: int,
    key_packed_size: int,
    value_bits: int,
    norm_correction: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather and dequantize paged TurboQuant cache into dense BNSD K/V."""
    if not has_turboquant_paged_dequant():
        raise RuntimeError(
            "The Ascend TurboQuant fused operator is not installed. Rebuild "
            "vllm-ascend with `pip install -e .` after sourcing the CANN environment."
        )
    return torch.ops._C_ascend.npu_turboquant_paged_dequant(
        query,
        kv_cache,
        block_table,
        seq_lens,
        page_table,
        centroids,
        max_seq_len,
        key_bits,
        key_packed_size,
        value_bits,
        norm_correction,
    )


def turboquant_paged_dequant_out(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    centroids: torch.Tensor,
    key_out: torch.Tensor,
    value_out: torch.Tensor,
    *,
    max_seq_len: int,
    key_bits: int,
    key_packed_size: int,
    value_bits: int,
    norm_correction: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize paged TurboQuant cache into caller-owned dense buffers."""
    if not has_turboquant_paged_dequant():
        raise RuntimeError(
            "The Ascend TurboQuant fused operator is not installed. Rebuild "
            "vllm-ascend with `pip install -e .` after sourcing the CANN environment."
        )
    return torch.ops._C_ascend.npu_turboquant_paged_dequant_out(
        query,
        kv_cache,
        block_table,
        seq_lens,
        page_table,
        centroids,
        max_seq_len,
        key_bits,
        key_packed_size,
        value_bits,
        norm_correction,
        key_out,
        value_out,
    )
