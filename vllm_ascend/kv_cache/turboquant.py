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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )


def is_turboquant_kv_cache_dtype(kv_cache_dtype: str | None) -> bool:
    return isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")


def get_turboquant_config(cache_dtype: str, head_dim: int) -> TurboQuantConfig:
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    return TurboQuantConfig.from_cache_dtype(cache_dtype, head_dim)


def get_turboquant_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype: str,
) -> tuple[int, ...]:
    config = get_turboquant_config(cache_dtype, head_size)
    return (
        num_blocks,
        block_size,
        num_kv_heads,
        config.slot_size_aligned,
    )


def validate_turboquant_layout(cache_dtype: str, head_dim: int) -> None:
    """Validate constraints shared by the Ascend Triton store/decode kernels."""
    config = get_turboquant_config(cache_dtype, head_dim)
    if config.key_fp8:
        raise NotImplementedError(
            "TurboQuant FP8 keys are not implemented by the Ascend Triton "
            "backend. Use turboquant_4bit_nc, turboquant_k3v4_nc, or "
            "turboquant_3bit_nc."
        )
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise NotImplementedError(
            f"Ascend TurboQuant currently requires a power-of-two attention head dimension, got {head_dim}."
        )
    if head_dim % 32 != 0:
        raise NotImplementedError(
            "Ascend TurboQuant currently requires head_dim to be divisible "
            f"by 32 for aligned 3-bit/4-bit metadata, got {head_dim}."
        )
    if config.key_quant_bits not in (3, 4):
        raise NotImplementedError(f"Unsupported TurboQuant key bit width: {config.key_quant_bits}.")
    if config.value_quant_bits not in (3, 4):
        raise NotImplementedError(f"Unsupported TurboQuant value bit width: {config.value_quant_bits}.")


def validate_turboquant_backend(
    cache_dtype: CacheDType | str | None,
    *,
    head_dim: int,
    use_mla: bool,
    use_sparse: bool,
    use_compress: bool,
    use_v2_runner: bool,
    is_310p_device: bool,
    has_sink: bool,
    use_mm_prefix: bool,
    use_non_causal: bool,
    use_batch_invariant: bool,
) -> None:
    if not is_turboquant_kv_cache_dtype(cache_dtype):
        return
    assert isinstance(cache_dtype, str)

    if use_v2_runner:
        raise NotImplementedError("Ascend TurboQuant currently supports only the vLLM V1 model runner.")
    if is_310p_device:
        raise NotImplementedError("Ascend TurboQuant is not implemented for 310P.")
    if use_mla or use_sparse or use_compress:
        raise NotImplementedError(
            "Ascend TurboQuant supports dense decoder attention only; MLA, "
            "sparse attention, and compressed attention are not supported."
        )
    if has_sink or use_mm_prefix or use_non_causal:
        raise NotImplementedError(
            "Ascend TurboQuant does not currently support attention sinks, "
            "multimodal prefix attention, or non-causal attention."
        )
    if use_batch_invariant:
        raise NotImplementedError("Ascend TurboQuant does not currently support batch-invariant mode.")
    validate_turboquant_layout(cache_dtype, head_dim)
