# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from vllm_ascend.kv_cache.higgs import (
    dequantize_higgs_vector,
    quantize_higgs_vector,
)

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType

TURBOQUANT_KV_CACHE_DTYPES = (
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
)
_TQ_PRESETS = {
    "turboquant_k8v4": {
        "key_quant_bits": 8,
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
}


@dataclass
class _FallbackTurboQuantConfig:
    head_dim: int
    key_quant_bits: int
    value_quant_bits: int
    norm_correction: bool = False

    @property
    def key_fp8(self) -> bool:
        return self.key_quant_bits == 8

    @property
    def key_mse_bits(self) -> int:
        return 0 if self.key_fp8 else self.key_quant_bits

    @property
    def key_packed_size(self) -> int:
        if self.key_fp8:
            return self.head_dim
        return math.ceil(self.head_dim * self.key_mse_bits / 8) + 2

    @property
    def value_packed_size(self) -> int:
        return math.ceil(self.head_dim * self.value_quant_bits / 8) + 4

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        return self.slot_size + (self.slot_size % 2)


class TurboQuantConfigLike(Protocol):
    head_dim: int
    key_quant_bits: int
    value_quant_bits: int

    @property
    def key_fp8(self) -> bool: ...

    @property
    def key_packed_size(self) -> int: ...

    @property
    def value_packed_size(self) -> int: ...

    @property
    def slot_size_aligned(self) -> int: ...


def is_turboquant_kv_cache_dtype(kv_cache_dtype: str | None) -> bool:
    return isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")


def get_turboquant_config(cache_dtype: str, head_dim: int) -> TurboQuantConfigLike:
    try:
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        return TurboQuantConfig.from_cache_dtype(cache_dtype, head_dim)
    except ModuleNotFoundError:
        if cache_dtype not in _TQ_PRESETS:
            valid = ", ".join(TURBOQUANT_KV_CACHE_DTYPES)
            raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype!r}. Valid presets: {valid}")
        preset = _TQ_PRESETS[cache_dtype]
        return _FallbackTurboQuantConfig(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            norm_correction=preset["norm_correction"],
        )


def get_turboquant_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str,
) -> tuple[int, ...]:
    tq_config = get_turboquant_config(cache_dtype_str, head_size)
    return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)


def validate_turboquant_config(
    cache_dtype: CacheDType | str | None,
    *,
    use_mla: bool,
    use_sparse: bool,
    use_compress: bool,
    is_310p: bool,
) -> str | None:
    """Return an unsupported reason, or None when the config can be routed."""
    if not is_turboquant_kv_cache_dtype(cache_dtype):
        return None
    if is_310p:
        return "TurboQuant KV cache is not implemented for Ascend 310P."
    if use_mla or use_sparse or use_compress:
        return (
            "TurboQuant KV cache currently supports dense decoder attention "
            "only; MLA, sparse attention, and compression backends are not "
            "supported."
        )
    return None


def _pack_bits(q_values: torch.Tensor, bits: int, packed_bytes: int) -> torch.Tensor:
    q_values = q_values.to(torch.uint8)
    last_dim = q_values.shape[-1]
    if bits == 4:
        if last_dim % 2 != 0:
            q_values = torch.nn.functional.pad(q_values, (0, 1))
        low = q_values[..., 0::2]
        high = q_values[..., 1::2]
        packed = low | (high << 4)
        return packed[..., :packed_bytes].contiguous()

    packed = torch.zeros(
        (*q_values.shape[:-1], packed_bytes),
        dtype=torch.uint8,
        device=q_values.device,
    )
    q_int = q_values.to(torch.int16)
    for coord in range(last_dim):
        bit_offset = coord * bits
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        current = q_int[..., coord]
        packed[..., byte_offset] |= ((current << shift) & 0xFF).to(torch.uint8)
        if shift + bits > 8 and byte_offset + 1 < packed_bytes:
            packed[..., byte_offset + 1] |= (current >> (8 - shift)).to(torch.uint8)
    return packed


def _unpack_bits(
    packed: torch.Tensor,
    bits: int,
    head_dim: int,
) -> torch.Tensor:
    if bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        values = torch.stack((low, high), dim=-1).flatten(-2)
        return values[..., :head_dim].contiguous()

    q_values = torch.empty(
        (*packed.shape[:-1], head_dim),
        dtype=torch.uint8,
        device=packed.device,
    )
    mask = (1 << bits) - 1
    packed_int = packed.to(torch.int16)
    for coord in range(head_dim):
        bit_offset = coord * bits
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        value = (packed_int[..., byte_offset] >> shift) & mask
        if shift + bits > 8 and byte_offset + 1 < packed.shape[-1]:
            value = value | (packed_int[..., byte_offset + 1] << (8 - shift))
        q_values[..., coord] = (value & mask).to(torch.uint8)
    return q_values


def _fp16_to_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float16).contiguous().view(torch.uint8)


def _bytes_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.float16)


def _fp8_key_to_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("TurboQuant k8v4 requires torch.float8_e4m3fn support.")
    return tensor.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)


def _bytes_to_fp8_key(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("TurboQuant k8v4 requires torch.float8_e4m3fn support.")
    return tensor.contiguous().view(torch.float8_e4m3fn).to(dtype)


def _quantize_uniform(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels = (1 << bits) - 1
    minimum = tensor.amin(dim=-1, keepdim=True)
    maximum = tensor.amax(dim=-1, keepdim=True)
    scale = ((maximum - minimum) / levels).clamp_min(torch.finfo(torch.float16).tiny)
    quantized = torch.round((tensor - minimum) / scale).clamp(0, levels)
    return quantized.to(torch.uint8), scale.to(torch.float16), minimum.to(torch.float16)


def _dequantize_uniform(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return quantized.to(torch.float32).mul(scale.to(torch.float32)).add(zero.to(torch.float32)).to(dtype)


def turboquant_store_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    tq_config: TurboQuantConfigLike,
) -> None:
    """Pack and store one token batch into the combined TurboQuant cache.

    This is a portable reference implementation. It follows the upstream TQ
    slot layout but uses plain Torch operations so the Python contract is
    available on Ascend before a fused CANN/custom op lands.
    """
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    key = key[:num_tokens].view(num_tokens, -1, tq_config.head_dim)
    value = value[:num_tokens].view(num_tokens, -1, tq_config.head_dim)
    slot_mapping = slot_mapping[:num_tokens].to(torch.long)
    valid_indices = torch.nonzero(slot_mapping >= 0, as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return

    key = key.index_select(0, valid_indices)
    value = value.index_select(0, valid_indices)
    slot_mapping = slot_mapping.index_select(0, valid_indices)

    key_data_bytes = tq_config.key_packed_size - (0 if tq_config.key_fp8 else 2)
    value_data_bytes = tq_config.value_packed_size - 4

    if tq_config.key_fp8:
        key_packed = _fp8_key_to_bytes(key)
    else:
        key_quant, key_norm = quantize_higgs_vector(
            key,
            tq_config.key_quant_bits,
        )
        key_packed = _pack_bits(key_quant, tq_config.key_quant_bits, key_data_bytes)
        key_packed = torch.cat((key_packed, _fp16_to_bytes(key_norm)), dim=-1)

    value_quant, value_scale, value_zero = _quantize_uniform(
        value.to(torch.float32),
        tq_config.value_quant_bits,
    )
    value_packed = torch.cat(
        (
            _pack_bits(value_quant, tq_config.value_quant_bits, value_data_bytes),
            _fp16_to_bytes(value_scale),
            _fp16_to_bytes(value_zero),
        ),
        dim=-1,
    )

    block_size = kv_cache.shape[1]
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size

    kv_cache[block_indices, block_offsets, :, : tq_config.key_packed_size] = key_packed
    value_start = tq_config.key_packed_size
    value_end = value_start + tq_config.value_packed_size
    kv_cache[block_indices, block_offsets, :, value_start:value_end] = value_packed


def turboquant_dequant_cache(
    kv_cache: torch.Tensor,
    tq_config: TurboQuantConfigLike,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the packed cache to standard K/V cache tensors."""
    head_dim = tq_config.head_dim
    key_data_bytes = tq_config.key_packed_size - (0 if tq_config.key_fp8 else 2)
    value_data_bytes = tq_config.value_packed_size - 4

    key_slot = kv_cache[..., : tq_config.key_packed_size]
    value_start = tq_config.key_packed_size
    value_slot = kv_cache[..., value_start : value_start + tq_config.value_packed_size]

    if tq_config.key_fp8:
        key_cache = _bytes_to_fp8_key(key_slot[..., :head_dim], dtype)
    else:
        key_quant = _unpack_bits(
            key_slot[..., :key_data_bytes],
            tq_config.key_quant_bits,
            head_dim,
        )
        key_norm = _bytes_to_fp16(key_slot[..., key_data_bytes : key_data_bytes + 2]).view(*key_quant.shape[:-1], 1)
        key_cache = dequantize_higgs_vector(
            key_quant,
            key_norm,
            tq_config.key_quant_bits,
            getattr(tq_config, "norm_correction", False),
            dtype,
        )

    value_quant = _unpack_bits(
        value_slot[..., :value_data_bytes],
        tq_config.value_quant_bits,
        head_dim,
    )
    value_scale = _bytes_to_fp16(value_slot[..., value_data_bytes : value_data_bytes + 2]).view(
        *value_quant.shape[:-1], 1
    )
    value_zero = _bytes_to_fp16(value_slot[..., value_data_bytes + 2 : value_data_bytes + 4]).view(
        *value_quant.shape[:-1], 1
    )
    value_cache = _dequantize_uniform(value_quant, value_scale, value_zero, dtype)
    return key_cache.contiguous(), value_cache.contiguous()
