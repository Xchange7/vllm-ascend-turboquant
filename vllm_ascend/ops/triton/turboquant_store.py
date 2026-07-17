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

"""Triton-Ascend kernels for writing the packed TurboQuant KV cache."""

import math

import torch
from vllm.triton_utils import tl, triton

ASCEND_UB_ALIGNMENT_BYTES = 32


@triton.jit
def _quantize_key_coordinates(
    rotated_key_ptr,
    midpoint_ptr,
    vector_base,
    coordinate_offsets,
    coordinate_mask,
    KEY_BITS: tl.constexpr,
    NUM_CENTROIDS: tl.constexpr,
    PACK_SIZE: tl.constexpr,
):
    rotated_key = tl.load(
        rotated_key_ptr + vector_base + coordinate_offsets,
        mask=coordinate_mask,
        other=0.0,
    )
    low = tl.zeros([PACK_SIZE], dtype=tl.int32)
    high = tl.full([PACK_SIZE], NUM_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(KEY_BITS):
        middle = (low + high) >> 1
        safe_middle = tl.minimum(middle, NUM_CENTROIDS - 2)
        midpoint = tl.load(
            midpoint_ptr + safe_middle,
            mask=coordinate_mask,
            other=0.0,
        )
        move_right = rotated_key >= midpoint
        low = tl.where(move_right, middle + 1, low)
        high = tl.where(move_right, high, middle)
    return tl.minimum(low, NUM_CENTROIDS - 1)


@triton.jit
def _quantize_value_coordinates(
    value_ptr,
    value_base,
    coordinate_offsets,
    coordinate_mask,
    value_min,
    scale,
    levels,
):
    value = tl.load(
        value_ptr + value_base + coordinate_offsets,
        mask=coordinate_mask,
        other=0.0,
    ).to(tl.float32)
    return tl.minimum(
        tl.maximum(
            ((value - value_min) / scale + 0.5).to(tl.int32),
            0,
        ),
        levels,
    )


@triton.jit
def _store_quantized_value(
    value_ptr,
    cache_u8_ptr,
    cache_f16_ptr,
    value_base,
    slot_base,
    d_offsets,
    d_mask,
    D: tl.constexpr,
    KPS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    value = tl.load(
        value_ptr + value_base + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    value_min = tl.min(
        tl.where(d_mask, value, float("inf")),
        axis=0,
    )
    value_max = tl.max(
        tl.where(d_mask, value, -float("inf")),
        axis=0,
    )
    levels = (1 << VALUE_BITS) - 1
    scale = tl.maximum((value_max - value_min) / levels, 1e-8)

    value_slot_base = slot_base + KPS
    if VALUE_BITS == 4:
        byte_offsets = tl.arange(0, BLOCK_D // 2)
        even_offsets = byte_offsets * 2
        odd_offsets = even_offsets + 1
        even = _quantize_value_coordinates(
            value_ptr,
            value_base,
            even_offsets,
            even_offsets < D,
            value_min,
            scale,
            levels,
        )
        odd = _quantize_value_coordinates(
            value_ptr,
            value_base,
            odd_offsets,
            odd_offsets < D,
            value_min,
            scale,
            levels,
        )
        packed = ((even & 0xF) | ((odd & 0xF) << 4)).to(tl.uint8)
        tl.store(
            cache_u8_ptr + value_slot_base + byte_offsets,
            packed,
            mask=byte_offsets < VALUE_DATA_BYTES,
        )
    else:
        group_offsets = tl.arange(0, BLOCK_GROUPS)
        group_mask = group_offsets < (D // 8)
        packed_24 = tl.zeros([BLOCK_GROUPS], dtype=tl.int32)
        for coordinate in range(8):
            coordinate_offsets = group_offsets * 8 + coordinate
            quantized = _quantize_value_coordinates(
                value_ptr,
                value_base,
                coordinate_offsets,
                group_mask,
                value_min,
                scale,
                levels,
            )
            packed_24 = packed_24 | ((quantized & 0x7) << (coordinate * 3))
        tl.store(
            cache_u8_ptr + value_slot_base + group_offsets * 3,
            (packed_24 & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            cache_u8_ptr + value_slot_base + group_offsets * 3 + 1,
            ((packed_24 >> 8) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            cache_u8_ptr + value_slot_base + group_offsets * 3 + 2,
            ((packed_24 >> 16) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )

    metadata_byte_offset = value_slot_base + VALUE_DATA_BYTES
    tl.store(
        cache_f16_ptr + metadata_byte_offset // 2,
        scale.to(tl.float16),
    )
    tl.store(
        cache_f16_ptr + metadata_byte_offset // 2 + 1,
        value_min.to(tl.float16),
    )


@triton.jit
def _turboquant_store_kernel(
    rotated_key_ptr,
    key_norm_ptr,
    value_ptr,
    midpoint_ptr,
    cache_u8_ptr,
    cache_f16_ptr,
    slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_position: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    KEY_BITS: tl.constexpr,
    NUM_CENTROIDS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    program_id = tl.program_id(0)
    token_index = program_id // NUM_KV_HEADS
    head_index = program_id % NUM_KV_HEADS
    slot = tl.load(slot_mapping_ptr + token_index)

    # Negative slots are graph/scheduler padding and must not write cache.
    if slot < 0:
        return

    block_index = tl.cast(slot // CACHE_BLOCK_SIZE, tl.int64)
    block_offset = tl.cast(slot % CACHE_BLOCK_SIZE, tl.int64)
    slot_base = (
        block_index * stride_cache_block
        + block_offset * stride_cache_position
        + tl.cast(head_index, tl.int64) * stride_cache_head
    )

    vector_base = program_id * D
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    if KEY_BITS == 4:
        byte_offsets = tl.arange(0, BLOCK_D // 2)
        even_offsets = byte_offsets * 2
        odd_offsets = even_offsets + 1
        even = _quantize_key_coordinates(
            rotated_key_ptr,
            midpoint_ptr,
            vector_base,
            even_offsets,
            even_offsets < D,
            KEY_BITS=KEY_BITS,
            NUM_CENTROIDS=NUM_CENTROIDS,
            PACK_SIZE=BLOCK_D // 2,
        )
        odd = _quantize_key_coordinates(
            rotated_key_ptr,
            midpoint_ptr,
            vector_base,
            odd_offsets,
            odd_offsets < D,
            KEY_BITS=KEY_BITS,
            NUM_CENTROIDS=NUM_CENTROIDS,
            PACK_SIZE=BLOCK_D // 2,
        )
        packed = ((even & 0xF) | ((odd & 0xF) << 4)).to(tl.uint8)
        tl.store(
            cache_u8_ptr + slot_base + byte_offsets,
            packed,
            mask=byte_offsets < MSE_BYTES,
        )
    else:
        group_offsets = tl.arange(0, BLOCK_GROUPS)
        group_mask = group_offsets < (D // 8)
        packed_24 = tl.zeros([BLOCK_GROUPS], dtype=tl.int32)
        for coordinate in range(8):
            coordinate_offsets = group_offsets * 8 + coordinate
            centroid_index = _quantize_key_coordinates(
                rotated_key_ptr,
                midpoint_ptr,
                vector_base,
                coordinate_offsets,
                group_mask,
                KEY_BITS=KEY_BITS,
                NUM_CENTROIDS=NUM_CENTROIDS,
                PACK_SIZE=BLOCK_GROUPS,
            )
            packed_24 = packed_24 | ((centroid_index & 0x7) << (coordinate * 3))
        tl.store(
            cache_u8_ptr + slot_base + group_offsets * 3,
            (packed_24 & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            cache_u8_ptr + slot_base + group_offsets * 3 + 1,
            ((packed_24 >> 8) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            cache_u8_ptr + slot_base + group_offsets * 3 + 2,
            ((packed_24 >> 16) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )

    tl.store(
        cache_f16_ptr + (slot_base + MSE_BYTES) // 2,
        tl.load(key_norm_ptr + program_id).to(tl.float16),
    )

    _store_quantized_value(
        value_ptr,
        cache_u8_ptr,
        cache_f16_ptr,
        vector_base,
        slot_base,
        d_offsets,
        d_mask,
        D=D,
        KPS=KPS,
        VALUE_BITS=VALUE_BITS,
        VALUE_DATA_BYTES=VALUE_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )


def triton_turboquant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    hadamard_transpose: torch.Tensor,
    midpoints: torch.Tensor,
    *,
    key_bits: int,
    key_packed_size: int,
    value_bits: int,
) -> None:
    """Quantize post-RoPE K/V and scatter packed bytes into paged cache."""
    if key_bits not in (3, 4) or value_bits not in (3, 4):
        raise NotImplementedError("The Ascend Triton TurboQuant store supports 3-bit and 4-bit K/V.")
    if key.shape != value.shape:
        raise ValueError(f"TurboQuant key/value shapes must match, got {key.shape} and {value.shape}.")
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"TurboQuant cache must use torch.uint8, got {kv_cache.dtype}.")

    num_tokens, num_kv_heads, head_dim = key.shape
    if num_tokens == 0:
        return
    if head_dim % ASCEND_UB_ALIGNMENT_BYTES != 0:
        raise ValueError(f"TurboQuant Triton packing requires head_dim % 32 == 0, got {head_dim}.")

    num_vectors = num_tokens * num_kv_heads
    key_float = key.float().reshape(num_vectors, head_dim)
    key_norm = torch.linalg.vector_norm(
        key_float,
        dim=1,
        keepdim=True,
    )
    normalized_key = key_float / (key_norm + 1e-8)
    rotated_key = (normalized_key @ hadamard_transpose).contiguous()
    value_contiguous = value.reshape(num_vectors, head_dim).contiguous()

    block_d = triton.next_power_of_2(head_dim)
    # Ascend vector operations and UB transfers require at least one aligned
    # 32-byte data block. Pad 3-bit packing lanes and mask inactive groups;
    # the external packed slot layout remains unchanged.
    block_groups = max(ASCEND_UB_ALIGNMENT_BYTES, block_d // 8)
    mse_bytes = math.ceil(head_dim * key_bits / 8)
    value_data_bytes = math.ceil(head_dim * value_bits / 8)
    metadata_offsets = (
        mse_bytes,
        key_packed_size + value_data_bytes,
    )
    if any(offset % 2 for offset in metadata_offsets):
        raise ValueError(f"TurboQuant FP16 metadata must start at an even byte offset; got offsets {metadata_offsets}.")

    grid = (num_vectors,)
    _turboquant_store_kernel[grid](
        rotated_key,
        key_norm.squeeze(1),
        value_contiguous,
        midpoints,
        kv_cache,
        kv_cache.view(torch.float16),
        slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_position=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=head_dim,
        NUM_KV_HEADS=num_kv_heads,
        CACHE_BLOCK_SIZE=kv_cache.shape[1],
        BLOCK_D=block_d,
        MSE_BYTES=mse_bytes,
        KPS=key_packed_size,
        KEY_BITS=key_bits,
        NUM_CENTROIDS=2**key_bits,
        VALUE_BITS=value_bits,
        VALUE_DATA_BYTES=value_data_bytes,
        BLOCK_GROUPS=block_groups,
        num_warps=4,
        num_stages=1,
    )
