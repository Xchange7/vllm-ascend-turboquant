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

"""Triton-Ascend attention that directly consumes packed TurboQuant pages."""

import math
from typing import Any

import torch
from vllm.triton_utils import tl, triton

_GROUPED_GQA_MIN_GROUP_SIZE = 4
_GROUPED_GQA_MAX_GROUP_SIZE = 32
_GROUPED_GQA_MAX_HEAD_DIM = 128
_GROUPED_GQA_BLOCK_KV = 16
_GROUPED_GQA_BLOCK_KV_OPTIONS = (16, 32)
_ASCEND_MAX_TRITON_GRID_SIZE = 65535

# Keep enough stage-1 programs to occupy the NPU without multiplying tiny
# split-KV programs at high concurrency. This mirrors upstream Triton MLA's
# sequence-length heuristic, with an additional cap for already-parallel
# batch/head rows. The constants are intentionally hardware-policy details,
# not user-facing knobs; profiling should validate them before they change.
_MIN_KV_TOKENS_PER_SPLIT = 512
_TARGET_DECODE_PROGRAMS = 128


@triton.jit
def _tanh(value):
    return 2 * tl.sigmoid(2 * value) - 1


@triton.jit
def _turboquant_decode_stage1(
    query_rotated_ptr,
    cache_u8_ptr,
    cache_f16_ptr,
    block_table_ptr,
    seq_lens_ptr,
    centroids_ptr,
    alibi_slopes_ptr,
    partial_output_ptr,
    stride_query_batch,
    stride_query_head,
    stride_cache_block,
    stride_cache_position,
    stride_cache_head,
    stride_block_table_batch,
    stride_partial_batch,
    stride_partial_head,
    stride_partial_split,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    ATTENTION_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    LOGITS_SOFT_CAP: tl.constexpr,
    SEQUENCE_LENGTH_DELTA: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    split_index = tl.program_id(2)
    kv_head = query_head // KV_GROUP_SIZE

    seq_len = tl.maximum(
        tl.load(seq_lens_ptr + batch_index) + SEQUENCE_LENGTH_DELTA,
        0,
    )
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * split_index
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    query_base = batch_index * stride_query_batch + query_head * stride_query_head
    query_rotated = tl.load(
        query_rotated_ptr + query_base + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    key_bit_offsets = d_offsets * KEY_BITS
    key_byte_indices = key_bit_offsets // 8
    key_bit_shifts = key_bit_offsets % 8
    key_mask = (1 << KEY_BITS) - 1

    if VALUE_BITS == 3:
        value_bit_offsets = d_offsets * 3
        value_byte_indices = value_bit_offsets // 8
        value_bit_shifts = value_bit_offsets % 8

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
    block_table_base = batch_index * stride_block_table_batch

    for tile_start in range(split_start, split_end, BLOCK_KV):
        positions = tile_start + kv_range
        position_mask = positions < split_end
        logical_pages = positions // CACHE_BLOCK_SIZE
        page_offsets = positions % CACHE_BLOCK_SIZE
        physical_blocks = tl.cast(
            tl.load(
                block_table_ptr + block_table_base + logical_pages,
                mask=position_mask,
                other=0,
            ),
            tl.int64,
        )
        slot_bases = (
            physical_blocks * stride_cache_block
            + tl.cast(page_offsets, tl.int64) * stride_cache_position
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        key_addresses = slot_bases[:, None] + key_byte_indices[None, :]
        key_byte_0 = tl.load(
            cache_u8_ptr + key_addresses,
            mask=position_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        key_byte_1 = tl.load(
            cache_u8_ptr + key_addresses + 1,
            mask=position_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        key_indices = ((key_byte_0 | (key_byte_1 << 8)) >> key_bit_shifts[None, :]) & key_mask
        centroid_values = tl.load(
            centroids_ptr + key_indices,
            mask=position_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        if NORM_CORRECTION:
            centroid_norm_sq = tl.sum(
                tl.where(
                    d_mask[None, :],
                    centroid_values * centroid_values,
                    0.0,
                ),
                axis=1,
            )
            centroid_values *= (1.0 / tl.sqrt(centroid_norm_sq + 1e-16))[:, None]

        key_dot = tl.sum(
            tl.where(
                d_mask[None, :],
                query_rotated[None, :] * centroid_values,
                0.0,
            ),
            axis=1,
        )
        key_norm = tl.load(
            cache_f16_ptr + (slot_bases + MSE_BYTES) // 2,
            mask=position_mask,
            other=0.0,
        ).to(tl.float32)
        scores = key_norm * key_dot * ATTENTION_SCALE
        if LOGITS_SOFT_CAP > 0:
            scores = LOGITS_SOFT_CAP * _tanh(scores / LOGITS_SOFT_CAP)
        if HAS_ALIBI:
            alibi_slope = tl.load(alibi_slopes_ptr + query_head)
            scores += alibi_slope * (positions - seq_len + 1)
        scores = tl.where(position_mask, scores, -float("inf"))

        new_max = tl.maximum(tl.max(scores, axis=0), running_max)
        previous_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)

        value_bases = slot_bases + KPS
        if VALUE_BITS == 3:
            value_addresses = value_bases[:, None] + value_byte_indices[None, :]
            value_byte_0 = tl.load(
                cache_u8_ptr + value_addresses,
                mask=position_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            value_byte_1 = tl.load(
                cache_u8_ptr + value_addresses + 1,
                mask=position_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            value_indices = ((value_byte_0 | (value_byte_1 << 8)) >> value_bit_shifts[None, :]) & 0x7
        else:
            value_byte_indices_4 = d_offsets // 2
            value_bit_shifts_4 = (d_offsets % 2) * 4
            value_byte = tl.load(
                cache_u8_ptr + value_bases[:, None] + value_byte_indices_4[None, :],
                mask=position_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            value_indices = (value_byte >> value_bit_shifts_4[None, :]) & 0xF

        value_metadata_base = value_bases + VALUE_DATA_BYTES
        value_scale = tl.load(
            cache_f16_ptr + value_metadata_base // 2,
            mask=position_mask,
            other=0.0,
        ).to(tl.float32)
        value_minimum = tl.load(
            cache_f16_ptr + value_metadata_base // 2 + 1,
            mask=position_mask,
            other=0.0,
        ).to(tl.float32)
        values = value_indices.to(tl.float32) * value_scale[:, None] + value_minimum[:, None]

        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=0)
        running_max = new_max

    partial_base = (
        batch_index * stride_partial_batch + query_head * stride_partial_head + split_index * stride_partial_split
    )
    safe_sum = tl.maximum(running_sum, 1e-20)
    tl.store(
        partial_output_ptr + partial_base + d_offsets,
        accumulator / safe_sum,
        mask=d_mask,
    )
    tl.store(
        partial_output_ptr + partial_base + HEAD_DIM,
        running_max + tl.log(safe_sum),
    )


@triton.jit
def _turboquant_grouped_gqa_stage1(
    query_rotated_ptr,
    cache_u8_ptr,
    cache_f16_ptr,
    block_table_ptr,
    seq_lens_ptr,
    centroids_ptr,
    alibi_slopes_ptr,
    partial_output_ptr,
    stride_query_batch,
    stride_query_head,
    stride_cache_block,
    stride_cache_position,
    stride_cache_head,
    stride_block_table_batch,
    stride_partial_batch,
    stride_partial_head,
    stride_partial_split,
    NUM_QUERY_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    ATTENTION_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    LOGITS_SOFT_CAP: tl.constexpr,
    SEQUENCE_LENGTH_DELTA: tl.constexpr,
):
    """Decode one GQA group while unpacking each compressed KV tile once."""
    batch_index = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_index = tl.program_id(2)

    seq_len = tl.maximum(
        tl.load(seq_lens_ptr + batch_index) + SEQUENCE_LENGTH_DELTA,
        0,
    )
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * split_index
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    query_offsets = tl.arange(0, BLOCK_Q)
    dimension_offsets = tl.arange(0, BLOCK_D)
    kv_offsets = tl.arange(0, BLOCK_KV)
    query_heads = kv_head * KV_GROUP_SIZE + query_offsets
    query_mask = (query_offsets < KV_GROUP_SIZE) & (query_heads < NUM_QUERY_HEADS)
    dimension_mask = dimension_offsets < HEAD_DIM
    query_addresses = (
        batch_index * stride_query_batch + query_heads[:, None] * stride_query_head + dimension_offsets[None, :]
    )
    query_rotated = tl.load(
        query_rotated_ptr + query_addresses,
        mask=query_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )

    key_bit_offsets = dimension_offsets * KEY_BITS
    key_byte_indices = key_bit_offsets // 8
    key_bit_shifts = key_bit_offsets % 8
    key_mask = (1 << KEY_BITS) - 1
    if VALUE_BITS == 3:
        value_bit_offsets = dimension_offsets * 3
        value_byte_indices = value_bit_offsets // 8
        value_bit_shifts = value_bit_offsets % 8

    running_max = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    running_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    block_table_base = batch_index * stride_block_table_batch

    for tile_start in range(split_start, split_end, BLOCK_KV):
        positions = tile_start + kv_offsets
        position_mask = positions < split_end
        logical_pages = positions // CACHE_BLOCK_SIZE
        page_offsets = positions % CACHE_BLOCK_SIZE
        physical_blocks = tl.cast(
            tl.load(
                block_table_ptr + block_table_base + logical_pages,
                mask=position_mask,
                other=0,
            ),
            tl.int64,
        )
        slot_bases = (
            physical_blocks * stride_cache_block
            + tl.cast(page_offsets, tl.int64) * stride_cache_position
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        key_addresses = slot_bases[:, None] + key_byte_indices[None, :]
        key_byte_0 = tl.load(
            cache_u8_ptr + key_addresses,
            mask=position_mask[:, None] & dimension_mask[None, :],
            other=0,
        ).to(tl.int32)
        key_byte_1 = tl.load(
            cache_u8_ptr + key_addresses + 1,
            mask=position_mask[:, None] & dimension_mask[None, :],
            other=0,
        ).to(tl.int32)
        key_indices = ((key_byte_0 | (key_byte_1 << 8)) >> key_bit_shifts[None, :]) & key_mask
        centroid_values = tl.load(
            centroids_ptr + key_indices,
            mask=position_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if NORM_CORRECTION:
            centroid_norm_sq = tl.sum(
                tl.where(
                    dimension_mask[None, :],
                    centroid_values * centroid_values,
                    0.0,
                ),
                axis=1,
            )
            centroid_values *= (1.0 / tl.sqrt(centroid_norm_sq + 1e-16))[:, None]

        key_dot = tl.dot(
            query_rotated,
            tl.trans(centroid_values.to(query_rotated_ptr.dtype.element_ty)),
        )
        key_norm = tl.load(
            cache_f16_ptr + (slot_bases + MSE_BYTES) // 2,
            mask=position_mask,
            other=0.0,
        ).to(tl.float32)
        scores = key_dot * key_norm[None, :] * ATTENTION_SCALE
        if LOGITS_SOFT_CAP > 0:
            scores = LOGITS_SOFT_CAP * _tanh(scores / LOGITS_SOFT_CAP)
        if HAS_ALIBI:
            alibi_slopes = tl.load(
                alibi_slopes_ptr + query_heads,
                mask=query_mask,
                other=0.0,
            )
            scores += alibi_slopes[:, None] * (positions[None, :] - seq_len + 1)
        scores = tl.where(
            position_mask[None, :],
            scores,
            -float("inf"),
        )
        scores = tl.where(query_mask[:, None], scores, 0.0)

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(tile_max, running_max)
        previous_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(
            query_mask[:, None] & position_mask[None, :],
            probabilities,
            0.0,
        )

        value_bases = slot_bases + KPS
        if VALUE_BITS == 3:
            value_addresses = value_bases[:, None] + value_byte_indices[None, :]
            value_byte_0 = tl.load(
                cache_u8_ptr + value_addresses,
                mask=position_mask[:, None] & dimension_mask[None, :],
                other=0,
            ).to(tl.int32)
            value_byte_1 = tl.load(
                cache_u8_ptr + value_addresses + 1,
                mask=position_mask[:, None] & dimension_mask[None, :],
                other=0,
            ).to(tl.int32)
            value_indices = ((value_byte_0 | (value_byte_1 << 8)) >> value_bit_shifts[None, :]) & 0x7
        else:
            value_byte_indices_4 = dimension_offsets // 2
            value_bit_shifts_4 = (dimension_offsets % 2) * 4
            value_byte = tl.load(
                cache_u8_ptr + value_bases[:, None] + value_byte_indices_4[None, :],
                mask=position_mask[:, None] & dimension_mask[None, :],
                other=0,
            ).to(tl.int32)
            value_indices = (value_byte >> value_bit_shifts_4[None, :]) & 0xF

        value_metadata_base = value_bases + VALUE_DATA_BYTES
        value_scale = tl.load(
            cache_f16_ptr + value_metadata_base // 2,
            mask=position_mask,
            other=0.0,
        ).to(tl.float32)
        value_minimum = tl.load(
            cache_f16_ptr + value_metadata_base // 2 + 1,
            mask=position_mask,
            other=0.0,
        ).to(tl.float32)
        values = value_indices.to(tl.float32) * value_scale[:, None] + value_minimum[:, None]
        weighted_values = tl.dot(
            probabilities.to(query_rotated_ptr.dtype.element_ty),
            values.to(query_rotated_ptr.dtype.element_ty),
        )

        accumulator = accumulator * previous_scale[:, None] + weighted_values
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
        running_max = new_max

    partial_bases = (
        batch_index * stride_partial_batch + query_heads * stride_partial_head + split_index * stride_partial_split
    )
    safe_sum = tl.maximum(running_sum, 1e-20)
    tl.store(
        partial_output_ptr + partial_bases[:, None] + dimension_offsets[None, :],
        accumulator / safe_sum[:, None],
        mask=query_mask[:, None] & dimension_mask[None, :],
    )
    tl.store(
        partial_output_ptr + partial_bases + HEAD_DIM,
        running_max + tl.log(safe_sum),
        mask=query_mask,
    )


@triton.jit
def _turboquant_decode_stage2(
    partial_output_ptr,
    output_ptr,
    lse_ptr,
    seq_lens_ptr,
    stride_partial_batch,
    stride_partial_head,
    stride_partial_split,
    stride_output_batch,
    stride_output_head,
    stride_lse_batch,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEQUENCE_LENGTH_DELTA: tl.constexpr,
):
    batch_index = tl.program_id(0)
    head_index = tl.program_id(1)
    seq_len = tl.maximum(
        tl.load(seq_lens_ptr + batch_index) + SEQUENCE_LENGTH_DELTA,
        0,
    )
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM
    running_sum = 0.0
    running_max = -float("inf")
    accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
    partial_base = batch_index * stride_partial_batch + head_index * stride_partial_head

    for split_index in range(NUM_KV_SPLITS):
        split_start = split_len * split_index
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_end > split_start:
            split_base = partial_base + split_index * stride_partial_split
            partial = tl.load(
                partial_output_ptr + split_base + d_offsets,
                mask=d_mask,
                other=0.0,
            )
            partial_lse = tl.load(partial_output_ptr + split_base + HEAD_DIM)
            new_max = tl.maximum(partial_lse, running_max)
            previous_scale = tl.exp(running_max - new_max)
            partial_scale = tl.exp(partial_lse - new_max)
            accumulator = accumulator * previous_scale + partial * partial_scale
            running_sum = running_sum * previous_scale + partial_scale
            running_max = new_max

    output_base = batch_index * stride_output_batch + head_index * stride_output_head
    safe_sum = tl.maximum(running_sum, 1e-20)
    tl.store(
        output_ptr + output_base + d_offsets,
        accumulator / safe_sum,
        mask=d_mask,
    )
    tl.store(
        lse_ptr + batch_index * stride_lse_batch + head_index,
        running_max + tl.log(safe_sum),
    )


@triton.jit
def _turboquant_full_dequant_kernel(
    cache_u8_ptr,
    cache_f16_ptr,
    block_table_ptr,
    seq_lens_ptr,
    seq_start_locs_ptr,
    centroids_ptr,
    key_output_ptr,
    value_output_ptr,
    program_offset,
    stride_cache_block,
    stride_cache_position,
    stride_cache_head,
    stride_block_table_batch,
    stride_key_token,
    stride_key_head,
    stride_value_token,
    stride_value_head,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    KEY_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
):
    flat_program = tl.program_id(0) + program_offset
    position = flat_program % MAX_SEQ_LEN
    batch_head = flat_program // MAX_SEQ_LEN
    batch_index = batch_head // NUM_KV_HEADS
    head_index = batch_head % NUM_KV_HEADS
    seq_len = tl.load(seq_lens_ptr + batch_index)
    if position >= seq_len:
        return

    logical_page = position // CACHE_BLOCK_SIZE
    page_offset = position % CACHE_BLOCK_SIZE
    physical_block = tl.cast(
        tl.load(block_table_ptr + batch_index * stride_block_table_batch + logical_page),
        tl.int64,
    )
    slot_base = (
        physical_block * stride_cache_block
        + tl.cast(page_offset, tl.int64) * stride_cache_position
        + tl.cast(head_index, tl.int64) * stride_cache_head
    )

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM
    key_bit_offsets = d_offsets * KEY_BITS
    key_byte_indices = key_bit_offsets // 8
    key_bit_shifts = key_bit_offsets % 8
    key_byte_0 = tl.load(
        cache_u8_ptr + slot_base + key_byte_indices,
        mask=d_mask,
        other=0,
    ).to(tl.int32)
    key_byte_1 = tl.load(
        cache_u8_ptr + slot_base + key_byte_indices + 1,
        mask=d_mask,
        other=0,
    ).to(tl.int32)
    key_indices = ((key_byte_0 | (key_byte_1 << 8)) >> key_bit_shifts) & ((1 << KEY_BITS) - 1)
    key_values = tl.load(
        centroids_ptr + key_indices,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    if NORM_CORRECTION:
        key_norm_sq = tl.sum(
            tl.where(d_mask, key_values * key_values, 0.0),
            axis=0,
        )
        key_values *= 1.0 / tl.sqrt(key_norm_sq + 1e-16)
    original_norm = tl.load(cache_f16_ptr + (slot_base + MSE_BYTES) // 2).to(tl.float32)
    key_values *= original_norm

    value_base = slot_base + KPS
    if VALUE_BITS == 3:
        value_bit_offsets = d_offsets * 3
        value_byte_indices = value_bit_offsets // 8
        value_bit_shifts = value_bit_offsets % 8
        value_byte_0 = tl.load(
            cache_u8_ptr + value_base + value_byte_indices,
            mask=d_mask,
            other=0,
        ).to(tl.int32)
        value_byte_1 = tl.load(
            cache_u8_ptr + value_base + value_byte_indices + 1,
            mask=d_mask,
            other=0,
        ).to(tl.int32)
        value_indices = ((value_byte_0 | (value_byte_1 << 8)) >> value_bit_shifts) & 0x7
    else:
        value_byte_indices = d_offsets // 2
        value_bit_shifts = (d_offsets % 2) * 4
        value_byte = tl.load(
            cache_u8_ptr + value_base + value_byte_indices,
            mask=d_mask,
            other=0,
        ).to(tl.int32)
        value_indices = (value_byte >> value_bit_shifts) & 0xF

    value_metadata_base = value_base + VALUE_DATA_BYTES
    value_scale = tl.load(cache_f16_ptr + value_metadata_base // 2).to(tl.float32)
    value_minimum = tl.load(cache_f16_ptr + value_metadata_base // 2 + 1).to(tl.float32)
    value_values = value_indices.to(tl.float32) * value_scale + value_minimum

    output_token = tl.load(seq_start_locs_ptr + batch_index) + position
    key_output_base = output_token * stride_key_token + head_index * stride_key_head
    value_output_base = output_token * stride_value_token + head_index * stride_value_head
    tl.store(
        key_output_ptr + key_output_base + d_offsets,
        key_values,
        mask=d_mask,
    )
    tl.store(
        value_output_ptr + value_output_base + d_offsets,
        value_values,
        mask=d_mask,
    )


def _layout(
    head_dim: int,
    key_bits: int,
    value_bits: int,
) -> tuple[int, int, int]:
    return (
        math.ceil(head_dim * key_bits / 8),
        math.ceil(head_dim * value_bits / 8),
        triton.next_power_of_2(head_dim),
    )


def _dequant_launch_ranges(total_programs: int) -> list[tuple[int, int]]:
    if total_programs < 0:
        raise ValueError(f"total_programs must be non-negative, got {total_programs}.")
    return [
        (
            program_start,
            min(program_start + _ASCEND_MAX_TRITON_GRID_SIZE, total_programs),
        )
        for program_start in range(
            0,
            total_programs,
            _ASCEND_MAX_TRITON_GRID_SIZE,
        )
    ]


def _supports_grouped_gqa(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> bool:
    group_size = num_query_heads // num_kv_heads
    return (
        _GROUPED_GQA_MIN_GROUP_SIZE <= group_size <= _GROUPED_GQA_MAX_GROUP_SIZE
        and head_dim <= _GROUPED_GQA_MAX_HEAD_DIM
        and head_dim % 16 == 0
    )


def select_turboquant_num_kv_splits(
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_num_kv_splits: int,
    max_sequence_length: int | None,
    implementation: str = "auto",
) -> int:
    """Select split-KV parallelism without oversubscribing high batches.

    The grouped kernel launches one stage-1 program per KV head and split;
    the reference kernel launches one per query head and split. Once the
    batch/head rows already expose enough parallel work, more splits only add
    program scheduling, partial-buffer traffic, and stage-2 reduction work.
    """
    if batch_size <= 0:
        raise ValueError(f"TurboQuant batch_size must be positive, got {batch_size}.")
    if num_query_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("TurboQuant query and KV head counts must be positive.")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"Query heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads}).")
    if max_num_kv_splits <= 0:
        raise ValueError(f"TurboQuant max_num_kv_splits must be positive, got {max_num_kv_splits}.")
    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError(f"TurboQuant max_sequence_length must be positive when provided, got {max_sequence_length}.")
    if implementation not in ("auto", "grouped_gqa", "reference"):
        raise ValueError(
            f"TurboQuant decode implementation must be auto, grouped_gqa, or reference; got {implementation}."
        )

    grouped = implementation != "reference" and _supports_grouped_gqa(
        num_query_heads,
        num_kv_heads,
        head_dim,
    )
    program_heads = num_kv_heads if grouped else num_query_heads
    parallel_rows = batch_size * program_heads
    parallel_limit = max(1, _TARGET_DECODE_PROGRAMS // parallel_rows)

    # Powers of two limit Triton specializations and produce balanced split
    # ranges. Round the concurrency limit down so the target is never exceeded.
    parallel_limit = 1 << (parallel_limit.bit_length() - 1)
    split_limit = min(max_num_kv_splits, parallel_limit)
    if max_sequence_length is None:
        return max(1, split_limit)

    ideal_splits = max(1, max_sequence_length // _MIN_KV_TOKENS_PER_SPLIT)
    ideal_splits = triton.next_power_of_2(ideal_splits)
    return max(1, min(split_limit, ideal_splits))


def _get_compute_rotation(
    holder: Any,
    rotation: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a cached low-precision rotation suitable for NPU Cube matmul."""
    name = "_tq_ascend_compute_rotation"
    cached = getattr(holder, name, None)
    if cached is not None and cached.device == rotation.device and cached.dtype == dtype:
        return cached
    converted = rotation.to(dtype=dtype)
    if holder is not None:
        setattr(holder, name, converted)
    return converted


def _get_workspace(
    holder: Any,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Reuse graph-safe layer storage or allocate an eager-only fallback."""
    buffer = getattr(holder, name, None)
    if (
        buffer is not None
        and buffer.dtype == torch.float32
        and buffer.device == device
        and buffer.ndim == len(shape)
        and all(actual >= required for actual, required in zip(buffer.shape, shape))
    ):
        return buffer[tuple(slice(0, size) for size in shape)]
    return torch.empty(shape, dtype=torch.float32, device=device)


def triton_turboquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    hadamard_transpose: torch.Tensor,
    centroids: torch.Tensor,
    *,
    scale: float,
    key_bits: int,
    key_packed_size: int,
    value_bits: int,
    norm_correction: bool,
    max_num_kv_splits: int,
    buffer_holder: Any,
    alibi_slopes: torch.Tensor | None = None,
    logits_soft_cap: float | None = None,
    output: torch.Tensor | None = None,
    sequence_length_delta: int = 0,
    implementation: str = "auto",
    compute_rotation: torch.Tensor | None = None,
    grouped_block_kv: int = _GROUPED_GQA_BLOCK_KV,
) -> torch.Tensor:
    """Run split-KV decode directly against packed TurboQuant pages."""
    if logits_soft_cap is not None and logits_soft_cap <= 0:
        raise ValueError(f"logits_soft_cap must be positive, got {logits_soft_cap}.")
    batch_size = seq_lens.shape[0]
    if batch_size == 0:
        return query[:0]
    if query.shape[0] < batch_size:
        raise ValueError(
            f"TurboQuant decode has fewer query rows than sequence lengths: {query.shape[0]} < {batch_size}."
        )
    if block_table.shape[0] < batch_size:
        raise ValueError(
            f"TurboQuant decode has fewer block-table rows than requests: {block_table.shape[0]} < {batch_size}."
        )
    if max_num_kv_splits <= 0:
        raise ValueError(f"TurboQuant max_num_kv_splits must be positive, got {max_num_kv_splits}.")
    if implementation not in ("auto", "grouped_gqa", "reference"):
        raise ValueError(
            f"TurboQuant decode implementation must be auto, grouped_gqa, or reference; got {implementation}."
        )
    if grouped_block_kv not in _GROUPED_GQA_BLOCK_KV_OPTIONS:
        raise ValueError(
            f"TurboQuant grouped BLOCK_KV must be one of {_GROUPED_GQA_BLOCK_KV_OPTIONS}, got {grouped_block_kv}."
        )
    query = query[:batch_size]
    _, num_query_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"Query heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads}).")
    if alibi_slopes is not None and alibi_slopes.numel() != num_query_heads:
        raise ValueError(
            "TurboQuant ALiBi slopes must contain one value per query head, "
            f"got {alibi_slopes.numel()} for {num_query_heads} heads."
        )
    grouped_gqa_supported = _supports_grouped_gqa(
        num_query_heads,
        num_kv_heads,
        head_dim,
    )
    if implementation == "grouped_gqa" and not grouped_gqa_supported:
        group_size = num_query_heads // num_kv_heads
        raise ValueError(
            "The grouped TurboQuant decode requires GQA group size in "
            f"[{_GROUPED_GQA_MIN_GROUP_SIZE}, "
            f"{_GROUPED_GQA_MAX_GROUP_SIZE}] and head_dim <= "
            f"{_GROUPED_GQA_MAX_HEAD_DIM}; got group_size={group_size}, "
            f"head_dim={head_dim}."
        )
    use_grouped_gqa = grouped_gqa_supported and implementation != "reference"

    mse_bytes, value_data_bytes, block_d = _layout(
        head_dim,
        key_bits,
        value_bits,
    )
    if compute_rotation is None and use_grouped_gqa:
        compute_rotation = _get_compute_rotation(
            buffer_holder,
            hadamard_transpose,
            query.dtype,
        )
    if compute_rotation is not None:
        if compute_rotation.dtype != query.dtype:
            raise TypeError(
                f"TurboQuant compute rotation dtype must match query, got {compute_rotation.dtype} and {query.dtype}."
            )
        if compute_rotation.device != query.device:
            raise ValueError(
                "TurboQuant compute rotation must be on the query device, got "
                f"{compute_rotation.device} and {query.device}."
            )
        if compute_rotation.shape != (head_dim, head_dim):
            raise ValueError(
                "TurboQuant compute rotation must be square with the attention "
                f"head dimension, got {compute_rotation.shape} for {head_dim}."
            )
        query_rotated = (query @ compute_rotation).contiguous()
    else:
        query_rotated = (query.float() @ hadamard_transpose).contiguous()
    num_splits = max_num_kv_splits

    # Layer buffers are sized for max_num_seqs and are stable during graph
    # replay. Continuation prefill can present more synthetic requests than
    # that, so eager execution must use a correctly sized temporary instead
    # of launching Triton against a truncated view.
    partial = _get_workspace(
        buffer_holder,
        "_tq_mid_o_buf",
        (batch_size, num_query_heads, num_splits, head_dim + 1),
        query.device,
    )
    lse = _get_workspace(
        buffer_holder,
        "_tq_lse_buf",
        (batch_size, num_query_heads),
        query.device,
    )
    if output is None:
        output = torch.empty(
            batch_size,
            num_query_heads,
            head_dim,
            dtype=query.dtype,
            device=query.device,
        )
    else:
        if output.dtype != query.dtype or output.device != query.device:
            raise TypeError(
                "TurboQuant output must match query dtype and device, got "
                f"{output.dtype}/{output.device} and "
                f"{query.dtype}/{query.device}."
            )
        if output.ndim != 3 or any(
            actual < required
            for actual, required in zip(
                output.shape,
                (batch_size, num_query_heads, head_dim),
            )
        ):
            raise ValueError(
                "TurboQuant output is smaller than the decode result: "
                f"{output.shape} vs "
                f"{(batch_size, num_query_heads, head_dim)}."
            )
        if output.stride(2) != 1:
            raise ValueError(
                f"TurboQuant output must be contiguous in the head dimension, got strides {output.stride()}."
            )
        output = output[
            :batch_size,
            :num_query_heads,
            :head_dim,
        ]

    common_args = (
        query_rotated,
        kv_cache,
        kv_cache.view(torch.float16),
        block_table,
        seq_lens,
        centroids,
        query if alibi_slopes is None else alibi_slopes,
        partial,
        query_rotated.stride(0),
        query_rotated.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
    )
    group_size = num_query_heads // num_kv_heads
    if use_grouped_gqa:
        grouped_grid = (batch_size, num_kv_heads, num_splits)
        _turboquant_grouped_gqa_stage1[grouped_grid](
            *common_args,
            NUM_QUERY_HEADS=num_query_heads,
            HEAD_DIM=head_dim,
            CACHE_BLOCK_SIZE=kv_cache.shape[1],
            NUM_KV_SPLITS=num_splits,
            KV_GROUP_SIZE=group_size,
            KEY_BITS=key_bits,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VALUE_BITS=value_bits,
            VALUE_DATA_BYTES=value_data_bytes,
            ATTENTION_SCALE=scale,
            BLOCK_Q=max(16, triton.next_power_of_2(group_size)),
            BLOCK_D=block_d,
            BLOCK_KV=grouped_block_kv,
            NORM_CORRECTION=norm_correction,
            HAS_ALIBI=alibi_slopes is not None,
            LOGITS_SOFT_CAP=logits_soft_cap or 0.0,
            SEQUENCE_LENGTH_DELTA=sequence_length_delta,
            num_warps=4,
            num_stages=1,
        )
    else:
        reference_grid = (batch_size, num_query_heads, num_splits)
        _turboquant_decode_stage1[reference_grid](
            *common_args,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            CACHE_BLOCK_SIZE=kv_cache.shape[1],
            NUM_KV_SPLITS=num_splits,
            KV_GROUP_SIZE=group_size,
            KEY_BITS=key_bits,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VALUE_BITS=value_bits,
            VALUE_DATA_BYTES=value_data_bytes,
            ATTENTION_SCALE=scale,
            BLOCK_D=block_d,
            BLOCK_KV=4,
            NORM_CORRECTION=norm_correction,
            HAS_ALIBI=alibi_slopes is not None,
            LOGITS_SOFT_CAP=logits_soft_cap or 0.0,
            SEQUENCE_LENGTH_DELTA=sequence_length_delta,
            num_warps=1,
            num_stages=1,
        )

    reduce_grid = (batch_size, num_query_heads)
    _turboquant_decode_stage2[reduce_grid](
        partial,
        output,
        lse,
        seq_lens,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_splits,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        SEQUENCE_LENGTH_DELTA=sequence_length_delta,
        num_warps=4,
        num_stages=1,
    )
    return output


def triton_turboquant_dequant_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_start_locs: torch.Tensor,
    centroids: torch.Tensor,
    key_output: torch.Tensor,
    value_output: torch.Tensor,
    *,
    max_seq_len: int,
    key_bits: int,
    key_packed_size: int,
    value_bits: int,
    norm_correction: bool,
) -> None:
    """Dequantize paged history to compact TND buffers for prefill fallback."""
    _, num_kv_heads, head_dim = key_output.shape
    if max_seq_len <= 0 or seq_lens.numel() == 0:
        return
    mse_bytes, value_data_bytes, block_d = _layout(
        head_dim,
        key_bits,
        value_bits,
    )
    total_programs = max_seq_len * seq_lens.shape[0] * num_kv_heads
    cache_f16 = kv_cache.view(torch.float16)
    for program_start, program_end in _dequant_launch_ranges(total_programs):
        grid = (program_end - program_start,)
        _turboquant_full_dequant_kernel[grid](
            kv_cache,
            cache_f16,
            block_table,
            seq_lens,
            seq_start_locs,
            centroids,
            key_output,
            value_output,
            program_start,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            key_output.stride(0),
            key_output.stride(1),
            value_output.stride(0),
            value_output.stride(1),
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            CACHE_BLOCK_SIZE=kv_cache.shape[1],
            KEY_BITS=key_bits,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VALUE_BITS=value_bits,
            VALUE_DATA_BYTES=value_data_bytes,
            BLOCK_D=block_d,
            NORM_CORRECTION=norm_correction,
            MAX_SEQ_LEN=max_seq_len,
            num_warps=4,
            num_stages=1,
        )
