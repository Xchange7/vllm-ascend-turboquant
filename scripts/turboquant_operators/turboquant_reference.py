#!/usr/bin/env python3

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

"""Device-independent reference for the packed TurboQuant cache.

This module intentionally does not import Triton, torch-npu, or the Ascend
custom operator. It reads the cache bytes on CPU and implements unpacking,
dequantization, GQA head expansion, and attention with ordinary PyTorch.
"""

from __future__ import annotations

import math

import torch


def _unpack_indices(payload: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    """Unpack little-endian bit fields from the final payload dimension."""
    if payload.device.type != "cpu" or payload.dtype != torch.uint8:
        raise TypeError("TurboQuant reference payload must be a CPU uint8 tensor.")
    if bits not in (3, 4):
        raise ValueError(f"TurboQuant reference supports 3-bit or 4-bit payloads, got {bits}.")

    bit_offsets = torch.arange(head_dim, dtype=torch.int64) * bits
    byte_indices = bit_offsets // 8
    bit_shifts = bit_offsets % 8
    # One zero sentinel makes the two-byte extraction valid at the payload end.
    padded = torch.nn.functional.pad(payload, (0, 1))
    low = padded[..., byte_indices].to(torch.int32)
    high = padded[..., byte_indices + 1].to(torch.int32)
    return ((low | (high << 8)) >> bit_shifts) & ((1 << bits) - 1)


def _decode_fp16(payload: torch.Tensor) -> torch.Tensor:
    """Interpret pairs of packed bytes as little-endian FP16 values."""
    if payload.shape[-1] != 2:
        raise ValueError(f"FP16 payload must contain two bytes, got {payload.shape[-1]}.")
    return payload.contiguous().view(torch.float16).squeeze(-1).float()


def gather_paged_slots(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
) -> torch.Tensor:
    """Gather active paged-cache slots in request-major token order."""
    if cache.device.type != "cpu" or cache.dtype != torch.uint8:
        raise TypeError("TurboQuant reference cache must be a CPU uint8 tensor.")
    if block_table.device.type != "cpu":
        raise TypeError("TurboQuant reference block table must be on CPU.")
    if cache.ndim != 4:
        raise ValueError(f"TurboQuant cache must be rank four, got {cache.shape}.")
    if block_table.ndim != 2 or block_table.shape[0] < len(seq_lens):
        raise ValueError(
            "TurboQuant block table must have one row per request, got "
            f"{block_table.shape} for {len(seq_lens)} requests."
        )

    block_size = cache.shape[1]
    num_kv_heads = cache.shape[2]
    flat_cache = cache.view(-1, cache.shape[-1])
    head_indices = torch.arange(num_kv_heads, dtype=torch.int64)
    requests = []
    for request_index, seq_len in enumerate(seq_lens):
        if seq_len <= 0:
            raise ValueError(f"TurboQuant sequence lengths must be positive, got {seq_lens}.")
        positions = torch.arange(seq_len, dtype=torch.int64)
        logical_blocks = positions // block_size
        if logical_blocks[-1] >= block_table.shape[1]:
            raise ValueError(
                f"Request {request_index} needs logical block {logical_blocks[-1].item()}, "
                f"but its block table has {block_table.shape[1]} entries."
            )
        physical_blocks = block_table[request_index, logical_blocks].long()
        if torch.any(physical_blocks < 0) or torch.any(physical_blocks >= cache.shape[0]):
            raise ValueError(f"Request {request_index} contains an invalid physical block.")
        # Cache layout V2 is physically [block, kv_head, token, slot] while
        # preserving the public [block, token, kv_head, slot] tensor shape.
        slot_indices = (
            (physical_blocks[:, None] * num_kv_heads + head_indices[None, :])
            * block_size
            + (positions % block_size)[:, None]
        )
        requests.append(flat_cache[slot_indices])
    return torch.cat(requests, dim=0)


def dequantize_paged_cache_reference(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    centroids: torch.Tensor,
    *,
    head_dim: int,
    key_bits: int,
    key_packed_size: int,
    value_bits: int,
    norm_correction: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode packed cache bytes without using either production implementation."""
    cache = cache.detach().cpu().contiguous()
    block_table = block_table.detach().cpu()
    centroids = centroids.detach().cpu().float()
    slots = gather_paged_slots(cache, block_table, seq_lens)

    key_data_bytes = math.ceil(head_dim * key_bits / 8)
    value_data_bytes = math.ceil(head_dim * value_bits / 8)
    expected_slot_size = key_packed_size + value_data_bytes + 4
    if slots.shape[-1] < expected_slot_size:
        raise ValueError(f"TurboQuant slot has {slots.shape[-1]} bytes, but {expected_slot_size} are required.")
    if key_packed_size != key_data_bytes + 2:
        raise ValueError(
            "TurboQuant reference expects an MSE key payload followed by an FP16 norm-correction scale, "
            f"got key_packed_size={key_packed_size} and key_data_bytes={key_data_bytes}."
        )

    key_indices = _unpack_indices(slots[..., :key_data_bytes], key_bits, head_dim)
    key = centroids[key_indices.long()]
    key_scale = _decode_fp16(slots[..., key_data_bytes:key_packed_size])
    if not norm_correction:
        decoded_norm = key.square().sum(dim=-1).clamp_min(1.0e-16).sqrt()
        key_scale = key_scale * decoded_norm
    key = key * key_scale.unsqueeze(-1)

    value_start = key_packed_size
    value_end = value_start + value_data_bytes
    value_indices = _unpack_indices(slots[..., value_start:value_end], value_bits, head_dim)
    value_scale = _decode_fp16(slots[..., value_end : value_end + 2])
    value_minimum = _decode_fp16(slots[..., value_end + 2 : value_end + 4])
    value = value_indices.float() * value_scale.unsqueeze(-1) + value_minimum.unsqueeze(-1)
    return key, value


def decode_attention_reference(
    query: torch.Tensor,
    key_rotated: torch.Tensor,
    value: torch.Tensor,
    seq_lens: list[int],
    rotation: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Compute one-token GQA attention from CPU-dequantized cache tensors."""
    query = query.detach().cpu().float()
    key_rotated = key_rotated.detach().cpu().float()
    value = value.detach().cpu().float()
    rotation = rotation.detach().cpu().float()
    if query.ndim != 3 or key_rotated.ndim != 3 or value.shape != key_rotated.shape:
        raise ValueError("TurboQuant reference expects Q=[B,Hq,D] and K/V=[T,Hkv,D].")
    if query.shape[0] != len(seq_lens) or key_rotated.shape[0] != sum(seq_lens):
        raise ValueError("TurboQuant reference tensor rows do not match sequence lengths.")

    num_query_heads = query.shape[1]
    num_kv_heads = key_rotated.shape[1]
    if num_query_heads % num_kv_heads:
        raise ValueError("TurboQuant reference requires query heads divisible by KV heads.")
    kv_head_indices = torch.arange(num_query_heads) // (num_query_heads // num_kv_heads)
    query_rotated = query @ rotation

    outputs = []
    token_start = 0
    for request_index, seq_len in enumerate(seq_lens):
        token_end = token_start + seq_len
        request_key = key_rotated[token_start:token_end, kv_head_indices]
        request_value = value[token_start:token_end, kv_head_indices]
        scores = torch.einsum("hd,shd->hs", query_rotated[request_index], request_key) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hs,shd->hd", probabilities, request_value))
        token_start = token_end
    return torch.stack(outputs)
