#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Validate the Ascend TurboQuant operator against Triton implementations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)

from vllm_ascend.attention.turboquant import _build_hadamard
from vllm_ascend.kv_cache.turboquant import get_turboquant_config
from vllm_ascend.ops.triton.turboquant_decode import (
    triton_turboquant_decode_attention,
    triton_turboquant_dequant_paged_cache,
)
from vllm_ascend.ops.triton.turboquant_store import triton_turboquant_store
from vllm_ascend.ops.turboquant import (
    has_turboquant_paged_dequant,
    turboquant_paged_dequant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TurboQuant operator correctness and quantization error.",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=(
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        ),
        default="turboquant_4bit_nc",
    )
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[133, 65],
        help="Per-request lengths. Unequal lengths exercise masking and paging.",
    )
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-kv-splits", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.sequence_lengths or any(length <= 0 for length in args.sequence_lengths):
        raise ValueError("--sequence-lengths must contain positive values.")
    if args.num_query_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("Head counts must be positive.")
    if args.num_query_heads % args.num_kv_heads:
        raise ValueError("--num-query-heads must be divisible by --num-kv-heads.")
    if args.head_dim < 32 or args.head_dim > 256 or args.head_dim & (args.head_dim - 1):
        raise ValueError("--head-dim must be a power of two in [32, 256].")
    if args.block_size <= 0 or args.num_kv_splits <= 0:
        raise ValueError("--block-size and --num-kv-splits must be positive.")


def activation_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float().reshape(-1)
    expected_float = expected.float().reshape(-1)
    difference = actual_float - expected_float
    mse = difference.square().mean()
    reference_energy = expected_float.square().mean().clamp_min(1.0e-20)
    return {
        "mse": mse.item(),
        "nmse": (mse / reference_energy).item(),
        "mae": difference.abs().mean().item(),
        "max_abs": difference.abs().max().item(),
        "cosine_similarity": F.cosine_similarity(
            actual_float.unsqueeze(0),
            expected_float.unsqueeze(0),
        ).item(),
    }


def close_result(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = error_metrics(actual, expected)
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        result["passed"] = False
        result["failure"] = str(error)
    else:
        result["passed"] = True
    result["atol"] = atol
    result["rtol"] = rtol
    return result


def build_inputs(args: argparse.Namespace):
    device = torch.device(f"npu:{args.device}")
    dtype = activation_dtype(args.activation_dtype)
    config = get_turboquant_config(args.cache_dtype, args.head_dim)
    hadamard = _build_hadamard(args.head_dim, str(device))
    compute_rotation = hadamard.to(dtype)
    centroids = get_centroids(args.head_dim, config.centroid_bits).to(
        device=device,
        dtype=torch.float32,
    )
    centroids, _ = centroids.sort()
    midpoints = (centroids[:-1] + centroids[1:]) / 2

    pages_per_request = [math.ceil(length / args.block_size) for length in args.sequence_lengths]
    total_blocks = sum(pages_per_request)
    max_pages = max(pages_per_request)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1)
    physical_blocks = torch.randperm(total_blocks, generator=generator).tolist()
    block_table_cpu = torch.full(
        (len(args.sequence_lengths), max_pages),
        -1,
        dtype=torch.int32,
    )
    block_cursor = 0
    for request_index, page_count in enumerate(pages_per_request):
        block_table_cpu[request_index, :page_count] = torch.tensor(
            physical_blocks[block_cursor : block_cursor + page_count],
            dtype=torch.int32,
        )
        block_cursor += page_count
    block_table = block_table_cpu.to(device)

    cache = torch.zeros(
        total_blocks,
        args.block_size,
        args.num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    key_parts = []
    value_parts = []
    slot_parts = []
    for request_index, sequence_length in enumerate(args.sequence_lengths):
        key = torch.randn(
            sequence_length,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        value = torch.randn(key.shape, dtype=key.dtype, device=key.device)
        positions = torch.arange(sequence_length, dtype=torch.int64, device=device)
        request_blocks = block_table[request_index, positions // args.block_size]
        slots = request_blocks.to(torch.int64) * args.block_size + positions % args.block_size
        key_parts.append(key)
        value_parts.append(value)
        slot_parts.append(slots)

    key = torch.cat(key_parts)
    value = torch.cat(value_parts)
    slot_mapping = torch.cat(slot_parts)
    triton_turboquant_store(
        key,
        value,
        cache,
        slot_mapping,
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=compute_rotation,
    )
    torch.npu.synchronize()

    seq_lens = torch.tensor(args.sequence_lengths, dtype=torch.int32, device=device)
    query = torch.randn(
        len(args.sequence_lengths),
        args.num_query_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    return (
        config,
        hadamard,
        compute_rotation,
        centroids,
        midpoints,
        cache,
        block_table,
        seq_lens,
        query,
        key,
        value,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.npu.is_available():
        raise RuntimeError("torch-npu cannot see an Ascend NPU.")
    if not has_turboquant_paged_dequant():
        raise RuntimeError(
            "TurboQuant paged-dequant return/out schemas are unavailable; "
            "clean and rebuild vLLM Ascend with SOC_VERSION=ascend910b4."
        )

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    (
        config,
        hadamard,
        compute_rotation,
        centroids,
        midpoints,
        cache,
        block_table,
        seq_lens,
        query,
        key,
        value,
    ) = build_inputs(args)
    max_seq_len = max(args.sequence_lengths)
    total_tokens = sum(args.sequence_lengths)

    cache_before_invalid_store = cache.clone()
    invalid_key = torch.randn(
        1,
        args.num_kv_heads,
        args.head_dim,
        dtype=key.dtype,
        device=key.device,
    )
    triton_turboquant_store(
        invalid_key,
        torch.randn(
            invalid_key.shape,
            dtype=invalid_key.dtype,
            device=invalid_key.device,
        ),
        cache,
        torch.tensor([-1], dtype=torch.int64, device=key.device),
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=compute_rotation,
    )
    torch.npu.synchronize()
    negative_slot_mapping_passed = torch.equal(cache, cache_before_invalid_store)
    del cache_before_invalid_store

    page_table = torch.tensor(
        [
            (request_index, page_index)
            for request_index, seq_len in enumerate(args.sequence_lengths)
            for page_index in range((seq_len + cache.shape[1] - 1) // cache.shape[1])
        ],
        dtype=torch.int32,
        device=query.device,
    )
    key_ascend, value_ascend = turboquant_paged_dequant(
        query,
        cache,
        block_table,
        seq_lens,
        page_table,
        centroids,
        max_seq_len=max_seq_len,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
    )
    sequence_start_locs = torch.tensor(
        [0, *[sum(args.sequence_lengths[: index + 1]) for index in range(len(args.sequence_lengths))]],
        dtype=torch.int32,
        device=key.device,
    )
    key_triton = torch.empty_like(key)
    value_triton = torch.empty_like(value)
    triton_turboquant_dequant_paged_cache(
        cache,
        block_table,
        seq_lens,
        sequence_start_locs,
        centroids,
        key_triton,
        value_triton,
        max_seq_len=max_seq_len,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
    )
    torch.npu.synchronize()

    key_ascend_compact = torch.cat(
        [key_ascend[index, :, :length].permute(1, 0, 2) for index, length in enumerate(args.sequence_lengths)]
    )
    value_ascend_compact = torch.cat(
        [value_ascend[index, :, :length].permute(1, 0, 2) for index, length in enumerate(args.sequence_lengths)]
    )
    dequant_tolerance = 2.0e-2 if key.dtype == torch.bfloat16 else 3.0e-3
    key_implementation = close_result(
        key_ascend_compact,
        key_triton,
        atol=dequant_tolerance,
        rtol=dequant_tolerance,
    )
    value_implementation = close_result(
        value_ascend_compact,
        value_triton,
        atol=dequant_tolerance,
        rtol=dequant_tolerance,
    )

    key_rotated_reference = (key.reshape(-1, args.head_dim) @ compute_rotation).reshape_as(key)
    key_quantization = error_metrics(key_ascend_compact, key_rotated_reference)
    value_quantization = error_metrics(value_ascend_compact, value)

    packed_output = triton_turboquant_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        hadamard,
        centroids,
        scale=1 / math.sqrt(args.head_dim),
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
        max_num_kv_splits=args.num_kv_splits,
        buffer_holder=SimpleNamespace(),
        compute_rotation=compute_rotation,
        implementation="auto",
    )
    query_rotated = (query @ compute_rotation).contiguous()
    fused_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query_rotated.unsqueeze(2),
        key=key_ascend,
        value=value_ascend,
        block_table=None,
        input_layout="BNSD",
        block_size=args.block_size,
        actual_seq_lengths_kv=args.sequence_lengths,
        num_key_value_heads=args.num_kv_heads,
        num_heads=args.num_query_heads,
        scale=1 / math.sqrt(args.head_dim),
        sparse_mode=0,
    )
    torch.npu.synchronize()
    attention_agreement = close_result(
        fused_output.squeeze(2),
        packed_output,
        atol=2.0e-2,
        rtol=2.0e-2,
    )

    native_capacity_bytes = (
        cache.shape[0] * args.block_size * args.num_kv_heads * 2 * args.head_dim * key.element_size()
    )
    packed_capacity_bytes = cache.numel() * cache.element_size()
    passed = (
        negative_slot_mapping_passed
        and key_implementation["passed"]
        and value_implementation["passed"]
        and attention_agreement["passed"]
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "configuration": {
            **vars(args),
            "output": str(args.output),
        },
        "device": torch.npu.get_device_name(args.device),
        "layout": {
            "slot_size_bytes": config.slot_size_aligned,
            "key_packed_size_bytes": config.key_packed_size,
            "value_packed_size_bytes": config.value_packed_size,
            "native_capacity_bytes": native_capacity_bytes,
            "packed_capacity_bytes": packed_capacity_bytes,
            "capacity_compression_ratio": native_capacity_bytes / packed_capacity_bytes,
            "valid_tokens": total_tokens,
            "physical_blocks": cache.shape[0],
        },
        "checks": {
            "negative_slot_mapping_preserves_cache": negative_slot_mapping_passed,
            "ascend_key_vs_triton": key_implementation,
            "ascend_value_vs_triton": value_implementation,
            "ascend_fia_vs_packed_decode": attention_agreement,
        },
        "quantization_error": {
            "rotated_key": key_quantization,
            "value": value_quantization,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Report: {args.output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
