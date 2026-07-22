#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Execute the AscendC TurboQuant operator in an isolated process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)

from vllm_ascend.kv_cache.turboquant import get_turboquant_config
from vllm_ascend.ops.turboquant import (
    has_turboquant_paged_dequant,
    turboquant_paged_dequant_out,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one minimal TurboQuant ACLNN launch and verify its outputs.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dtype", default="turboquant_4bit_nc")
    parser.add_argument("--activation-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_turboquant_libraries() -> list[dict[str, Any]]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        return []
    candidates: set[Path] = set()
    for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        path_text = line.rsplit(maxsplit=1)[-1]
        if not path_text.startswith("/"):
            continue
        lowered = path_text.lower()
        if "vllm_ascend_c" in lowered or "libopapi" in lowered or "libcust_opapi" in lowered:
            candidates.add(Path(path_text))

    libraries = []
    for path in sorted(candidates):
        entry: dict[str, Any] = {"path": str(path)}
        try:
            entry["sha256"] = sha256_file(path)
            entry["size_bytes"] = path.stat().st_size
        except OSError as error:
            entry["inspection_error"] = str(error)
        libraries.append(entry)
    return libraries


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.npu.is_available():
        raise RuntimeError("torch-npu cannot see an Ascend NPU.")
    if not has_turboquant_paged_dequant():
        raise RuntimeError("TurboQuant return/out operator schemas are not registered.")

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = torch.float16 if args.activation_dtype == "float16" else torch.bfloat16
    config = get_turboquant_config(args.cache_dtype, args.head_dim)
    query = torch.zeros(1, 1, args.head_dim, dtype=dtype, device=device)
    cache = torch.zeros(
        1,
        args.block_size,
        1,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    seq_lens = torch.ones(1, dtype=torch.int32, device=device)
    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    centroids = get_centroids(args.head_dim, config.centroid_bits).to(
        device=device,
        dtype=torch.float32,
    )
    centroids, _ = centroids.sort()
    output_shape = (1, 1, 1, args.head_dim)
    key_out = torch.full(output_shape, float("nan"), dtype=dtype, device=device)
    value_out = torch.full_like(key_out, float("nan"))

    libraries_before_launch = loaded_turboquant_libraries()
    print(
        json.dumps({"loaded_libraries_before_launch": libraries_before_launch}, indent=2),
        flush=True,
    )
    print("Launching npu_turboquant_paged_dequant_out...", flush=True)
    returned_key, returned_value = turboquant_paged_dequant_out(
        query,
        cache,
        block_table,
        seq_lens,
        page_table,
        centroids,
        key_out,
        value_out,
        max_seq_len=1,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
    )
    torch.npu.synchronize()
    if returned_key.data_ptr() != key_out.data_ptr() or returned_value.data_ptr() != value_out.data_ptr():
        raise RuntimeError("Out-style operator did not return the caller-owned output tensors.")
    key_cpu = key_out.float().cpu()
    value_cpu = value_out.float().cpu()
    if not torch.isfinite(key_cpu).all() or not torch.isfinite(value_cpu).all():
        raise RuntimeError("TurboQuant ACLNN probe produced non-finite output.")
    if not torch.equal(key_cpu, torch.zeros_like(key_cpu)) or not torch.equal(
        value_cpu,
        torch.zeros_like(value_cpu),
    ):
        raise RuntimeError("A zero packed cache did not dequantize to zero.")

    return {
        "schema_version": 1,
        "passed": True,
        "device": torch.npu.get_device_name(args.device),
        "configuration": {
            "cache_dtype": args.cache_dtype,
            "activation_dtype": args.activation_dtype,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "slot_size_bytes": config.slot_size_aligned,
        },
        "checks": {
            "schemas_registered": True,
            "kernel_launch_completed": True,
            "caller_owned_outputs_returned": True,
            "outputs_are_finite": True,
            "zero_cache_dequantizes_to_zero": True,
        },
        "loaded_libraries_before_launch": libraries_before_launch,
        "loaded_libraries_after_launch": loaded_turboquant_libraries(),
    }


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
