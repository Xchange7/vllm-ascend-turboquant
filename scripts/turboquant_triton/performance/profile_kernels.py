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
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch_npu
import vllm
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)

from vllm_ascend.attention.turboquant import _build_hadamard
from vllm_ascend.kv_cache.turboquant import get_turboquant_config
from vllm_ascend.ops.triton.turboquant_decode import (
    select_turboquant_num_kv_splits,
    triton_turboquant_decode_attention,
    triton_turboquant_dequant_paged_cache,
)
from vllm_ascend.ops.triton.turboquant_store import triton_turboquant_store


@dataclass
class BenchmarkResult:
    operation: str
    iterations: int
    min_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    throughput_name: str
    throughput: float
    memory_baseline_bytes: int
    peak_memory_bytes: int
    peak_memory_increase_bytes: int


@dataclass
class BenchmarkCase:
    name: str
    run: Callable[[], Any]
    work_per_iteration: int
    throughput_name: str
    tensors: tuple[Any, ...]


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "editable/unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the Triton-Ascend TurboQuant KV cache kernels.",
    )
    parser.add_argument(
        "--operation",
        choices=(
            "all",
            "store",
            "decode",
            "decode_step",
            "dequant",
            "native_decode",
        ),
        default="all",
    )
    parser.add_argument(
        "--native-baseline",
        action="store_true",
        help="Benchmark native BF16/FP16 paged attention beside TurboQuant decode.",
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--store-tokens", type=int, default=1024)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-kv-splits", type=int, default=8)
    parser.add_argument(
        "--adaptive-splits",
        action="store_true",
        help="Use the production TurboQuant split selection policy.",
    )
    parser.add_argument(
        "--decode-implementation",
        choices=("auto", "grouped_gqa", "reference"),
        default="auto",
    )
    parser.add_argument(
        "--grouped-block-kv",
        choices=(16, 32),
        type=int,
        default=16,
    )
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Profiler output directory. Defaults to profiles/turboquant_<timestamp>.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Only collect NPU Event timing; do not run torch_npu.profiler.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "batch_size",
        "sequence_length",
        "store_tokens",
        "num_query_heads",
        "num_kv_heads",
        "head_dim",
        "block_size",
        "num_kv_splits",
        "iterations",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.warmup < 0 or args.profile_iterations < 0:
        raise ValueError("--warmup and --profile-iterations cannot be negative.")
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("--num-query-heads must be divisible by --num-kv-heads.")
    if args.head_dim <= 0 or args.head_dim & (args.head_dim - 1):
        raise ValueError("--head-dim must be a power of two.")
    if args.head_dim % 32:
        raise ValueError("--head-dim must be divisible by 32.")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def make_constants(args: argparse.Namespace):
    config = get_turboquant_config(args.cache_dtype, args.head_dim)
    device = torch.device(f"npu:{args.device}")
    hadamard = _build_hadamard(args.head_dim, str(device))
    centroids = get_centroids(args.head_dim, config.centroid_bits).to(
        device=device,
        dtype=torch.float32,
    )
    centroids, _ = centroids.sort()
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    compute_rotation = hadamard.to(activation_dtype(args))
    return config, hadamard, centroids, midpoints, compute_rotation


def activation_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if args.activation_dtype == "float16" else torch.bfloat16


def build_store_case(args: argparse.Namespace, constants) -> BenchmarkCase:
    config, hadamard, _, midpoints, compute_rotation = constants
    device = torch.device(f"npu:{args.device}")
    key = torch.randn(
        args.store_tokens,
        args.num_kv_heads,
        args.head_dim,
        dtype=activation_dtype(args),
        device=device,
    )
    value = torch.randn_like(key)
    num_blocks = math.ceil(args.store_tokens / args.block_size)
    cache = torch.empty(
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    slot_mapping = torch.arange(
        args.store_tokens,
        dtype=torch.int64,
        device=device,
    )
    active_rotation = None if args.decode_implementation == "reference" else compute_rotation

    def run() -> None:
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
            compute_rotation=active_rotation,
        )

    return BenchmarkCase(
        name="store",
        run=run,
        work_per_iteration=args.store_tokens,
        throughput_name="input_tokens_per_second",
        tensors=(key, value, cache, slot_mapping),
    )


def build_paged_cache(args: argparse.Namespace, constants):
    config, hadamard, centroids, midpoints, compute_rotation = constants
    device = torch.device(f"npu:{args.device}")
    pages_per_request = math.ceil(args.sequence_length / args.block_size)
    total_blocks = args.batch_size * pages_per_request
    block_table = torch.arange(
        total_blocks,
        dtype=torch.int32,
        device=device,
    ).view(args.batch_size, pages_per_request)
    cache = torch.empty(
        total_blocks,
        args.block_size,
        args.num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )

    positions = torch.arange(
        args.sequence_length,
        dtype=torch.int64,
        device=device,
    )
    logical_pages = positions // args.block_size
    page_offsets = positions % args.block_size
    physical_blocks = block_table[:, logical_pages]
    slot_mapping = (physical_blocks.to(torch.int64) * args.block_size + page_offsets.unsqueeze(0)).reshape(-1)
    total_tokens = args.batch_size * args.sequence_length
    key = torch.randn(
        total_tokens,
        args.num_kv_heads,
        args.head_dim,
        dtype=activation_dtype(args),
        device=device,
    )
    value = torch.randn_like(key)
    active_rotation = None if args.decode_implementation == "reference" else compute_rotation
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
        compute_rotation=active_rotation,
    )
    torch.npu.synchronize()
    del key, value

    seq_lens = torch.full(
        (args.batch_size,),
        args.sequence_length,
        dtype=torch.int32,
        device=device,
    )
    return cache, block_table, seq_lens, slot_mapping, centroids


def build_decode_case(
    args: argparse.Namespace,
    constants,
    paged_cache,
    *,
    include_store: bool = False,
) -> BenchmarkCase:
    config, hadamard, centroids, midpoints, compute_rotation = constants
    cache, block_table, seq_lens, slot_mapping, _ = paged_cache
    device = torch.device(f"npu:{args.device}")
    query = torch.randn(
        args.batch_size,
        args.num_query_heads,
        args.head_dim,
        dtype=activation_dtype(args),
        device=device,
    )
    buffers = SimpleNamespace(
        _tq_mid_o_buf=torch.empty(
            args.batch_size,
            args.num_query_heads,
            args.num_kv_splits,
            args.head_dim + 1,
            dtype=torch.float32,
            device=device,
        ),
        _tq_lse_buf=torch.empty(
            args.batch_size,
            args.num_query_heads,
            dtype=torch.float32,
            device=device,
        ),
    )
    output = torch.empty_like(query)
    active_rotation = None if args.decode_implementation == "reference" else compute_rotation
    current_key = None
    current_value = None
    current_slots = None
    if include_store:
        current_key = torch.randn(
            args.batch_size,
            args.num_kv_heads,
            args.head_dim,
            dtype=activation_dtype(args),
            device=device,
        )
        current_value = torch.randn_like(current_key)
        current_slots = slot_mapping.view(args.batch_size, args.sequence_length)[:, -1].contiguous()

    def run() -> torch.Tensor:
        if include_store:
            assert current_key is not None
            assert current_value is not None
            assert current_slots is not None
            triton_turboquant_store(
                current_key,
                current_value,
                cache,
                current_slots,
                hadamard,
                midpoints,
                key_bits=config.key_quant_bits,
                key_packed_size=config.key_packed_size,
                value_bits=config.value_quant_bits,
                compute_rotation=active_rotation,
            )
        return triton_turboquant_decode_attention(
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
            buffer_holder=buffers,
            output=output,
            implementation=args.decode_implementation,
            compute_rotation=active_rotation,
            grouped_block_kv=args.grouped_block_kv,
        )

    return BenchmarkCase(
        name="decode_step" if include_store else "decode",
        run=run,
        work_per_iteration=args.batch_size,
        throughput_name="generated_tokens_per_second",
        tensors=(
            query,
            cache,
            block_table,
            seq_lens,
            buffers,
            output,
            current_key,
            current_value,
            current_slots,
        ),
    )


def build_native_decode_case(args: argparse.Namespace) -> BenchmarkCase:
    device = torch.device(f"npu:{args.device}")
    pages_per_request = math.ceil(args.sequence_length / args.block_size)
    total_blocks = args.batch_size * pages_per_request
    block_table = torch.arange(
        total_blocks,
        dtype=torch.int32,
        device=device,
    ).view(args.batch_size, pages_per_request)
    cache_shape = (
        total_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
    )
    key_cache = torch.randn(
        cache_shape,
        dtype=activation_dtype(args),
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        args.batch_size,
        args.num_query_heads,
        args.head_dim,
        dtype=activation_dtype(args),
        device=device,
    )
    seq_lens = torch.full(
        (args.batch_size,),
        args.sequence_length,
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty_like(query)

    def run() -> torch.Tensor:
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=args.num_kv_heads,
            num_heads=args.num_query_heads,
            scale_value=1 / math.sqrt(args.head_dim),
            block_table=block_table,
            context_lens=seq_lens,
            out=output,
        )
        return output

    return BenchmarkCase(
        name="native_decode",
        run=run,
        work_per_iteration=args.batch_size,
        throughput_name="generated_tokens_per_second",
        tensors=(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            output,
        ),
    )


def build_dequant_case(args: argparse.Namespace, constants, paged_cache) -> BenchmarkCase:
    config, _, centroids, _, _ = constants
    cache, block_table, seq_lens, _, _ = paged_cache
    device = torch.device(f"npu:{args.device}")
    total_tokens = args.batch_size * args.sequence_length
    seq_start_locs = torch.arange(
        0,
        total_tokens + 1,
        args.sequence_length,
        dtype=torch.int32,
        device=device,
    )
    key_output = torch.empty(
        total_tokens,
        args.num_kv_heads,
        args.head_dim,
        dtype=activation_dtype(args),
        device=device,
    )
    value_output = torch.empty_like(key_output)

    def run() -> None:
        triton_turboquant_dequant_paged_cache(
            cache,
            block_table,
            seq_lens,
            seq_start_locs,
            centroids,
            key_output,
            value_output,
            max_seq_len=args.sequence_length,
            key_bits=config.key_quant_bits,
            key_packed_size=config.key_packed_size,
            value_bits=config.value_quant_bits,
            norm_correction=config.norm_correction,
        )

    return BenchmarkCase(
        name="dequant",
        run=run,
        work_per_iteration=total_tokens,
        throughput_name="cache_tokens_per_second",
        tensors=(cache, block_table, seq_lens, key_output, value_output),
    )


def benchmark(case: BenchmarkCase, args: argparse.Namespace) -> BenchmarkResult:
    for _ in range(args.warmup):
        case.run()
    torch.npu.synchronize()

    memory_baseline_bytes = torch.npu.memory_allocated(args.device)
    torch.npu.reset_peak_memory_stats(args.device)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, end in zip(starts, ends):
        start.record()
        case.run()
        end.record()
    torch.npu.synchronize()

    durations_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    mean_ms = statistics.fmean(durations_ms)
    peak_memory_bytes = torch.npu.max_memory_allocated(args.device)
    return BenchmarkResult(
        operation=case.name,
        iterations=args.iterations,
        min_ms=min(durations_ms),
        mean_ms=mean_ms,
        p50_ms=percentile(durations_ms, 0.50),
        p90_ms=percentile(durations_ms, 0.90),
        p99_ms=percentile(durations_ms, 0.99),
        max_ms=max(durations_ms),
        throughput_name=case.throughput_name,
        throughput=case.work_per_iteration / (mean_ms / 1000),
        memory_baseline_bytes=memory_baseline_bytes,
        peak_memory_bytes=peak_memory_bytes,
        peak_memory_increase_bytes=max(
            0,
            peak_memory_bytes - memory_baseline_bytes,
        ),
    )


def collect_trace(case: BenchmarkCase, args: argparse.Namespace, trace_dir: Path) -> None:
    if args.no_trace or args.profile_iterations == 0:
        return
    operation_dir = trace_dir / case.name
    operation_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=False,
        profile_memory=True,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(operation_dir),
            worker_name=f"turboquant_{case.name}",
        ),
    )
    profiler.start()
    with torch.profiler.record_function(f"turboquant_{case.name}"):
        for _ in range(args.profile_iterations):
            case.run()
    torch.npu.synchronize()
    profiler.stop()


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.operation:>7}: mean={result.mean_ms:.4f} ms "
        f"p50={result.p50_ms:.4f} ms p90={result.p90_ms:.4f} ms "
        f"p99={result.p99_ms:.4f} ms "
        f"{result.throughput_name}={result.throughput:,.2f} "
        f"memory_baseline={result.memory_baseline_bytes / 2**20:.2f} MiB "
        f"peak_increase={result.peak_memory_increase_bytes / 2**20:.2f} MiB"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.adaptive_splits:
        configured_max_splits = args.num_kv_splits
        args.num_kv_splits = select_turboquant_num_kv_splits(
            batch_size=args.batch_size,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            max_num_kv_splits=configured_max_splits,
            max_sequence_length=args.sequence_length,
            implementation=args.decode_implementation,
        )
        print(
            f"Adaptive split selection: max={configured_max_splits}, selected={args.num_kv_splits}",
            flush=True,
        )
    if not torch.npu.is_available():
        raise RuntimeError("torch-npu cannot see an Ascend NPU.")

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    trace_dir = args.trace_dir or Path(
        "profiles",
        f"turboquant_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    constants = make_constants(args)

    cases: list[BenchmarkCase] = []
    if args.operation in ("all", "store"):
        cases.append(build_store_case(args, constants))
    if args.operation in ("all", "decode", "decode_step", "dequant"):
        paged_cache = build_paged_cache(args, constants)
        if args.operation in ("all", "decode"):
            cases.append(build_decode_case(args, constants, paged_cache))
        if args.operation in ("all", "decode_step"):
            cases.append(
                build_decode_case(
                    args,
                    constants,
                    paged_cache,
                    include_store=True,
                )
            )
        if args.operation in ("all", "dequant"):
            cases.append(build_dequant_case(args, constants, paged_cache))
    if args.operation == "native_decode" or (args.native_baseline and args.operation in ("all", "decode")):
        cases.append(build_native_decode_case(args))

    results = []
    for case in cases:
        result = benchmark(case, args)
        print_result(result)
        collect_trace(case, args, trace_dir)
        results.append(asdict(result))

    result_by_operation = {result["operation"]: result for result in results}
    comparison: dict[str, float] = {}
    if "decode" in result_by_operation and "native_decode" in result_by_operation:
        tq_decode = result_by_operation["decode"]
        native_decode = result_by_operation["native_decode"]
        config = get_turboquant_config(args.cache_dtype, args.head_dim)
        native_slot_bytes = 2 * args.head_dim * torch.empty((), dtype=activation_dtype(args)).element_size()
        comparison.update(
            {
                "decode_speedup_vs_native": (native_decode["mean_ms"] / tq_decode["mean_ms"]),
                "native_mean_ms": native_decode["mean_ms"],
                "turboquant_mean_ms": tq_decode["mean_ms"],
                "theoretical_cache_compression_ratio": (native_slot_bytes / config.slot_size_aligned),
            }
        )
        print(
            "compare: "
            f"decode_speedup={comparison['decode_speedup_vs_native']:.3f}x "
            "cache_compression="
            f"{comparison['theoretical_cache_compression_ratio']:.3f}x"
        )
    if "decode" in result_by_operation and "decode_step" in result_by_operation:
        decode = result_by_operation["decode"]
        decode_step = result_by_operation["decode_step"]
        store_overhead_ms = max(0.0, decode_step["mean_ms"] - decode["mean_ms"])
        comparison.update(
            {
                "decode_step_mean_ms": decode_step["mean_ms"],
                "current_token_store_overhead_ms": store_overhead_ms,
            }
        )
        print(f"decode step: store_overhead={store_overhead_ms:.4f} ms total={decode_step['mean_ms']:.4f} ms")

    report = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": torch.npu.get_device_name(args.device),
        "software": {
            "vllm": vllm.__version__,
            "vllm_ascend": distribution_version("vllm-ascend"),
            "torch": torch.__version__,
            "torch_npu": distribution_version("torch-npu"),
            "triton_ascend": distribution_version("triton-ascend"),
        },
        "trace_dir": str(trace_dir),
        "results": results,
        "comparison": comparison,
    }
    report_path = trace_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    if not args.no_trace and args.profile_iterations:
        print(f"Profiler traces: {trace_dir}")


if __name__ == "__main__":
    main()
