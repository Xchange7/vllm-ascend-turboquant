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

"""Measure serving TTFT/TPOT and compare native and TurboQuant KV caches."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

KV_CAPACITY_PATTERN = re.compile(
    r"(?:GPU|NPU) KV cache size:\s*([\d,]+)\s*tokens",
    re.IGNORECASE,
)
KV_MEMORY_PATTERN = re.compile(
    r"Available KV cache memory:\s*([\d.]+)\s*GiB",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Benchmark one running server.")
    run.add_argument("--base-url", default="http://127.0.0.1:18001")
    run.add_argument("--model", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--cache-dtype", required=True)
    run.add_argument("--input-tokens", type=int, default=1024)
    run.add_argument("--output-tokens", type=int, default=128)
    run.add_argument("--max-model-len", type=int, default=2048)
    run.add_argument("--warmup-requests", type=int, default=2)
    run.add_argument("--requests", type=int, default=10)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="Compare two benchmark reports and server logs.")
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--turboquant", type=Path, required=True)
    compare.add_argument("--native-log", type=Path, required=True)
    compare.add_argument("--turboquant-log", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--summary-markdown", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def build_prompt(model: str, target_tokens: int) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("--input-tokens must be positive")

    from transformers import AutoTokenizer

    model_path = Path(model).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=model_path.exists(),
    )
    seed = "TurboQuant serving benchmark measures prefill and decode latency with prefix caching disabled. "
    seed_tokens = tokenizer.encode(seed, add_special_tokens=False)
    repeats = math.ceil(target_tokens / len(seed_tokens))
    prompt = tokenizer.decode((seed_tokens * repeats)[:target_tokens])
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual_tokens


def stream_completion(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_tokens = 0
    observed_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                logprobs = choice.get("logprobs") or {}
                chunk_tokens = len(logprobs.get("tokens") or [])
                if chunk_tokens == 0 and choice.get("text"):
                    chunk_tokens = 1
                if chunk_tokens:
                    token_arrival = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = token_arrival
                    last_token_at = token_arrival
                    observed_tokens += chunk_tokens
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error

    finished = time.perf_counter()
    if first_token_at is None or last_token_at is None:
        raise RuntimeError("The streaming response did not contain an output token")
    completion_tokens = completion_tokens or observed_tokens
    if completion_tokens < 2:
        raise RuntimeError(f"TPOT requires at least two output tokens; got {completion_tokens}")
    ttft = first_token_at - started
    end_to_end = finished - started
    decode_seconds = last_token_at - first_token_at
    if decode_seconds <= 0:
        raise RuntimeError("Cannot measure TPOT because tokens arrived in one chunk")
    tpot = decode_seconds / (completion_tokens - 1)
    return {
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "end_to_end_seconds": end_to_end,
        "completion_tokens": completion_tokens,
        "output_tokens_per_second": (completion_tokens - 1) / decode_seconds,
    }


def run_benchmark(args: argparse.Namespace) -> int:
    if args.requests <= 0 or args.warmup_requests < 0:
        raise ValueError("Request count must be positive and warmup count non-negative")
    prompt, actual_input_tokens = build_prompt(args.model, args.input_tokens)
    total_budget = actual_input_tokens + args.output_tokens
    if total_budget > args.max_model_len:
        raise ValueError(
            f"input_tokens={actual_input_tokens} + output_tokens={args.output_tokens} "
            f"exceeds max_model_len={args.max_model_len}"
        )

    endpoint = f"{args.base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "logprobs": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for index in range(args.warmup_requests):
        print(f"warmup {index + 1}/{args.warmup_requests}", flush=True)
        stream_completion(endpoint, payload, args.timeout)

    samples = []
    for index in range(args.requests):
        sample = stream_completion(endpoint, payload, args.timeout)
        sample["index"] = index
        samples.append(sample)
        print(
            f"request {index + 1}/{args.requests}: "
            f"TTFT={sample['ttft_seconds'] * 1000:.2f} ms, "
            f"TPOT={sample['tpot_seconds'] * 1000:.2f} ms",
            flush=True,
        )

    report = {
        "label": args.label,
        "cache_dtype": args.cache_dtype,
        "model": args.model,
        "prefix_caching": False,
        "requested_input_tokens": args.input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "output_tokens": args.output_tokens,
        "max_model_len": args.max_model_len,
        "warmup_requests": args.warmup_requests,
        "measured_requests": args.requests,
        "samples": samples,
        "summary": {
            "ttft_seconds": summarize([item["ttft_seconds"] for item in samples]),
            "tpot_seconds": summarize([item["tpot_seconds"] for item in samples]),
            "end_to_end_seconds": summarize([item["end_to_end_seconds"] for item in samples]),
            "output_tokens_per_second": summarize([item["output_tokens_per_second"] for item in samples]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.output}")
    return 0


def last_match(path: Path, pattern: re.Pattern[str], cast: type) -> Any | None:
    matches = pattern.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        return None
    return cast(matches[-1].replace(",", ""))


def relative_change(native: float, turboquant: float) -> float:
    return (turboquant / native - 1.0) * 100.0


def compare_reports(args: argparse.Namespace) -> int:
    native = json.loads(args.native.read_text(encoding="utf-8"))
    turboquant = json.loads(args.turboquant.read_text(encoding="utf-8"))
    comparable_fields = (
        "model",
        "actual_input_tokens",
        "output_tokens",
        "measured_requests",
        "prefix_caching",
    )
    for field in comparable_fields:
        if native[field] != turboquant[field]:
            raise ValueError(f"Benchmark field differs between reports: {field}")

    native_capacity = last_match(args.native_log, KV_CAPACITY_PATTERN, int)
    tq_capacity = last_match(args.turboquant_log, KV_CAPACITY_PATTERN, int)
    native_memory_gib = last_match(args.native_log, KV_MEMORY_PATTERN, float)
    tq_memory_gib = last_match(args.turboquant_log, KV_MEMORY_PATTERN, float)
    compression = None
    if native_capacity is not None and tq_capacity is not None:
        capacity_ratio = tq_capacity / native_capacity
        compression = {
            "native_capacity_tokens": native_capacity,
            "turboquant_capacity_tokens": tq_capacity,
            "capacity_ratio": capacity_ratio,
            "estimated_bytes_per_token_ratio": 1.0 / capacity_ratio,
            "estimated_kv_memory_reduction_percent": (1.0 - 1.0 / capacity_ratio) * 100.0,
        }

    metrics = {}
    for metric in (
        "ttft_seconds",
        "tpot_seconds",
        "end_to_end_seconds",
        "output_tokens_per_second",
    ):
        native_mean = native["summary"][metric]["mean"]
        tq_mean = turboquant["summary"][metric]["mean"]
        metrics[metric] = {
            "native_mean": native_mean,
            "turboquant_mean": tq_mean,
            "turboquant_change_percent": relative_change(native_mean, tq_mean),
        }

    report = {
        "configuration": {field: native[field] for field in comparable_fields},
        "available_kv_cache_memory_gib": {
            "native": native_memory_gib,
            "turboquant": tq_memory_gib,
        },
        "kv_cache_compression": compression,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# TurboQuant Serving Benchmark",
        "",
        f"- Model: `{native['model']}`",
        f"- Input/output tokens: {native['actual_input_tokens']}/{native['output_tokens']}",
        f"- Measured requests: {native['measured_requests']}",
        "- Prefix caching: disabled",
        "",
        "| Metric | Native | TurboQuant | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "ttft_seconds": "Mean TTFT (ms)",
        "tpot_seconds": "Mean TPOT (ms)",
        "end_to_end_seconds": "Mean E2E (ms)",
        "output_tokens_per_second": "Output throughput (token/s)",
    }
    for metric, label in labels.items():
        item = metrics[metric]
        scale = 1.0 if metric == "output_tokens_per_second" else 1000.0
        lines.append(
            f"| {label} | {item['native_mean'] * scale:.3f} | "
            f"{item['turboquant_mean'] * scale:.3f} | "
            f"{item['turboquant_change_percent']:+.2f}% |"
        )
    lines.extend(["", "## KV Cache Capacity", ""])
    if compression is None:
        lines.append(
            "The server logs did not contain `GPU/NPU KV cache size: ... tokens`; inspect both server logs manually."
        )
    else:
        lines.extend(
            [
                f"- Native capacity: {native_capacity:,} tokens",
                f"- TurboQuant capacity: {tq_capacity:,} tokens",
                f"- Capacity ratio: {compression['capacity_ratio']:.4f}x",
                f"- Estimated bytes/token ratio: {compression['estimated_bytes_per_token_ratio']:.4f}",
                f"- Estimated KV memory reduction: {compression['estimated_kv_memory_reduction_percent']:.2f}%",
            ]
        )
    lines.append("")
    args.summary_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Summary: {args.summary_markdown}")
    return 0


def main() -> None:
    args = parse_args()
    status = run_benchmark(args) if args.command == "run" else compare_reports(args)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
