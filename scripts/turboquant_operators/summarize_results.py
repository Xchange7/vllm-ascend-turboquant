#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Summarize TurboQuant operator accuracy and performance reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def relative_case(path: Path, root: Path) -> str:
    return str(path.parent.relative_to(root))


def load_accuracy(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("accuracy.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report["configuration"]
        checks = report["checks"]
        quantization = report["quantization_error"]
        rows.append(
            {
                "case": relative_case(path, root),
                "cache_dtype": config["cache_dtype"],
                "activation_dtype": config["activation_dtype"],
                "sequence_lengths": ",".join(map(str, config["sequence_lengths"])),
                "passed": report["passed"],
                "negative_slot_mapping": checks["negative_slot_mapping_preserves_cache"],
                "key_impl_max_abs": checks["ascend_key_vs_triton"]["max_abs"],
                "value_impl_max_abs": checks["ascend_value_vs_triton"]["max_abs"],
                "attention_max_abs": checks["ascend_fia_vs_packed_decode"]["max_abs"],
                "key_cpu_max_abs": checks["triton_key_vs_cpu_reference"]["max_abs"],
                "value_cpu_max_abs": checks["triton_value_vs_cpu_reference"]["max_abs"],
                "attention_cpu_max_abs": checks["packed_decode_vs_cpu_reference"]["max_abs"],
                "key_quant_nmse": quantization["rotated_key"]["nmse"],
                "value_quant_nmse": quantization["value"]["nmse"],
                "compression_ratio": report["layout"]["capacity_compression_ratio"],
                "report": str(path),
            }
        )
    return rows


def load_benchmarks(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    comparisons = []
    for path in sorted(root.rglob("benchmark.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report["configuration"]
        common = {
            "case": relative_case(path, root),
            "cache_dtype": config["cache_dtype"],
            "activation_dtype": config["activation_dtype"],
            "batch_size": int(config["batch_size"]),
            "sequence_length": int(config["sequence_length"]),
            "report": str(path),
        }
        for result in report["results"]:
            rows.append(
                {
                    **common,
                    "operation": result["operation"],
                    "mean_ms": result["mean_ms"],
                    "p50_ms": result["p50_ms"],
                    "p90_ms": result["p90_ms"],
                    "p99_ms": result["p99_ms"],
                    "throughput_name": result["throughput_name"],
                    "throughput": result["throughput"],
                    "peak_memory_increase_bytes": result["peak_memory_increase_bytes"],
                }
            )
        comparisons.append({**common, **report.get("comparison", {})})
    return rows, comparisons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(
    accuracy: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> str:
    lines = ["# TurboQuant Operator Summary", ""]
    if accuracy:
        lines.extend(
            [
                "## Correctness",
                "",
                "| Case | Cache | Dtype | Pass | K CPU max | V CPU max | "
                "Attention CPU max | K impl max | V impl max | Attention impl max | "
                "K NMSE | V NMSE | Compression |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in accuracy:
            lines.append(
                "| {case} | {cache_dtype} | {activation_dtype} | {passed} | "
                "{key_cpu_max_abs:.3e} | {value_cpu_max_abs:.3e} | "
                "{attention_cpu_max_abs:.3e} | "
                "{key_impl_max_abs:.3e} | {value_impl_max_abs:.3e} | "
                "{attention_max_abs:.3e} | {key_quant_nmse:.3e} | "
                "{value_quant_nmse:.3e} | {compression_ratio:.3f}x |".format(**row)
            )
        lines.append("")
    if benchmarks:
        lines.extend(
            [
                "## Latency",
                "",
                "| Case | Operation | Batch | Context | Mean ms | P99 ms | Throughput | Peak delta MiB |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in benchmarks:
            lines.append(
                "| {case} | {operation} | {batch_size} | {sequence_length} | "
                "{mean_ms:.4f} | {p99_ms:.4f} | {throughput:.2f} | {peak:.2f} |".format(
                    **row,
                    peak=row["peak_memory_increase_bytes"] / 2**20,
                )
            )
        lines.append("")
    comparable = [row for row in comparisons if "decode_speedup_vs_native" in row]
    if comparable:
        lines.extend(
            [
                "## Comparison",
                "",
                "Speedup greater than 1.0 means TurboQuant is faster than the named baseline.",
                "",
                "| Case | Packed/native | Ascend/Triton dequant | Ascend/packed decode | Compression |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparable:
            lines.append(
                "| {case} | {packed:.3f}x | {dequant:.3f}x | {decode:.3f}x | {compression:.3f}x |".format(
                    case=row["case"],
                    packed=row["decode_speedup_vs_native"],
                    dequant=row.get("ascend_fused_dequant_speedup_vs_triton", float("nan")),
                    decode=row.get("ascend_fused_speedup_vs_packed_decode", float("nan")),
                    compression=row["theoretical_cache_compression_ratio"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    accuracy = load_accuracy(args.result_root)
    benchmarks, comparisons = load_benchmarks(args.result_root)
    if not accuracy and not benchmarks:
        raise RuntimeError(f"No operator reports found under {args.result_root}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_json = args.output_prefix.with_suffix(".json")
    output_markdown = args.output_prefix.with_suffix(".md")
    output_json.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "benchmarks": benchmarks,
                "comparisons": comparisons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_markdown.write_text(
        markdown(accuracy, benchmarks, comparisons),
        encoding="utf-8",
    )
    write_csv(args.output_prefix.with_name(f"{args.output_prefix.name}_accuracy.csv"), accuracy)
    write_csv(args.output_prefix.with_name(f"{args.output_prefix.name}_benchmarks.csv"), benchmarks)
    print(output_markdown.read_text(encoding="utf-8"))
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_markdown}")


if __name__ == "__main__":
    main()
