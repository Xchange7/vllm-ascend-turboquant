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

"""Summarize TurboQuant kernel benchmark reports into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_rows(profile_root: Path) -> list[dict[str, Any]]:
    rows = []
    for report_path in sorted(profile_root.rglob("benchmark.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report["configuration"]
        comparison = report.get("comparison")
        if not comparison:
            continue
        rows.append(
            {
                "case": str(report_path.parent.relative_to(profile_root)),
                "cache_dtype": config["cache_dtype"],
                "activation_dtype": config["activation_dtype"],
                "batch_size": int(config["batch_size"]),
                "sequence_length": int(config["sequence_length"]),
                "num_query_heads": int(config["num_query_heads"]),
                "num_kv_heads": int(config["num_kv_heads"]),
                "head_dim": int(config["head_dim"]),
                "num_kv_splits": int(config["num_kv_splits"]),
                "turboquant_mean_ms": comparison["turboquant_mean_ms"],
                "native_mean_ms": comparison["native_mean_ms"],
                "decode_speedup_vs_native": comparison["decode_speedup_vs_native"],
                "theoretical_cache_compression_ratio": comparison["theoretical_cache_compression_ratio"],
                "report": str(report_path),
            }
        )
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# TurboQuant Kernel Performance Summary",
        "",
        "Speedup greater than 1.0 means packed TurboQuant decode is faster.",
        "",
        "| Case | Cache | Dtype | Batch | Context | Splits | TQ ms | Native ms | Speedup |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {cache_dtype} | {activation_dtype} | {batch_size} | "
            "{sequence_length} | {num_kv_splits} | {turboquant_mean_ms:.4f} | "
            "{native_mean_ms:.4f} | {decode_speedup_vs_native:.3f}x |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.profile_root)
    if not rows:
        raise RuntimeError(f"No benchmark comparison found under {args.profile_root}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
