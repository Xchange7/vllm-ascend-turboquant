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

"""Build one compact report from all TurboQuant E2E comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def optional_number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "N/A"


def collect(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for report_path in sorted(input_dir.glob("*/*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status_path = report_path.with_suffix(".status")
        status = None
        if status_path.exists():
            try:
                status = int(status_path.read_text(encoding="utf-8").strip())
            except ValueError:
                status = None
        pair = report_path.parent.name
        base, separator, candidate = pair.partition("_vs_")
        if not separator:
            continue
        rows.append(
            {
                "base": base,
                "candidate": candidate,
                "suite": report_path.stem,
                "status": status,
                "summary": report["summary"],
                "report": str(report_path),
            }
        )
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# TurboQuant End-to-End Correctness Summary",
        "",
        "`native -> native_repeat` is the runtime noise control. "
        "`native -> tq_reference` isolates the TurboQuant reference Triton "
        "algorithm. `tq_reference -> tq_auto` isolates optimized Triton drift. "
        "Comparisons against `tq_ascend_fused` isolate fused-path drift.",
        "",
        "## Teacher Forcing",
        "",
        "| Pair | Status | Positions | Mean abs LP | P95 abs LP | NLL delta "
        "| PPL ratio | Prompt top-1 | First top-1 | Top-k overlap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["suite"] != "teacher_forcing":
            continue
        summary = row["summary"]
        status = "PASS" if row["status"] == 0 else "FAIL"
        if row["status"] is None:
            status = "UNKNOWN"
        lines.append(
            f"| {row['base']} -> {row['candidate']} | {status} | "
            f"{summary['prompt_logprob_positions']} | "
            f"{optional_number(summary['prompt_mean_abs_logprob_diff'], 6)} | "
            f"{optional_number(summary['prompt_p95_abs_logprob_diff'], 6)} | "
            f"{optional_number(summary['prompt_mean_nll_delta'], 6)} | "
            f"{optional_number(summary['prompt_perplexity_ratio'], 6)} | "
            f"{optional_number(summary['prompt_top1_match_rate'])} | "
            f"{optional_number(summary['first_token_top1_match_rate'])} | "
            f"{optional_number(summary['first_token_topk_overlap'])} |"
        )

    lines.extend(
        [
            "",
            "## Ground-Truth And Generation",
            "",
            "| Pair | Status | Cases | Base accuracy | Candidate accuracy | "
            "Regressions | Exact text | Token prefix | First top-1 | Top-k overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["suite"] != "quality":
            continue
        summary = row["summary"]
        status = "PASS" if row["status"] == 0 else "FAIL"
        if row["status"] is None:
            status = "UNKNOWN"
        lines.append(
            f"| {row['base']} -> {row['candidate']} | {status} | "
            f"{summary['graded_cases']} | "
            f"{optional_number(summary['native_accuracy'])} | "
            f"{optional_number(summary['turboquant_accuracy'])} | "
            f"{summary['quality_regressions']} | "
            f"{optional_number(summary['exact_text_match_rate'])} | "
            f"{optional_number(summary['token_prefix_rate'])} | "
            f"{optional_number(summary['first_token_top1_match_rate'])} | "
            f"{optional_number(summary['first_token_topk_overlap'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = collect(args.input_dir)
    result = {
        "comparisons": rows,
        "failed_comparisons": sum(row["status"] not in (None, 0) for row in rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))


if __name__ == "__main__":
    main()
