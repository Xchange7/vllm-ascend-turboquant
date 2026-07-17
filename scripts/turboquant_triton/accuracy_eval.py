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

"""Collect and compare deterministic OpenAI completion responses."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect or compare native and TurboQuant completion results.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--base-url", default="http://127.0.0.1:8000")
    collect.add_argument("--model", required=True)
    collect.add_argument("--prompts", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--label", required=True)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--logprobs", type=int, default=5)
    collect.add_argument("--timeout", type=float, default=600.0)
    collect.add_argument(
        "--warm-prefix",
        action="store_true",
        help="Send each prompt once before the recorded request to exercise prefix reuse.",
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--turboquant", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--summary-markdown", type=Path, required=True)
    compare.add_argument("--min-exact-match-rate", type=float, default=0.0)
    compare.add_argument("--min-token-prefix-rate", type=float, default=0.0)
    compare.add_argument("--max-mean-logprob-diff", type=float, default=None)
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number}: missing non-empty string id")
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {case_id!r}")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{path}:{line_number}: missing non-empty prompt")
        repeat = item.get("repeat", 1)
        if not isinstance(repeat, int) or repeat <= 0:
            raise ValueError(f"{path}:{line_number}: repeat must be positive")
        suffix = item.get("suffix", "")
        if not isinstance(suffix, str):
            raise ValueError(f"{path}:{line_number}: suffix must be a string")
        max_tokens = item.get("max_tokens", 64)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"{path}:{line_number}: max_tokens must be positive")
        prompts.append(
            {
                "id": case_id,
                "prompt": prompt * repeat + suffix,
                "max_tokens": max_tokens,
            }
        )
        seen_ids.add(case_id)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error


def collect(args: argparse.Namespace) -> int:
    cases = []
    failures = 0
    endpoint = f"{args.base_url.rstrip('/')}/v1/completions"
    for prompt in load_prompts(args.prompts):
        payload = {
            "model": args.model,
            "prompt": prompt["prompt"],
            "max_tokens": prompt["max_tokens"],
            "temperature": 0,
            "seed": args.seed,
            "logprobs": args.logprobs,
        }
        started = time.perf_counter()
        try:
            if args.warm_prefix:
                post_json(endpoint, payload, args.timeout)
            response = post_json(endpoint, payload, args.timeout)
            error = None
        except Exception as exception:  # Preserve all cases for offline review.
            response = None
            error = f"{type(exception).__name__}: {exception}"
            failures += 1
        elapsed_seconds = time.perf_counter() - started
        cases.append(
            {
                "id": prompt["id"],
                "request": payload,
                "response": response,
                "error": error,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        status = "PASS" if error is None else "FAIL"
        print(f"{prompt['id']}: {status} ({elapsed_seconds:.3f}s)")

    report = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "warm_prefix": args.warm_prefix,
        "prompt_file": str(args.prompts),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {args.output}")
    return 1 if failures else 0


def first_choice(case: dict[str, Any]) -> dict[str, Any] | None:
    response = case.get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    return choices[0] if isinstance(choices[0], dict) else None


def sequence_prefix_length(left: list[Any], right: list[Any]) -> int:
    prefix = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        prefix += 1
    return prefix


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def compare_case(native: dict[str, Any], turboquant: dict[str, Any]) -> dict[str, Any]:
    native_choice = first_choice(native)
    tq_choice = first_choice(turboquant)
    if native_choice is None or tq_choice is None:
        return {
            "id": native["id"],
            "valid": False,
            "native_error": native.get("error"),
            "turboquant_error": turboquant.get("error"),
        }

    native_logprobs = native_choice.get("logprobs") or {}
    tq_logprobs = tq_choice.get("logprobs") or {}
    native_tokens = native_logprobs.get("tokens") or []
    tq_tokens = tq_logprobs.get("tokens") or []
    prefix_tokens = sequence_prefix_length(native_tokens, tq_tokens)
    denominator = max(len(native_tokens), len(tq_tokens), 1)

    native_token_lps = native_logprobs.get("token_logprobs") or []
    tq_token_lps = tq_logprobs.get("token_logprobs") or []
    logprob_diffs = []
    for index in range(prefix_tokens):
        if index >= len(native_token_lps) or index >= len(tq_token_lps):
            break
        native_lp = finite_number(native_token_lps[index])
        tq_lp = finite_number(tq_token_lps[index])
        if native_lp is not None and tq_lp is not None:
            logprob_diffs.append(abs(native_lp - tq_lp))

    return {
        "id": native["id"],
        "valid": True,
        "exact_text_match": native_choice.get("text") == tq_choice.get("text"),
        "native_text": native_choice.get("text"),
        "turboquant_text": tq_choice.get("text"),
        "native_token_count": len(native_tokens),
        "turboquant_token_count": len(tq_tokens),
        "common_prefix_tokens": prefix_tokens,
        "token_prefix_rate": prefix_tokens / denominator,
        "mean_common_token_logprob_diff": (statistics.fmean(logprob_diffs) if logprob_diffs else None),
        "max_common_token_logprob_diff": max(logprob_diffs) if logprob_diffs else None,
        "native_elapsed_seconds": native.get("elapsed_seconds"),
        "turboquant_elapsed_seconds": turboquant.get("elapsed_seconds"),
    }


def markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# TurboQuant Accuracy Comparison",
        "",
        f"- Cases: {summary['total_cases']}",
        f"- Valid cases: {summary['valid_cases']}",
        f"- Exact text match rate: {summary['exact_text_match_rate']:.4f}",
        f"- Aggregate token prefix rate: {summary['token_prefix_rate']:.4f}",
        f"- Mean common-token logprob difference: {summary['mean_logprob_diff']}",
        f"- Max common-token logprob difference: {summary['max_logprob_diff']}",
        "",
        "| Case | Valid | Exact | Prefix rate | Mean logprob diff |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for case in report["cases"]:
        mean_diff = case.get("mean_common_token_logprob_diff")
        lines.append(
            "| {id} | {valid} | {exact} | {prefix:.4f} | {diff} |".format(
                id=case["id"],
                valid=case["valid"],
                exact=case.get("exact_text_match", "N/A"),
                prefix=case.get("token_prefix_rate", 0.0),
                diff="N/A" if mean_diff is None else f"{mean_diff:.6f}",
            )
        )
    lines.append("")
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> int:
    native_report = json.loads(args.native.read_text(encoding="utf-8"))
    tq_report = json.loads(args.turboquant.read_text(encoding="utf-8"))
    native_by_id = {case["id"]: case for case in native_report["cases"]}
    tq_by_id = {case["id"]: case for case in tq_report["cases"]}
    if native_by_id.keys() != tq_by_id.keys():
        raise ValueError("Native and TurboQuant reports contain different case IDs.")

    cases = [compare_case(native_by_id[case_id], tq_by_id[case_id]) for case_id in native_by_id]
    valid_cases = [case for case in cases if case["valid"]]
    exact_matches = sum(case["exact_text_match"] for case in valid_cases)
    total_prefix = sum(case["common_prefix_tokens"] for case in valid_cases)
    total_tokens = sum(max(case["native_token_count"], case["turboquant_token_count"]) for case in valid_cases)
    logprob_diffs = [
        case["mean_common_token_logprob_diff"]
        for case in valid_cases
        if case["mean_common_token_logprob_diff"] is not None
    ]
    max_logprob_diffs = [
        case["max_common_token_logprob_diff"]
        for case in valid_cases
        if case["max_common_token_logprob_diff"] is not None
    ]
    summary = {
        "total_cases": len(cases),
        "valid_cases": len(valid_cases),
        "request_failures": len(cases) - len(valid_cases),
        "exact_text_matches": exact_matches,
        "exact_text_match_rate": exact_matches / max(len(valid_cases), 1),
        "common_prefix_tokens": total_prefix,
        "compared_tokens": total_tokens,
        "token_prefix_rate": total_prefix / max(total_tokens, 1),
        "mean_logprob_diff": statistics.fmean(logprob_diffs) if logprob_diffs else None,
        "max_logprob_diff": max(max_logprob_diffs) if max_logprob_diffs else None,
    }
    report = {"summary": summary, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary_markdown.write_text(markdown_summary(report), encoding="utf-8")

    failures = []
    if summary["request_failures"]:
        failures.append(f"{summary['request_failures']} request(s) failed")
    if summary["exact_text_match_rate"] < args.min_exact_match_rate:
        failures.append("exact text match rate is below threshold")
    if summary["token_prefix_rate"] < args.min_token_prefix_rate:
        failures.append("token prefix rate is below threshold")
    if (
        args.max_mean_logprob_diff is not None
        and summary["mean_logprob_diff"] is not None
        and summary["mean_logprob_diff"] > args.max_mean_logprob_diff
    ):
        failures.append("mean logprob difference is above threshold")

    print(json.dumps(summary, indent=2))
    print(f"Report: {args.output}")
    print(f"Summary: {args.summary_markdown}")
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    return 0


def main() -> None:
    args = parse_args()
    status = collect(args) if args.command == "collect" else compare(args)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
