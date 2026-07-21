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

"""Collect, grade, and compare deterministic OpenAI responses."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
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
    collect.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Write prompts and unmodified model answers to a readable text file.",
    )
    collect.add_argument("--label", required=True)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--logprobs", type=int, default=5)
    collect.add_argument(
        "--prompt-logprobs",
        type=int,
        default=0,
        help=("Request per-position prompt logprobs for teacher-forcing tests. Zero disables prompt logprobs."),
    )
    collect.add_argument("--timeout", type=float, default=600.0)
    collect.add_argument(
        "--request-mode",
        choices=("completion", "chat"),
        default="completion",
    )
    collect.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Reject cases whose prompt and generation budget exceed this limit.",
    )
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
    compare.add_argument(
        "--max-prompt-mean-abs-logprob-diff",
        type=float,
        default=None,
    )
    compare.add_argument(
        "--max-prompt-p95-abs-logprob-diff",
        type=float,
        default=None,
    )
    compare.add_argument(
        "--max-prompt-abs-mean-nll-delta",
        type=float,
        default=None,
    )
    compare.add_argument(
        "--min-first-token-top1-match-rate",
        type=float,
        default=None,
    )
    compare.add_argument(
        "--min-first-token-topk-overlap",
        type=float,
        default=None,
    )
    compare.add_argument("--min-turboquant-accuracy", type=float, default=None)
    compare.add_argument("--max-quality-regressions", type=int, default=None)
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
        category = item.get("category", "uncategorized")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{path}:{line_number}: category must be a string")
        system_prompt = item.get(
            "system_prompt",
            "Follow the user instruction exactly. Use only supplied context when the user restricts the source.",
        )
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError(f"{path}:{line_number}: system_prompt must be a string")
        evaluation = item.get("evaluation")
        if evaluation is not None and not isinstance(evaluation, dict):
            raise ValueError(f"{path}:{line_number}: evaluation must be an object")
        score_last_tokens = item.get("score_last_tokens")
        if score_last_tokens is not None and (not isinstance(score_last_tokens, int) or score_last_tokens <= 0):
            raise ValueError(f"{path}:{line_number}: score_last_tokens must be positive")
        prompts.append(
            {
                "id": case_id,
                "category": category,
                "system_prompt": system_prompt,
                "prompt": prompt * repeat + suffix,
                "max_tokens": max_tokens,
                "evaluation": evaluation,
                "score_last_tokens": score_last_tokens,
            }
        )
        seen_ids.add(case_id)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def validate_context_budgets(
    prompts: list[dict[str, Any]],
    model: str,
    max_model_len: int | None,
    request_mode: str = "completion",
) -> list[dict[str, Any]]:
    if max_model_len is None:
        return prompts
    if max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")

    # Import lazily so report comparison does not require model dependencies.
    from transformers import AutoTokenizer

    model_path = Path(model).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=model_path.exists(),
    )
    validated_prompts = []
    errors = []
    for prompt in prompts:
        if request_mode == "chat":
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["prompt"]},
            ]
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            prompt_tokens = len(token_ids)
        else:
            prompt_tokens = len(tokenizer.encode(prompt["prompt"], add_special_tokens=False))
        total_token_budget = prompt_tokens + prompt["max_tokens"]
        validated_prompts.append(
            {
                **prompt,
                "prompt_tokens": prompt_tokens,
                "total_token_budget": total_token_budget,
            }
        )
        if total_token_budget > max_model_len:
            errors.append(
                f"{prompt['id']}: prompt_tokens={prompt_tokens} + "
                f"max_tokens={prompt['max_tokens']} = {total_token_budget} > "
                f"max_model_len={max_model_len}"
            )
    if errors:
        raise ValueError("Context budget exceeded:\n" + "\n".join(errors))
    return validated_prompts


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    hostname = urllib.parse.urlparse(url).hostname
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        open_request = opener.open
    else:
        open_request = urllib.request.urlopen
    try:
        with open_request(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error


def choice_text(choice: dict[str, Any] | None) -> str | None:
    if choice is None:
        return None
    text = choice.get("text")
    if isinstance(text, str):
        return text
    message = choice.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return None


def normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().split()).casefold()


def evaluate_answer(
    answer: str | None,
    evaluation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    if answer is None:
        return {"correct": False, "reason": "missing answer"}

    evaluation_type = evaluation.get("type")
    expected = evaluation.get("expected")
    if evaluation_type == "exact":
        if not isinstance(expected, str):
            raise ValueError("exact evaluation requires a string expected value")
        correct = normalize_answer(answer) == normalize_answer(expected)
    elif evaluation_type == "one_of":
        if not isinstance(expected, list) or not all(isinstance(value, str) for value in expected):
            raise ValueError("one_of evaluation requires a string list")
        normalized_answer = normalize_answer(answer)
        correct = normalized_answer in {normalize_answer(value) for value in expected}
    elif evaluation_type == "json_exact":
        try:
            parsed_answer = json.loads(answer.strip())
        except json.JSONDecodeError:
            parsed_answer = None
        correct = parsed_answer == expected
    else:
        raise ValueError(f"Unsupported evaluation type: {evaluation_type!r}")

    return {
        "correct": correct,
        "evaluation_type": evaluation_type,
        "expected": expected,
        "normalized_answer": normalize_answer(answer),
    }


def answer_text_report(label: str, cases: list[dict[str, Any]]) -> str:
    """Render model answers without applying evaluator normalization."""
    lines = [f"TurboQuant accuracy raw answers: {label}", ""]
    for case in cases:
        evaluation_result = case.get("evaluation_result")
        if case.get("error") is not None:
            status = "ERROR"
        elif isinstance(evaluation_result, dict):
            status = "CORRECT" if evaluation_result.get("correct") else "WRONG"
        else:
            status = "NOT_GRADED"

        request = case.get("request") or {}
        prompt = request.get("prompt")
        if prompt is None:
            messages = request.get("messages") or []
            prompt = "\n".join(
                f"[{message.get('role', 'unknown')}] {message.get('content', '')}"
                for message in messages
                if isinstance(message, dict)
            )
        expected = (case.get("evaluation") or {}).get("expected")
        answer = choice_text(first_choice(case))

        lines.extend(
            [
                f"===== {case['id']} =====",
                f"category: {case.get('category', 'uncategorized')}",
                f"status: {status}",
                "prompt:",
                str(prompt or ""),
                "expected:",
                json.dumps(expected, ensure_ascii=False) if expected is not None else "<not graded>",
                "answer:",
                "<no model output>" if answer is None else answer,
            ]
        )
        if case.get("error") is not None:
            lines.extend(["error:", str(case["error"])])
        lines.extend([f"===== end {case['id']} =====", ""])
    return "\n".join(lines)


def collect(args: argparse.Namespace) -> int:
    if args.logprobs < 0:
        raise ValueError("--logprobs must be non-negative")
    if args.prompt_logprobs < 0:
        raise ValueError("--prompt-logprobs must be non-negative")
    cases = []
    failures = 0
    endpoint_path = "chat/completions" if args.request_mode == "chat" else "completions"
    endpoint = f"{args.base_url.rstrip('/')}/v1/{endpoint_path}"
    prompts = validate_context_budgets(
        load_prompts(args.prompts),
        args.model,
        args.max_model_len,
        args.request_mode,
    )
    for prompt in prompts:
        token_budget = ""
        if "prompt_tokens" in prompt:
            token_budget = (
                f" ({prompt['prompt_tokens']} prompt + {prompt['max_tokens']} output <= {args.max_model_len})"
            )
        print(f"{prompt['id']}: sending request{token_budget}", flush=True)
        payload: dict[str, Any] = {
            "model": args.model,
            "max_tokens": prompt["max_tokens"],
            "temperature": 0,
            "seed": args.seed,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }
        if args.prompt_logprobs:
            payload["prompt_logprobs"] = args.prompt_logprobs
        if args.request_mode == "chat":
            payload.update(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt["system_prompt"],
                        },
                        {"role": "user", "content": prompt["prompt"]},
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "logprobs": args.logprobs > 0,
                    "top_logprobs": args.logprobs if args.logprobs else 0,
                }
            )
        else:
            payload.update(
                {
                    "prompt": prompt["prompt"],
                    "logprobs": args.logprobs,
                }
            )
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
        answer = None
        if response is not None:
            response_choices = response.get("choices") or []
            if response_choices and isinstance(response_choices[0], dict):
                answer = choice_text(response_choices[0])
        evaluation_result = evaluate_answer(answer, prompt["evaluation"])
        cases.append(
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "evaluation": prompt["evaluation"],
                "evaluation_result": evaluation_result,
                "request": payload,
                "response": response,
                "error": error,
                "elapsed_seconds": elapsed_seconds,
                "prompt_tokens": prompt.get("prompt_tokens"),
                "total_token_budget": prompt.get("total_token_budget"),
                "score_last_tokens": prompt.get("score_last_tokens"),
            }
        )
        if error is not None:
            status = "FAIL"
        elif evaluation_result is None:
            status = "PASS"
        else:
            status = "CORRECT" if evaluation_result["correct"] else "WRONG"
        print(f"{prompt['id']}: {status} ({elapsed_seconds:.3f}s)", flush=True)

    report = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "request_mode": args.request_mode,
        "warm_prefix": args.warm_prefix,
        "max_model_len": args.max_model_len,
        "prompt_file": str(args.prompts),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text_output = args.text_output or args.output.with_name(f"{args.output.stem}_answers.txt")
    text_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.write_text(answer_text_report(args.label, cases), encoding="utf-8")
    print(f"Report: {args.output}")
    print(f"Raw answers: {text_output}")
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


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def generated_token_trace(choice: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize completion and chat logprobs into one token trace."""
    logprobs = choice.get("logprobs") or {}
    completion_tokens = logprobs.get("tokens")
    if isinstance(completion_tokens, list):
        token_logprobs = logprobs.get("token_logprobs") or []
        top_logprobs = logprobs.get("top_logprobs") or []
        return [
            {
                "token": token,
                "logprob": finite_number(token_logprobs[index] if index < len(token_logprobs) else None),
                "top_logprobs": (
                    top_logprobs[index] if index < len(top_logprobs) and isinstance(top_logprobs[index], dict) else {}
                ),
            }
            for index, token in enumerate(completion_tokens)
        ]

    content = logprobs.get("content")
    if not isinstance(content, list):
        return []
    trace = []
    for item in content:
        if not isinstance(item, dict):
            continue
        top = {}
        for candidate in item.get("top_logprobs") or []:
            if not isinstance(candidate, dict):
                continue
            token = candidate.get("token")
            logprob = finite_number(candidate.get("logprob"))
            if isinstance(token, str) and logprob is not None:
                top[token] = logprob
        trace.append(
            {
                "token": item.get("token"),
                "logprob": finite_number(item.get("logprob")),
                "top_logprobs": top,
            }
        )
    return trace


def first_token_distribution(choice: dict[str, Any]) -> dict[str, float]:
    trace = generated_token_trace(choice)
    if not trace:
        return {}
    return {
        str(token): float(logprob)
        for token, logprob in trace[0]["top_logprobs"].items()
        if finite_number(logprob) is not None
    }


def prompt_token_trace(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract scored target-token logprobs from a vLLM response."""
    response = case.get("response")
    choice = first_choice(case)
    if not isinstance(response, dict) or choice is None:
        return []

    prompt_token_ids = response.get("prompt_token_ids")
    prompt_logprobs = response.get("prompt_logprobs")
    if not isinstance(prompt_token_ids, list):
        prompt_token_ids = choice.get("prompt_token_ids")
    if not isinstance(prompt_logprobs, list):
        prompt_logprobs = choice.get("prompt_logprobs")
    if not isinstance(prompt_token_ids, list) or not isinstance(prompt_logprobs, list):
        return []
    if len(prompt_token_ids) != len(prompt_logprobs):
        return []

    count = len(prompt_token_ids)
    score_last_tokens = case.get("score_last_tokens")
    if not isinstance(score_last_tokens, int):
        score_last_tokens = max(count - 1, 0)
    start = max(1, count - score_last_tokens)
    trace = []
    for position in range(start, count):
        target_token_id = prompt_token_ids[position]
        candidates = prompt_logprobs[position]
        if not isinstance(target_token_id, int) or not isinstance(candidates, dict):
            continue
        target = candidates.get(str(target_token_id), candidates.get(target_token_id))
        if not isinstance(target, dict):
            continue
        logprob = finite_number(target.get("logprob"))
        if logprob is None:
            continue
        top1_token_id = None
        for candidate_token_id, candidate in candidates.items():
            if isinstance(candidate, dict) and candidate.get("rank") == 1:
                try:
                    top1_token_id = int(candidate_token_id)
                except (TypeError, ValueError):
                    top1_token_id = None
                break
        trace.append(
            {
                "position": position,
                "target_token_id": target_token_id,
                "logprob": logprob,
                "rank": target.get("rank"),
                "top1_token_id": top1_token_id,
            }
        )
    expected_positions = max(count - start, 0)
    return trace if len(trace) == expected_positions else []


def compare_prompt_traces(
    native: dict[str, Any],
    turboquant: dict[str, Any],
) -> dict[str, Any]:
    native_trace = prompt_token_trace(native)
    tq_trace = prompt_token_trace(turboquant)
    if not native_trace and not tq_trace:
        return {"available": False}
    if len(native_trace) != len(tq_trace):
        return {
            "available": True,
            "valid": False,
            "reason": "different scored prompt lengths",
            "native_positions": len(native_trace),
            "turboquant_positions": len(tq_trace),
        }

    abs_diffs = []
    native_logprobs = []
    tq_logprobs = []
    top1_matches = 0
    rank_diffs = []
    for native_item, tq_item in zip(native_trace, tq_trace):
        if native_item["target_token_id"] != tq_item["target_token_id"]:
            return {
                "available": True,
                "valid": False,
                "reason": "different prompt token IDs",
                "position": native_item["position"],
            }
        native_lp = native_item["logprob"]
        tq_lp = tq_item["logprob"]
        native_logprobs.append(native_lp)
        tq_logprobs.append(tq_lp)
        abs_diffs.append(abs(native_lp - tq_lp))
        top1_matches += (
            native_item["top1_token_id"] is not None and native_item["top1_token_id"] == tq_item["top1_token_id"]
        )
        native_rank = native_item.get("rank")
        tq_rank = tq_item.get("rank")
        if isinstance(native_rank, int) and isinstance(tq_rank, int):
            rank_diffs.append(abs(native_rank - tq_rank))

    if not abs_diffs:
        return {
            "available": True,
            "valid": False,
            "reason": "no finite target-token logprobs",
        }
    native_nll = -statistics.fmean(native_logprobs)
    tq_nll = -statistics.fmean(tq_logprobs)
    nll_delta = tq_nll - native_nll
    return {
        "available": True,
        "valid": True,
        "positions": len(abs_diffs),
        "native_mean_nll": native_nll,
        "turboquant_mean_nll": tq_nll,
        "mean_nll_delta": nll_delta,
        "perplexity_ratio": math.exp(max(min(nll_delta, 50.0), -50.0)),
        "mean_abs_logprob_diff": statistics.fmean(abs_diffs),
        "p95_abs_logprob_diff": percentile(abs_diffs, 0.95),
        "max_abs_logprob_diff": max(abs_diffs),
        "top1_match_rate": top1_matches / len(abs_diffs),
        "mean_target_rank_diff": (statistics.fmean(rank_diffs) if rank_diffs else None),
        "abs_logprob_diffs": abs_diffs,
    }


def compare_case(native: dict[str, Any], turboquant: dict[str, Any]) -> dict[str, Any]:
    native_choice = first_choice(native)
    tq_choice = first_choice(turboquant)
    native_evaluation = native.get("evaluation_result")
    tq_evaluation = turboquant.get("evaluation_result")
    native_correct = native_evaluation.get("correct") if isinstance(native_evaluation, dict) else None
    tq_correct = tq_evaluation.get("correct") if isinstance(tq_evaluation, dict) else None
    if native_choice is None or tq_choice is None:
        return {
            "id": native["id"],
            "category": native.get("category", "uncategorized"),
            "valid": False,
            "native_correct": native_correct,
            "turboquant_correct": tq_correct,
            "native_error": native.get("error"),
            "turboquant_error": turboquant.get("error"),
        }

    native_trace = generated_token_trace(native_choice)
    tq_trace = generated_token_trace(tq_choice)
    native_tokens = [item["token"] for item in native_trace]
    tq_tokens = [item["token"] for item in tq_trace]
    prefix_tokens = sequence_prefix_length(native_tokens, tq_tokens)
    denominator = max(len(native_tokens), len(tq_tokens), 1)

    logprob_diffs = []
    for index in range(prefix_tokens):
        native_lp = finite_number(native_trace[index]["logprob"])
        tq_lp = finite_number(tq_trace[index]["logprob"])
        if native_lp is not None and tq_lp is not None:
            logprob_diffs.append(abs(native_lp - tq_lp))

    native_first_token = first_token_distribution(native_choice)
    tq_first_token = first_token_distribution(tq_choice)
    first_token_comparison: dict[str, Any] = {"available": False}
    if native_first_token and tq_first_token:
        native_top1 = max(native_first_token, key=native_first_token.get)
        tq_top1 = max(tq_first_token, key=tq_first_token.get)
        native_tokens_set = set(native_first_token)
        tq_tokens_set = set(tq_first_token)
        union = native_tokens_set | tq_tokens_set
        common = native_tokens_set & tq_tokens_set
        common_diffs = [abs(native_first_token[token] - tq_first_token[token]) for token in common]
        first_token_comparison = {
            "available": True,
            "native_top1": native_top1,
            "turboquant_top1": tq_top1,
            "top1_match": native_top1 == tq_top1,
            "topk_overlap": len(common) / max(len(union), 1),
            "common_tokens": len(common),
            "union_tokens": len(union),
            "mean_common_abs_logprob_diff": (statistics.fmean(common_diffs) if common_diffs else None),
            "max_common_abs_logprob_diff": (max(common_diffs) if common_diffs else None),
        }

    prompt_comparison = compare_prompt_traces(native, turboquant)

    return {
        "id": native["id"],
        "category": native.get("category", "uncategorized"),
        "valid": True,
        "exact_text_match": choice_text(native_choice) == choice_text(tq_choice),
        "native_text": choice_text(native_choice),
        "turboquant_text": choice_text(tq_choice),
        "native_correct": native_correct,
        "turboquant_correct": tq_correct,
        "quality_regression": native_correct is True and tq_correct is False,
        "both_wrong": native_correct is False and tq_correct is False,
        "expected": (native.get("evaluation") or {}).get("expected"),
        "native_token_count": len(native_tokens),
        "turboquant_token_count": len(tq_tokens),
        "common_prefix_tokens": prefix_tokens,
        "token_prefix_rate": prefix_tokens / denominator,
        "mean_common_token_logprob_diff": (statistics.fmean(logprob_diffs) if logprob_diffs else None),
        "max_common_token_logprob_diff": max(logprob_diffs) if logprob_diffs else None,
        "first_token": first_token_comparison,
        "prompt_logprobs": prompt_comparison,
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
    ]
    if summary["graded_cases"]:
        lines.extend(
            [
                f"- Graded cases: {summary['graded_cases']}",
                f"- Native accuracy: {summary['native_accuracy']:.4f}",
                f"- TurboQuant accuracy: {summary['turboquant_accuracy']:.4f}",
                f"- Quality regressions: {summary['quality_regressions']}",
                f"- Both wrong: {summary['both_wrong']}",
            ]
        )
    if summary["prompt_logprob_cases"]:
        lines.extend(
            [
                f"- Teacher-forced positions: {summary['prompt_logprob_positions']}",
                f"- Teacher-forced mean absolute logprob difference: {summary['prompt_mean_abs_logprob_diff']:.6f}",
                f"- Teacher-forced p95 absolute logprob difference: {summary['prompt_p95_abs_logprob_diff']:.6f}",
                f"- Native mean NLL: {summary['native_prompt_mean_nll']:.6f}",
                f"- TurboQuant mean NLL: {summary['turboquant_prompt_mean_nll']:.6f}",
                f"- Mean NLL delta: {summary['prompt_mean_nll_delta']:+.6f}",
                f"- Perplexity ratio: {summary['prompt_perplexity_ratio']:.6f}",
                f"- Teacher-forced top-1 match rate: {summary['prompt_top1_match_rate']:.4f}",
            ]
        )
    if summary["first_token_distribution_cases"]:
        lines.extend(
            [
                f"- First-token top-1 match rate: {summary['first_token_top1_match_rate']:.4f}",
                f"- Mean first-token top-k overlap: {summary['first_token_topk_overlap']:.4f}",
            ]
        )
    lines.extend(
        [
            f"- Exact text match rate: {summary['exact_text_match_rate']:.4f}",
            f"- Aggregate token prefix rate: {summary['token_prefix_rate']:.4f}",
            f"- Mean common-token logprob difference: {summary['mean_logprob_diff']}",
            f"- Max common-token logprob difference: {summary['max_logprob_diff']}",
            "",
            "| Case | Category | Valid | Native correct | TQ correct | Regression | Exact | Prefix rate |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for case in report["cases"]:
        lines.append(
            "| {id} | {category} | {valid} | {native_correct} | "
            "{tq_correct} | {regression} | {exact} | {prefix:.4f} |".format(
                id=case["id"],
                category=case["category"],
                valid=case["valid"],
                native_correct=case.get("native_correct", "N/A"),
                tq_correct=case.get("turboquant_correct", "N/A"),
                regression=case.get("quality_regression", "N/A"),
                exact=case.get("exact_text_match", "N/A"),
                prefix=case.get("token_prefix_rate", 0.0),
            )
        )
    if summary["category_accuracy"]:
        lines.extend(
            [
                "",
                "## Category Accuracy",
                "",
                "| Category | Cases | Native | TurboQuant | Regressions |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for category, metrics in summary["category_accuracy"].items():
            lines.append(
                f"| {category} | {metrics['cases']} | "
                f"{metrics['native_accuracy']:.4f} | "
                f"{metrics['turboquant_accuracy']:.4f} | "
                f"{metrics['quality_regressions']} |"
            )
    if summary["prompt_logprob_cases"]:
        lines.extend(
            [
                "",
                "## Teacher-Forcing Drift",
                "",
                "| Case | Positions | Mean abs diff | P95 abs diff | NLL delta | PPL ratio | Top-1 match |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in report["cases"]:
            prompt = case.get("prompt_logprobs") or {}
            if not prompt.get("valid"):
                continue
            lines.append(
                f"| {case['id']} | {prompt['positions']} | "
                f"{prompt['mean_abs_logprob_diff']:.6f} | "
                f"{prompt['p95_abs_logprob_diff']:.6f} | "
                f"{prompt['mean_nll_delta']:+.6f} | "
                f"{prompt['perplexity_ratio']:.6f} | "
                f"{prompt['top1_match_rate']:.4f} |"
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
    graded_cases = [
        case
        for case in valid_cases
        if case.get("native_correct") is not None and case.get("turboquant_correct") is not None
    ]
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
    prompt_cases = [case["prompt_logprobs"] for case in valid_cases if case.get("prompt_logprobs", {}).get("valid")]
    prompt_failures = sum(
        case.get("prompt_logprobs", {}).get("available", False)
        and not case.get("prompt_logprobs", {}).get("valid", False)
        for case in valid_cases
    )
    prompt_abs_diffs = [difference for prompt in prompt_cases for difference in prompt["abs_logprob_diffs"]]
    prompt_positions = sum(prompt["positions"] for prompt in prompt_cases)
    native_prompt_nll_sum = sum(prompt["native_mean_nll"] * prompt["positions"] for prompt in prompt_cases)
    tq_prompt_nll_sum = sum(prompt["turboquant_mean_nll"] * prompt["positions"] for prompt in prompt_cases)
    native_prompt_mean_nll = native_prompt_nll_sum / prompt_positions if prompt_positions else None
    tq_prompt_mean_nll = tq_prompt_nll_sum / prompt_positions if prompt_positions else None
    prompt_mean_nll_delta = (
        tq_prompt_mean_nll - native_prompt_mean_nll
        if tq_prompt_mean_nll is not None and native_prompt_mean_nll is not None
        else None
    )
    prompt_top1_matches = sum(prompt["top1_match_rate"] * prompt["positions"] for prompt in prompt_cases)
    distribution_cases = [case["first_token"] for case in valid_cases if case.get("first_token", {}).get("available")]
    category_accuracy = {}
    for category in sorted({case["category"] for case in graded_cases}):
        category_cases = [case for case in graded_cases if case["category"] == category]
        category_accuracy[category] = {
            "cases": len(category_cases),
            "native_accuracy": sum(case["native_correct"] for case in category_cases) / len(category_cases),
            "turboquant_accuracy": sum(case["turboquant_correct"] for case in category_cases) / len(category_cases),
            "quality_regressions": sum(case["quality_regression"] for case in category_cases),
        }
    native_correct = sum(case["native_correct"] for case in graded_cases)
    tq_correct = sum(case["turboquant_correct"] for case in graded_cases)
    quality_regressions = sum(case["quality_regression"] for case in graded_cases)
    both_wrong = sum(case["both_wrong"] for case in graded_cases)
    summary = {
        "total_cases": len(cases),
        "valid_cases": len(valid_cases),
        "request_failures": len(cases) - len(valid_cases),
        "graded_cases": len(graded_cases),
        "native_correct": native_correct,
        "turboquant_correct": tq_correct,
        "native_accuracy": native_correct / max(len(graded_cases), 1),
        "turboquant_accuracy": tq_correct / max(len(graded_cases), 1),
        "quality_regressions": quality_regressions,
        "both_wrong": both_wrong,
        "category_accuracy": category_accuracy,
        "exact_text_matches": exact_matches,
        "exact_text_match_rate": exact_matches / max(len(valid_cases), 1),
        "common_prefix_tokens": total_prefix,
        "compared_tokens": total_tokens,
        "token_prefix_rate": total_prefix / max(total_tokens, 1),
        "mean_logprob_diff": statistics.fmean(logprob_diffs) if logprob_diffs else None,
        "max_logprob_diff": max(max_logprob_diffs) if max_logprob_diffs else None,
        "prompt_logprob_cases": len(prompt_cases),
        "prompt_logprob_failures": prompt_failures,
        "prompt_logprob_positions": prompt_positions,
        "prompt_mean_abs_logprob_diff": (statistics.fmean(prompt_abs_diffs) if prompt_abs_diffs else None),
        "prompt_p95_abs_logprob_diff": percentile(prompt_abs_diffs, 0.95),
        "prompt_max_abs_logprob_diff": (max(prompt_abs_diffs) if prompt_abs_diffs else None),
        "native_prompt_mean_nll": native_prompt_mean_nll,
        "turboquant_prompt_mean_nll": tq_prompt_mean_nll,
        "prompt_mean_nll_delta": prompt_mean_nll_delta,
        "prompt_perplexity_ratio": (
            math.exp(max(min(prompt_mean_nll_delta, 50.0), -50.0)) if prompt_mean_nll_delta is not None else None
        ),
        "prompt_top1_match_rate": (prompt_top1_matches / prompt_positions if prompt_positions else None),
        "first_token_distribution_cases": len(distribution_cases),
        "first_token_top1_match_rate": (
            sum(distribution["top1_match"] for distribution in distribution_cases) / len(distribution_cases)
            if distribution_cases
            else None
        ),
        "first_token_topk_overlap": (
            statistics.fmean(distribution["topk_overlap"] for distribution in distribution_cases)
            if distribution_cases
            else None
        ),
    }
    report = {"summary": summary, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary_markdown.write_text(markdown_summary(report), encoding="utf-8")

    failures = []
    if summary["request_failures"]:
        failures.append(f"{summary['request_failures']} request(s) failed")
    if summary["prompt_logprob_failures"]:
        failures.append(f"{summary['prompt_logprob_failures']} prompt logprob comparison(s) failed")
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
    prompt_thresholds = (
        (
            "max_prompt_mean_abs_logprob_diff",
            "prompt_mean_abs_logprob_diff",
            "teacher-forced mean absolute logprob difference",
        ),
        (
            "max_prompt_p95_abs_logprob_diff",
            "prompt_p95_abs_logprob_diff",
            "teacher-forced p95 absolute logprob difference",
        ),
    )
    for argument, metric, description in prompt_thresholds:
        threshold = getattr(args, argument, None)
        value = summary[metric]
        if threshold is not None and value is None:
            failures.append(f"{description} is unavailable")
        elif threshold is not None and value > threshold:
            failures.append(f"{description} is above threshold")
    max_abs_nll_delta = getattr(args, "max_prompt_abs_mean_nll_delta", None)
    if max_abs_nll_delta is not None and summary["prompt_mean_nll_delta"] is None:
        failures.append("teacher-forced mean NLL delta is unavailable")
    elif max_abs_nll_delta is not None and abs(summary["prompt_mean_nll_delta"]) > max_abs_nll_delta:
        failures.append("absolute teacher-forced mean NLL delta is above threshold")
    distribution_thresholds = (
        (
            "min_first_token_top1_match_rate",
            "first_token_top1_match_rate",
            "first-token top-1 match rate",
        ),
        (
            "min_first_token_topk_overlap",
            "first_token_topk_overlap",
            "first-token top-k overlap",
        ),
    )
    for argument, metric, description in distribution_thresholds:
        threshold = getattr(args, argument, None)
        value = summary[metric]
        if threshold is not None and value is None:
            failures.append(f"{description} is unavailable")
        elif threshold is not None and value < threshold:
            failures.append(f"{description} is below threshold")
    if (
        args.min_turboquant_accuracy is not None
        and summary["graded_cases"]
        and summary["turboquant_accuracy"] < args.min_turboquant_accuracy
    ):
        failures.append("TurboQuant ground-truth accuracy is below threshold")
    if args.max_quality_regressions is not None and summary["quality_regressions"] > args.max_quality_regressions:
        failures.append("quality regression count is above threshold")

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
