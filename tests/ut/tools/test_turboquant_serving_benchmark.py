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

import argparse
import importlib.util
import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / "turboquant_triton" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_report(ttft: float, tpot: float) -> dict:
    def metric(value: float) -> dict[str, float]:
        return {"mean": value, "p50": value, "p90": value, "min": value, "max": value}

    return {
        "model": "/models/qwen3",
        "actual_input_tokens": 1024,
        "output_tokens": 128,
        "measured_requests": 10,
        "concurrency": 4,
        "prefix_caching": False,
        "summary": {
            "ttft_seconds": metric(ttft),
            "tpot_seconds": metric(tpot),
            "end_to_end_seconds": metric(2.0),
            "output_tokens_per_second": metric(100.0),
            "request_throughput": metric(4.0),
            "aggregate_output_tokens_per_second": metric(400.0),
        },
    }


def test_percentile_interpolates() -> None:
    benchmark = load_script("serving_benchmark.py")
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_stream_completion_measures_between_first_and_last_token() -> None:
    benchmark = load_script("serving_benchmark.py")
    events = [
        b'data: {"choices":[{"text":"a","logprobs":{"tokens":["a"]}}]}\n',
        b'data: {"choices":[{"text":"b","logprobs":{"tokens":["b"]}}]}\n',
        b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
        b"data: [DONE]\n",
    ]
    response = MagicMock()
    response.__enter__.return_value = events
    response.__exit__.return_value = False
    opener = MagicMock()
    opener.open.return_value = response

    with (
        patch.object(benchmark.urllib.request, "build_opener", return_value=opener),
        patch.object(
            benchmark.time,
            "perf_counter",
            side_effect=[10.0, 10.2, 10.3, 10.5],
        ),
    ):
        result = benchmark.stream_completion("http://server", {}, 1.0)

    assert result["ttft_seconds"] == pytest.approx(0.2)
    assert result["tpot_seconds"] == pytest.approx(0.1)
    assert result["end_to_end_seconds"] == pytest.approx(0.5)


def test_execute_requests_reaches_requested_concurrency(capsys) -> None:
    benchmark = load_script("serving_benchmark.py")
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_completion(*_args):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return {
            "ttft_seconds": 0.1,
            "tpot_seconds": 0.01,
            "completion_tokens": 8,
        }

    with patch.object(benchmark, "stream_completion", side_effect=fake_completion):
        samples, elapsed = benchmark.execute_requests(8, 4, "http://server", {}, 1.0, "request")

    assert max_active == 4
    assert [sample["index"] for sample in samples] == list(range(8))
    assert elapsed > 0
    output = capsys.readouterr().out
    assert "request: submitting 8 request(s) with concurrency 4" in output
    assert "mean TTFT=" in output


def test_compare_reports_calculates_effective_compression(tmp_path: Path) -> None:
    benchmark = load_script("serving_benchmark.py")
    native_path = tmp_path / "native.json"
    tq_path = tmp_path / "turboquant.json"
    native_log = tmp_path / "native.log"
    tq_log = tmp_path / "turboquant.log"
    output = tmp_path / "comparison.json"
    summary = tmp_path / "summary.md"
    native_path.write_text(json.dumps(benchmark_report(0.10, 0.01)), encoding="utf-8")
    tq_path.write_text(json.dumps(benchmark_report(0.09, 0.008)), encoding="utf-8")
    native_log.write_text(
        "Available KV cache memory: 8.00 GiB\nGPU KV cache size: 10,000 tokens\n",
        encoding="utf-8",
    )
    tq_log.write_text(
        "Available KV cache memory: 8.00 GiB\nGPU KV cache size: 40,000 tokens\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        native=native_path,
        turboquant=tq_path,
        native_log=native_log,
        turboquant_log=tq_log,
        output=output,
        summary_markdown=summary,
    )
    assert benchmark.compare_reports(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    compression = report["kv_cache_compression"]
    assert compression["capacity_ratio"] == 4.0
    assert compression["estimated_bytes_per_token_ratio"] == 0.25
    assert compression["estimated_kv_memory_reduction_percent"] == 75.0
    assert report["metrics"]["ttft_seconds"]["turboquant_change_percent"] == pytest.approx(-10.0)

    mismatched = benchmark_report(0.09, 0.008)
    mismatched["concurrency"] = 8
    tq_path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="concurrency"):
        benchmark.compare_reports(args)


def test_accuracy_context_budget_rejects_oversized_case() -> None:
    accuracy = load_script("accuracy_eval.py")
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(2040))
    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained.return_value = tokenizer
    transformers = MagicMock(AutoTokenizer=auto_tokenizer)
    prompts = [{"id": "long", "prompt": "content", "max_tokens": 16}]

    with (
        patch.dict(sys.modules, {"transformers": transformers}),
        pytest.raises(ValueError, match=r"2040 \+ max_tokens=16 = 2056"),
    ):
        accuracy.validate_context_budgets(prompts, "/models/qwen3", 2048)


def test_accuracy_evaluates_exact_and_json_answers() -> None:
    accuracy = load_script("accuracy_eval.py")
    exact = accuracy.evaluate_answer(
        "  Canberra\n",
        {"type": "exact", "expected": "canberra"},
    )
    structured = accuracy.evaluate_answer(
        '{"red": 3, "blue": 2}',
        {"type": "json_exact", "expected": {"blue": 2, "red": 3}},
    )
    assert exact["correct"] is True
    assert structured["correct"] is True


def test_accuracy_marks_native_pass_turboquant_fail_as_regression() -> None:
    accuracy = load_script("accuracy_eval.py")

    def case(answer: str, correct: bool) -> dict:
        return {
            "id": "grounded",
            "category": "hallucination_resistance",
            "evaluation": {
                "type": "exact",
                "expected": "INSUFFICIENT_INFORMATION",
            },
            "evaluation_result": {"correct": correct},
            "response": {
                "choices": [{"message": {"content": answer}}],
            },
            "error": None,
        }

    compared = accuracy.compare_case(
        case("INSUFFICIENT_INFORMATION", True),
        case("The answer is 2020", False),
    )
    assert compared["native_correct"] is True
    assert compared["turboquant_correct"] is False
    assert compared["quality_regression"] is True


def test_quality_case_file_has_valid_ground_truth() -> None:
    accuracy = load_script("accuracy_eval.py")
    cases = accuracy.load_prompts(REPO_ROOT / "scripts" / "turboquant_triton" / "quality_cases.jsonl")
    assert len(cases) >= 15
    assert {case["category"] for case in cases} >= {
        "arithmetic",
        "context_retrieval",
        "hallucination_resistance",
        "long_context",
    }
    for case in cases:
        assert case["evaluation"] is not None


def test_quality_comparison_threshold_fails_on_regression(tmp_path: Path) -> None:
    accuracy = load_script("accuracy_eval.py")

    def report(answer: str, correct: bool) -> dict:
        return {
            "cases": [
                {
                    "id": "grounded",
                    "category": "hallucination_resistance",
                    "evaluation": {
                        "type": "exact",
                        "expected": "INSUFFICIENT_INFORMATION",
                    },
                    "evaluation_result": {"correct": correct},
                    "response": {"choices": [{"message": {"content": answer}}]},
                    "error": None,
                }
            ]
        }

    native = tmp_path / "native.json"
    turboquant = tmp_path / "turboquant.json"
    output = tmp_path / "comparison.json"
    summary_markdown = tmp_path / "summary.md"
    native.write_text(
        json.dumps(report("INSUFFICIENT_INFORMATION", True)),
        encoding="utf-8",
    )
    turboquant.write_text(
        json.dumps(report("The answer is 2020", False)),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        native=native,
        turboquant=turboquant,
        output=output,
        summary_markdown=summary_markdown,
        min_exact_match_rate=0.0,
        min_token_prefix_rate=0.0,
        max_mean_logprob_diff=None,
        min_turboquant_accuracy=None,
        max_quality_regressions=0,
    )
    assert accuracy.compare(args) == 1
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["summary"]["native_accuracy"] == 1.0
    assert comparison["summary"]["turboquant_accuracy"] == 0.0
    assert comparison["summary"]["quality_regressions"] == 1
