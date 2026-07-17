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
        "prefix_caching": False,
        "summary": {
            "ttft_seconds": metric(ttft),
            "tpot_seconds": metric(tpot),
            "end_to_end_seconds": metric(2.0),
            "output_tokens_per_second": metric(100.0),
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
