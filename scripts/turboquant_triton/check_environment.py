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

import importlib.metadata
import inspect
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton  # type: ignore[import-untyped]
import vllm
from packaging.version import Version
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.v1.kv_cache_interface import TQFullAttentionSpec

import vllm_ascend
from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionBackend


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "editable/unknown"


def _source_commit(module_file: str) -> str:
    repository = Path(module_file).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    if Version(vllm.__version__).release[:3] != (0, 20, 2):
        raise RuntimeError(f"This branch requires vLLM 0.20.2 or 0.20.2+empty; found {vllm.__version__}.")
    if not torch.npu.is_available():
        raise RuntimeError("torch-npu cannot see an Ascend NPU.")

    config = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", 128)
    spec = TQFullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        tq_slot_size=config.slot_size_aligned,
    )
    expected_page_bytes = 128 * 2 * config.slot_size_aligned
    if spec.page_size_bytes != expected_page_bytes:
        raise RuntimeError(
            "vLLM TQFullAttentionSpec has an incompatible page-size contract: "
            f"{spec.page_size_bytes} != {expected_page_bytes}."
        )

    init_source = inspect.getsource(Attention._init_turboquant_buffers)
    required_workspaces = (
        "_tq_mid_o_buf",
        "_tq_output_buf",
        "_tq_lse_buf",
    )
    missing_workspaces = [name for name in required_workspaces if name not in init_source]
    if missing_workspaces:
        raise RuntimeError(
            "The loaded vLLM Attention implementation is missing TurboQuant "
            f"workspaces: {', '.join(missing_workspaces)}."
        )

    print(f"vLLM:          {vllm.__version__}")
    print(f"vLLM source:   {vllm.__file__}")
    print(f"vLLM Ascend:   {_version('vllm-ascend')}")
    print(f"Ascend source: {vllm_ascend.__file__}")
    print(f"Ascend commit: {_source_commit(vllm_ascend.__file__)}")
    print(f"torch:         {torch.__version__}")
    print(f"torch-npu:     {_version('torch-npu')}")
    print(f"Triton:        {triton.__version__}")
    print(f"Triton Ascend: {_version('triton-ascend')}")
    print(f"NPU:           {torch.npu.get_device_name(0)}")
    print(f"Backend:       {AscendTurboQuantAttentionBackend.get_name()}")
    print(f"TQ slot bytes: {config.slot_size_aligned}")
    print(f"TQ page bytes: {spec.page_size_bytes}")
    print("TQ core API:    compatible")


if __name__ == "__main__":
    main()
