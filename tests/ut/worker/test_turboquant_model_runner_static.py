# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from pathlib import Path


def test_turboquant_int8_cache_allocator_is_defined():
    source = Path("vllm_ascend/worker/model_runner_v1.py").read_text()

    assert "def _allocate_int8_cache_tensor(" in source
    assert "self._allocate_int8_cache_tensor(" in source
