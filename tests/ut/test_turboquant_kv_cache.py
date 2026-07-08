# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from vllm_ascend.kv_cache.turboquant import (
    get_turboquant_config,
    get_turboquant_kv_cache_shape,
    is_turboquant_kv_cache_dtype,
    turboquant_dequant_cache,
    turboquant_store_kv,
)


def test_turboquant_dtype_and_shape():
    assert is_turboquant_kv_cache_dtype("turboquant_4bit_nc")
    assert not is_turboquant_kv_cache_dtype("bfloat16")

    assert get_turboquant_kv_cache_shape(
        num_blocks=2,
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        cache_dtype_str="turboquant_4bit_nc",
    ) == (2, 128, 8, 134)


def test_turboquant_store_dequant_roundtrip_contract():
    head_dim = 8
    for cache_dtype in (
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ):
        tq_config = get_turboquant_config(cache_dtype, head_dim)
        kv_cache = torch.zeros(
            (2, 4, 2, tq_config.slot_size_aligned),
            dtype=torch.uint8,
        )
        key = torch.randn(3, 2, head_dim, dtype=torch.float16)
        value = torch.randn(3, 2, head_dim, dtype=torch.float16)
        slot_mapping = torch.tensor([0, 3, 5], dtype=torch.int64)

        turboquant_store_kv(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            tq_config=tq_config,
        )
        key_cache, value_cache = turboquant_dequant_cache(
            kv_cache,
            tq_config,
            torch.float16,
        )

        assert key_cache.shape == (2, 4, 2, head_dim)
        assert value_cache.shape == (2, 4, 2, head_dim)
        assert key_cache.dtype == torch.float16
        assert value_cache.dtype == torch.float16
        assert key_cache[0, 0].abs().sum() > 0
        assert value_cache[1, 1].abs().sum() > 0
