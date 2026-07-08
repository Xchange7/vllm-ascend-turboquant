# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

from vllm_ascend.kv_cache.higgs import (
    build_hadamard,
    dequantize_higgs_vector,
    get_higgs_centroids,
    quantize_higgs_vector,
)


def test_higgs_centroids_are_sorted():
    centroids = get_higgs_centroids(8, 3)

    assert centroids.shape == (8,)
    assert centroids.dtype == torch.float32
    assert torch.all(centroids[1:] > centroids[:-1])


def test_higgs_hadamard_is_orthonormal():
    hadamard = build_hadamard(8)
    identity = hadamard @ hadamard.T

    assert torch.allclose(identity, torch.eye(8), atol=1e-6)


def test_higgs_quant_dequant_roundtrip_contract():
    tensor = torch.randn(3, 2, 8, dtype=torch.float16)

    for bits in (2, 3, 4):
        quantized, norm = quantize_higgs_vector(tensor, bits)
        restored = dequantize_higgs_vector(
            quantized,
            norm,
            bits,
            norm_correction=True,
            dtype=torch.float16,
        )

        assert quantized.shape == tensor.shape
        assert quantized.dtype == torch.uint8
        assert norm.shape == (3, 2, 1)
        assert norm.dtype == torch.float16
        assert restored.shape == tensor.shape
        assert restored.dtype == torch.float16
        assert restored.abs().sum() > 0


def test_higgs_rejects_non_power_of_two_head_dim():
    with pytest.raises(ValueError, match="power-of-two"):
        build_hadamard(7)

    with pytest.raises(ValueError, match="power-of-two"):
        quantize_higgs_vector(torch.randn(1, 7, dtype=torch.float16), 3)
