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

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = REPO_ROOT / "scripts" / "turboquant_operators" / "turboquant_reference.py"
QUALITY_GATE_PATH = REPO_ROOT / "scripts" / "turboquant_operators" / "operator_quality_gate.py"


def load_reference():
    spec = importlib.util.spec_from_file_location("turboquant_operator_reference", REFERENCE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_quality_gate():
    spec = importlib.util.spec_from_file_location("turboquant_operator_quality_gate", QUALITY_GATE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack_indices(indices: list[int], bits: int) -> list[int]:
    packed = [0] * math.ceil(len(indices) * bits / 8)
    for coordinate, index in enumerate(indices):
        bit_offset = coordinate * bits
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        value = index << shift
        packed[byte_index] |= value & 0xFF
        if byte_index + 1 < len(packed):
            packed[byte_index + 1] |= (value >> 8) & 0xFF
    return packed


def fp16_bytes(value: float) -> list[int]:
    return torch.tensor([value], dtype=torch.float16).view(torch.uint8).tolist()


@pytest.mark.parametrize(("key_bits", "value_bits"), [(4, 4), (3, 3), (3, 4)])
def test_cpu_reference_dequantizes_permuted_pages(key_bits: int, value_bits: int) -> None:
    reference = load_reference()
    head_dim = 8
    key_data_bytes = math.ceil(head_dim * key_bits / 8)
    key_packed_size = key_data_bytes + 2
    value_data_bytes = math.ceil(head_dim * value_bits / 8)
    slot_size = key_packed_size + value_data_bytes + 4
    cache = torch.zeros(2, 2, 1, slot_size, dtype=torch.uint8)
    block_table = torch.tensor([[1]], dtype=torch.int32)
    centroids = torch.linspace(-0.5, 0.5, 2**key_bits)

    expected_keys = []
    expected_values = []
    for position in range(2):
        key_indices = [(position + coordinate) % (2**key_bits) for coordinate in range(head_dim)]
        value_indices = [(2 * position + coordinate) % (2**value_bits) for coordinate in range(head_dim)]
        key_norm = 1.5 + position
        value_scale = 0.25 + position * 0.125
        value_minimum = -1.0 + position * 0.5
        slot = (
            pack_indices(key_indices, key_bits)
            + fp16_bytes(key_norm)
            + pack_indices(value_indices, value_bits)
            + fp16_bytes(value_scale)
            + fp16_bytes(value_minimum)
        )
        cache[1, position, 0] = torch.tensor(slot, dtype=torch.uint8)
        expected_keys.append(centroids[key_indices] * key_norm)
        expected_values.append(torch.tensor(value_indices).float() * value_scale + value_minimum)

    key, value = reference.dequantize_paged_cache_reference(
        cache,
        block_table,
        [2],
        centroids,
        head_dim=head_dim,
        key_bits=key_bits,
        key_packed_size=key_packed_size,
        value_bits=value_bits,
        norm_correction=False,
    )

    torch.testing.assert_close(key[:, 0], torch.stack(expected_keys))
    torch.testing.assert_close(value[:, 0], torch.stack(expected_values))


def test_cpu_reference_attention_matches_torch_sdpa() -> None:
    reference = load_reference()
    torch.manual_seed(3)
    batch_size = 2
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 8
    seq_lens = [3, 2]
    query = torch.randn(batch_size, num_query_heads, head_dim)
    key = torch.randn(sum(seq_lens), num_kv_heads, head_dim)
    value = torch.randn_like(key)
    rotation, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
    scale = 1 / math.sqrt(head_dim)

    actual = reference.decode_attention_reference(
        query,
        key,
        value,
        seq_lens,
        rotation,
        scale=scale,
    )

    expected = []
    token_start = 0
    kv_head_indices = torch.arange(num_query_heads) // (num_query_heads // num_kv_heads)
    for request_index, seq_len in enumerate(seq_lens):
        token_end = token_start + seq_len
        request_query = (query[request_index] @ rotation).view(1, num_query_heads, 1, head_dim)
        request_key = key[token_start:token_end, kv_head_indices].permute(1, 0, 2).unsqueeze(0)
        request_value = value[token_start:token_end, kv_head_indices].permute(1, 0, 2).unsqueeze(0)
        expected.append(
            F.scaled_dot_product_attention(
                request_query,
                request_key,
                request_value,
                scale=scale,
            )[0, :, 0]
        )
        token_start = token_end

    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("cache_dtype", "key_nmse", "value_nmse"),
    [
        ("turboquant_4bit_nc", 0.01, 0.01),
        ("turboquant_k3v4_nc", 0.04, 0.01),
        ("turboquant_3bit_nc", 0.04, 0.05),
    ],
)
def test_quantization_quality_gate_accepts_expected_error(
    cache_dtype: str,
    key_nmse: float,
    value_nmse: float,
) -> None:
    quality_gate = load_quality_gate()
    result = quality_gate.evaluate_quantization_quality(
        {"nmse": key_nmse, "cosine_similarity": 0.99},
        {"nmse": value_nmse, "cosine_similarity": 0.99},
        quality_gate.resolve_thresholds(cache_dtype),
    )

    assert result["passed"]
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("key_nmse", "key_cosine"),
    [(0.50, 0.99), (0.01, 0.50), (float("nan"), 0.99)],
)
def test_quantization_quality_gate_rejects_corruption(
    key_nmse: float,
    key_cosine: float,
) -> None:
    quality_gate = load_quality_gate()
    result = quality_gate.evaluate_quantization_quality(
        {"nmse": key_nmse, "cosine_similarity": key_cosine},
        {"nmse": 0.01, "cosine_similarity": 0.99},
        quality_gate.resolve_thresholds("turboquant_4bit_nc"),
    )

    assert not result["passed"]


def test_quantization_quality_thresholds_can_be_overridden() -> None:
    quality_gate = load_quality_gate()
    thresholds = quality_gate.resolve_thresholds(
        "turboquant_4bit_nc",
        max_key_nmse=0.5,
        min_value_cosine=0.5,
    )

    assert thresholds.max_key_nmse == 0.5
    assert thresholds.max_value_nmse == 0.03
    assert thresholds.min_key_cosine == 0.98
    assert thresholds.min_value_cosine == 0.5


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_key_nmse": -0.1}, "NMSE"),
        ({"max_value_nmse": float("inf")}, "NMSE"),
        ({"min_key_cosine": 1.1}, "Cosine"),
        ({"min_value_cosine": float("nan")}, "Cosine"),
    ],
)
def test_quantization_quality_rejects_invalid_thresholds(
    override: dict[str, float],
    message: str,
) -> None:
    quality_gate = load_quality_gate()

    with pytest.raises(ValueError, match=message):
        quality_gate.resolve_thresholds("turboquant_4bit_nc", **override)
