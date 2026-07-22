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

import math
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)

import vllm_ascend.ops.triton.turboquant_decode as turboquant_decode_module
import vllm_ascend.ops.triton.turboquant_store as turboquant_store_module
from tests.ut.conftest import npu_test
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.attention.turboquant import (
    AscendTurboQuantAttentionImpl,
    _build_hadamard,
    _build_turboquant_page_table_cpu,
    _TurboQuantFusedWorkspace,
)
from vllm_ascend.kv_cache.turboquant import get_turboquant_config
from vllm_ascend.ops.triton.turboquant_decode import (
    _ASCEND_MAX_TRITON_GRID_SIZE,
    _dequant_launch_ranges,
    _supports_grouped_gqa,
    select_turboquant_num_kv_splits,
    triton_turboquant_decode_attention,
    triton_turboquant_dequant_paged_cache,
)
from vllm_ascend.ops.triton.turboquant_store import (
    ASCEND_MAX_TRITON_GRID_SIZE,
    _store_launch_token_ranges,
    triton_turboquant_store,
)
from vllm_ascend.ops.turboquant import (
    turboquant_paged_dequant,
    turboquant_paged_dequant_out,
)

_DECODE_CORRECTNESS_CASES = [
    (cache_dtype, query_len, torch.float16, 128, False, False, None, "auto")
    for cache_dtype in (
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    )
    for query_len in (1, 2, 4, 8)
] + [
    ("turboquant_4bit_nc", 4, torch.bfloat16, 128, False, False, None, "auto"),
    ("turboquant_k3v4_nc", 4, torch.bfloat16, 128, False, False, None, "auto"),
    ("turboquant_3bit_nc", 4, torch.bfloat16, 128, False, False, None, "auto"),
    ("turboquant_4bit_nc", 1, torch.float16, 64, False, False, None, "auto"),
    ("turboquant_4bit_nc", 1, torch.float16, 256, False, False, None, "reference"),
    ("turboquant_4bit_nc", 4, torch.float16, 128, True, False, None, "auto"),
    ("turboquant_4bit_nc", 2, torch.float16, 128, False, True, 3.0, "auto"),
]


def _constants(cache_dtype: str, head_dim: int):
    config = get_turboquant_config(cache_dtype, head_dim)
    hadamard = _build_hadamard(head_dim, "npu:0")
    centroids = get_centroids(head_dim, config.centroid_bits).to(
        device="npu",
        dtype=torch.float32,
    )
    centroids, _ = centroids.sort()
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    return config, hadamard, centroids, midpoints


def _build_fused_dequant_inputs(
    cache_dtype: str,
    activation_dtype: torch.dtype,
    *,
    head_dim: int = 128,
):
    torch.manual_seed(7)
    batch_size = 2
    num_kv_heads = 2
    seq_lens_list = [133, 65]
    max_seq_len = max(seq_lens_list)
    config, hadamard, centroids, midpoints = _constants(cache_dtype, head_dim)
    block_table = torch.tensor(
        [[2, 0], [3, 1]],
        dtype=torch.int32,
        device="npu",
    )
    cache = torch.zeros(
        4,
        128,
        num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device="npu",
    )
    key_parts = []
    value_parts = []
    slot_parts = []
    for request_index, sequence_length in enumerate(seq_lens_list):
        positions = torch.arange(sequence_length, dtype=torch.int64, device="npu")
        physical_blocks = block_table[request_index, positions // 128]
        slot_parts.append(physical_blocks.to(torch.int64) * 128 + positions % 128)
        key_parts.append(
            torch.randn(
                sequence_length,
                num_kv_heads,
                head_dim,
                dtype=activation_dtype,
                device="npu",
            )
        )
        value_parts.append(torch.randn_like(key_parts[-1]))

    key = torch.cat(key_parts)
    value = torch.cat(value_parts)
    rotated_key_workspace = torch.empty(
        key.shape[0] * num_kv_heads,
        head_dim,
        dtype=activation_dtype,
        device="npu",
    )
    triton_turboquant_store(
        key,
        value,
        cache,
        torch.cat(slot_parts),
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=hadamard.to(activation_dtype),
        rotated_key_out=rotated_key_workspace,
    )
    torch.npu.synchronize()
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="npu")
    query = torch.randn(
        batch_size,
        16,
        head_dim,
        dtype=activation_dtype,
        device="npu",
    )
    return (
        config,
        hadamard,
        centroids,
        cache,
        block_table,
        seq_lens,
        seq_lens_list,
        max_seq_len,
        query,
    )


@npu_test(num_npus=1, npu_type="a2")
@pytest.mark.parametrize(
    ("cache_dtype", "activation_dtype", "head_dim", "norm_correction"),
    [
        ("turboquant_4bit_nc", torch.float16, 32, True),
        ("turboquant_4bit_nc", torch.float16, 64, True),
        ("turboquant_4bit_nc", torch.float16, 128, True),
        ("turboquant_4bit_nc", torch.float16, 128, False),
        ("turboquant_4bit_nc", torch.float16, 256, True),
        ("turboquant_4bit_nc", torch.bfloat16, 128, True),
        ("turboquant_k3v4_nc", torch.float16, 128, True),
        ("turboquant_3bit_nc", torch.float16, 128, True),
    ],
)
def test_turboquant_ascend_fused_dequant_matches_triton(
    cache_dtype,
    activation_dtype,
    head_dim,
    norm_correction,
):
    (
        config,
        _,
        centroids,
        cache,
        block_table,
        seq_lens,
        seq_lens_list,
        max_seq_len,
        query,
    ) = _build_fused_dequant_inputs(cache_dtype, activation_dtype, head_dim=head_dim)
    page_table = _build_turboquant_page_table_cpu(seq_lens_list, cache.shape[1]).to(query.device)
    key_bnsd, value_bnsd = turboquant_paged_dequant(
        query,
        cache,
        block_table,
        seq_lens,
        page_table,
        centroids,
        max_seq_len=max_seq_len,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=norm_correction,
    )

    total_tokens = sum(seq_lens_list)
    key_reference = torch.empty(
        total_tokens,
        cache.shape[2],
        query.shape[-1],
        dtype=activation_dtype,
        device="npu",
    )
    value_reference = torch.empty_like(key_reference)
    triton_turboquant_dequant_paged_cache(
        cache,
        block_table,
        seq_lens,
        torch.tensor(
            [0, seq_lens_list[0], total_tokens],
            dtype=torch.int32,
            device="npu",
        ),
        centroids,
        key_reference,
        value_reference,
        max_seq_len=max_seq_len,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=norm_correction,
    )
    torch.npu.synchronize()

    key_dense = torch.cat([key_bnsd[i, :, :seq_len].permute(1, 0, 2) for i, seq_len in enumerate(seq_lens_list)])
    value_dense = torch.cat([value_bnsd[i, :, :seq_len].permute(1, 0, 2) for i, seq_len in enumerate(seq_lens_list)])
    tolerance = 2e-2 if activation_dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(key_dense, key_reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(value_dense, value_reference, atol=tolerance, rtol=tolerance)

    key_out = torch.empty_like(key_bnsd)
    value_out = torch.empty_like(value_bnsd)
    returned_key, returned_value = turboquant_paged_dequant_out(
        query,
        cache,
        block_table,
        seq_lens,
        page_table,
        centroids,
        key_out,
        value_out,
        max_seq_len=max_seq_len,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=norm_correction,
    )
    assert returned_key.data_ptr() == key_out.data_ptr()
    assert returned_value.data_ptr() == value_out.data_ptr()
    key_out_dense = torch.cat([key_out[i, :, :seq_len].permute(1, 0, 2) for i, seq_len in enumerate(seq_lens_list)])
    value_out_dense = torch.cat([value_out[i, :, :seq_len].permute(1, 0, 2) for i, seq_len in enumerate(seq_lens_list)])
    torch.testing.assert_close(key_out_dense, key_dense)
    torch.testing.assert_close(value_out_dense, value_dense)


@npu_test(num_npus=1, npu_type="a2")
def test_turboquant_ascend_fused_decode_matches_packed_decode():
    (
        config,
        hadamard,
        centroids,
        cache,
        block_table,
        seq_lens,
        seq_lens_list,
        max_seq_len,
        query,
    ) = _build_fused_dequant_inputs("turboquant_4bit_nc", torch.float16)
    layer = SimpleNamespace(
        _tq_ascend_centroids=centroids,
        _tq_ascend_compute_rotation=hadamard.to(query.dtype),
    )
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_kv_heads = cache.shape[2]
    impl.num_heads = query.shape[1]
    impl.scale = 1 / math.sqrt(query.shape[-1])
    impl.tq_config = config
    page_table = _build_turboquant_page_table_cpu(seq_lens_list, cache.shape[1]).to(query.device)
    fused_output = impl._run_ascend_fused_decode(
        layer,
        query,
        cache,
        block_table,
        seq_lens,
        page_table,
        seq_lens_list,
        torch.empty_like(query),
        max_seq_len,
        _TurboQuantFusedWorkspace(),
    )
    packed_output = triton_turboquant_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        hadamard,
        centroids,
        scale=1 / math.sqrt(query.shape[-1]),
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
        max_num_kv_splits=4,
        buffer_holder=SimpleNamespace(),
        compute_rotation=hadamard.to(query.dtype),
        implementation="auto",
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        fused_output,
        packed_output,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim", "expected"),
    [
        (16, 2, 128, True),
        (8, 2, 64, True),
        (4, 2, 128, False),
        (64, 1, 128, False),
        (16, 2, 256, False),
    ],
)
def test_turboquant_grouped_gqa_dispatch(
    num_query_heads,
    num_kv_heads,
    head_dim,
    expected,
):
    assert _supports_grouped_gqa(num_query_heads, num_kv_heads, head_dim) is expected


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "implementation", "expected"),
    [
        (1, 16384, "auto", 32),
        (2, 16384, "auto", 32),
        (4, 16384, "auto", 32),
        (8, 16384, "auto", 32),
        (16, 16384, "auto", 32),
        (20, 16384, "auto", 32),
        (16, 16384, "reference", 32),
        (20, 512, "auto", 4),
        (20, 4096, "auto", 8),
        (1, 1024, "auto", 32),
        (1, 128, "auto", 32),
        (1, 8, "auto", 8),
        (1, 1, "auto", 1),
    ],
)
def test_turboquant_split_selection_for_qwen3_tp4(
    batch_size,
    sequence_length,
    implementation,
    expected,
):
    assert (
        select_turboquant_num_kv_splits(
            batch_size=batch_size,
            num_query_heads=16,
            num_kv_heads=2,
            head_dim=128,
            max_num_kv_splits=32,
            max_sequence_length=sequence_length,
            implementation=implementation,
        )
        == expected
    )


def test_turboquant_split_selection_without_host_sequence_length():
    assert (
        select_turboquant_num_kv_splits(
            batch_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            head_dim=128,
            max_num_kv_splits=32,
            max_sequence_length=None,
        )
        == 4
    )


def test_turboquant_graph_split_selection_ignores_padded_batch_size():
    assert (
        select_turboquant_num_kv_splits(
            batch_size=20,
            num_query_heads=16,
            num_kv_heads=2,
            head_dim=128,
            max_num_kv_splits=32,
            max_sequence_length=40960,
            use_static_graph_splits=True,
        )
        == 32
    )


def test_turboquant_graph_split_selection_respects_ascend_grid_limit():
    splits = select_turboquant_num_kv_splits(
        batch_size=2048,
        num_query_heads=16,
        num_kv_heads=2,
        head_dim=128,
        max_num_kv_splits=32,
        max_sequence_length=40960,
        use_static_graph_splits=True,
    )

    assert splits == 8
    assert 2048 * 2 * splits <= 65535


def test_turboquant_split_selection_rejects_oversized_base_grid():
    with pytest.raises(ValueError, match="grid exceeds the Ascend launch limit"):
        select_turboquant_num_kv_splits(
            batch_size=32768,
            num_query_heads=16,
            num_kv_heads=2,
            head_dim=128,
            max_num_kv_splits=32,
            max_sequence_length=40960,
        )


def test_turboquant_store_launch_ranges_respect_ascend_grid_limit():
    ranges = _store_launch_token_ranges(65536, 2)

    assert ranges == [(0, 32767), (32767, 65534), (65534, 65536)]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 65536
    assert all((token_end - token_start) * 2 <= ASCEND_MAX_TRITON_GRID_SIZE for token_start, token_end in ranges)


def test_turboquant_dequant_launch_ranges_respect_ascend_grid_limit():
    ranges = _dequant_launch_ranges(131072)

    assert ranges == [(0, 65535), (65535, 131070), (131070, 131072)]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 131072
    assert all(program_end - program_start <= _ASCEND_MAX_TRITON_GRID_SIZE for program_start, program_end in ranges)


@npu_test(num_npus=1, npu_type="a2")
def test_turboquant_negative_slot_mapping_does_not_write_cache():
    torch.manual_seed(0)
    head_dim = 128
    config, hadamard, _, midpoints = _constants(
        "turboquant_4bit_nc",
        head_dim,
    )
    key = torch.randn(2, 2, head_dim, dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    cache = torch.zeros(
        1,
        128,
        2,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device="npu",
    )
    slot_mapping = torch.full((2,), -1, dtype=torch.int64, device="npu")

    triton_turboquant_store(
        key,
        value,
        cache,
        slot_mapping,
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
    )

    assert torch.count_nonzero(cache.cpu()) == 0


@npu_test(num_npus=1, npu_type="a2")
@pytest.mark.parametrize(
    ("cache_dtype", "head_dim"),
    [
        pytest.param("turboquant_4bit_nc", 64, id="4bit-d64"),
        pytest.param("turboquant_4bit_nc", 128, id="4bit-d128"),
        pytest.param("turboquant_k3v4_nc", 128, id="k3v4-d128"),
        pytest.param("turboquant_3bit_nc", 128, id="3bit-d128"),
    ],
)
def test_turboquant_store_writes_valid_slots(cache_dtype, head_dim):
    """Compile and execute only the positive-slot store path."""
    torch.manual_seed(1)
    config, hadamard, _, midpoints = _constants(cache_dtype, head_dim)
    key = torch.randn(2, 2, head_dim, dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    cache = torch.zeros(
        2,
        128,
        2,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device="npu",
    )
    slot_mapping = torch.tensor([0, 129], dtype=torch.int64, device="npu")

    triton_turboquant_store(
        key,
        value,
        cache,
        slot_mapping,
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
    )
    torch.npu.synchronize()

    assert torch.count_nonzero(cache[0, 0].cpu()) > 0
    assert torch.count_nonzero(cache[1, 1].cpu()) > 0


@npu_test(num_npus=1, npu_type="a2")
def test_turboquant_chunked_store_matches_single_launch(monkeypatch):
    torch.manual_seed(2)
    head_dim = 128
    num_tokens = 5
    num_kv_heads = 2
    config, hadamard, _, midpoints = _constants(
        "turboquant_4bit_nc",
        head_dim,
    )
    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="npu",
    )
    value = torch.randn_like(key)
    reference_cache = torch.zeros(
        1,
        128,
        num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device="npu",
    )
    chunked_cache = torch.zeros_like(reference_cache)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    store_kwargs = {
        "key_bits": config.key_quant_bits,
        "key_packed_size": config.key_packed_size,
        "value_bits": config.value_quant_bits,
    }

    triton_turboquant_store(
        key,
        value,
        reference_cache,
        slot_mapping,
        hadamard,
        midpoints,
        **store_kwargs,
    )
    monkeypatch.setattr(
        turboquant_store_module,
        "ASCEND_MAX_TRITON_GRID_SIZE",
        4,
    )
    triton_turboquant_store(
        key,
        value,
        chunked_cache,
        slot_mapping,
        hadamard,
        midpoints,
        **store_kwargs,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(chunked_cache.cpu(), reference_cache.cpu())


@npu_test(num_npus=1, npu_type="a2")
def test_turboquant_chunked_dequant_matches_single_launch(monkeypatch):
    (
        config,
        _,
        centroids,
        cache,
        block_table,
        seq_lens,
        seq_lens_list,
        max_seq_len,
        query,
    ) = _build_fused_dequant_inputs("turboquant_4bit_nc", torch.float16)
    total_tokens = sum(seq_lens_list)
    seq_start_locs = torch.tensor(
        [0, seq_lens_list[0], total_tokens],
        dtype=torch.int32,
        device="npu",
    )
    reference_key = torch.empty(
        total_tokens,
        cache.shape[2],
        query.shape[-1],
        dtype=query.dtype,
        device="npu",
    )
    reference_value = torch.empty_like(reference_key)
    chunked_key = torch.empty_like(reference_key)
    chunked_value = torch.empty_like(reference_value)
    dequant_kwargs = {
        "max_seq_len": max_seq_len,
        "key_bits": config.key_quant_bits,
        "key_packed_size": config.key_packed_size,
        "value_bits": config.value_quant_bits,
        "norm_correction": config.norm_correction,
    }

    triton_turboquant_dequant_paged_cache(
        cache,
        block_table,
        seq_lens,
        seq_start_locs,
        centroids,
        reference_key,
        reference_value,
        **dequant_kwargs,
    )
    monkeypatch.setattr(
        turboquant_decode_module,
        "_ASCEND_MAX_TRITON_GRID_SIZE",
        100,
    )
    triton_turboquant_dequant_paged_cache(
        cache,
        block_table,
        seq_lens,
        seq_start_locs,
        centroids,
        chunked_key,
        chunked_value,
        **dequant_kwargs,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(chunked_key, reference_key, rtol=0, atol=0)
    torch.testing.assert_close(chunked_value, reference_value, rtol=0, atol=0)


@npu_test(num_npus=1, npu_type="a2")
@pytest.mark.parametrize(
    (
        "cache_dtype",
        "query_len",
        "activation_dtype",
        "head_dim",
        "use_noncontiguous_pages",
        "use_alibi",
        "logits_soft_cap",
        "implementation",
    ),
    _DECODE_CORRECTNESS_CASES,
)
def test_turboquant_store_and_decode_match_dequantized_reference(
    cache_dtype,
    query_len,
    activation_dtype,
    head_dim,
    use_noncontiguous_pages,
    use_alibi,
    logits_soft_cap,
    implementation,
):
    torch.manual_seed(1)
    num_tokens = 133 if use_noncontiguous_pages else max(8, query_len)
    num_kv_heads = 2
    # Qwen3-32B uses a GQA group size of eight after tensor parallelism.
    num_query_heads = 16
    config, hadamard, centroids, midpoints = _constants(
        cache_dtype,
        head_dim,
    )
    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=activation_dtype,
        device="npu",
    )
    value = torch.randn_like(key)
    query = torch.randn(
        query_len,
        num_query_heads,
        head_dim,
        dtype=activation_dtype,
        device="npu",
    )
    num_logical_blocks = (num_tokens + 127) // 128
    if use_noncontiguous_pages:
        physical_blocks = torch.tensor([2, 0], dtype=torch.int32, device="npu")
        num_physical_blocks = 3
    else:
        physical_blocks = torch.arange(
            num_logical_blocks,
            dtype=torch.int32,
            device="npu",
        )
        num_physical_blocks = num_logical_blocks
    cache = torch.zeros(
        num_physical_blocks,
        128,
        num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device="npu",
    )
    logical_positions = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    slot_mapping = physical_blocks[logical_positions // 128].to(torch.int64) * 128 + logical_positions % 128
    block_table = physical_blocks.view(1, num_logical_blocks)
    cache_seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    decode_seq_lens = torch.arange(
        num_tokens - query_len + 1,
        num_tokens + 1,
        dtype=torch.int32,
        device="npu",
    )
    decode_block_table = block_table.expand(query_len, -1)

    triton_turboquant_store(
        key,
        value,
        cache,
        slot_mapping,
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=None if implementation == "reference" else hadamard.to(activation_dtype),
    )
    # Triton launches asynchronously. Surface a store-kernel failure here so
    # it cannot poison the stream and masquerade as a later dequant/FIA error.
    torch.npu.synchronize()

    key_rotated = torch.empty_like(key)
    value_dense = torch.empty_like(value)
    triton_turboquant_dequant_paged_cache(
        cache,
        block_table,
        cache_seq_lens,
        torch.tensor([0, num_tokens], dtype=torch.int32, device="npu"),
        centroids,
        key_rotated,
        value_dense,
        max_seq_len=num_tokens,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
    )
    torch.npu.synchronize()
    num_splits = 4
    # A continuation chunk can be larger than max_num_seqs. Exercise the
    # eager fallback instead of giving every case an exactly sized workspace.
    workspace_batch = 2 if query_len == 8 else query_len
    buffers = SimpleNamespace(
        _tq_mid_o_buf=torch.empty(
            workspace_batch,
            num_query_heads,
            num_splits,
            head_dim + 1,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_lse_buf=torch.empty(
            workspace_batch,
            num_query_heads,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_ascend_query_rotation_buf=torch.empty_like(
            query,
            dtype=torch.float32 if implementation == "reference" else activation_dtype,
        ),
        _tq_ascend_query_float_buf=torch.empty_like(query, dtype=torch.float32),
    )
    scale = 1 / math.sqrt(head_dim)
    alibi_slopes = (
        torch.linspace(
            0.1,
            0.4,
            num_query_heads,
            dtype=torch.float32,
            device="npu",
        )
        if use_alibi
        else None
    )
    output = torch.empty_like(query)
    compute_rotation = None if implementation == "reference" else hadamard.to(activation_dtype)
    result = triton_turboquant_decode_attention(
        query,
        cache,
        decode_block_table,
        decode_seq_lens,
        hadamard,
        centroids,
        scale=scale,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
        max_num_kv_splits=num_splits,
        buffer_holder=buffers,
        alibi_slopes=alibi_slopes,
        logits_soft_cap=logits_soft_cap,
        output=output,
        compute_rotation=compute_rotation,
        implementation=implementation,
    )
    torch.npu.synchronize()

    kv_head_indices = torch.arange(num_query_heads, device="npu") // (num_query_heads // num_kv_heads)
    expanded_key = key_rotated[:, kv_head_indices].float()
    expanded_value = value_dense[:, kv_head_indices].float()
    query_rotated = (query.float() @ hadamard if compute_rotation is None else query @ compute_rotation).float()
    scores = torch.einsum("qhd,thd->qht", query_rotated, expanded_key) * scale
    if logits_soft_cap is not None:
        scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
    positions = torch.arange(num_tokens, device="npu")
    if alibi_slopes is not None:
        relative_positions = positions[None, :] - decode_seq_lens[:, None] + 1
        scores += alibi_slopes[None, :, None] * relative_positions[:, None, :]
    causal_mask = positions[None, :] < decode_seq_lens[:, None]
    scores = scores.masked_fill(~causal_mask[:, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.einsum("qht,thd->qhd", probabilities, expanded_value)

    torch.testing.assert_close(
        result.float().cpu(),
        reference.cpu(),
        rtol=2e-2,
        atol=2e-2,
    )
    assert buffers._tq_mid_o_buf.shape[0] == workspace_batch
    assert result.data_ptr() == output.data_ptr()


@npu_test(num_npus=1, npu_type="a2")
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_turboquant_zero_keys_and_constant_values_remain_finite(
    activation_dtype,
):
    num_tokens = 5
    num_kv_heads = 2
    num_query_heads = 4
    head_dim = 128
    num_splits = 4
    config, hadamard, centroids, midpoints = _constants(
        "turboquant_4bit_nc",
        head_dim,
    )
    compute_rotation = hadamard.to(activation_dtype)
    key = torch.zeros(
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=activation_dtype,
        device="npu",
    )
    value = torch.full_like(key, 123)
    query = torch.randn(
        1,
        num_query_heads,
        head_dim,
        dtype=activation_dtype,
        device="npu",
    )
    cache = torch.zeros(
        1,
        128,
        num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device="npu",
    )
    triton_turboquant_store(
        key,
        value,
        cache,
        torch.arange(num_tokens, dtype=torch.int64, device="npu"),
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=compute_rotation,
    )
    torch.npu.synchronize()
    buffers = SimpleNamespace(
        _tq_mid_o_buf=torch.empty(
            1,
            num_query_heads,
            num_splits,
            head_dim + 1,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_lse_buf=torch.empty(
            1,
            num_query_heads,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_ascend_query_rotation_buf=torch.empty_like(query),
    )
    result = triton_turboquant_decode_attention(
        query,
        cache,
        torch.zeros(1, 1, dtype=torch.int32, device="npu"),
        torch.tensor([num_tokens], dtype=torch.int32, device="npu"),
        hadamard,
        centroids,
        scale=1 / math.sqrt(head_dim),
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        norm_correction=config.norm_correction,
        max_num_kv_splits=num_splits,
        buffer_holder=buffers,
        output=torch.empty_like(query),
        compute_rotation=compute_rotation,
    )
    torch.npu.synchronize()

    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result.float().cpu(),
        torch.full(result.shape, 123, dtype=torch.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@npu_test(num_npus=1, npu_type="a2")
def test_turboquant_store_and_decode_aclgraph_replay_matches_eager():
    torch.manual_seed(2)
    num_reqs = 2
    query_len = 4
    history_len = 3
    final_seq_len = history_len + query_len
    num_kv_heads = 2
    num_query_heads = 16
    head_dim = 128
    num_splits = 4
    config, hadamard, centroids, midpoints = _constants(
        "turboquant_4bit_nc",
        head_dim,
    )
    compute_rotation = hadamard.to(torch.float16)
    history_key = torch.randn(
        num_reqs * history_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="npu",
    )
    history_value = torch.randn_like(history_key)
    key = torch.randn(
        num_reqs * query_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="npu",
    )
    value = torch.randn_like(key)
    query = torch.randn(
        num_reqs * query_len,
        num_query_heads,
        head_dim,
        dtype=torch.float16,
        device="npu",
    )
    cache = torch.zeros(
        num_reqs * 2,
        128,
        num_kv_heads,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device="npu",
    )
    block_table = torch.arange(num_reqs, dtype=torch.int32, device="npu").view(num_reqs, 1)
    seq_lens = torch.full(
        (num_reqs,),
        final_seq_len,
        dtype=torch.int32,
        device="npu",
    )
    history_slot_mapping = torch.cat(
        [
            request_index * 128 + torch.arange(history_len, dtype=torch.int64, device="npu")
            for request_index in range(num_reqs)
        ]
    )
    slot_mapping = torch.cat(
        [
            request_index * 128
            + torch.arange(
                history_len,
                final_seq_len,
                dtype=torch.int64,
                device="npu",
            )
            for request_index in range(num_reqs)
        ]
    )

    # Warm up store compilation before entering the graph capture scope.
    triton_turboquant_store(
        key,
        value,
        cache,
        slot_mapping,
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=compute_rotation,
    )
    torch.npu.synchronize()
    cache.zero_()
    triton_turboquant_store(
        history_key,
        history_value,
        cache,
        history_slot_mapping,
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=compute_rotation,
    )
    torch.npu.synchronize()

    layer = SimpleNamespace(
        _tq_mid_o_buf=torch.empty(
            num_reqs,
            num_query_heads,
            num_splits,
            head_dim + 1,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_lse_buf=torch.empty(
            num_reqs,
            num_query_heads,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_ascend_hadamard=hadamard,
        _tq_ascend_compute_rotation=compute_rotation,
        _tq_ascend_centroids=centroids,
    )
    metadata = AscendMetadata(
        attn_state=AscendAttentionState.SpecDecoding,
        num_actual_tokens=num_reqs * query_len,
        num_decode_tokens=num_reqs * query_len,
        num_decodes=num_reqs,
        seq_lens=seq_lens,
        seq_lens_list=[final_seq_len] * num_reqs,
        actual_seq_lengths_q=[query_len * request_index for request_index in range(1, num_reqs + 1)],
        query_start_loc=torch.arange(
            0,
            (num_reqs + 1) * query_len,
            query_len,
            dtype=torch.int32,
            device="npu",
        ),
        block_tables=block_table,
    )
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_heads = num_query_heads
    impl.head_size = head_dim
    impl.scale = 1 / math.sqrt(head_dim)
    impl.alibi_slopes = None
    impl.logits_soft_cap = None
    impl.tq_config = config
    impl.max_num_kv_splits = num_splits
    impl.decode_implementation = "auto"
    output = torch.empty_like(query)

    # Warm up compilation before entering the graph capture scope.
    impl._decode_attention(
        layer,
        query,
        cache,
        metadata,
        output,
    )
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        triton_turboquant_store(
            key,
            value,
            cache,
            slot_mapping,
            hadamard,
            midpoints,
            key_bits=config.key_quant_bits,
            key_packed_size=config.key_packed_size,
            value_bits=config.value_quant_bits,
            compute_rotation=compute_rotation,
        )
        graph_result = impl._decode_attention(
            layer,
            query,
            cache,
            metadata,
            output,
        )

    graph.replay()
    torch.npu.synchronize()
    first_graph_result = graph_result.clone()
    first_eager_result = impl._decode_attention(
        layer,
        query,
        cache,
        metadata,
        torch.empty_like(output),
    )
    torch.testing.assert_close(first_graph_result, first_eager_result)

    query.copy_(torch.randn_like(query))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    alternate_blocks = torch.arange(
        num_reqs,
        num_reqs * 2,
        dtype=torch.int32,
        device="npu",
    ).view(num_reqs, 1)
    alternate_history_slots = torch.cat(
        [
            (num_reqs + request_index) * 128 + torch.arange(history_len, dtype=torch.int64, device="npu")
            for request_index in range(num_reqs)
        ]
    )
    triton_turboquant_store(
        torch.randn_like(history_key),
        torch.randn_like(history_value),
        cache,
        alternate_history_slots,
        hadamard,
        midpoints,
        key_bits=config.key_quant_bits,
        key_packed_size=config.key_packed_size,
        value_bits=config.value_quant_bits,
        compute_rotation=compute_rotation,
    )
    block_table.copy_(alternate_blocks)
    slot_mapping.copy_(
        torch.cat(
            [
                (num_reqs + request_index) * 128
                + torch.arange(
                    history_len,
                    final_seq_len,
                    dtype=torch.int64,
                    device="npu",
                )
                for request_index in range(num_reqs)
            ]
        )
    )
    seq_lens.fill_(final_seq_len - 1)
    cache_before_replay = cache.clone()
    graph.replay()
    torch.npu.synchronize()
    assert not torch.equal(cache, cache_before_replay)
    replay_result = graph_result.clone()
    eager_result = impl._decode_attention(
        layer,
        query,
        cache,
        metadata,
        torch.empty_like(output),
    )
    torch.testing.assert_close(replay_result, eager_result)
