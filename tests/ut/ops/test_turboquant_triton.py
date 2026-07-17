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

from tests.ut.conftest import npu_test
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.attention.turboquant import (
    AscendTurboQuantAttentionImpl,
    _build_hadamard,
)
from vllm_ascend.kv_cache.turboquant import get_turboquant_config
from vllm_ascend.ops.triton.turboquant_decode import (
    triton_turboquant_decode_attention,
    triton_turboquant_dequant_paged_cache,
)
from vllm_ascend.ops.triton.turboquant_store import triton_turboquant_store

_DECODE_CORRECTNESS_CASES = [
    (cache_dtype, query_len, torch.float16, 128, False, False, None)
    for cache_dtype in (
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    )
    for query_len in (1, 2, 4, 8)
] + [
    ("turboquant_4bit_nc", 4, torch.bfloat16, 128, False, False, None),
    ("turboquant_k3v4_nc", 4, torch.bfloat16, 128, False, False, None),
    ("turboquant_3bit_nc", 4, torch.bfloat16, 128, False, False, None),
    ("turboquant_4bit_nc", 1, torch.float16, 64, False, False, None),
    ("turboquant_4bit_nc", 1, torch.float16, 256, False, False, None),
    ("turboquant_4bit_nc", 4, torch.float16, 128, True, False, None),
    ("turboquant_4bit_nc", 2, torch.float16, 128, False, True, 3.0),
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
@pytest.mark.parametrize(
    (
        "cache_dtype",
        "query_len",
        "activation_dtype",
        "head_dim",
        "use_noncontiguous_pages",
        "use_alibi",
        "logits_soft_cap",
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
):
    torch.manual_seed(1)
    num_tokens = 133 if use_noncontiguous_pages else max(8, query_len)
    num_kv_heads = 2
    num_query_heads = 4
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
    key_dense = (key_rotated.float().reshape(-1, head_dim) @ hadamard).view_as(key_rotated)

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
        _tq_output_buf=torch.empty(
            workspace_batch,
            num_query_heads,
            head_dim,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_lse_buf=torch.empty(
            workspace_batch,
            num_query_heads,
            dtype=torch.float32,
            device="npu",
        ),
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
    )
    torch.npu.synchronize()

    kv_head_indices = torch.arange(num_query_heads, device="npu") // (num_query_heads // num_kv_heads)
    expanded_key = key_dense[:, kv_head_indices].float()
    expanded_value = value_dense[:, kv_head_indices].float()
    scores = torch.einsum("qhd,thd->qht", query.float(), expanded_key) * scale
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
        _tq_output_buf=torch.empty(
            1,
            num_query_heads,
            head_dim,
            dtype=torch.float32,
            device="npu",
        ),
        _tq_lse_buf=torch.empty(
            1,
            num_query_heads,
            dtype=torch.float32,
            device="npu",
        ),
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
    num_query_heads = 4
    head_dim = 128
    num_splits = 4
    config, hadamard, centroids, midpoints = _constants(
        "turboquant_4bit_nc",
        head_dim,
    )
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
        _tq_output_buf=torch.empty(
            num_reqs,
            num_query_heads,
            head_dim,
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
