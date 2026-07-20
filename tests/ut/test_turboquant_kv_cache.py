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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.kv_cache.turboquant import (
    get_turboquant_kv_cache_shape,
    is_turboquant_kv_cache_dtype,
    validate_turboquant_backend,
    validate_turboquant_layout,
)


def _copy_decode_output(
    _layer,
    step_query,
    _cache,
    _blocks,
    _seq_lens,
    *,
    output=None,
    **_kwargs,
):
    assert output is not None
    output.copy_(step_query)
    return output


@pytest.mark.parametrize(
    ("cache_dtype", "slot_size"),
    [
        ("turboquant_4bit_nc", 134),
        ("turboquant_k3v4_nc", 118),
        ("turboquant_3bit_nc", 102),
    ],
)
def test_turboquant_cache_shape_uses_packed_slot(cache_dtype, slot_size):
    assert get_turboquant_kv_cache_shape(8, 128, 8, 128, cache_dtype) == (
        8,
        128,
        8,
        slot_size,
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_turboquant_supports_common_dense_head_dimensions(head_dim):
    validate_turboquant_layout("turboquant_4bit_nc", head_dim)
    shape = get_turboquant_kv_cache_shape(
        2,
        128,
        4,
        head_dim,
        "turboquant_4bit_nc",
    )
    assert shape[:3] == (2, 128, 4)
    assert shape[3] > 0


def test_turboquant_dtype_detection_does_not_match_normal_cache():
    assert is_turboquant_kv_cache_dtype("turboquant_4bit_nc")
    assert not is_turboquant_kv_cache_dtype("auto")
    assert not is_turboquant_kv_cache_dtype(None)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_turboquant_randomized_hadamard_is_orthogonal(head_dim):
    from vllm_ascend.attention.turboquant import _build_hadamard

    rotation = _build_hadamard(head_dim, "cpu")
    identity = torch.eye(head_dim, dtype=torch.float32)

    torch.testing.assert_close(rotation @ rotation.T, identity)


def test_turboquant_randomized_hadamard_spreads_constant_vectors():
    from vllm_ascend.attention.turboquant import _build_hadamard

    head_dim = 128
    rotation = _build_hadamard(head_dim, "cpu")
    rotated = torch.ones(1, head_dim) @ rotation

    # A plain Sylvester Hadamard maps this input to one coordinate with
    # magnitude sqrt(D). Random signs before mixing keep the largest coordinate
    # below 30% of the vector norm for the fixed production seed.
    assert rotated.abs().amax() / torch.linalg.vector_norm(rotated) < 0.3
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated),
        torch.tensor(head_dim**0.5),
    )


def test_turboquant_rejects_unimplemented_fp8_key_path():
    with pytest.raises(NotImplementedError, match="FP8 keys"):
        validate_turboquant_layout("turboquant_k8v4", 128)


@pytest.mark.parametrize("head_dim", [0, 80, 96])
def test_turboquant_rejects_unsupported_head_dimension(head_dim):
    with pytest.raises(NotImplementedError):
        validate_turboquant_layout("turboquant_4bit_nc", head_dim)


@pytest.mark.parametrize(
    "unsupported_option",
    [
        "use_mla",
        "use_sparse",
        "use_compress",
        "use_v2_runner",
        "is_310p_device",
        "has_sink",
        "use_mm_prefix",
        "use_non_causal",
        "use_batch_invariant",
    ],
)
def test_turboquant_backend_rejects_unsupported_features(unsupported_option):
    options = {
        "head_dim": 128,
        "use_mla": False,
        "use_sparse": False,
        "use_compress": False,
        "use_v2_runner": False,
        "is_310p_device": False,
        "has_sink": False,
        "use_mm_prefix": False,
        "use_non_causal": False,
        "use_batch_invariant": False,
    }
    options[unsupported_option] = True
    with pytest.raises(NotImplementedError):
        validate_turboquant_backend("turboquant_4bit_nc", **options)


def test_backend_validation_is_noop_for_uncompressed_cache():
    validate_turboquant_backend(
        "auto",
        head_dim=96,
        use_mla=True,
        use_sparse=True,
        use_compress=True,
        use_v2_runner=True,
        is_310p_device=True,
        has_sink=True,
        use_mm_prefix=True,
        use_non_causal=True,
        use_batch_invariant=True,
    )


def test_turboquant_model_runner_allocates_and_reshapes_packed_cache():
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        TQFullAttentionSpec,
    )

    from vllm_ascend.attention.turboquant import (
        AscendTurboQuantAttentionBackend,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    layer_name = "model.layers.0.self_attn.attn"
    num_blocks = 2
    spec = TQFullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        tq_slot_size=134,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                shared_by=[layer_name],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
        ],
    )
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.use_compress = False
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cache_dtype="turboquant_4bit_nc"),
    )
    runner._get_layer_kv_cache_specs = lambda _config: {layer_name: spec}
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                backend=AscendTurboQuantAttentionBackend,
                kv_cache_spec=spec,
                layer_names=[layer_name],
            )
        ]
    )

    raw_caches = NPUModelRunner._allocate_kv_cache_tensors(runner, config)
    shaped_caches = NPUModelRunner._reshape_kv_cache_tensors(
        runner,
        config,
        raw_caches,
    )

    cache = shaped_caches[layer_name]
    assert cache.dtype == torch.uint8
    assert cache.shape == (num_blocks, 128, 2, 134)
    assert cache.untyped_storage().data_ptr() == raw_caches[layer_name].untyped_storage().data_ptr()


def test_turboquant_copy_blocks_preserves_packed_pages():
    from vllm_ascend.attention.turboquant import (
        AscendTurboQuantAttentionBackend,
    )

    cache = torch.arange(4 * 2 * 3, dtype=torch.uint8).view(4, 2, 1, 3)
    expected_sources = cache[:2].clone()
    source_to_destinations = torch.tensor([[0, 2], [1, 3]])

    AscendTurboQuantAttentionBackend.copy_blocks(
        [cache],
        source_to_destinations,
    )

    torch.testing.assert_close(cache[2:], expected_sources)


def test_turboquant_copy_blocks_snapshots_overlapping_sources():
    from vllm_ascend.attention.turboquant import (
        AscendTurboQuantAttentionBackend,
    )

    cache = torch.arange(4, dtype=torch.uint8).view(4, 1, 1, 1)
    original = cache.clone()

    AscendTurboQuantAttentionBackend.copy_blocks(
        [cache],
        torch.tensor([[0, 1], [1, 2]]),
    )

    torch.testing.assert_close(cache[1], original[0])
    torch.testing.assert_close(cache[2], original[1])


def test_turboquant_swap_blocks_copies_opaque_pages_between_tensors():
    from vllm_ascend.attention.turboquant import (
        AscendTurboQuantAttentionBackend,
    )

    source = torch.arange(4 * 3, dtype=torch.uint8).view(4, 1, 1, 3)
    destination = torch.zeros_like(source)

    AscendTurboQuantAttentionBackend.swap_blocks(
        [source],
        [destination],
        torch.tensor([[3, 0], [1, 2]]),
    )

    torch.testing.assert_close(destination[0], source[3])
    torch.testing.assert_close(destination[2], source[1])


def test_turboquant_graph_capture_metadata_uses_device_sequence_lengths():
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
    from vllm_ascend.attention.turboquant import AscendTurboQuantMetadataBuilder

    metadata = AscendMetadata(
        seq_lens=torch.tensor([7, 11], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    builder = object.__new__(AscendTurboQuantMetadataBuilder)
    common_attn_metadata = MagicMock()

    with patch.object(
        AscendTurboQuantMetadataBuilder,
        "build",
        return_value=metadata,
    ) as build:
        captured = builder.build_for_cudagraph_capture(common_attn_metadata)

    build.assert_called_once_with(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )
    assert captured.attn_state is AscendAttentionState.DecodeOnly
    torch.testing.assert_close(
        captured.seq_lens,
        torch.full((2,), 4, dtype=torch.int32),
    )


def test_turboquant_builder_routes_all_decode_tokens_to_packed_path():
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
        AscendAttentionState,
        AscendMetadata,
    )
    from vllm_ascend.attention.turboquant import AscendTurboQuantMetadataBuilder

    metadata = AscendMetadata(
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_actual_tokens=8,
        num_decode_tokens=8,
        num_decodes=2,
    )
    common_attn_metadata = MagicMock(
        num_reqs=2,
        num_actual_tokens=8,
        seq_lens=torch.tensor([7, 9], dtype=torch.int32),
    )
    builder = object.__new__(AscendTurboQuantMetadataBuilder)

    with patch.object(
        AscendAttentionMetadataBuilder,
        "build",
        return_value=metadata,
    ):
        result = builder.build(0, common_attn_metadata)

    assert result.attn_state is AscendAttentionState.DecodeOnly
    assert result.seq_lens.data_ptr() == common_attn_metadata.seq_lens.data_ptr()
    torch.testing.assert_close(
        result.seq_lens,
        common_attn_metadata.seq_lens[: common_attn_metadata.num_reqs],
    )


@pytest.mark.parametrize(
    ("cumulative_query_lens", "num_reqs", "expected"),
    [
        ([1, 2, 3], 3, [1, 1, 1]),
        ([4, 8], 2, [4, 4]),
        ([2, 5], 2, [2, 3]),
        ([], 0, []),
    ],
)
def test_turboquant_query_lens_from_cumulative(
    cumulative_query_lens,
    num_reqs,
    expected,
):
    from vllm_ascend.attention.turboquant import _query_lens_from_cumulative

    assert _query_lens_from_cumulative(cumulative_query_lens, num_reqs) == expected


def test_turboquant_query_lens_rejects_invalid_metadata():
    from vllm_ascend.attention.turboquant import _query_lens_from_cumulative

    with pytest.raises(ValueError, match="fewer query boundaries"):
        _query_lens_from_cumulative([1], 2)
    with pytest.raises(ValueError, match="positive query lengths"):
        _query_lens_from_cumulative([2, 2], 2)


def test_turboquant_uniform_multi_token_decode_reuses_request_workspace():
    from vllm_ascend.attention.attention_v1 import AscendMetadata
    from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionImpl

    num_reqs = 2
    query_len = 4
    query = torch.arange(num_reqs * query_len, dtype=torch.float32).view(-1, 1, 1)
    output = torch.empty_like(query)
    metadata = AscendMetadata(
        num_decodes=num_reqs,
        num_decode_tokens=num_reqs * query_len,
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([7, 9], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
    )
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_heads = 1
    impl.head_size = 1

    impl._launch_decode = MagicMock(side_effect=_copy_decode_output)

    result = impl._decode_attention(
        MagicMock(),
        query,
        torch.empty(0),
        metadata,
        output,
    )

    torch.testing.assert_close(result, query)
    assert impl._launch_decode.call_count == query_len
    for call in impl._launch_decode.call_args_list:
        assert call.args[4].data_ptr() == metadata.seq_lens.data_ptr()
    assert [call.kwargs["sequence_length_delta"] for call in impl._launch_decode.call_args_list] == [-3, -2, -1, 0]


def test_turboquant_uniform_decode_passes_padded_lengths_to_device():
    from vllm_ascend.attention.attention_v1 import AscendMetadata
    from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionImpl

    query = torch.arange(4, dtype=torch.float32).view(-1, 1, 1)
    output = torch.empty_like(query)
    metadata = AscendMetadata(
        num_decodes=2,
        num_decode_tokens=2,
        actual_seq_lengths_q=[2, 4],
        seq_lens=torch.tensor([5, 0], dtype=torch.int32),
        block_tables=torch.tensor([[0], [0]], dtype=torch.int32),
    )
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_heads = 1
    impl.head_size = 1

    impl._launch_decode = MagicMock(side_effect=_copy_decode_output)

    impl._decode_attention(
        MagicMock(),
        query,
        torch.empty(0),
        metadata,
        output,
    )

    for call in impl._launch_decode.call_args_list:
        assert call.args[4].data_ptr() == metadata.seq_lens.data_ptr()
    assert [call.kwargs["sequence_length_delta"] for call in impl._launch_decode.call_args_list] == [-1, 0]


def test_turboquant_nonuniform_decode_uses_per_request_causal_lengths():
    from vllm_ascend.attention.attention_v1 import AscendMetadata
    from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionImpl

    query = torch.arange(5, dtype=torch.float32).view(-1, 1, 1)
    output = torch.empty_like(query)
    metadata = AscendMetadata(
        num_decodes=2,
        num_decode_tokens=5,
        actual_seq_lengths_q=[2, 5],
        seq_lens=torch.tensor([6, 9], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
    )
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_heads = 1
    impl.head_size = 1

    impl._launch_decode = MagicMock(side_effect=_copy_decode_output)

    result = impl._decode_attention(
        MagicMock(),
        query,
        torch.empty(0),
        metadata,
        output,
    )

    torch.testing.assert_close(result, query)
    calls = impl._launch_decode.call_args_list
    assert [call.args[4].tolist() for call in calls] == [
        [6],
        [6],
        [9],
        [9],
        [9],
    ]
    assert [call.kwargs["sequence_length_delta"] for call in calls] == [
        -1,
        0,
        -2,
        -1,
        0,
    ]


def test_turboquant_forward_splits_mixed_decode_and_prefill_outputs():
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
    from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionImpl

    query = torch.zeros(4, 1, 1)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    output = torch.empty_like(query)
    metadata = AscendMetadata(
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_actual_tokens=4,
        num_decode_tokens=1,
        num_decodes=1,
        num_prefills=1,
        actual_seq_lengths_q=[1, 4],
        seq_lens=torch.tensor([5, 3], dtype=torch.int32),
        seq_lens_list=[5, 3],
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
    )
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl._ensure_constants = MagicMock()

    def decode_side_effect(_layer, _query, _cache, _metadata, decode_output):
        decode_output.fill_(10)
        return decode_output

    def prefill_side_effect(
        _layer,
        _query,
        _key,
        _value,
        _cache,
        _metadata,
        prefill_output,
        *,
        request_start,
        token_start,
    ):
        assert request_start == 1
        assert token_start == 1
        prefill_output[token_start:].fill_(20)

    impl._decode_attention = MagicMock(side_effect=decode_side_effect)
    impl._prefill_requests = MagicMock(side_effect=prefill_side_effect)

    result = impl.forward(
        MagicMock(),
        query,
        key,
        value,
        torch.empty(0),
        metadata,
        output,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([10, 20, 20, 20], dtype=query.dtype).view(-1, 1, 1),
    )
    assert impl._decode_attention.call_args.args[1].shape[0] == 1


def test_turboquant_small_continuation_uses_packed_decode():
    from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionImpl

    query = torch.arange(3, dtype=torch.float32).view(-1, 1, 1)
    output = torch.empty_like(query)
    impl = object.__new__(AscendTurboQuantAttentionImpl)

    impl._launch_decode = MagicMock(side_effect=_copy_decode_output)

    impl._continuation_prefill(
        MagicMock(),
        query,
        torch.zeros_like(query),
        torch.zeros_like(query),
        torch.empty(1, 128, 1, 1),
        torch.tensor([[0]], dtype=torch.int32),
        10,
        None,
        output,
    )

    torch.testing.assert_close(output, query)
    call = impl._launch_decode.call_args
    assert call.args[3].shape == (3, 1)
    torch.testing.assert_close(call.args[4], torch.tensor([8, 9, 10], dtype=torch.int32))
    assert call.kwargs["output"].data_ptr() == output.data_ptr()


def test_turboquant_large_continuation_dequantizes_history_only():
    from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionImpl

    query_len = 129
    history_len = 7
    seq_len = query_len + history_len
    query = torch.zeros(query_len, 1, 1)
    key = torch.full_like(query, 2)
    value = torch.full_like(query, 3)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        _tq_ascend_centroids=torch.zeros(16),
        _tq_ascend_hadamard=torch.ones(1, 1),
    )
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_heads = 1
    impl.num_kv_heads = 1
    impl.head_size = 1
    impl.tq_config = SimpleNamespace(
        key_quant_bits=4,
        key_packed_size=4,
        value_quant_bits=4,
        norm_correction=False,
    )
    impl._run_fia = MagicMock()

    def dequant_side_effect(
        _cache,
        _block_table,
        _seq_lens,
        _start_locs,
        _centroids,
        key_output,
        value_output,
        **_kwargs,
    ):
        key_output.fill_(4)
        value_output.fill_(5)

    with patch(
        "vllm_ascend.attention.turboquant.triton_turboquant_dequant_paged_cache",
        side_effect=dequant_side_effect,
    ) as dequant:
        impl._continuation_prefill(
            layer,
            query,
            key,
            value,
            torch.empty(2, 128, 1, 8, dtype=torch.uint8),
            torch.tensor([[0, 1]], dtype=torch.int32),
            seq_len,
            None,
            output,
        )

    assert dequant.call_args.kwargs["max_seq_len"] == history_len
    assert dequant.call_args.args[5].shape[0] == history_len
    fia_call = impl._run_fia.call_args
    assert fia_call.args[1].shape[0] == seq_len
    assert fia_call.args[2].shape[0] == seq_len
    assert fia_call.args[3] == [query_len]
    assert fia_call.args[4] == [seq_len]
    torch.testing.assert_close(fia_call.args[1][:history_len], torch.full((history_len, 1, 1), 4.0))
    torch.testing.assert_close(fia_call.args[1][history_len:], key)
    assert not hasattr(layer, "_tq_ascend_history_key_buf")
    assert not hasattr(layer, "_tq_ascend_history_value_buf")


def test_turboquant_feature_prefill_applies_softcap_alibi_and_causal_mask():
    from vllm_ascend.attention.turboquant import AscendTurboQuantAttentionImpl

    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 1
    impl.scale = 1.0
    impl.logits_soft_cap = 2.0
    impl.alibi_slopes = torch.tensor([1.0, 0.5])
    query = torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]]])
    key = torch.tensor([[[1.0]], [[-1.0]]])
    value = torch.tensor([[[1.0]], [[3.0]]])
    output = torch.empty_like(query)

    impl._run_feature_prefill(
        query,
        key,
        value,
        [2],
        [2],
        output,
    )

    torch.testing.assert_close(output[0], torch.ones(2, 1))
    capped_scores = 2 * torch.tanh(torch.tensor([1.0, -1.0]) / 2)
    expected = []
    for slope in impl.alibi_slopes:
        alibi_bias = torch.stack((-slope, slope.new_zeros(())))
        probabilities = torch.softmax(
            capped_scores + alibi_bias,
            dim=0,
        )
        expected.append(probabilities[0] + 3 * probabilities[1])
    torch.testing.assert_close(output[1, :, 0], torch.stack(expected))


def test_turboquant_feature_prefill_bounds_score_workspace_by_query_tile():
    from vllm_ascend.attention.turboquant import (
        _FEATURE_PREFILL_QUERY_TILE_SIZE,
        AscendTurboQuantAttentionImpl,
    )

    query_len = _FEATURE_PREFILL_QUERY_TILE_SIZE * 2 + 1
    impl = object.__new__(AscendTurboQuantAttentionImpl)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 1
    impl.scale = 1.0
    impl.logits_soft_cap = 2.0
    impl.alibi_slopes = None
    query = torch.randn(query_len, 2, 1)
    key = torch.randn(query_len, 1, 1)
    value = torch.ones(query_len, 1, 1)
    output = torch.empty_like(query)

    with patch(
        "vllm_ascend.attention.turboquant.F.softmax",
        wraps=torch.nn.functional.softmax,
    ) as softmax:
        impl._run_feature_prefill(
            query,
            key,
            value,
            [query_len],
            [query_len],
            output,
        )

    assert softmax.call_count == 3
    assert all(call.args[0].shape[-2] <= _FEATURE_PREFILL_QUERY_TILE_SIZE for call in softmax.call_args_list)
    torch.testing.assert_close(output, torch.ones_like(output))
