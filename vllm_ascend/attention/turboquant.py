# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from typing import ClassVar

import torch
import vllm.envs as envs_vllm
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import AttentionLayer, AttentionType

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.kv_cache.turboquant import (
    TURBOQUANT_KV_CACHE_DTYPES,
    get_turboquant_config,
    get_turboquant_kv_cache_shape,
    turboquant_dequant_cache,
    turboquant_store_kv,
)


class AscendTurboQuantAttentionBackend(AscendAttentionBackend):
    """Ascend attention backend for upstream TurboQuant KV-cache specs."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(TURBOQUANT_KV_CACHE_DTYPES)

    @staticmethod
    def get_name() -> str:
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[AscendTurboQuantAttentionImpl]:
        return AscendTurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        return get_turboquant_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str,
        )

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")


class AscendTurboQuantAttentionImpl(AscendAttentionBackendImpl):
    """Portable Ascend implementation for TurboQuant packed KV cache.

    The cache layout and sizing follow upstream vLLM TurboQuant. The current
    store/dequant path is a reference Torch implementation; it keeps the
    Python/backend contract stable for a later fused NPU kernel replacement.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
            **kwargs,
        )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("TurboQuant KV cache currently supports decoder attention only.")
        self.tq_config = get_turboquant_config(kv_cache_dtype, head_size)

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if key is None or value is None or kv_cache is None:
            return
        turboquant_store_kv(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            tq_config=self.tq_config,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if key is not None and value is not None and kv_cache is not None:
            slot_mapping = attn_metadata.slot_mapping
            num_actual_tokens = attn_metadata.num_actual_tokens
            turboquant_store_kv(
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                kv_cache=kv_cache,
                slot_mapping=slot_mapping[:num_actual_tokens],
                tq_config=self.tq_config,
            )
        return query, key, value, output

    def _materialize_standard_kv_cache(
        self,
        kv_cache: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = turboquant_dequant_cache(
            kv_cache,
            self.tq_config,
            dtype,
        )
        self.key_cache = key_cache
        self.value_cache = value_cache
        return key_cache, value_cache

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if self._use_layer_aware_fia_graph_replay:
            self._layer_name = layer.layer_name
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported for Ascend TurboQuant.")
        if attn_metadata is None:
            return output.fill_(0)

        num_tokens = query.shape[0]
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache and key is not None and value is not None:
            attn_output = self.forward_fused_infer_attention(
                query,
                key,
                value,
                attn_metadata,
                output,
                kv_cache=None,
            )
        else:
            standard_kv_cache = self._materialize_standard_kv_cache(
                kv_cache,
                query.dtype,
            )
            attn_output = self.forward_impl(
                query,
                key,
                value,
                standard_kv_cache,
                attn_metadata,
                output,
            )

        output[:num_tokens] = attn_output[:num_tokens]
        return output
