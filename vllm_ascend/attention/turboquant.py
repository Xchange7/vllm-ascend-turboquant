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

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import ClassVar, NamedTuple

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.kv_cache.turboquant import (
    get_turboquant_config,
    get_turboquant_kv_cache_shape,
    validate_turboquant_layout,
)
from vllm_ascend.ops.triton.turboquant_decode import (
    select_turboquant_grouped_block_kv,
    select_turboquant_num_kv_splits,
    triton_turboquant_decode_attention,
    triton_turboquant_dequant_paged_cache,
)
from vllm_ascend.ops.triton.turboquant_store import triton_turboquant_store
from vllm_ascend.ops.turboquant import (
    has_turboquant_paged_attention,
    has_turboquant_paged_dequant,
    turboquant_paged_attention_out,
    turboquant_paged_dequant_out,
)

_TURBOQUANT_CACHE_DTYPES: list[CacheDType] = [
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
]

# Match the upstream TurboQuant continuation policy. Small chunks are cheaper
# as sequential packed decodes; larger chunks amortize history dequantization.
_CONTINUATION_DECODE_THRESHOLD = 128

# Bound the FP32 score workspace used only when FIA cannot express ALiBi or
# logits soft cap. The K dimension remains full so each tile is exact.
_FEATURE_PREFILL_QUERY_TILE_SIZE = 32

# Keep this in sync with the AscendC paged-dequant tiling constraints. The
# Triton path supports a wider set of power-of-two head dimensions.
_ASCEND_FUSED_MIN_HEAD_DIM = 32
_ASCEND_FUSED_MAX_HEAD_DIM = 256

# Grow the dense FIA workspace in coarse sequence-length buckets. Without
# bucketing, decode can reallocate K/V and query FIA workspace every token.
_ASCEND_FUSED_SEQUENCE_WORKSPACE_GRANULARITY = 1024

# The Lloyd-Max codebook models coordinates produced by a randomized
# orthogonal transform. Keep the seed stable so cache writes and reads use the
# same transform across workers and graph captures.
_ROTATION_SEED = 42


@functools.cache
def _build_hadamard(head_dim: int, device_string: str) -> torch.Tensor:
    """Build a deterministic randomized Hadamard rotation ``D @ H``.

    The random sign diagonal must precede Hadamard mixing for row vectors.
    Applying signs after ``H`` would leave structured inputs such as constant
    vectors concentrated in one coordinate and invalidate the Gaussian
    assumption used to construct the scalar quantizer.
    """
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < head_dim:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    matrix /= math.sqrt(head_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_ROTATION_SEED)
    signs = torch.randint(
        0,
        2,
        (head_dim,),
        generator=generator,
        dtype=torch.int8,
    ).to(torch.float32)
    signs.mul_(2).sub_(1)
    matrix.mul_(signs[:, None])
    return matrix.to(torch.device(device_string))


@functools.cache
def _build_compute_rotation(
    head_dim: int,
    device_string: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Share the low-precision Cube operand across attention layers."""
    return _build_hadamard(head_dim, device_string).to(dtype=dtype)


@functools.cache
def _build_pipelined_compute_rotation(
    head_dim: int,
    device_string: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Emit rotated Q in the packed cache's low/high-nibble order.

    A 4-bit cache byte stores an even coordinate in its low nibble and the
    following odd coordinate in its high nibble. Producing Q as all even
    coordinates followed by all odd coordinates lets the AscendC pipeline
    skip two 128-element deinterleaves for every cached KV token.
    """
    rotation = _build_compute_rotation(head_dim, device_string, dtype)
    return torch.cat((rotation[:, 0::2], rotation[:, 1::2]), dim=1).contiguous()


def _query_lens_from_cumulative(
    cumulative_query_lens: list[int],
    num_reqs: int,
) -> list[int]:
    if num_reqs == 0:
        return []
    if len(cumulative_query_lens) < num_reqs:
        raise ValueError(
            "TurboQuant decode metadata has fewer query boundaries than requests: "
            f"{len(cumulative_query_lens)} < {num_reqs}."
        )

    query_lens: list[int] = []
    previous_end = 0
    for request_end in cumulative_query_lens[:num_reqs]:
        query_len = request_end - previous_end
        if query_len <= 0:
            raise ValueError(f"TurboQuant decode requires positive query lengths, got {query_len}.")
        query_lens.append(query_len)
        previous_end = request_end
    return query_lens


def _validate_ascend_fused_head_dim(head_dim: int) -> None:
    if not _supports_ascend_fused_head_dim(head_dim):
        raise NotImplementedError(
            "Ascend TurboQuant fused decode requires head_dim to be a "
            f"power of two in [32, 256], got {head_dim}. Use the default "
            "auto decode implementation for other supported head dimensions."
        )


def _supports_ascend_fused_head_dim(head_dim: int) -> bool:
    return _ASCEND_FUSED_MIN_HEAD_DIM <= head_dim <= _ASCEND_FUSED_MAX_HEAD_DIM and head_dim & (head_dim - 1) == 0


def _is_aclgraph_forward() -> bool:
    try:
        forward_context = get_forward_context()
    except (AssertionError, LookupError, RuntimeError):
        return False
    runtime_mode = getattr(
        forward_context,
        "cudagraph_runtime_mode",
        CUDAGraphMode.NONE,
    )
    return runtime_mode is not None and runtime_mode != CUDAGraphMode.NONE


def _should_use_ascend_fused_decode(
    implementation: str,
    fused_available: bool,
    single_token_decode: bool,
    aclgraph_forward: bool,
) -> bool:
    if not single_token_decode:
        return False
    if implementation == "ascend_fused":
        return True
    return implementation == "auto" and fused_available and not aclgraph_forward


def _build_turboquant_page_table_cpu(seq_lens: list[int], block_size: int) -> torch.Tensor:
    """Build compact ``(request, logical_page)`` rows without device sync."""
    if block_size <= 0:
        raise ValueError(f"TurboQuant block_size must be positive, got {block_size}.")
    if not seq_lens:
        raise ValueError("TurboQuant sequence lengths must not be empty.")
    if any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError(f"TurboQuant sequence lengths must be positive, got {seq_lens}.")
    rows = [
        (request_index, page_index)
        for request_index, seq_len in enumerate(seq_lens)
        for page_index in range((seq_len + block_size - 1) // block_size)
    ]
    return torch.tensor(rows, dtype=torch.int32).reshape(-1, 2)


class _TurboQuantPageTableSlot:
    """One asynchronous host/device staging pair."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.capacity = 0
        self.host_buffer: torch.Tensor | None = None
        self.page_indices: torch.Tensor | None = None
        self.device_buffer: torch.Tensor | None = None
        self.copy_event: torch.npu.Event | None = None
        self.copy_pending = False

    def ready_for_host_write(self) -> bool:
        if not self.copy_pending:
            return True
        assert self.copy_event is not None
        if self.copy_event.query():
            self.copy_pending = False
            return True
        return False


class _TurboQuantPageTableBuilder:
    """Build active-page metadata without synchronizing the model thread."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.slots = [
            _TurboQuantPageTableSlot(device),
            _TurboQuantPageTableSlot(device),
        ]
        self.active_slot = -1
        self.last_page_counts: tuple[int, ...] | None = None

    @staticmethod
    def _reserve(slot: _TurboQuantPageTableSlot, required_pages: int) -> None:
        if required_pages <= slot.capacity:
            return
        slot.capacity = max(required_pages, max(16, slot.capacity * 2))
        slot.host_buffer = torch.empty(
            (slot.capacity, 2),
            dtype=torch.int32,
            pin_memory=slot.device.type == "npu",
        )
        slot.page_indices = torch.empty(
            slot.capacity,
            dtype=torch.int32,
            pin_memory=slot.device.type == "npu",
        )
        torch.arange(slot.capacity, dtype=torch.int32, out=slot.page_indices)
        slot.device_buffer = torch.empty(
            (slot.capacity, 2),
            dtype=torch.int32,
            device=slot.device,
        )

    def _next_writable_slot(self) -> tuple[int, _TurboQuantPageTableSlot]:
        for offset in range(1, len(self.slots) + 1):
            slot_index = (self.active_slot + offset) % len(self.slots)
            slot = self.slots[slot_index]
            if slot.ready_for_host_write():
                return slot_index, slot

        # The CPU can occasionally enqueue more than two model steps before
        # either staging copy completes. Add a slot instead of blocking it.
        slot = _TurboQuantPageTableSlot(self.device)
        self.slots.append(slot)
        return len(self.slots) - 1, slot

    def build(self, seq_lens: list[int], block_size: int) -> torch.Tensor:
        if block_size <= 0:
            raise ValueError(f"TurboQuant block_size must be positive, got {block_size}.")
        if not seq_lens:
            raise ValueError("TurboQuant sequence lengths must not be empty.")
        page_counts = tuple((seq_len + block_size - 1) // block_size for seq_len in seq_lens)
        if any(seq_len <= 0 for seq_len in seq_lens):
            raise ValueError(f"TurboQuant sequence lengths must be positive, got {seq_lens}.")
        active_pages = sum(page_counts)
        if page_counts == self.last_page_counts:
            active = self.slots[self.active_slot]
            assert active.device_buffer is not None
            return active.device_buffer[:active_pages]

        slot_index, slot = self._next_writable_slot()
        self._reserve(slot, active_pages)
        assert slot.host_buffer is not None
        assert slot.page_indices is not None
        assert slot.device_buffer is not None
        offset = 0
        for request_index, page_count in enumerate(page_counts):
            next_offset = offset + page_count
            slot.host_buffer[offset:next_offset, 0].fill_(request_index)
            slot.host_buffer[offset:next_offset, 1].copy_(slot.page_indices[:page_count])
            offset = next_offset
        slot.device_buffer[:active_pages].copy_(
            slot.host_buffer[:active_pages],
            non_blocking=True,
        )
        if self.device.type == "npu":
            if slot.copy_event is None:
                slot.copy_event = torch.npu.Event()
            slot.copy_event.record()
            slot.copy_pending = True
        self.active_slot = slot_index
        self.last_page_counts = page_counts
        return slot.device_buffer[:active_pages]


class _TurboQuantDenseBuffers(NamedTuple):
    key: torch.Tensor
    value: torch.Tensor
    query: torch.Tensor
    softmax_lse: torch.Tensor
    key_capacity: torch.Tensor
    value_capacity: torch.Tensor
    query_capacity: torch.Tensor
    batch_capacity: int
    sequence_capacity: int


class _TurboQuantFusedWorkspace:
    """Step-shared buffers for dense dequantization and eager FIA decode."""

    def __init__(self) -> None:
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.query: torch.Tensor | None = None
        self.softmax_lse: torch.Tensor | None = None
        self.fia_workspace: torch.Tensor | None = None
        self.fia_workspace_shapes: set[tuple[object, ...]] = set()
        self.batch_capacity = 0
        self.sequence_capacity = 0
        self.layout: tuple[object, ...] | None = None

    @staticmethod
    def _grow_capacity(current: int, required: int, maximum: int) -> int:
        if current >= required:
            return current
        if current == 0:
            return required
        return min(maximum, max(required, current * 2))

    @staticmethod
    def _flat_buffer(
        buffer: torch.Tensor | None,
        required_elements: int,
        template: torch.Tensor,
    ) -> torch.Tensor:
        if (
            buffer is None
            or buffer.device != template.device
            or buffer.dtype != template.dtype
            or buffer.numel() < required_elements
        ):
            return torch.empty(
                required_elements,
                dtype=template.dtype,
                device=template.device,
            )
        return buffer

    def dense_buffers(
        self,
        query: torch.Tensor,
        num_kv_heads: int,
        max_seq_len: int,
        reserve_seq_len: int | None = None,
        max_batch_size: int | None = None,
        max_sequence_capacity: int | None = None,
    ) -> _TurboQuantDenseBuffers:
        batch_size, num_query_heads, head_dim = query.shape
        if reserve_seq_len is None:
            reserve_seq_len = max_seq_len
        if reserve_seq_len < max_seq_len:
            raise ValueError(f"TurboQuant reserve_seq_len {reserve_seq_len} is smaller than max_seq_len {max_seq_len}.")
        if max_batch_size is None:
            max_batch_size = batch_size
        if max_sequence_capacity is None:
            max_sequence_capacity = reserve_seq_len
        if max_batch_size < batch_size or max_sequence_capacity < reserve_seq_len:
            raise ValueError("TurboQuant fused workspace capacity does not cover the requested shape.")

        layout = (
            query.device,
            query.dtype,
            num_query_heads,
            num_kv_heads,
            head_dim,
        )
        if layout != self.layout:
            self.key = None
            self.value = None
            self.query = None
            self.softmax_lse = None
            self.fia_workspace = None
            self.fia_workspace_shapes.clear()
            self.batch_capacity = 0
            self.sequence_capacity = 0
            self.layout = layout
        self.batch_capacity = self._grow_capacity(
            self.batch_capacity,
            batch_size,
            max_batch_size,
        )
        self.sequence_capacity = self._grow_capacity(
            self.sequence_capacity,
            reserve_seq_len,
            max_sequence_capacity,
        )
        dense_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        reserve_shape = (
            self.batch_capacity,
            num_kv_heads,
            self.sequence_capacity,
            head_dim,
        )
        dense_elements = math.prod(dense_shape)
        reserve_elements = math.prod(reserve_shape)
        self.key = self._flat_buffer(self.key, reserve_elements, query)
        self.value = self._flat_buffer(self.value, reserve_elements, query)
        query_capacity_elements = self.batch_capacity * num_query_heads * head_dim
        self.query = self._flat_buffer(self.query, query_capacity_elements, query)
        self.softmax_lse = self._flat_buffer(self.softmax_lse, 1, query)
        return _TurboQuantDenseBuffers(
            key=self.key[:dense_elements].view(dense_shape),
            value=self.value[:dense_elements].view(dense_shape),
            query=self.query[: query.numel()].view(batch_size, num_query_heads, head_dim),
            softmax_lse=self.softmax_lse[:1],
            key_capacity=self.key[:reserve_elements].view(reserve_shape),
            value_capacity=self.value[:reserve_elements].view(reserve_shape),
            query_capacity=self.query[:query_capacity_elements].view(
                self.batch_capacity,
                num_query_heads,
                head_dim,
            ),
            batch_capacity=self.batch_capacity,
            sequence_capacity=self.sequence_capacity,
        )

    def get_fia_workspace(
        self,
        shape_key: tuple[object, ...],
        factory: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if shape_key not in self.fia_workspace_shapes:
            candidate = factory()
            if (
                self.fia_workspace is None
                or candidate.device != self.fia_workspace.device
                or candidate.dtype != self.fia_workspace.dtype
            ):
                self.fia_workspace = candidate
                self.fia_workspace_shapes.clear()
            elif candidate.numel() > self.fia_workspace.numel():
                self.fia_workspace = candidate
            self.fia_workspace_shapes.add(shape_key)
        assert self.fia_workspace is not None
        return self.fia_workspace


class AscendTurboQuantMetadataBuilder(AscendAttentionMetadataBuilder):
    """Build Ascend metadata while retaining device-side sequence lengths."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._turboquant_page_table_builder = _TurboQuantPageTableBuilder(self.device)
        self._turboquant_workspace = _TurboQuantFusedWorkspace()

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        implementation = envs.VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION
        if implementation == "ascend_fused":
            return AttentionCGSupport.NEVER
        head_size = getattr(kv_cache_spec, "head_size", None)
        if (
            implementation == "auto"
            and isinstance(head_size, int)
            and _supports_ascend_fused_head_dim(head_size)
            and (
                has_turboquant_paged_attention()
                or has_turboquant_paged_dequant()
            )
        ):
            # ACLGraph would force auto onto the much slower packed path.
            # Prefer eager AscendC+CANN when that implementation is available;
            # grouped_gqa remains the explicit graph-capable selection.
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        metadata.seq_lens = common_attn_metadata.seq_lens[: common_attn_metadata.num_reqs]
        metadata.max_seq_len = common_attn_metadata.max_seq_len
        if envs.VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION in ("auto", "ascend_fused") and metadata.num_decodes:
            # Defer page-table construction until the implementation selects
            # eager fused decode. ACLGraph auto mode therefore does no unused
            # host work and captures only the packed Triton path.
            metadata.turboquant_page_table_builder = self._turboquant_page_table_builder
            metadata.turboquant_workspace = self._turboquant_workspace
        if (
            metadata.attn_state != AscendAttentionState.PrefillNoCache
            and metadata.num_decodes == common_attn_metadata.num_reqs
            and metadata.num_decode_tokens == common_attn_metadata.num_actual_tokens
        ):
            # Non-MTP speculative decoding is exposed as ChunkedPrefill by the
            # Ascend runner. All-decode metadata can still use the packed
            # sequential decode path without materializing dense K/V.
            metadata.attn_state = AscendAttentionState.DecodeOnly
        return metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
    ) -> AscendMetadata:
        metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        metadata.attn_state = AscendAttentionState.DecodeOnly
        # Capture with the shortest valid context for every query length.
        # Replay updates this persistent input with real sequence lengths.
        query_lens = metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]
        metadata.seq_lens.copy_(query_lens)
        return metadata


class AscendTurboQuantAttentionBackend(AscendAttentionBackend):
    """Ascend backend backed by packed KV operators and native FIA."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = _TURBOQUANT_CACHE_DTYPES

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type[AscendTurboQuantAttentionImpl]:
        return AscendTurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AscendTurboQuantMetadataBuilder]:
        return AscendTurboQuantMetadataBuilder

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

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_kv_cache_dtype(
        cls,
        kv_cache_dtype: CacheDType | None,
    ) -> bool:
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]
        for source, destination in zip(src_kv_cache, dst_kv_cache):
            source_pages = source[src_indices].to(destination.device).clone()
            destination[dst_indices] = source_pages

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]
        for kv_cache in kv_caches:
            source_pages = kv_cache[src_indices].clone()
            kv_cache[dst_indices] = source_pages


class AscendTurboQuantAttentionImpl(AscendAttentionBackendImpl):
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
            raise NotImplementedError("Ascend TurboQuant supports decoder self-attention only.")
        if sliding_window is not None:
            raise NotImplementedError("Ascend TurboQuant does not currently support sliding-window attention.")
        if logits_soft_cap is not None and logits_soft_cap <= 0:
            raise ValueError(f"logits_soft_cap must be positive, got {logits_soft_cap}.")
        self.logits_soft_cap = logits_soft_cap

        validate_turboquant_layout(kv_cache_dtype, head_size)
        self.tq_config = get_turboquant_config(
            kv_cache_dtype,
            head_size,
        )
        vllm_config = get_current_vllm_config()
        attention_config = vllm_config.attention_config
        self.max_num_kv_splits = attention_config.tq_max_kv_splits_for_cuda_graph
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.decode_implementation = envs.VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION
        if self.decode_implementation not in (
            "auto",
            "ascend_fused",
            "grouped_gqa",
            "reference",
        ):
            raise ValueError(
                "VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION must be auto, "
                "ascend_fused, grouped_gqa, or reference; got "
                f"{self.decode_implementation}."
            )
        fused_features_supported = self.alibi_slopes is None and self.logits_soft_cap is None
        fused_shape_supported = _supports_ascend_fused_head_dim(head_size)
        pipelined_shape_supported = (
            head_size == 128
            and num_heads == num_kv_heads * 8
            and self.tq_config.key_quant_bits == 4
            and self.tq_config.value_quant_bits == 4
            and self.tq_config.norm_correction
        )
        paged_attention_available = (
            pipelined_shape_supported
            and has_turboquant_paged_attention()
        )
        paged_dequant_available = has_turboquant_paged_dequant()
        self.ascend_pipelined_available = (
            self.decode_implementation in ("auto", "ascend_fused")
            and fused_features_supported
            and paged_attention_available
        )
        self.ascend_fused_available = (
            self.decode_implementation in ("auto", "ascend_fused")
            and fused_features_supported
            and fused_shape_supported
            and (paged_attention_available or paged_dequant_available)
        )
        if self.ascend_pipelined_available:
            eager_single_token_path = "ascend_pipelined"
        elif self.ascend_fused_available:
            eager_single_token_path = "ascend_dequant_fia"
        else:
            eager_single_token_path = "packed_triton"
        logger.info_once(
            "TurboQuant decode dispatch: requested=%s, eager_single_token=%s, "
            "paged_attention_schema=%s, paged_dequant_schema=%s, head_dim=%s, "
            "alibi=%s, logits_soft_cap=%s.",
            self.decode_implementation,
            eager_single_token_path,
            paged_attention_available,
            paged_dequant_available,
            head_size,
            self.alibi_slopes is not None,
            self.logits_soft_cap,
        )
        if self.decode_implementation == "auto" and not self.ascend_fused_available:
            logger.warning_once(
                "TurboQuant auto decode is falling back to packed Triton. "
                "Qwen3-32B long-context decode can be extremely slow on this "
                "path. Rebuild the paged-dequant custom op for the physical "
                "NPU, or set VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION="
                "ascend_fused to fail fast during startup."
            )
        if self.decode_implementation == "ascend_fused" and not fused_features_supported:
            raise NotImplementedError("Ascend TurboQuant fused decode does not support ALiBi or logits soft cap.")
        if self.decode_implementation == "ascend_fused":
            _validate_ascend_fused_head_dim(head_size)
            if not self.ascend_fused_available:
                raise RuntimeError(
                    "Ascend TurboQuant fused decode requires the pipelined "
                    "attention operator or the paged-dequant out operator. "
                    "Rebuild vllm-ascend after sourcing CANN."
                )

    def _ensure_constants(
        self,
        layer: AttentionLayer,
        device: torch.device,
        activation_dtype: torch.dtype,
    ) -> None:
        workspace_signature = (
            device,
            activation_dtype,
            self.decode_implementation,
            self.max_num_seqs,
        )
        if hasattr(layer, "_tq_ascend_constants_ready"):
            compute_rotation = getattr(layer, "_tq_ascend_compute_rotation", None)
            compute_rotation_ready = (
                compute_rotation is not None
                and compute_rotation.dtype == activation_dtype
                and compute_rotation.device == device
            )
            pipelined_rotation = getattr(layer, "_tq_ascend_pipelined_compute_rotation", None)
            pipelined_rotation_ready = (
                pipelined_rotation is not None
                and pipelined_rotation.dtype == activation_dtype
                and pipelined_rotation.device == device
            )
            if (
                compute_rotation_ready
                and pipelined_rotation_ready
                and getattr(layer, "_tq_ascend_workspace_signature", None) == workspace_signature
            ):
                return
            if not compute_rotation_ready:
                layer._tq_ascend_compute_rotation = _build_compute_rotation(
                    self.head_size,
                    str(device),
                    activation_dtype,
                )
            if not pipelined_rotation_ready:
                layer._tq_ascend_pipelined_compute_rotation = _build_pipelined_compute_rotation(
                    self.head_size,
                    str(device),
                    activation_dtype,
                )
        else:
            rotation = _build_hadamard(self.head_size, str(device))
            layer._tq_ascend_hadamard = rotation
            layer._tq_ascend_compute_rotation = _build_compute_rotation(
                self.head_size,
                str(device),
                activation_dtype,
            )
            layer._tq_ascend_pipelined_compute_rotation = _build_pipelined_compute_rotation(
                self.head_size,
                str(device),
                activation_dtype,
            )
            centroids = layer._tq_centroids.to(  # type: ignore[attr-defined]
                device=device,
                dtype=torch.float32,
            )
            centroids, _ = centroids.sort()
            layer._tq_ascend_centroids = centroids
            layer._tq_ascend_midpoints = (centroids[:-1] + centroids[1:]) / 2
            layer._tq_ascend_constants_ready = True

        rotation_dtype = torch.float32 if self.decode_implementation == "reference" else activation_dtype
        self._ensure_layer_workspace(
            layer,
            "_tq_ascend_query_rotation_buf",
            (self.max_num_seqs, self.num_heads, self.head_size),
            rotation_dtype,
            device,
        )
        self._ensure_layer_workspace(
            layer,
            "_tq_ascend_key_rotation_buf",
            (self.max_num_seqs * self.num_kv_heads, self.head_size),
            rotation_dtype,
            device,
        )
        if self.decode_implementation == "reference":
            self._ensure_layer_workspace(
                layer,
                "_tq_ascend_query_float_buf",
                (self.max_num_seqs, self.num_heads, self.head_size),
                torch.float32,
                device,
            )
        layer._tq_ascend_workspace_signature = workspace_signature

    @staticmethod
    def _ensure_layer_workspace(
        layer: AttentionLayer,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = getattr(layer, name, None)
        if (
            buffer is None
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.ndim != len(shape)
            or any(actual < required for actual, required in zip(buffer.shape, shape))
        ):
            buffer = torch.empty(shape, dtype=dtype, device=device)
            setattr(layer, name, buffer)
        return buffer

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        num_tokens = slot_mapping.shape[0]
        if num_tokens == 0:
            return
        self._ensure_constants(layer, key.device, key.dtype)
        key = key[:num_tokens].view(
            num_tokens,
            self.num_kv_heads,
            self.head_size,
        )
        value = value[:num_tokens].view(
            num_tokens,
            self.num_kv_heads,
            self.head_size,
        )
        key_rotation_workspace = layer._tq_ascend_key_rotation_buf
        num_key_vectors = num_tokens * self.num_kv_heads
        if key_rotation_workspace.shape[0] < num_key_vectors:
            key_rotation_workspace = None
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping[:num_tokens],
            layer._tq_ascend_hadamard,
            layer._tq_ascend_midpoints,
            key_bits=self.tq_config.key_quant_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_bits=self.tq_config.value_quant_bits,
            compute_rotation=(None if self.decode_implementation == "reference" else layer._tq_ascend_compute_rotation),
            rotated_key_out=key_rotation_workspace,
        )

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = metadata.num_actual_tokens
        self._run_fia(
            query[:num_tokens],
            key[:num_tokens],
            value[:num_tokens],
            metadata.actual_seq_lengths_q,
            metadata.actual_seq_lengths_q,
            metadata.attn_mask,
            output[:num_tokens],
            kv_block_size=128,
        )
        return output

    def _run_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        actual_seq_lengths_q: list[int],
        actual_seq_lengths_kv: list[int],
        attn_mask: torch.Tensor | None,
        output: torch.Tensor,
        *,
        kv_block_size: int,
    ) -> None:
        if self.alibi_slopes is not None or self.logits_soft_cap is not None:
            self._run_feature_prefill(
                query,
                key,
                value,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                output,
            )
            return
        attention_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=kv_block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        output.copy_(
            attention_output.view(
                query.shape[0],
                self.num_heads,
                self.head_size,
            )
        )

    def _run_ascend_fused_decode(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens_list: list[int],
        output: torch.Tensor,
        max_seq_len: int,
        workspace: _TurboQuantFusedWorkspace,
    ) -> torch.Tensor:
        """Decode packed cache with AscendC dequantization followed by FIA."""
        if len(seq_lens_list) != query.shape[0]:
            raise ValueError(
                "Ascend TurboQuant fused decode requires one sequence length "
                f"per query, got {len(seq_lens_list)} and {query.shape[0]}."
            )
        if max_seq_len <= 0 or max_seq_len < max(seq_lens_list):
            raise ValueError(
                "Ascend TurboQuant fused decode max_seq_len must cover all "
                f"sequences, got {max_seq_len} for {seq_lens_list}."
            )

        cache_sequence_capacity = block_tables.shape[1] * kv_cache.shape[1]
        if max(seq_lens_list) > cache_sequence_capacity:
            raise ValueError(
                "Ascend TurboQuant fused decode sequence length exceeds "
                f"block-table capacity {cache_sequence_capacity}: {seq_lens_list}."
            )
        reserve_seq_len = min(
            (
                (max_seq_len + _ASCEND_FUSED_SEQUENCE_WORKSPACE_GRANULARITY - 1)
                // _ASCEND_FUSED_SEQUENCE_WORKSPACE_GRANULARITY
            )
            * _ASCEND_FUSED_SEQUENCE_WORKSPACE_GRANULARITY,
            cache_sequence_capacity,
        )

        buffers = workspace.dense_buffers(
            query,
            self.num_kv_heads,
            max_seq_len,
            reserve_seq_len,
            max_batch_size=self.max_num_seqs,
            max_sequence_capacity=cache_sequence_capacity,
        )
        query_bnsd = buffers.query.unsqueeze(2)
        output_bnsd = output.unsqueeze(2)
        fia_kwargs = dict(
            query=query_bnsd,
            key=buffers.key,
            value=buffers.value,
            block_table=None,
            input_layout="BNSD",
            block_size=kv_cache.shape[1],
            actual_seq_lengths_kv=seq_lens_list,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=0,
        )
        fia_workspace_kwargs = {
            **fia_kwargs,
            "query": buffers.query_capacity.unsqueeze(2),
            "key": buffers.key_capacity,
            "value": buffers.value_capacity,
            "actual_seq_lengths_kv": [buffers.sequence_capacity] * buffers.batch_capacity,
        }
        fia_workspace = workspace.get_fia_workspace(
            (
                query.device.type,
                query.device.index,
                query.dtype,
                buffers.batch_capacity,
                self.num_heads,
                self.head_size,
                buffers.sequence_capacity,
            ),
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(**fia_workspace_kwargs),
        )
        turboquant_paged_dequant_out(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            page_table,
            layer._tq_ascend_centroids,
            buffers.key,
            buffers.value,
            max_seq_len=max_seq_len,
            key_bits=self.tq_config.key_quant_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_bits=self.tq_config.value_quant_bits,
            norm_correction=self.tq_config.norm_correction,
        )
        # K is stored in randomized Hadamard coordinates. Rotating Q preserves
        # QK exactly and avoids materializing an inverse-rotated dense K tensor.
        torch.matmul(
            query,
            layer._tq_ascend_compute_rotation,
            out=buffers.query,
        )
        torch_npu.npu_fused_infer_attention_score.out(
            **fia_kwargs,
            workspace=fia_workspace,
            out=[output_bnsd, buffers.softmax_lse],
        )
        return output

    def _run_ascend_pipelined_decode(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Run packed dequant, QK, online softmax, and PV in one AscendC op."""
        if max_seq_len <= 0:
            raise ValueError(
                "Ascend TurboQuant pipelined decode requires a positive "
                f"max_seq_len, got {max_seq_len}."
            )
        cache_sequence_capacity = block_tables.shape[1] * kv_cache.shape[1]
        if max_seq_len > cache_sequence_capacity:
            raise ValueError(
                "Ascend TurboQuant pipelined decode max_seq_len exceeds "
                f"block-table capacity {cache_sequence_capacity}: "
                f"{max_seq_len}."
            )
        rotated_query = layer._tq_ascend_query_rotation_buf[: query.shape[0]]
        torch.matmul(
            query,
            layer._tq_ascend_pipelined_compute_rotation,
            out=rotated_query,
        )
        turboquant_paged_attention_out(
            rotated_query,
            kv_cache,
            block_tables,
            seq_lens,
            layer._tq_ascend_centroids,
            output,
            scale=self.scale,
            max_seq_len=max_seq_len,
            key_bits=self.tq_config.key_quant_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_bits=self.tq_config.value_quant_bits,
            norm_correction=self.tq_config.norm_correction,
            max_num_splits=self.max_num_kv_splits,
        )
        return output

    def _run_feature_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        actual_seq_lengths_q: list[int],
        actual_seq_lengths_kv: list[int],
        output: torch.Tensor,
    ) -> None:
        if len(actual_seq_lengths_q) != len(actual_seq_lengths_kv):
            raise ValueError("TurboQuant feature prefill requires matching query/KV sequence boundaries.")
        query_start = 0
        kv_start = 0
        queries_per_kv_head = self.num_heads // self.num_kv_heads
        alibi_slopes = (
            self.alibi_slopes.view(
                self.num_kv_heads,
                queries_per_kv_head,
            )
            if self.alibi_slopes is not None
            else None
        )
        for query_end, kv_end in zip(
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
        ):
            query_seq = query[query_start:query_end]
            key_seq = key[kv_start:kv_end].float()
            value_seq = value[kv_start:kv_end].float()
            query_len = query_end - query_start
            kv_len = kv_end - kv_start
            history_len = kv_len - query_len
            if query_len <= 0 or history_len < 0:
                raise ValueError(
                    "TurboQuant feature prefill received invalid sequence "
                    f"lengths: query_len={query_len}, kv_len={kv_len}."
                )

            key_positions = torch.arange(kv_len, device=query.device)
            for tile_start in range(
                0,
                query_len,
                _FEATURE_PREFILL_QUERY_TILE_SIZE,
            ):
                tile_end = min(
                    tile_start + _FEATURE_PREFILL_QUERY_TILE_SIZE,
                    query_len,
                )
                query_tile = (
                    query_seq[tile_start:tile_end]
                    .float()
                    .reshape(
                        tile_end - tile_start,
                        self.num_kv_heads,
                        queries_per_kv_head,
                        self.head_size,
                    )
                )
                # [KV heads, query groups, query tile, KV tokens]. Keeping
                # GQA grouped avoids materializing Hq copies of K and V.
                scores = torch.einsum(
                    "tngd,knd->ngtk",
                    query_tile,
                    key_seq,
                )
                scores.mul_(self.scale)
                if self.logits_soft_cap is not None:
                    scores = self.logits_soft_cap * torch.tanh(scores / self.logits_soft_cap)

                query_positions = history_len + torch.arange(
                    tile_start,
                    tile_end,
                    device=query.device,
                )
                if alibi_slopes is not None:
                    relative_positions = key_positions[None, :] - query_positions[:, None]
                    scores.add_(alibi_slopes[:, :, None, None] * relative_positions)

                causal_mask = key_positions[None, :] <= query_positions[:, None]
                scores.masked_fill_(~causal_mask[None, None, :, :], float("-inf"))
                scores = F.softmax(scores, dim=-1, dtype=torch.float32)
                result = torch.einsum(
                    "ngtk,knd->tngd",
                    scores,
                    value_seq,
                ).reshape(
                    tile_end - tile_start,
                    self.num_heads,
                    self.head_size,
                )
                output[query_start + tile_start : query_start + tile_end].copy_(result.to(output.dtype))
            query_start = query_end
            kv_start = kv_end

        if query_start != query.shape[0] or kv_start != key.shape[0]:
            raise ValueError(
                "TurboQuant feature prefill boundaries do not consume all "
                f"tokens: query={query_start}/{query.shape[0]}, "
                f"KV={kv_start}/{key.shape[0]}."
            )

    def _decode_attention(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = metadata.num_decodes
        query_lens = _query_lens_from_cumulative(
            metadata.actual_seq_lengths_q,
            num_reqs,
        )
        if not query_lens:
            return output

        single_token_decode = all(query_len == 1 for query_len in query_lens)
        use_ascend_fused = _should_use_ascend_fused_decode(
            self.decode_implementation,
            self.ascend_fused_available,
            single_token_decode,
            _is_aclgraph_forward(),
        )
        if use_ascend_fused:
            max_seq_len = metadata.max_seq_len
            if max_seq_len is None:
                max_seq_len = max(metadata.seq_lens_list[:num_reqs])
            if self.ascend_pipelined_available:
                self._run_ascend_pipelined_decode(
                    layer,
                    query[:num_reqs],
                    kv_cache,
                    metadata.block_tables[:num_reqs],
                    metadata.seq_lens[:num_reqs],
                    output[:num_reqs],
                    max_seq_len,
                )
                return output
            if not isinstance(metadata.turboquant_workspace, _TurboQuantFusedWorkspace) or not isinstance(
                metadata.turboquant_page_table_builder,
                _TurboQuantPageTableBuilder,
            ):
                raise RuntimeError("Ascend TurboQuant fused metadata is missing its page-table builder or workspace.")
            if metadata.turboquant_page_table is None:
                metadata.turboquant_page_table = metadata.turboquant_page_table_builder.build(
                    metadata.seq_lens_list[:num_reqs],
                    kv_cache.shape[1],
                )
            self._run_ascend_fused_decode(
                layer,
                query[:num_reqs],
                kv_cache,
                metadata.block_tables[:num_reqs],
                metadata.seq_lens[:num_reqs],
                metadata.turboquant_page_table,
                metadata.seq_lens_list[:num_reqs],
                output[:num_reqs],
                max_seq_len,
                metadata.turboquant_workspace,
            )
            return output

        if all(query_len == query_lens[0] for query_len in query_lens):
            self._uniform_multi_token_decode(
                layer,
                query,
                kv_cache,
                metadata,
                output,
                query_lens[0],
            )
            return output

        token_start = 0
        for request_index, query_len in enumerate(query_lens):
            request_query = query[token_start : token_start + query_len]
            request_output = output[token_start : token_start + query_len]
            for token_index in range(query_len):
                self._launch_decode(
                    layer,
                    request_query[token_index : token_index + 1],
                    kv_cache,
                    metadata.block_tables[request_index : request_index + 1],
                    metadata.seq_lens[request_index : request_index + 1],
                    output=request_output[token_index : token_index + 1],
                    sequence_length_delta=-query_len + token_index + 1,
                    max_sequence_length=metadata.max_seq_len,
                )
            token_start += query_len
        return output

    def _uniform_multi_token_decode(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: AscendMetadata,
        output: torch.Tensor,
        query_len: int,
    ) -> None:
        num_reqs = metadata.num_decodes
        num_query_tokens = num_reqs * query_len
        query_by_request = query[:num_query_tokens].view(
            num_reqs,
            query_len,
            self.num_heads,
            self.head_size,
        )
        output_by_request = output[:num_query_tokens].view_as(query_by_request)
        block_tables = metadata.block_tables[:num_reqs]
        final_seq_lens = metadata.seq_lens[:num_reqs]

        # Keep workspace proportional to requests, not requests * query_len.
        # The static loop is captured in full for uniform multi-token graphs.
        for token_index in range(query_len):
            self._launch_decode(
                layer,
                query_by_request[:, token_index],
                kv_cache,
                block_tables,
                final_seq_lens,
                output=output_by_request[:, token_index],
                sequence_length_delta=-query_len + token_index + 1,
                max_sequence_length=metadata.max_seq_len,
            )

    def _launch_decode(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
        sequence_length_delta: int = 0,
        max_sequence_length: int | None = None,
    ) -> torch.Tensor:
        triton_implementation = "auto" if self.decode_implementation == "ascend_fused" else self.decode_implementation
        num_kv_splits = select_turboquant_num_kv_splits(
            batch_size=seq_lens.shape[0],
            num_query_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            max_num_kv_splits=self.max_num_kv_splits,
            max_sequence_length=max_sequence_length,
            implementation=triton_implementation,
            use_static_graph_splits=_is_aclgraph_forward(),
        )
        return triton_turboquant_decode_attention(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            layer._tq_ascend_hadamard,
            layer._tq_ascend_centroids,
            scale=self.scale,
            key_bits=self.tq_config.key_quant_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_bits=self.tq_config.value_quant_bits,
            norm_correction=self.tq_config.norm_correction,
            max_num_kv_splits=num_kv_splits,
            buffer_holder=layer,
            alibi_slopes=self.alibi_slopes,
            logits_soft_cap=self.logits_soft_cap,
            output=output,
            sequence_length_delta=sequence_length_delta,
            compute_rotation=(None if self.decode_implementation == "reference" else layer._tq_ascend_compute_rotation),
            implementation=triton_implementation,
            grouped_block_kv=select_turboquant_grouped_block_kv(
                max_sequence_length,
                num_kv_splits,
            ),
        )

    def _continuation_prefill(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        attn_mask: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        query_len = query.shape[0]
        history_len = seq_len - query_len
        if history_len <= 0:
            self._run_fia(
                query,
                key,
                value,
                [query_len],
                [query_len],
                attn_mask,
                output,
                kv_block_size=kv_cache.shape[1],
            )
            return

        if query_len <= _CONTINUATION_DECODE_THRESHOLD:
            seq_lens = torch.arange(
                history_len + 1,
                seq_len + 1,
                dtype=torch.int32,
                device=query.device,
            )
            expanded_block_table = block_table.expand(query_len, -1)
            self._launch_decode(
                layer,
                query,
                kv_cache,
                expanded_block_table,
                seq_lens,
                output=output,
                max_sequence_length=seq_len,
            )
            return

        block_size = kv_cache.shape[1]
        history_shape = (
            history_len,
            self.num_kv_heads,
            self.head_size,
        )
        # Do not retain context-sized buffers on every layer. The NPU caching
        # allocator can recycle these eager-only temporaries after FIA has
        # consumed them, while stream ordering preserves their lifetime.
        key_rotated = torch.empty(
            history_shape,
            dtype=query.dtype,
            device=query.device,
        )
        value_history = torch.empty_like(key_rotated)

        history_seq_lens = torch.tensor(
            [history_len],
            dtype=torch.int32,
            device=query.device,
        )
        history_start_locs = torch.tensor(
            [0, history_len],
            dtype=torch.int32,
            device=query.device,
        )
        triton_turboquant_dequant_paged_cache(
            kv_cache,
            block_table,
            history_seq_lens,
            history_start_locs,
            layer._tq_ascend_centroids,
            key_rotated,
            value_history,
            max_seq_len=history_len,
            key_bits=self.tq_config.key_quant_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_bits=self.tq_config.value_quant_bits,
            norm_correction=self.tq_config.norm_correction,
        )
        key_history = (
            key_rotated.float()
            .reshape(-1, self.head_size)
            .matmul(layer._tq_ascend_hadamard.T)
            .to(query.dtype)
            .view(history_len, self.num_kv_heads, self.head_size)
        )
        full_shape = (
            seq_len,
            self.num_kv_heads,
            self.head_size,
        )
        key_full = torch.empty(full_shape, dtype=query.dtype, device=query.device)
        value_full = torch.empty_like(key_full)
        key_full[:history_len].copy_(key_history)
        key_full[history_len:].copy_(key)
        value_full[:history_len].copy_(value_history)
        value_full[history_len:].copy_(value)
        self._run_fia(
            query,
            key_full,
            value_full,
            [query_len],
            [seq_len],
            attn_mask,
            output,
            kv_block_size=block_size,
        )

    def _prefill_requests(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: AscendMetadata,
        output: torch.Tensor,
        *,
        request_start: int,
        token_start: int,
    ) -> None:
        query_lens = _query_lens_from_cumulative(
            metadata.actual_seq_lengths_q,
            len(metadata.seq_lens_list),
        )
        current_token = token_start
        for request_index in range(request_start, len(query_lens)):
            query_len = query_lens[request_index]
            token_end = current_token + query_len
            self._continuation_prefill(
                layer,
                query[current_token:token_end],
                key[current_token:token_end],
                value[current_token:token_end],
                kv_cache,
                metadata.block_tables[request_index : request_index + 1],
                metadata.seq_lens_list[request_index],
                metadata.attn_mask,
                output[current_token:token_end],
            )
            current_token = token_end

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Ascend TurboQuant does not support fused output quantization.")
        if attn_metadata is None:
            return output.fill_(0)

        self._ensure_constants(layer, query.device, query.dtype)
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            if key is None or value is None:
                raise RuntimeError("TurboQuant first prefill requires the current raw K/V.")
            return self._prefill_attention(
                query,
                key,
                value,
                attn_metadata,
                output,
            )
        if (
            attn_metadata.attn_state
            in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            )
            and attn_metadata.num_prefills == 0
        ):
            return self._decode_attention(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )

        num_decode_tokens = attn_metadata.num_decode_tokens
        if num_decode_tokens > 0:
            self._decode_attention(
                layer,
                query[:num_decode_tokens],
                kv_cache,
                attn_metadata,
                output[:num_decode_tokens],
            )

        if attn_metadata.num_prefills > 0:
            if key is None or value is None:
                raise RuntimeError("TurboQuant continuation prefill requires current raw K/V.")
            self._prefill_requests(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                request_start=attn_metadata.num_decodes,
                token_start=num_decode_tokens,
            )
        return output
