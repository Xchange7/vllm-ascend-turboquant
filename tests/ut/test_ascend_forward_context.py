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

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ascend_forward_context import (
    _get_attention_slot_mapping,
    set_ascend_forward_context,
)


def test_attention_slot_mapping_preserves_per_layer_cache_groups():
    native_slots = torch.tensor([128, 129], dtype=torch.int32)
    turboquant_slots = torch.tensor([384, 385], dtype=torch.int32)
    attn_metadata = {
        "model.layers.0.self_attn.attn": SimpleNamespace(
            slot_mapping=native_slots,
        ),
        "model.layers.2.self_attn.attn": SimpleNamespace(
            slot_mapping=turboquant_slots,
        ),
    }

    slot_mapping = _get_attention_slot_mapping(attn_metadata)

    assert set(slot_mapping) == {
        "model.layers.0.self_attn.attn",
        "model.layers.2.self_attn.attn",
    }
    assert slot_mapping["model.layers.0.self_attn.attn"] is native_slots
    assert slot_mapping["model.layers.2.self_attn.attn"] is turboquant_slots


def test_attention_slot_mapping_ignores_metadata_without_cache_addresses():
    slots = torch.tensor([128], dtype=torch.int32)
    attn_metadata = {
        "attention": SimpleNamespace(slot_mapping=slots),
        "linear_attention": SimpleNamespace(),
    }

    slot_mapping = _get_attention_slot_mapping(attn_metadata)

    assert set(slot_mapping) == {"attention"}
    assert slot_mapping["attention"] is slots


def test_attention_slot_mapping_is_empty_without_per_layer_metadata():
    assert _get_attention_slot_mapping(None) == {}
    assert _get_attention_slot_mapping([{}]) == {}


def test_ascend_forward_context_passes_attention_slot_mapping_to_vllm():
    slots = torch.tensor([128, 129], dtype=torch.int32)
    attn_metadata = {
        "model.layers.2.self_attn.attn": SimpleNamespace(
            slot_mapping=slots,
        ),
    }
    captured_kwargs = {}
    forward_context = SimpleNamespace()

    @contextmanager
    def capture_forward_context(**kwargs):
        captured_kwargs.update(kwargs)
        yield

    with (
        patch(
            "vllm_ascend.ascend_forward_context.set_forward_context",
            capture_forward_context,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.select_moe_comm_method",
            return_value=None,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.is_moe_model",
            return_value=False,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.enable_sp",
            return_value=False,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.flashcomm2_enable",
            return_value=False,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_dp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        set_ascend_forward_context(
            attn_metadata,
            SimpleNamespace(),
            num_tokens=2,
        ),
    ):
        pass

    slot_mapping = captured_kwargs["slot_mapping"]
    assert set(slot_mapping) == {"model.layers.2.self_attn.attn"}
    assert slot_mapping["model.layers.2.self_attn.attn"] is slots
