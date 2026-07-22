#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Screen TurboQuant quantization error independently of implementation parity."""

from __future__ import annotations

import math
from typing import Any, NamedTuple


class QuantizationThresholds(NamedTuple):
    max_key_nmse: float
    max_value_nmse: float
    min_key_cosine: float
    min_value_cosine: float


# These are regression-screening limits, not paper-level accuracy claims. They
# leave headroom above the deterministic random-input measurements while still
# catching broken packing, scaling, rotation, or centroid selection.
DEFAULT_THRESHOLDS = {
    "turboquant_4bit_nc": QuantizationThresholds(0.03, 0.03, 0.98, 0.98),
    "turboquant_k3v4_nc": QuantizationThresholds(0.08, 0.03, 0.95, 0.98),
    "turboquant_3bit_nc": QuantizationThresholds(0.08, 0.08, 0.95, 0.95),
}


def resolve_thresholds(
    cache_dtype: str,
    *,
    max_key_nmse: float | None = None,
    max_value_nmse: float | None = None,
    min_key_cosine: float | None = None,
    min_value_cosine: float | None = None,
) -> QuantizationThresholds:
    try:
        thresholds = DEFAULT_THRESHOLDS[cache_dtype]
    except KeyError as error:
        raise ValueError(f"No quantization thresholds for {cache_dtype!r}.") from error
    overrides = {
        "max_key_nmse": max_key_nmse,
        "max_value_nmse": max_value_nmse,
        "min_key_cosine": min_key_cosine,
        "min_value_cosine": min_value_cosine,
    }
    resolved = thresholds._replace(**{name: value for name, value in overrides.items() if value is not None})
    nmse_limits = (resolved.max_key_nmse, resolved.max_value_nmse)
    cosine_limits = (resolved.min_key_cosine, resolved.min_value_cosine)
    if not all(math.isfinite(value) and value >= 0 for value in nmse_limits):
        raise ValueError("NMSE thresholds must be finite and non-negative.")
    if not all(math.isfinite(value) and -1 <= value <= 1 for value in cosine_limits):
        raise ValueError("Cosine thresholds must be finite values in [-1, 1].")
    return resolved


def evaluate_quantization_quality(
    key_metrics: dict[str, float],
    value_metrics: dict[str, float],
    thresholds: QuantizationThresholds,
) -> dict[str, Any]:
    metric_values = {
        "key_nmse": float(key_metrics["nmse"]),
        "value_nmse": float(value_metrics["nmse"]),
        "key_cosine": float(key_metrics["cosine_similarity"]),
        "value_cosine": float(value_metrics["cosine_similarity"]),
    }
    checks = {
        "finite": all(math.isfinite(value) for value in metric_values.values()),
        "key_nmse": metric_values["key_nmse"] <= thresholds.max_key_nmse,
        "value_nmse": metric_values["value_nmse"] <= thresholds.max_value_nmse,
        "key_cosine": metric_values["key_cosine"] >= thresholds.min_key_cosine,
        "value_cosine": metric_values["value_cosine"] >= thresholds.min_value_cosine,
    }
    return {
        "passed": all(checks.values()),
        "thresholds": thresholds._asdict(),
        "metrics": metric_values,
        "checks": checks,
    }
