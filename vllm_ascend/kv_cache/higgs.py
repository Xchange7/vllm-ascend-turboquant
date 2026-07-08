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

import math
from collections.abc import Callable
from functools import lru_cache

import torch

HIGGS_SUPPORTED_BITS = (2, 3, 4)
_LLOYD_MAX_INTEGRATION_STEPS = 200
_LLOYD_MAX_MAX_ITER = 200
_LLOYD_MAX_TOL = 1e-10
_LLOYD_MAX_MIN_DENOMINATOR = 1e-15


def _validate_higgs_params(head_dim: int, bits: int) -> None:
    if head_dim <= 0:
        raise ValueError(f"HIGGS head_dim must be positive, got {head_dim}.")
    if head_dim & (head_dim - 1):
        raise ValueError(f"HIGGS Hadamard transform requires a power-of-two head_dim, got {head_dim}.")
    if bits not in HIGGS_SUPPORTED_BITS:
        raise ValueError(f"HIGGS supports {HIGGS_SUPPORTED_BITS} quantization bits, got {bits}.")


def build_hadamard(
    head_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized Hadamard matrix for one HIGGS vector group."""
    if head_dim <= 0:
        raise ValueError(f"HIGGS head_dim must be positive, got {head_dim}.")
    if head_dim & (head_dim - 1):
        raise ValueError(f"HIGGS Hadamard transform requires a power-of-two head_dim, got {head_dim}.")

    matrix = torch.tensor([[1.0]], device=device, dtype=dtype)
    while matrix.shape[0] < head_dim:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(head_dim)


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _trapz(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    n: int = _LLOYD_MAX_INTEGRATION_STEPS,
) -> float:
    """Trapezoidal numerical integration used by Lloyd-Max centroid solving."""
    step = (upper - lower) / n
    result = 0.5 * (f(lower) + f(upper))
    for i in range(1, n):
        result += f(lower + i * step)
    return result * step


def solve_lloyd_max(
    head_dim: int,
    bits: int,
    max_iter: int = _LLOYD_MAX_MAX_ITER,
    tol: float = _LLOYD_MAX_TOL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max centroids for HIGGS coordinates.

    After Hadamard rotation, each normalized coordinate is approximated as
    N(0, 1 / head_dim). This mirrors the upstream TurboQuant centroid solver
    while keeping vLLM Ascend's HIGGS reference path self-contained.
    """
    _validate_higgs_params(head_dim, bits)
    n_levels = 1 << bits
    sigma2 = 1.0 / head_dim
    sigma = math.sqrt(sigma2)
    lower = -3.5 * sigma
    upper = 3.5 * sigma

    def pdf(x: float) -> float:
        return _gaussian_pdf(x, sigma2)

    centroids = [lower + (upper - lower) * (level + 0.5) / n_levels for level in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[level] + centroids[level + 1]) / 2.0 for level in range(n_levels - 1)]
        edges = [lower * 3] + boundaries + [upper * 3]
        new_centroids = []
        for level in range(n_levels):
            a, b = edges[level], edges[level + 1]
            numerator = _trapz(lambda x: x * pdf(x), a, b)
            denominator = _trapz(pdf, a, b)
            if denominator > _LLOYD_MAX_MIN_DENOMINATOR:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[level])

        max_delta = max(abs(new_centroids[level] - centroids[level]) for level in range(n_levels))
        centroids = new_centroids
        if max_delta < tol:
            break

    boundaries = [(centroids[level] + centroids[level + 1]) / 2.0 for level in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=32)
def get_higgs_centroids(head_dim: int, bits: int) -> torch.Tensor:
    """Return cached HIGGS Lloyd-Max centroids for one vector group."""
    centroids, _ = solve_lloyd_max(head_dim, bits)
    return centroids


def quantize_higgs_vector(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference HIGGS quantization for vectors in the last dimension."""
    head_dim = tensor.shape[-1]
    _validate_higgs_params(head_dim, bits)

    hadamard = build_hadamard(head_dim, tensor.device).to(torch.float32)
    rotated = torch.matmul(tensor.to(torch.float32), hadamard)
    norm = rotated.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float16).tiny)
    normalized = rotated / norm

    centroids = get_higgs_centroids(head_dim, bits).to(
        device=tensor.device,
        dtype=torch.float32,
    )
    centroid_shape = (1,) * (normalized.dim() - 1) + (1, -1)
    distances = (normalized.unsqueeze(-1) - centroids.view(centroid_shape)).abs()
    quantized = distances.argmin(dim=-1).to(torch.uint8)
    return quantized, norm.to(torch.float16)


def dequantize_higgs_vector(
    quantized: torch.Tensor,
    norm: torch.Tensor,
    bits: int,
    norm_correction: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reference HIGGS dequantization for vectors in the last dimension."""
    head_dim = quantized.shape[-1]
    _validate_higgs_params(head_dim, bits)

    centroids = get_higgs_centroids(head_dim, bits).to(
        device=quantized.device,
        dtype=torch.float32,
    )
    rotated = centroids[quantized.to(torch.long)]
    if norm_correction:
        rotated_norm = rotated.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float16).tiny)
        rotated = rotated / rotated_norm
    rotated = rotated * norm.to(torch.float32)
    hadamard = build_hadamard(head_dim, quantized.device).to(torch.float32)
    return torch.matmul(rotated, hadamard).to(dtype)
