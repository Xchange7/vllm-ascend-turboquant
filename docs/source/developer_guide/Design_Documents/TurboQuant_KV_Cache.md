# TurboQuant KV Cache Support Plan

## Background

TurboQuant is a KV cache quantization feature. It should be integrated through
`cache_config.cache_dtype`, KV cache specification, cache allocation, and dense
attention cache read/write paths. It should not be registered as a normal model
weight quantization method such as `--quantization ascend`.

The goal of this design is to support TurboQuant in vLLM Ascend as a reusable
community feature. Qwen3 dense models should be the first validation target, but
the implementation must not hard-code Qwen3-only assumptions into the common
cache layout, attention backend, or parser.

## Scope

The first merged implementation should provide a generic TurboQuant KV-cache
framework and enable it for the first safe model family:

- First target: Qwen3 dense models, including `Qwen/Qwen3-32B`
- Attention type: standard decoder full attention with GQA/MHA
- Weight dtype: FP16/BF16
- KV cache dtype presets:
  - `turboquant_k8v4`
  - `turboquant_4bit_nc`
  - `turboquant_k3v4_nc`
  - `turboquant_3bit_nc`
- Engine path: vLLM V1 model runner first
- Cache layout: upstream-compatible packed TurboQuant slot cache

The implementation should be generic over model families that use the same
standard decoder attention path. Model-specific enablement should be expressed
through capability checks, not by embedding model names inside the core
TurboQuant cache code.

The first implementation may explicitly reject the following features while
leaving clear extension points:

- MLA, DSA, SFA, and other compressed or sparse attention paths
- Sliding window attention
- Speculative decoding
- Context parallel or pipeline/context interleaving features that require
  different cache movement semantics
- Prefix caching and KV transfer until the packed cache format is validated
- ACLGraph performance guarantees

The rejection should be done with clear error messages. Silent fallback to BF16
KV cache must be avoided because it makes the user believe TurboQuant is enabled
when it is not.

## Current Implementation Status

The initial vLLM Ascend implementation adds the generic framework described in
this design:

- `vllm_ascend/kv_cache/turboquant.py`
  Provides TurboQuant dtype detection, preset/config helpers, packed cache shape
  calculation, capability validation, and a portable Torch reference
  store/dequant implementation.
- `vllm_ascend/kv_cache/higgs.py`
  Provides the reusable HIGGS reference quantizer used by TurboQuant key
  quantization, including the normalized Hadamard transform and cached
  Lloyd-Max centroids. This keeps the algorithmic quantizer separate from the
  TurboQuant packed cache layout.
- `vllm_ascend/attention/turboquant.py`
  Adds `AscendTurboQuantAttentionBackend`, using the upstream-compatible packed
  cache tensor shape
  `(num_blocks, block_size, num_kv_heads, slot_size_aligned)`.
- `vllm_ascend/platform.py`
  Routes `--kv-cache-dtype turboquant_*` to the Ascend TurboQuant backend for
  dense attention and rejects unsupported MLA/sparse/compression/310P
  combinations.
- `vllm_ascend/worker/model_runner_v1.py`
  Allocates and reshapes TurboQuant attention cache as one packed `uint8`
  tensor instead of splitting it into standard K/V tensors.
- `tests/ut/test_turboquant_kv_cache.py`
  Covers dtype detection, packed shape calculation, and CPU reference
  store/dequant cache contract.
- `tests/ut/test_higgs_kv_cache.py`
  Covers HIGGS centroid metadata, Hadamard orthonormality, quant/dequant
  contracts, and unsupported group dimensions.

The current store/dequant path is intentionally a correctness-first reference
implementation. It is not yet a fused TurboQuant NPU kernel and should be
replaced by a CANN/custom-op implementation before claiming production
performance. The Python interface, packed layout, and backend routing are
intended to remain stable across that replacement.

## Upstream vLLM Reference

The implementation should follow upstream vLLM's TurboQuant architecture as
closely as possible.

Relevant upstream files in `../vllm`:

- `vllm/config/cache.py`
  Defines TurboQuant presets as valid `CacheDType` values.
- `vllm/utils/torch_utils.py`
  Maps TurboQuant cache dtypes to `torch.uint8` raw storage.
- `vllm/model_executor/layers/quantization/turboquant/config.py`
  Defines `TurboQuantConfig`, preset parameters, packed key/value sizes, and
  `slot_size_aligned`.
- `vllm/v1/kv_cache_interface.py`
  Defines `TQFullAttentionSpec`, whose `page_size_bytes` is based on
  `block_size * num_kv_heads * tq_slot_size`.
- `vllm/model_executor/layers/attention/attention.py`
  Returns `TQFullAttentionSpec` when `kv_cache_dtype.startswith("turboquant_")`.
- `vllm/v1/attention/backends/turboquant_attn.py`
  Provides `TurboQuantAttentionBackend`, packed cache shape, metadata builder,
  store path, and decode path.
- `vllm/v1/attention/ops/triton_turboquant_store.py`
  Implements CUDA/Triton packed K/V store.
- `vllm/v1/attention/ops/triton_turboquant_decode.py`
  Implements CUDA/Triton decode attention over packed TurboQuant cache.

The Ascend implementation should not copy CUDA/Triton kernels directly. It
should reuse upstream Python config/spec concepts and provide Ascend-specific
operators and backend logic.

## Current vLLM Ascend Code Review

The current vLLM Ascend code already has most of the places needed to integrate
a reusable TurboQuant backend:

- `vllm_ascend/platform.py`
  Selects Ascend attention backend classes and registers Ascend weight
  quantization methods. TurboQuant should not be added to
  `NPUPlatform.supported_quantization` because it is a KV-cache dtype, not a
  weight quantization method.
- `vllm_ascend/attention/attention_v1.py`
  Defines `AscendAttentionBackend`, metadata builders, and dense attention
  implementation. This is where an Ascend TurboQuant backend should be added or
  wired.
- `vllm_ascend/core/kv_cache_interface.py`
  Adds Ascend-specific KV cache specs such as `AscendMLAAttentionSpec`. If the
  installed upstream vLLM already provides `TQFullAttentionSpec`, vLLM Ascend
  should reuse it. If not, add a compatibility spec with the same semantics.
- `vllm_ascend/worker/model_runner_v1.py`
  Allocates and reshapes KV cache buffers. The dense attention path currently
  splits normal attention cache into `(k_cache, v_cache)`. TurboQuant should
  follow upstream and use one packed cache tensor:

  ```text
  (num_blocks, block_size, num_kv_heads, slot_size_aligned)
  ```

- `vllm_ascend/worker/v2/attn_utils.py`
  Has a separate V2 allocation/reshape path. V2 support should be mirrored
  after the V1 path is stable.
- `vllm_ascend/quantization/methods/kv_c8.py`
  Shows an existing pattern for KV-cache quantization integration. It should be
  used as a reference for capability checks and backend switching, but
  TurboQuant should not be implemented as a model weight quantization scheme.

The previous four-tensor layout proposal:

```text
(
    k_quant_cache,
    k_scale_cache,
    v_quant_cache,
    v_scale_cache,
)
```

should not be used as the default design. It diverges from upstream and makes
future compatibility harder. The upstream-compatible packed slot layout should
be the baseline.

## Design Principles

1. Keep TurboQuant separate from weight quantization.
   `NPUPlatform.supported_quantization` is for model weight quantization methods
   such as `ascend`, `compressed-tensors`, and `fp8`. TurboQuant should be
   handled as a KV cache dtype.

2. Match upstream public behavior.
   The user-facing interface should be `--kv-cache-dtype turboquant_*`. Preset
   names, packed size calculation, and cache spec semantics should match
   upstream vLLM unless there is a documented Ascend hardware reason to differ.

3. Keep model enablement separate from core cache logic.
   Qwen3 should be the first supported model family, but the core implementation
   should validate attention capabilities such as full decoder attention,
   supported head size, dtype, and cache feature compatibility. Future model
   support should be added by extending capability checks and tests.

4. Start with correctness before fused performance.
   The first Ascend backend can use clear reference operators if needed. A later
   patch should replace them with fused NPU operators for store and decode.

5. Keep the cache layout auditable.
   Packed slot size, page size, tensor dtype, and backend cache shape must be
   covered by unit tests.

## Required Code Changes

### 1. Reuse or backport upstream TurboQuant config

Prefer importing upstream:

```python
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)
```

If the pinned vLLM version does not include these symbols, add a small
compatibility module in vLLM Ascend:

```text
vllm_ascend/kv_cache/turboquant_config.py
```

The compatibility implementation must keep the same preset names and fields as
upstream `TurboQuantConfig`, especially `key_packed_size`,
`value_packed_size`, `slot_size`, and `slot_size_aligned`.

Do not create a Qwen3-specific parser.

### 2. Add generic TurboQuant capability validation

Add validation helpers, for example:

```text
vllm_ascend/kv_cache/turboquant.py
```

Suggested APIs:

```python
def is_turboquant_kv_cache_dtype(cache_dtype: str) -> bool: ...

def get_turboquant_config(cache_dtype: str, head_dim: int) -> TurboQuantConfig: ...

def validate_turboquant_layer(
    *,
    layer_name: str,
    attn_type: str,
    head_size: int,
    head_size_v: int,
    dtype: torch.dtype,
    sliding_window: int | None,
) -> None: ...

def validate_turboquant_runtime(vllm_config: VllmConfig) -> None: ...
```

The layer validation should check capabilities, not model names:

- decoder full attention only
- FP16/BF16 model dtype
- `head_size == head_size_v`
- supported TurboQuant head size according to `TurboQuantConfig`
- no sliding-window path in the first patch

The runtime validation should reject temporarily unsupported global features:

- speculative decoding
- context parallel
- prefix caching, until block hashing and packed-cache reuse are validated
- KV transfer, until packed cache transfer is validated
- ACLGraph, unless explicitly tested

Qwen3 dense model support should be documented in the support matrix and tests,
not embedded as the only accepted model type in core helpers.

### 3. Use an upstream-compatible TurboQuant KV cache spec

Upstream uses `TQFullAttentionSpec` with:

```python
@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    tq_slot_size: int = 0

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.tq_slot_size
```

vLLM Ascend should reuse `vllm.v1.kv_cache_interface.TQFullAttentionSpec` when
available. If a compatibility implementation is required, it should have the
same fields and merge semantics.

The spec should represent one packed cache slot per token/head:

```text
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

Do not represent TurboQuant as separate K/V/scale tensors in the default path.

### 4. Make dense attention return the TurboQuant cache spec

The dense attention layer should return `TQFullAttentionSpec` when
`self.kv_cache_dtype.startswith("turboquant_")`, matching upstream.

The desired logic is:

```python
if self.kv_cache_dtype.startswith("turboquant_"):
    tq_config = TurboQuantConfig.from_cache_dtype(
        self.kv_cache_dtype,
        self.head_size,
    )
    return TQFullAttentionSpec(
        block_size=block_size,
        num_kv_heads=self.num_kv_heads,
        head_size=self.head_size,
        head_size_v=self.head_size_v,
        dtype=self.kv_cache_torch_dtype,
        tq_slot_size=tq_config.slot_size_aligned,
    )
```

This should be done in the shared dense attention layer path. Do not add a
Qwen3-only branch.

### 5. Add an Ascend TurboQuant attention backend

Add an Ascend-specific backend, for example:

```text
vllm_ascend/attention/turboquant.py
```

Suggested classes:

```python
class AscendTurboQuantAttentionBackend(AttentionBackend): ...

class AscendTurboQuantMetadata(AttentionMetadata): ...

class AscendTurboQuantMetadataBuilder(
    AttentionMetadataBuilder[AscendTurboQuantMetadata]
): ...

class AscendTurboQuantAttentionImpl(
    AttentionImpl[AscendTurboQuantMetadata]
): ...
```

The backend should expose an upstream-compatible cache shape:

```python
@staticmethod
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "turboquant_4bit_nc",
) -> tuple[int, ...]:
    tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
    return (
        num_blocks,
        block_size,
        num_kv_heads,
        tq_config.slot_size_aligned,
    )
```

The backend should report support only for TurboQuant cache dtypes and decoder
attention.

### 6. Wire backend selection without hard-coding model names

Update:

```text
vllm_ascend/platform.py
```

or the dense attention construction path that calls `get_attn_backend`.

When `kv_cache_dtype.startswith("turboquant_")`, select
`AscendTurboQuantAttentionBackend` for standard dense decoder attention.

The selection should be based on attention capability and cache dtype:

- dense full attention
- not MLA/DSA/SFA
- not sliding window for the first patch
- supported dtype/head size

Do not dispatch on `model_type == "qwen3"` in the backend selector. Qwen3 should
be one model family that satisfies the generic checks.

### 7. Update KV cache allocation and reshape

Update:

```text
vllm_ascend/worker/model_runner_v1.py
```

The current normal dense attention path splits each attention cache allocation
into separate raw K and V tensors. Add a specific branch for
`TQFullAttentionSpec` before the generic `AttentionSpec` split path.

Allocation behavior:

- allocate one raw `torch.int8` buffer for the entire packed cache tensor
- share it across layers in the same `KVCacheTensor.shared_by` group as normal
- preserve existing alignment behavior when KV transfer is enabled, even if
  TurboQuant KV transfer remains disabled initially

Reshape behavior:

- reshape the raw buffer as `torch.uint8`
- use `attn_backend.get_kv_cache_shape(...)`
- pass `cache_dtype_str` so the backend can compute the correct slot size
- return a single packed cache tensor for the layer, not `(k_cache, v_cache)`

The implementation must verify:

- `kv_cache_tensor.size` is divisible by `page_size_bytes`
- the computed number of blocks equals `kv_cache_config.num_blocks`
- `page_size_bytes == block_size * num_kv_heads * tq_slot_size`
- raw byte length matches the reshaped tensor byte length

### 8. Add Ascend TurboQuant store and decode operators

Add a module for operator wrappers:

```text
vllm_ascend/attention/turboquant_ops.py
```

The backend should call wrapper functions instead of directly embedding packing
details in the attention implementation:

```python
def ascend_turboquant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    tq_config: TurboQuantConfig,
) -> None: ...

def ascend_turboquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: AscendTurboQuantMetadata,
    tq_config: TurboQuantConfig,
) -> torch.Tensor: ...
```

The first version may use a reference implementation if no NPU fused operator is
available. The reference implementation must still write and read the same
packed cache layout as upstream.

Avoid `tensor.item()` on NPU tensors in hot paths.

### 9. Use boundary skip layers through upstream semantics

Upstream `TurboQuantConfig.get_boundary_skip_layers(...)` can skip first/last
dense layers for aggressive presets. vLLM Ascend should preserve this behavior
when the current vLLM dependency provides it.

This makes future support for larger or more sensitive models easier because
layer skip policy remains consistent with upstream rather than encoded as
Qwen3-specific special cases.

### 10. Keep v2 runner as a follow-up

Do not modify:

```text
vllm_ascend/worker/v2/model_runner.py
```

in the first patch unless the serving path requires it. Once V1 is validated,
mirror the same `TQFullAttentionSpec` allocation and reshape behavior in:

```text
vllm_ascend/worker/v2/attn_utils.py
```

and the V2 model runner.

## Test Plan

### Unit tests

Add tests under `tests/ut/`, for example:

```text
tests/ut/kv_cache/test_turboquant.py
tests/ut/attention/test_turboquant.py
```

Required unit coverage:

- TurboQuant dtype detection
- unsupported dtype rejection
- all upstream preset `slot_size_aligned` calculations
- `TQFullAttentionSpec.page_size_bytes`
- dense attention returns `TQFullAttentionSpec` for TurboQuant dtypes
- Qwen3 dense model validation success
- non-full-attention or sliding-window validation failure
- V1 model runner allocates one packed cache tensor for `TQFullAttentionSpec`
- packed cache shape and dtype
- store/decode reference round trip on small tensors
- unsupported runtime feature rejection messages

### NPU smoke tests

Run a real Qwen3 dense model service on NPU:

```bash
vllm serve Qwen/Qwen3-32B \
  --kv-cache-dtype turboquant_4bit_nc \
  --no-enable-prefix-caching
```

Then send an OpenAI-compatible request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "介绍一下 TurboQuant。"}],
    "max_tokens": 64
  }'
```

The smoke test must verify:

- service starts successfully
- first request returns HTTP 200
- output is non-empty
- logs show TurboQuant cache dtype is enabled
- allocated KV cache memory is lower than BF16 KV cache
- packed cache shape matches `(num_blocks, block_size, num_kv_heads, slot_size)`

### Regression comparison

Compare BF16 KV cache and TurboQuant KV cache on the same prompt set:

- short prompt
- long prompt near the target context length
- multi-request batch

Check:

- no runtime crash
- no shape or cache block assertion failure
- acceptable output quality drift for the selected preset
- measurable KV cache memory reduction

### Extension tests

After Qwen3 is stable, enable one additional dense decoder model that shares the
same standard attention path. The purpose is to prove that the implementation is
not Qwen3-specific. Add the model to the support matrix only after NPU smoke
validation passes.

## Performance Follow-up

The reference implementation may be slower than BF16 KV cache. After correctness
is validated, optimize in this order:

1. Replace Python/Torch packing with NPU custom operators.
2. Fuse TurboQuant store with existing KV scatter semantics.
3. Implement NPU decode attention directly over the packed cache.
4. Add continuation prefill optimization after decode is stable.
5. Re-enable ACLGraph only after graph capture supports the packed cache and
   operators.

## Risks

- Incorrect `page_size_bytes` will cause wrong block allocation or invalid cache
  views.
- Packed cache layout mismatch will produce silent accuracy issues.
- Per-token dequantization may introduce performance regressions.
- Prefix caching and KV transfer may require serialization support for the
  packed single-tensor cache.
- Context parallel paths load and gather KV cache in specialized ways and should
  remain disabled until explicitly adapted.
- Diverging from upstream `TurboQuantConfig` or `TQFullAttentionSpec` will make
  future vLLM upgrades harder.

## Expected Patch Breakdown

Recommended patch order:

1. Reuse or backport upstream `TurboQuantConfig` and `TQFullAttentionSpec`.
2. Add generic TurboQuant capability/runtime validation.
3. Add Ascend TurboQuant backend skeleton and backend selection.
4. Add V1 packed cache allocation/reshape support.
5. Add Ascend reference store/decode operators.
6. Add Qwen3 dense NPU smoke validation evidence.
7. Add a second dense-model validation target.
8. Add fused NPU operator optimization in later patches.

## Support Matrix

| Model family | Attention path | Status | Notes |
| --- | --- | --- | --- |
| Qwen3 dense | Full decoder attention | First target | Includes `Qwen/Qwen3-32B`; validates GQA and head_dim 128. |
| Other dense decoder models | Full decoder attention | Planned | Enable after capability checks and NPU smoke tests pass. |
| Sliding-window models | Sliding-window attention | Not supported initially | Needs separate cache spec/backend behavior. |
| MLA/DSA/SFA models | Compressed/sparse attention | Not supported initially | Needs dedicated integration, not dense TurboQuant backend. |
| Hybrid attention/Mamba models | Mixed cache groups | Not supported initially | Requires layer skip policy and cache group validation. |
