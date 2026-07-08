# TurboQuant KV Cache 支持方案

## 背景

TurboQuant 是 KV cache 量化能力，应通过 `cache_config.cache_dtype`、KV
cache spec、cache 分配以及 dense attention 的 cache 读写路径接入。它不
应该作为普通模型权重量化方法注册，例如不应该通过 `--quantization ascend`
这类入口启用。

本设计的目标是把 TurboQuant 做成 vLLM Ascend 社区可维护、可扩展的通用
KV-cache 能力。Qwen3 dense 模型是第一阶段验证目标，但通用 cache layout、
attention backend 和 parser 中不能写死 Qwen3 专属假设。

## 范围

第一阶段合入目标是提供一套通用 TurboQuant KV-cache 框架，并优先启用一个
安全的模型族：

- 首个目标：Qwen3 dense 模型，包括 `Qwen/Qwen3-32B`
- Attention 类型：标准 decoder full attention，支持 GQA/MHA
- 权重 dtype：FP16/BF16
- KV cache dtype preset：
  - `turboquant_k8v4`
  - `turboquant_4bit_nc`
  - `turboquant_k3v4_nc`
  - `turboquant_3bit_nc`
- Engine 路径：优先支持 vLLM V1 model runner
- Cache layout：与 upstream vLLM 对齐的 packed TurboQuant slot cache

实现应面向使用相同标准 decoder attention 路径的模型族保持通用。模型启用
策略应通过能力检查表达，而不是把模型名写进 TurboQuant 核心 cache 逻辑。

第一阶段可以显式拒绝以下能力，但要保留清晰扩展点：

- MLA、DSA、SFA 等 compressed/sparse attention 路径
- Sliding window attention
- Speculative decoding
- Context parallel 或需要特殊 cache 搬运语义的 pipeline/context interleaving
- Prefix caching 和 KV transfer，直到 packed cache 格式经过验证
- ACLGraph 性能承诺

拒绝路径必须给出清晰错误信息。禁止静默 fallback 到 BF16 KV cache，否则用户
会误以为 TurboQuant 已经生效。

## 当前实现状态

当前 vLLM Ascend 已经落下第一版通用框架：

- `vllm_ascend/kv_cache/turboquant.py`
  提供 TurboQuant dtype 识别、preset/config helper、packed cache shape 计算、
  能力校验，以及纯 Torch reference store/dequant 实现。
- `vllm_ascend/kv_cache/higgs.py`
  提供可复用的 HIGGS reference quantizer，包括 normalized Hadamard 变换和带
  cache 的 Lloyd-Max centroid 求解。TurboQuant key 量化复用该模块，避免把
  算法 quantizer 和 packed cache layout 写在同一个文件里。
- `vllm_ascend/attention/turboquant.py`
  新增 `AscendTurboQuantAttentionBackend`，使用与 upstream 兼容的 packed cache
  tensor shape：
  `(num_blocks, block_size, num_kv_heads, slot_size_aligned)`。
- `vllm_ascend/platform.py`
  当用户设置 `--kv-cache-dtype turboquant_*` 时，将 dense attention 路由到
  Ascend TurboQuant backend，并显式拒绝 MLA、sparse、compression、310P 等暂
  不支持组合。
- `vllm_ascend/worker/model_runner_v1.py`
  对 TurboQuant attention cache 使用单个 packed `uint8` tensor 分配和 reshape，
  不再拆成标准 K/V 两个 tensor。
- `tests/ut/test_turboquant_kv_cache.py`
  覆盖 dtype 识别、packed shape 计算，以及 CPU reference store/dequant cache
  contract。
- `tests/ut/test_higgs_kv_cache.py`
  覆盖 HIGGS centroid metadata、Hadamard 正交性、quant/dequant contract，以及
  不支持 group 维度时的显式错误。

当前 store/dequant 路径是正确性优先的 reference 实现，还不是 fused
TurboQuant NPU kernel。后续在声明生产性能前，应替换为 CANN/custom-op 实现。
Python 接口、packed layout 和 backend 路由预计在替换高性能 kernel 时保持稳
定。

## upstream vLLM 参考

实现应尽量对齐 upstream vLLM 的 TurboQuant 架构。

`../vllm` 中相关文件：

- `vllm/config/cache.py`
  定义 TurboQuant preset 作为合法 `CacheDType`。
- `vllm/utils/torch_utils.py`
  将 TurboQuant cache dtype 映射到 `torch.uint8` 原始存储。
- `vllm/model_executor/layers/quantization/turboquant/config.py`
  定义 `TurboQuantConfig`、preset 参数、packed key/value size 和
  `slot_size_aligned`。
- `vllm/v1/kv_cache_interface.py`
  定义 `TQFullAttentionSpec`，其 `page_size_bytes` 基于
  `block_size * num_kv_heads * tq_slot_size`。
- `vllm/model_executor/layers/attention/attention.py`
  当 `kv_cache_dtype.startswith("turboquant_")` 时返回
  `TQFullAttentionSpec`。
- `vllm/v1/attention/backends/turboquant_attn.py`
  提供 `TurboQuantAttentionBackend`、packed cache shape、metadata builder、
  store 路径和 decode 路径。
- `vllm/v1/attention/ops/triton_turboquant_store.py`
  CUDA/Triton packed K/V store 实现。
- `vllm/v1/attention/ops/triton_turboquant_decode.py`
  基于 packed TurboQuant cache 的 CUDA/Triton decode attention 实现。

Ascend 实现不能直接复制 CUDA/Triton kernel。应复用 upstream 的 Python
config/spec 概念，并提供 Ascend 专属 operator 和 backend 逻辑。

## 当前 vLLM Ascend 代码审视

当前 vLLM Ascend 已经具备接入通用 TurboQuant backend 所需的大部分位置：

- `vllm_ascend/platform.py`
  负责选择 Ascend attention backend，并注册 Ascend 权重量化方法。TurboQuant
  不应加入 `NPUPlatform.supported_quantization`，因为它是 KV-cache dtype，
  不是权重量化方法。
- `vllm_ascend/attention/attention_v1.py`
  定义 `AscendAttentionBackend`、metadata builder 和 dense attention 实现。
  Ascend TurboQuant backend 应在这里新增或接入。
- `vllm_ascend/core/kv_cache_interface.py`
  定义 Ascend 专属 KV cache spec，例如 `AscendMLAAttentionSpec`。如果当前
  upstream vLLM 已经提供 `TQFullAttentionSpec`，vLLM Ascend 应直接复用；若
  没有，则添加语义一致的兼容 spec。
- `vllm_ascend/worker/model_runner_v1.py`
  负责 KV cache buffer 的分配和 reshape。当前 dense attention 路径会把普通
  attention cache 拆成 `(k_cache, v_cache)`。TurboQuant 应对齐 upstream，使用
  单个 packed cache tensor：

  ```text
  (num_blocks, block_size, num_kv_heads, slot_size_aligned)
  ```

- `vllm_ascend/worker/v2/attn_utils.py`
  有单独的 V2 allocation/reshape 路径。V2 支持应在 V1 稳定后镜像实现。
- `vllm_ascend/quantization/methods/kv_c8.py`
  展示了已有 KV-cache quantization 的接入模式。可以参考其 capability check
  和 backend 切换方式，但 TurboQuant 不应实现成模型权重量化 scheme。

之前提出的四 tensor layout：

```text
(
    k_quant_cache,
    k_scale_cache,
    v_quant_cache,
    v_scale_cache,
)
```

不应作为默认方案。它偏离 upstream，会增加未来同步和扩展成本。默认设计应以
upstream 兼容的 packed slot layout 为基线。

## 设计原则

1. TurboQuant 与权重量化解耦。
   `NPUPlatform.supported_quantization` 用于 `ascend`、`compressed-tensors`、
   `fp8` 等模型权重量化方法。TurboQuant 应作为 KV cache dtype 处理。

2. 对齐 upstream 用户接口。
   用户侧入口应为 `--kv-cache-dtype turboquant_*`。Preset 名称、packed size
   计算和 cache spec 语义应与 upstream vLLM 保持一致，除非存在明确的 Ascend
   硬件原因需要差异化。

3. 模型启用策略与核心 cache 逻辑分离。
   Qwen3 是首个支持模型族，但核心实现应检查 full decoder attention、head
   size、dtype、cache feature 兼容性等能力。后续支持新模型时，应扩展能力检
   查和测试，而不是修改核心实现里的模型名判断。

4. 先正确性，后融合性能。
   第一版 Ascend backend 可以使用清晰的 reference operator。后续 patch 再替
   换为 NPU fused store/decode operator。

5. Cache layout 必须可审计。
   Packed slot size、page size、tensor dtype 和 backend cache shape 必须有单
   元测试覆盖。

## 需要的代码改动

### 1. 复用或 backport upstream TurboQuant config

优先直接导入 upstream：

```python
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)
```

如果当前 pin 的 vLLM 版本没有这些符号，则在 vLLM Ascend 中添加一个小的兼容
模块：

```text
vllm_ascend/kv_cache/turboquant_config.py
```

兼容实现必须保持与 upstream `TurboQuantConfig` 一致的 preset 名称和字段，
尤其是 `key_packed_size`、`value_packed_size`、`slot_size` 和
`slot_size_aligned`。

不要创建 Qwen3 专属 parser。

### 2. 添加通用 TurboQuant 能力校验

添加校验 helper，例如：

```text
vllm_ascend/kv_cache/turboquant.py
```

建议 API：

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

Layer 校验应检查能力，而不是模型名：

- 仅 decoder full attention
- FP16/BF16 模型 dtype
- `head_size == head_size_v`
- head size 满足 `TurboQuantConfig` 支持范围
- 第一阶段不支持 sliding-window 路径

Runtime 校验应暂时拒绝以下全局能力：

- speculative decoding
- context parallel
- prefix caching，直到 block hashing 和 packed-cache reuse 验证完成
- KV transfer，直到 packed cache transfer 验证完成
- ACLGraph，除非已经显式测试

Qwen3 dense 支持应体现在支持矩阵和测试中，而不是作为 core helper 的唯一接受
模型类型。

### 3. 使用 upstream 兼容的 TurboQuant KV cache spec

Upstream 使用 `TQFullAttentionSpec`：

```python
@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    tq_slot_size: int = 0

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.tq_slot_size
```

vLLM Ascend 应在可用时复用 `vllm.v1.kv_cache_interface.TQFullAttentionSpec`。
如果需要兼容实现，应保持相同字段和 merge 语义。

该 spec 表示每个 token/head 对应一个 packed cache slot：

```text
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

默认路径不要把 TurboQuant 表示为拆开的 K/V/scale tensors。

### 4. 让 dense attention 返回 TurboQuant cache spec

当 `self.kv_cache_dtype.startswith("turboquant_")` 时，dense attention layer
应返回 `TQFullAttentionSpec`，与 upstream 保持一致。

目标逻辑：

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

这应在 shared dense attention layer 路径中完成。不要添加 Qwen3-only 分支。

### 5. 添加 Ascend TurboQuant attention backend

新增 Ascend 专属 backend，例如：

```text
vllm_ascend/attention/turboquant.py
```

建议类：

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

Backend 应暴露 upstream 兼容的 cache shape：

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

Backend 只声明支持 TurboQuant cache dtype 和 decoder attention。

### 6. 不通过模型名硬编码 backend 选择

更新：

```text
vllm_ascend/platform.py
```

或调用 `get_attn_backend` 的 dense attention 构造路径。

当 `kv_cache_dtype.startswith("turboquant_")` 时，为标准 dense decoder
attention 选择 `AscendTurboQuantAttentionBackend`。

选择条件应基于 attention 能力和 cache dtype：

- dense full attention
- 非 MLA/DSA/SFA
- 第一阶段不支持 sliding window
- dtype/head size 被支持

不要在 backend selector 中通过 `model_type == "qwen3"` 派发。Qwen3 应只是满
足这些通用检查的一个模型族。

### 7. 更新 KV cache 分配和 reshape

更新：

```text
vllm_ascend/worker/model_runner_v1.py
```

当前普通 dense attention 路径会把 attention cache allocation 拆成独立的 K
和 V raw tensor。应在 generic `AttentionSpec` split path 之前，为
`TQFullAttentionSpec` 添加专门分支。

分配行为：

- 为整个 packed cache tensor 分配一个 raw `torch.int8` buffer
- 和普通路径一样，在同一个 `KVCacheTensor.shared_by` group 内共享
- 即使第一阶段禁用 TurboQuant KV transfer，也保留 KV transfer 场景下的对齐
  行为

Reshape 行为：

- 将 raw buffer view 为 `torch.uint8`
- 使用 `attn_backend.get_kv_cache_shape(...)`
- 传入 `cache_dtype_str`，让 backend 计算正确 slot size
- 每层返回单个 packed cache tensor，而不是 `(k_cache, v_cache)`

实现必须校验：

- `kv_cache_tensor.size` 可被 `page_size_bytes` 整除
- 计算出的 block 数等于 `kv_cache_config.num_blocks`
- `page_size_bytes == block_size * num_kv_heads * tq_slot_size`
- raw byte 长度与 reshape 后 tensor byte 长度一致

### 8. 添加 Ascend TurboQuant store 和 decode operator wrapper

新增 operator wrapper 模块：

```text
vllm_ascend/attention/turboquant_ops.py
```

Backend 应调用 wrapper function，而不是把 packing 细节直接嵌在 attention 实
现里：

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

如果暂时没有 NPU fused operator，第一版可以使用 reference 实现。但 reference
实现仍必须读写与 upstream 一致的 packed cache layout。

热路径中避免对 NPU tensor 调用 `tensor.item()`。

### 9. 复用 upstream boundary skip layer 语义

Upstream `TurboQuantConfig.get_boundary_skip_layers(...)` 可为激进 preset 跳过
dense 模型的首尾若干层。vLLM Ascend 应在当前 vLLM 依赖提供该能力时保留这一
行为。

这样后续支持更大或更敏感模型时，layer skip policy 能与 upstream 保持一致，
而不是写成 Qwen3 专属特殊逻辑。

### 10. V2 runner 作为后续任务

第一阶段除非 serving 路径必须，否则不修改：

```text
vllm_ascend/worker/v2/model_runner.py
```

V1 稳定后，再在以下路径镜像相同的 `TQFullAttentionSpec` allocation/reshape：

```text
vllm_ascend/worker/v2/attn_utils.py
```

以及 V2 model runner。

## 测试计划

### 单元测试

新增测试，例如：

```text
tests/ut/kv_cache/test_turboquant.py
tests/ut/attention/test_turboquant.py
```

需要覆盖：

- TurboQuant dtype 识别
- unsupported dtype 拒绝
- 所有 upstream preset 的 `slot_size_aligned` 计算
- `TQFullAttentionSpec.page_size_bytes`
- TurboQuant dtype 下 dense attention 返回 `TQFullAttentionSpec`
- Qwen3 dense 模型校验成功
- 非 full-attention 或 sliding-window 校验失败
- V1 model runner 为 `TQFullAttentionSpec` 分配单个 packed cache tensor
- packed cache shape 和 dtype
- 小 tensor 上 store/decode reference round trip
- unsupported runtime feature 的错误信息

### NPU smoke test

在 NPU 上启动真实 Qwen3 dense 模型：

```bash
vllm serve Qwen/Qwen3-32B \
  --kv-cache-dtype turboquant_4bit_nc \
  --no-enable-prefix-caching
```

发送 OpenAI-compatible 请求：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "介绍一下 TurboQuant。"}],
    "max_tokens": 64
  }'
```

Smoke test 必须验证：

- 服务成功启动
- 首个请求返回 HTTP 200
- 输出非空
- 日志显示 TurboQuant cache dtype 已启用
- KV cache 显存分配低于 BF16 KV cache
- packed cache shape 匹配 `(num_blocks, block_size, num_kv_heads, slot_size)`

### 回归对比

使用同一组 prompt 对比 BF16 KV cache 和 TurboQuant KV cache：

- 短 prompt
- 接近目标上下文长度的长 prompt
- 多请求 batch

检查：

- 无运行时崩溃
- 无 shape 或 cache block assertion 失败
- 对选定 preset，输出质量漂移在可接受范围内
- KV cache 显存下降可观测

### 扩展性测试

Qwen3 稳定后，启用另一个共享标准 attention 路径的 dense decoder 模型。目的
是证明实现不是 Qwen3 专属。只有 NPU smoke 验证通过后，才把模型加入支持矩阵。

## 性能后续

Reference 实现可能比 BF16 KV cache 更慢。正确性验证完成后，按以下顺序优化：

1. 用 NPU custom operator 替换 Python/Torch packing。
2. 将 TurboQuant store 与现有 KV scatter 语义融合。
3. 实现直接读取 packed cache 的 NPU decode attention。
4. Decode 稳定后，再添加 continuation prefill 优化。
5. 只有 graph capture 支持 packed cache 和相关 operator 后，才重新启用 ACLGraph。

## 风险

- `page_size_bytes` 错误会导致 block allocation 或 cache view 错误。
- Packed cache layout 不匹配可能产生隐蔽精度问题。
- Per-token dequantization 可能带来性能回退。
- Prefix caching 和 KV transfer 可能需要为 packed single-tensor cache 添加序列
  化/传输支持。
- Context parallel 路径有专门的 KV cache load/gather 逻辑，需要独立适配后再启
  用。
- 偏离 upstream `TurboQuantConfig` 或 `TQFullAttentionSpec` 会增加未来 vLLM
  升级成本。

## 推荐 patch 拆分

推荐按以下顺序提交：

1. 复用或 backport upstream `TurboQuantConfig` 和 `TQFullAttentionSpec`。
2. 添加通用 TurboQuant capability/runtime validation。
3. 添加 Ascend TurboQuant backend skeleton 和 backend selection。
4. 添加 V1 packed cache allocation/reshape 支持。
5. 添加 Ascend reference store/decode operator。
6. 提供 Qwen3 dense NPU smoke 验证证据。
7. 添加第二个 dense 模型验证目标。
8. 后续 patch 添加 fused NPU operator 优化。

## 支持矩阵

| 模型族 | Attention 路径 | 状态 | 说明 |
| --- | --- | --- | --- |
| Qwen3 dense | Full decoder attention | 第一目标 | 包括 `Qwen/Qwen3-32B`，验证 GQA 和 head_dim 128。 |
| 其他 dense decoder 模型 | Full decoder attention | 计划中 | capability check 和 NPU smoke test 通过后启用。 |
| Sliding-window 模型 | Sliding-window attention | 初始不支持 | 需要单独的 cache spec/backend 行为。 |
| MLA/DSA/SFA 模型 | Compressed/sparse attention | 初始不支持 | 需要专门集成，不能直接复用 dense TurboQuant backend。 |
| Hybrid attention/Mamba 模型 | Mixed cache groups | 初始不支持 | 需要验证 layer skip policy 和 cache group。 |
