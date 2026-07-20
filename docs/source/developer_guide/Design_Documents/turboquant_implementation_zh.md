# TurboQuant 在 vLLM Ascend 中的实现逻辑

## 1. 文档范围

本文说明 TurboQuant KV Cache 在 vLLM Ascend 中的代码结构、数据布局和运行流程。
当前实现基于 vLLM `0.20.2`/`0.20.2+empty` 提供的 TurboQuant core 接口，并使用
Triton-Ascend 实现 NPU 侧的 cache 写入、packed decode 和完整反量化算子。
当前能力边界和后续解阻条件见 [TurboQuant Ascend 当前功能限制](turboquant_limitations_zh.md)。

TurboQuant 在这里量化的是推理时动态生成的 KV Cache，而不是模型权重。因此：

- 不需要先用 ModelSlim 生成 TurboQuant 模型；
- 模型权重仍按照原有 BF16、FP16 或独立的权重量化方式加载；
- `--kv-cache-dtype turboquant_4bit_nc` 控制的是 KV Cache 存储格式和 attention backend。

## 2. 代码分层

| 层次 | 文件 | 职责 |
| --- | --- | --- |
| vLLM core | `vllm/config/cache.py` | 定义 TurboQuant cache dtype |
| vLLM core | `vllm/model_executor/layers/quantization/turboquant/` | 定义配置、码本和 slot 大小 |
| vLLM core | `vllm/v1/kv_cache_interface.py` | 定义 `TQFullAttentionSpec` |
| 平台路由 | `vllm_ascend/platform.py` | 校验能力并选择 Ascend TurboQuant backend |
| Cache 工具 | `vllm_ascend/kv_cache/turboquant.py` | 解析配置、计算 shape、集中校验限制 |
| Model runner | `vllm_ascend/worker/model_runner_v1.py` | 分配并 reshape 单一 packed cache tensor |
| Attention | `vllm_ascend/attention/turboquant.py` | metadata、cache update 和 attention 路径调度 |
| Store 算子 | `vllm_ascend/ops/triton/turboquant_store.py` | K/V 量化、bit packing 和分页写入 |
| Decode 算子 | `vllm_ascend/ops/triton/turboquant_decode.py` | packed decode 和完整 cache 反量化 |

这个分层刻意避免模型名称判断。Qwen3 是首个验证目标，但 backend 根据 cache dtype、
attention spec、head dimension 和能力组合工作，后续普通 dense decoder 可以复用同一实现。

## 3. 初始化调用链

### 3.1 解析 cache dtype

用户通过以下参数启用 TurboQuant：

```bash
vllm serve MODEL \
    --kv-cache-dtype turboquant_4bit_nc \
    --block-size 128
```

首次正确性验证建议加 `--enforce-eager`。完成
`scripts/turboquant_triton/correctness/run_aclgraph_smoke.sh` 后可去掉该参数，TurboQuant backend 会仅对
uniform single/multi-token decode 声明 ACLGraph 支持；prefill 和不满足条件的 batch 不进入该图路径。

vLLM core 将参数写入 `CacheConfig.cache_dtype`。在构造每个 attention layer 时，vLLM 会：

1. 根据 preset 创建 `TurboQuantConfig`；
2. 注册 Lloyd-Max centroids；
3. 预分配 split-KV decode workspace；
4. 为启用 TurboQuant 的 layer 返回 `TQFullAttentionSpec`；
5. 自动将首两层和末两层放入 `kv_cache_dtype_skip_layers`，这些层继续使用普通 cache。

边界层保护是 vLLM core 的策略，不是 Ascend backend 中写死的 Qwen 层号。

### 3.2 选择 Ascend backend

`NPUPlatform.get_attn_backend_cls()` 收到 per-layer `AttentionSelectorConfig` 后进行能力校验。
当该 layer 的 cache dtype 以 `turboquant_` 开头时，返回：

```text
vllm_ascend.attention.turboquant.AscendTurboQuantAttentionBackend
```

被跳过的首尾层，其 per-layer cache dtype 是 `auto`，仍然选择普通
`AscendAttentionBackend`。因此一个模型中可以同时存在 native KV Cache group 和
TurboQuant KV Cache group。

### 3.3 分配 packed cache

普通 Ascend attention 将 K cache 和 V cache 分成两个 tensor。TurboQuant 将 K、V 和
量化元数据打包在同一个 slot 中，因此 `NPUModelRunner` 对 `TQFullAttentionSpec` 使用单一
byte buffer，并 reshape 为：

```text
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

这里没有普通 cache 的 leading K/V dimension。scheduler 只关心每个 page 的字节数，
不解析 slot 内部字段；backend 和 Triton 算子共同维护 slot layout 契约。

## 4. Packed slot 布局

设 attention head dimension 为 `D`。

### 4.1 Key 区域

当前 Ascend 路径支持 3-bit 或 4-bit Lloyd-Max key：

```text
[packed centroid indices | original key norm(fp16)]
```

字段大小为：

```text
key_index_bytes = ceil(D * key_bits / 8)
key_packed_size = key_index_bytes + 2
```

### 4.2 Value 区域

Value 使用 per-token、per-KV-head 的 affine quantization：

```text
[packed value indices | scale(fp16) | minimum(fp16)]
```

反量化公式为：

```text
value = quantized_index * scale + minimum
```

字段大小为：

```text
value_data_bytes = ceil(D * value_bits / 8)
value_packed_size = value_data_bytes + 4
```

整个 slot 会向偶数字节对齐，使 FP16 metadata 可以通过同一 buffer 的 FP16 view 访问。

当 `D=128` 时：

| Cache dtype | Key bits | Value bits | Slot bytes |
| --- | ---: | ---: | ---: |
| `turboquant_4bit_nc` | 4 | 4 | 134 |
| `turboquant_k3v4_nc` | 3 | 4 | 118 |
| `turboquant_3bit_nc` | 3 | 3 | 102 |

## 5. KV Cache 写入逻辑

`Attention.forward()` 在 attention 计算前通过 `unified_kv_cache_update()` 调用
`AscendTurboQuantAttentionImpl.do_kv_cache_update()`。主要步骤如下。

### 5.1 Key 处理

对每个 token 和 KV head：

1. 计算原始 key 的 L2 norm；
2. 将 key 归一化；
3. 乘确定性的随机符号 Hadamard rotation `R = D @ H`，得到旋转后的 key；
4. 使用相邻 centroids 的 midpoint 做二分查找；
5. 得到每个维度的 Lloyd-Max centroid index；
6. 将 3-bit 或 4-bit index 打包为 byte；
7. 将原始 key norm 以 FP16 写入 key payload 末尾。

随机符号 Hadamard rotation 保持内积结构，同时避免结构化 key 在纯 Sylvester Hadamard
下发生能量集中，使各维度分布更适合标量量化。原始 norm 单独保存，使 decode 时可以恢复
key 的尺度。

### 5.2 Value 处理

对每个 token 和 KV head：

1. 计算该 value vector 的 minimum 和 maximum；
2. 根据 3-bit 或 4-bit levels 计算 scale；
3. 执行 affine quantization；
4. 将 indices 打包为 byte；
5. 保存 FP16 scale 和 minimum。

### 5.3 Paged cache 定位

`slot_mapping` 中的逻辑 slot 被转换为：

```text
physical_block = slot // block_size
position       = slot % block_size
```

最终地址由 block stride、position stride 和 KV-head stride 组合。`slot_mapping < 0`
表示 scheduler padding，Triton kernel 必须直接返回，不能写入任何 cache page。

## 6. Decode 计算逻辑

Decode 不会先创建完整 FP16 K/V tensor，而是直接读取 packed cache。

### 6.1 Query rotation

Query 乘同一个 rotation `R`。由于 matrix 是正交的：

```text
(QR) · (KR) = Q · K
```

因此可以在旋转空间计算 attention score。

### 6.2 Stage 1: split-KV attention

`_turboquant_decode_stage1` 使用固定数量的 KV splits。每个 program 处理：

```text
(batch index, query head, split index)
```

处理流程为：

1. 从 block table 将 token position 映射到 physical block；
2. 读取并 unpack key centroid indices；
3. 根据 indices gather centroids；
4. 可选执行 norm correction，将 centroid vector 重新归一化；
5. 乘保存的原始 key norm，计算 QK score；
6. 按配置执行 logits soft cap，并按 query head 加入 ALiBi 相对位置 bias；
7. 使用 online softmax 更新当前 split 的 max、sum 和 value accumulator；
8. 按需 unpack 并反量化 value；
9. 写出 partial output 和该 split 的 log-sum-exp。

该路径不会物化完整历史 K/V，因此是 TurboQuant decode 节省显存和带宽的核心。

### 6.3 Stage 2: 合并 splits

`_turboquant_decode_stage2` 根据各 split 的 log-sum-exp 重新缩放 partial outputs，合并成
最终 attention output。该过程保持跨 split softmax 的数值稳定性。

## 7. Prefill 路径

### 7.1 首次 prefill

`PrefillNoCache` 时，本轮原始 K/V 已经存在，无需从 packed cache 读回。实现直接调用
`torch_npu.npu_fused_infer_attention_score()`，同时 cache update 算子将 K/V 压缩写入 cache，
供后续 decode 使用。

### 7.2 Continuation prefill

发生 prefix hit、chunked prefill 或 mixed batch 时，attention 必须同时读取历史 cache 和本轮
token。当前实现按 query chunk 大小选择：

1. `query_len <= 128` 时，为每个 query token 构造递增的有效 KV 长度，直接复用 packed
   decode；
2. 更大的 chunk 只根据 block table 反量化历史 prefix，本轮 K/V 保持原始 dtype；
3. 对历史旋转空间 K 乘 inverse rotation `R.T`，再与当前 K/V 拼接并调用 NPU FIA；
4. 历史反量化使用 eager-only 临时 buffer，不在每个 layer 上长期保留；
5. mixed batch 显式切分 decode 和 prefill token ranges，分别写回输出。

该路径已避免短 continuation 的重复全历史反量化。大 chunk 仍需要与历史长度成正比的 dense
buffer，但不会随 layer 数量持久累积；它不是最终的 fused packed-prefill 性能方案。

## 8. Metadata

TurboQuant metadata builder 复用 `AscendAttentionMetadataBuilder`，但保留 device-side
`seq_lens`，避免 decode kernel 访问 CPU tensor。关键字段包括：

- `slot_mapping`：本轮 K/V 写入地址；
- `block_tables`：历史 token 到 physical page 的映射；
- `seq_lens`：每个 request 的总上下文长度；
- `actual_seq_lengths_q`：TND query 的累计长度；
- `attn_state`：选择首次 prefill、decode 或反量化 fallback。

写 cache 和读 cache 使用不同 metadata：`slot_mapping` 只描述本轮写入，block table 与
sequence length 描述所有历史读取。

## 9. 关键正确性约束

修改 layout 或算子时必须同时满足：

1. `TQFullAttentionSpec.page_size_bytes` 与实际 cache shape 的字节数一致；
2. store 与 decode 对 key/value metadata offset 的计算完全一致；
3. slot stride 必须为偶数，FP16 metadata offset 必须对齐；
4. query heads 必须能整除 KV heads，GQA head 映射为连续分组；
5. `slot_mapping < 0` 不得产生写入；
6. block table 中只有 `position < seq_len` 的 page 可以被访问；
7. centroids 在 midpoint 查找和 decode gather 中必须使用相同排序；
8. graph decode workspace 必须在 memory profiling 前预分配；eager continuation 必须先检查
   容量，不足时使用精确大小的临时 workspace。

## 10. 测试与 Profiling

先执行算子正确性 smoke test：

```bash
bash scripts/turboquant_triton/correctness/run_kernel_smoke.sh
```

该测试覆盖 packed layout、负 slot、store/dequant round trip，以及 packed decode 与
反量化 attention reference 的一致性。

执行算子 profiling：

```bash
bash scripts/turboquant_triton/performance/profile_kernels.sh \
    --operation all \
    --batch-size 4 \
    --sequence-length 4096 \
    --num-query-heads 32 \
    --num-kv-heads 4
```

head 数是当前 TP rank 的本地 head 数。输出包括 NPU Event 延迟统计、吞吐、显存变化和
`torch_npu.profiler` trace。

## 11. 推荐阅读顺序

1. `vllm_ascend/kv_cache/turboquant.py`：先理解支持范围和 shape；
2. `vllm_ascend/platform.py`：理解 backend 如何被选择；
3. `vllm_ascend/worker/model_runner_v1.py`：理解 cache 如何分配；
4. `vllm_ascend/attention/turboquant.py`：理解整体状态机；
5. `turboquant_store.py`：理解量化和 bit packing；
6. `turboquant_decode.py`：理解 paged addressing、online softmax 和 split reduce；
7. 对照 vLLM CUDA TurboQuant backend，确认算法语义而不是直接复制硬件实现。
