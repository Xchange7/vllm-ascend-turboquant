# TurboQuant Ascend 代码逐块导读

## 1. 文档目的

本文用于逐块阅读 vLLM Ascend 中的 TurboQuant 实现。重点不是重复算法论文，
而是回答以下代码问题：

- 用户传入 `--kv-cache-dtype turboquant_4bit_nc` 后，配置如何进入 backend；
- KV Cache 为什么是一个 `uint8` packed tensor，它的尺寸如何计算；
- K/V 在哪个函数中量化、打包并写入 paged cache；
- decode 如何直接读取压缩 cache，而不先恢复完整 FP16 K/V；
- prefill、continuation prefill、mixed batch 和 ACLGraph 分别走哪条路径；
- 每个测试和诊断脚本覆盖什么故障。

本文针对当前 checkout 中的实现。相关代码分为两部分：

1. vLLM core 提供 TurboQuant preset、centroid、layer workspace 和
   `TQFullAttentionSpec`；
2. vLLM Ascend 提供 NPU backend、metadata 路由、KV Cache 分配和
   Triton-Ascend kernel。

TurboQuant 在这里压缩的是运行时 KV Cache，不是模型权重。因此启用该功能不需要先使用
ModelSlim 处理模型权重。

## 2. 建议阅读顺序

建议按照下列顺序阅读，而不是直接从 Triton kernel 开始：

1. `vllm_ascend/kv_cache/turboquant.py`：理解配置和 packed shape；
2. `vllm_ascend/platform.py`：理解 backend 如何被选中；
3. `vllm_ascend/worker/model_runner_v1.py`：理解 cache 如何分配；
4. `vllm_ascend/attention/turboquant.py`：理解运行时路径调度；
5. `vllm_ascend/ops/triton/turboquant_store.py`：理解量化和写 cache；
6. `vllm_ascend/ops/triton/turboquant_decode.py`：理解 packed decode；
7. `tests/ut/test_turboquant_kv_cache.py` 和
   `tests/ut/ops/test_turboquant_triton.py`：用测试确认理解是否正确；
8. `scripts/turboquant_triton/`：在 910B4 上验证和 profiling。

## 3. 一次请求的完整调用链

```text
--kv-cache-dtype turboquant_4bit_nc
  |
  v
vLLM TurboQuantConfig + TQFullAttentionSpec
  |
  v
NPUPlatform.get_attn_backend_cls()
  |
  +-- validate_turboquant_backend()
  |
  `-- AscendTurboQuantAttentionBackend
          |
          +-- MetadataBuilder.build()
          |
          +-- NPUModelRunner 分配 packed uint8 KV Cache
          |
          +-- do_kv_cache_update()
          |     `-- triton_turboquant_store()
          |
          `-- forward()
                +-- 首次 prefill -> NPU FIA
                +-- decode -> packed Triton decode
                +-- 短 continuation -> 多次 packed decode
                `-- 长 continuation -> 历史反量化 + NPU FIA
```

需要区分两个动作：

- `do_kv_cache_update()` 负责把本轮新产生的 K/V 写进 cache；
- `forward()` 负责读取当前请求可见的 K/V 并计算 attention。

写 cache 使用 `slot_mapping`，读 cache 使用 `block_tables` 和 `seq_lens`。
这两组 metadata 不能混为一谈。

## 4. vLLM core 提供的契约

以下代码不属于 vLLM Ascend，但 Ascend 实现依赖这些接口。服务器上的 vLLM core
必须与本分支匹配。

### 4.1 `TurboQuantConfig`

文件：`vllm/model_executor/layers/quantization/turboquant/config.py`

`TQ_PRESETS` 把用户可见的 cache dtype 转换为算法参数：

| Cache dtype | Key | Value | Norm correction |
| --- | ---: | ---: | --- |
| `turboquant_4bit_nc` | 4 bit | 4 bit | 开启 |
| `turboquant_k3v4_nc` | 3 bit | 4 bit | 开启 |
| `turboquant_3bit_nc` | 3 bit | 3 bit | 开启 |

当前 Ascend backend 不支持 `turboquant_k8v4` 的 FP8 key 路径。

`key_packed_size`、`value_packed_size` 和 `slot_size_aligned` 是 store、decode、
cache spec 必须共同遵守的布局契约：

```text
key_packed_size   = ceil(D * key_bits / 8) + 2
value_packed_size = ceil(D * value_bits / 8) + 4
slot_size         = key_packed_size + value_packed_size
slot_size_aligned = slot_size 向偶数字节对齐
```

额外的 2 字节是 key norm，额外的 4 字节是 value scale 和 minimum，三者均使用 FP16。

### 4.2 `Attention._init_turboquant_buffers()`

文件：`vllm/model_executor/layers/attention/attention.py`

这个函数在每个 TurboQuant attention layer 上注册：

- `_tq_centroids`：Lloyd-Max centroid；
- `_tq_mid_o_buf`：split-KV 的 partial output 和 LSE；
- `_tq_output_buf`：FP32 decode 输出 workspace；
- `_tq_lse_buf`：每个 request、每个 head 的 LSE workspace。

workspace 在模型 placement 和内存 profiling 之前创建。这样 KV Cache block 规划时会把
decode workspace 的显存计入，避免第一次 decode 才临时申请显存并发生 OOM。

### 4.3 `TQFullAttentionSpec`

文件：`vllm/v1/kv_cache_interface.py`

普通 `FullAttentionSpec` 根据两个 dense K/V tensor 计算 page bytes。TurboQuant 使用单个
packed slot，因此 `TQFullAttentionSpec.real_page_size_bytes` 改为：

```text
block_size * num_kv_heads * tq_slot_size
```

scheduler 只管理 page 数量和 page 字节数，不理解 slot 内部字段。packed 布局由
TurboQuant backend 和 Triton kernel 负责。

## 5. Cache 配置工具

文件：`vllm_ascend/kv_cache/turboquant.py`

### 5.1 `is_turboquant_kv_cache_dtype()`

判断 cache dtype 是否以 `turboquant_` 开头。它是平台路由的统一入口，避免在多个文件中
重复字符串判断。

非字符串、`None` 和普通 `auto` cache 都返回 `False`。

### 5.2 `get_turboquant_config()`

调用 vLLM core 的 `TurboQuantConfig.from_cache_dtype()`。该函数故意使用延迟 import，
避免普通非 TurboQuant 启动路径提前加载量化模块。

输入是 cache dtype 和 head dimension，输出包含 bit 数、norm correction、packed size
等全部布局参数。

### 5.3 `get_turboquant_kv_cache_shape()`

返回物理 cache shape：

```text
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

最后一维表示每个 token、每个 KV head 的全部 packed bytes。它不是 attention 的
`head_dim`。

以 `D=128` 为例：

| Preset | Slot bytes | Cache shape 最后一维 |
| --- | ---: | ---: |
| 4-bit K / 4-bit V | 134 | 134 |
| 3-bit K / 4-bit V | 118 | 118 |
| 3-bit K / 3-bit V | 102 | 102 |

### 5.4 `validate_turboquant_layout()`

集中检查 Triton kernel 的物理布局约束：

- 拒绝尚未实现的 FP8 key；
- `head_dim` 必须是 2 的幂并且能被 32 整除；
- key/value bit 数只能是 3 或 4。

这些错误在 backend 初始化阶段抛出，而不是等 kernel 编译或运行后才失败。

### 5.5 `validate_turboquant_backend()`

集中检查功能组合。普通 cache dtype 会直接返回，不影响原有 backend。

当前显式拒绝的组合包括 V2 runner、310P、MLA、sparse attention、compressed
attention、attention sink、multimodal prefix、non-causal 和 batch-invariant mode。

集中校验的目的有两个：

1. 不让不支持的组合静默进入错误路径；
2. 后续实现一项能力时，只需删除对应限制并补齐测试，不需要搜索模型名称分支。

## 6. 平台路由

文件：`vllm_ascend/platform.py`

### 6.1 `check_and_update_config()` 中的全局校验

当全局 cache dtype 是 TurboQuant 时，这一块拒绝：

- parallel drafting；
- KV transfer；
- context parallelism。

这些能力涉及 model runner、跨 worker cache 或 attention 并行语义，不能只靠单层
attention backend 实现。

### 6.2 `get_attn_backend_cls()` 中的逐层路由

该函数收到每层的 `AttentionSelectorConfig` 后：

1. 调用 `validate_turboquant_backend()`；
2. 如果该层 cache dtype 是 TurboQuant，返回
   `AscendTurboQuantAttentionBackend`；
3. 否则继续原有 Ascend backend 选择流程。

这使一个模型可以同时包含普通 cache layer 和 TurboQuant layer。实现没有写死 Qwen3
层号，也不依赖模型类名。

## 7. Model runner 中的 KV Cache 分配

文件：`vllm_ascend/worker/model_runner_v1.py`

### 7.1 `_allocate_kv_cache_tensors()`

遇到 `TQFullAttentionSpec` 时，runner 按 `KVCacheTensor.size` 分配单个 `int8` 原始
buffer。`int8` 在这里只用于申请确定数量的字节，不代表 packed 数据是有符号量化值。

同一 cache group 中共享 storage 的 layer 会指向同一个原始 buffer。TurboQuant 不走普通
Ascend attention 的独立 K tensor 和 V tensor 分配路径。

KV transfer 在这里再次被拒绝，是为了防止未来绕过 platform 校验后使用不兼容布局。

### 7.2 `_reshape_kv_cache_tensors()`

该函数完成三项检查和转换：

1. 原始字节数必须能被 `page_size_bytes` 整除；
2. 由字节数计算出的 `num_blocks` 不能小于 scheduler 配置；
3. 使用 TurboQuant backend 的 `get_kv_cache_shape()` reshape 为 packed shape。

最终 tensor dtype 是 spec 中的 `torch.uint8`，底层 storage 与原始 buffer 相同，没有复制。

这里解决了普通 Ascend INT8 cache helper 无法表达 TurboQuant combined K/V slot 的问题。

## 8. Attention metadata builder

文件：`vllm_ascend/attention/turboquant.py`

### 8.1 `_TURBOQUANT_CACHE_DTYPES`

声明 Ascend backend 真正支持的三种 preset。它同时用于 backend capability reporting，
不能包含尚无 kernel 的 `turboquant_k8v4`。

### 8.2 `_CONTINUATION_DECODE_THRESHOLD`

值为 128。continuation query 不超过该值时，复用 packed decode；更大时，反量化历史
并调用 FIA。

这个阈值控制的是策略选择，不是模型限制。后续应根据 910B4 profiling 数据调整。

### 8.3 `_FEATURE_PREFILL_QUERY_TILE_SIZE`

值为 32。ALiBi 或 logits soft cap 不能直接交给当前 FIA 参数组合时，fallback 会以
32 个 query token 为一块计算 FP32 score，限制峰值 workspace。

### 8.4 `_build_hadamard()`

从 `[[1]]` 开始递归构造 Sylvester Hadamard matrix：

```text
H(2n) = [H(n)   H(n)]
        [H(n)  -H(n)]
```

最后除以 `sqrt(head_dim)`，得到正交矩阵。`functools.cache` 使用
`(head_dim, device_string)` 作为 key，同一设备和 head dimension 只构造一次。

当前实现使用 dense matrix multiplication。它保持逻辑正确，但未来可以替换为 FWT
以减少旋转计算量。

### 8.5 `_query_lens_from_cumulative()`

`actual_seq_lengths_q` 保存 TND batch 的累计 query 结束位置，例如 `[2, 5, 6]`。
该函数将其转换为 `[2, 3, 1]`。

它同时检查：

- boundary 数量不能少于 request 数；
- 每个 request 的 query length 必须大于零。

## 9. `AscendTurboQuantMetadataBuilder`

### 9.1 `_cudagraph_support`

声明 `AttentionCGSupport.UNIFORM_BATCH`。含义是 TurboQuant 支持 uniform single-token
和 uniform multi-token decode 的 ACLGraph capture，不声明任意 mixed batch 都可捕获。

接口名称沿用 vLLM 的 CUDAGraph 抽象；在 Ascend 平台上实际由 ACLGraph/NPUGraph
基础设施执行。

### 9.2 `build()`

先调用普通 Ascend metadata builder，再把 device-side `seq_lens` 保存到 metadata。
packed decode kernel 直接读取这个 NPU tensor，避免在 attention 热路径调用 `.item()`
或执行 NPU 到 CPU 同步。

当 metadata 实际上全部是 decode token，但 Ascend runner 将其标记成
`ChunkedPrefill` 时，这里改写为 `DecodeOnly`。这样非 MTP speculative decode 可以继续
走 packed sequential decode，不需要反量化完整历史。

### 9.3 `build_for_cudagraph_capture()`

capture 时强制设置为 `DecodeOnly`，并用每个 request 的 query length 初始化
`seq_lens`。这是最短合法 context，避免 capture 阶段访问无效 page。

真实 replay 前，runner 会更新持久化 `seq_lens`、block table、slot mapping 和输入 tensor。

## 10. `AscendTurboQuantAttentionBackend`

这个类描述 backend 能力和 cache 管理接口，不实现 attention 数学。

### 10.1 类属性

- `accept_output_buffer=True`：由 runner 提供 output tensor；
- `forward_includes_kv_cache_update=False`：cache update 由统一更新流程单独调用；
- `supported_dtypes`：支持 FP16 和 BF16 activation；
- `supported_kv_cache_dtypes`：只包含三个已实现 preset。

### 10.2 `get_name()`、`get_impl_cls()`、`get_builder_cls()`

这三个函数分别向 vLLM 注册 backend 名称、运行实现类和 metadata builder。运行日志中的
backend 名称是 `TURBOQUANT`，不是一个新的 `AttentionBackendEnum` 成员。

### 10.3 `get_kv_cache_shape()`

把 shape 计算委托给 cache 工具模块，确保 model runner 与 kernel 使用同一个
`slot_size_aligned`。

### 10.4 capability 方法

- block size 固定为 128；
- 只接受 decoder self-attention；
- 不支持 per-head quant scale；
- cache dtype 必须属于支持列表。

### 10.5 `swap_blocks()`

在两个 cache tensor 之间复制完整 packed page。先读取 source pages、迁移到 destination
device 并 `clone()`，再写目标位置。

backend 把 page 当作不透明字节，不解析内部 K/V。该接口用于 cache swap 等管理动作。

### 10.6 `copy_blocks()`

在同一个 cache tensor 内复制 packed page。先 `clone()` source，防止 source 和
destination 重叠时前一个写操作污染后续 source。

该接口是 prefix cache、preemption 和 block reuse 正确性的基础之一。

## 11. `AscendTurboQuantAttentionImpl`

### 11.1 `__init__()`

初始化流程如下：

1. 复用 `AscendAttentionBackendImpl` 保存 head 数、scale 和 attention 属性；
2. 拒绝非 decoder attention 和 sliding window；
3. 校验 logits soft cap 为正数；
4. 校验 TurboQuant layout 并创建 `tq_config`；
5. 从 vLLM attention config 读取固定 split 数。

固定 split 数使 decode workspace shape 在 ACLGraph capture/replay 之间保持稳定。

### 11.2 `_ensure_constants()`

第一次使用某层时，把该层需要的 NPU 常量准备好：

- 构建 Hadamard matrix；
- 将上游 vLLM 注册的 centroids 移到当前 NPU 并转为 FP32；
- 对 centroids 排序；
- 计算相邻 centroid midpoint；
- 设置 ready 标记，后续 forward 不再重复创建。

store 使用 midpoint 做 centroid index 二分查找；decode 使用 centroid 本身恢复数值。

### 11.3 `do_kv_cache_update()`

这个函数只负责写入本轮 K/V：

1. 根据 `slot_mapping` 确定有效 token 数；
2. 将 K/V reshape 为 `[tokens, KV heads, head_dim]`；
3. 调用 `triton_turboquant_store()` 完成量化、packing 和 scatter。

传给 store 的 key 是 post-RoPE key，因此后续 decode 不需要再次处理历史 key 的 RoPE。

### 11.4 `_prefill_attention()`

首次 prefill 时，当前完整 K/V 仍是 FP16/BF16。函数直接调用 `_run_fia()`，不从刚写入的
packed cache 读回数据。

这避免了不必要的“量化后立即反量化”。同一轮 cache update 仍会保存压缩 K/V，供后续
decode 使用。

### 11.5 `_run_fia()`

普通 prefill 走 `torch_npu.npu_fused_infer_attention_score()`：

- 输入 layout 是 `TND`；
- Q 和 KV 的累计长度分别传入；
- `sparse_mode=3` 表示 causal attention mask 语义；
- 结果 reshape 后复制到 runner 提供的 output。

如果模型启用了 ALiBi 或 logits soft cap，则转入 `_run_feature_prefill()`，因为当前 FIA
调用不能完整表达这两个语义组合。

### 11.6 `_run_feature_prefill()`

该函数实现精确的 PyTorch fallback。它按 request 和 query tile 处理：

1. 将 K/V 转成 FP32，但保持 `[KV tokens, KV heads, D]`；
2. 将 query tile reshape 为 `[tile, KV heads, GQA groups, D]`；
3. 用 `einsum` 计算 `[KV heads, groups, tile, KV tokens]` score；
4. 依次应用 attention scale、soft cap、ALiBi 和 causal mask；
5. 在 KV token 维做 FP32 softmax；
6. 与 value 做 `einsum`，恢复 `[tile, query heads, D]`；
7. 写入对应 output slice。

它不把 K/V 扩展到 query head 数，也不创建完整 `[Hq, Q, K]` score。query tile 大小固定
为 32，但 K 维保持完整，所以 softmax 结果是精确的，不是分块近似。

循环结束后检查累计 boundaries 是否消费了所有 Q/KV token，防止 metadata 与 tensor
悄悄错位。

### 11.7 `_decode_attention()`

先从累计 boundary 得到每个 request 的 query length。

- 所有 request query length 相同时，调用 uniform multi-token 路径；
- 不同时，逐 request、逐 token 调用 packed decode。

逐 token 时使用：

```text
effective_seq_len = final_seq_len - query_len + token_index + 1
```

这保证 speculative 或 multi-token query 中第一个 token 看不到后面的 token。

### 11.8 `_uniform_multi_token_decode()`

把 query/output reshape 为：

```text
[num_requests, query_len, num_query_heads, head_dim]
```

然后以静态 Python 循环处理每个 token step。每一步只把 request 维传给 packed decode，
因此 workspace 与 request 数成正比，而不是与 `request * query_len` 成正比。

query length 在 graph capture 时固定，所以这段循环会整体进入 uniform multi-token
ACLGraph。

### 11.9 `_launch_decode()`

这是 attention 层与 Triton decode launcher 的单一连接点。它传递：

- packed cache、block table、device-side sequence lengths；
- Hadamard、centroid；
- K/V bit 数和 packed offset；
- norm correction、split 数、ALiBi 和 soft cap。

集中调用可避免 uniform、non-uniform 和 continuation 路径各自维护 kernel 参数。

### 11.10 `_continuation_prefill()`

continuation 表示请求已有历史 cache，本轮又一次调度多个 query token。

首先计算：

```text
history_len = final_seq_len - query_len
```

随后分三种情况：

1. 无历史：直接对当前 K/V 调 FIA；
2. `query_len <= 128`：构造递增 `seq_lens`，把每个 query token 当成一个 packed decode；
3. 大于 128：只反量化历史 cache，和当前原始 K/V 合并后调用 FIA。

大 continuation 中，反量化 kernel 输出的是旋转空间 key。Hadamard 矩阵满足
`H^-1 = H`，因此再乘一次 Hadamard 恢复原 key 空间。

历史和合并 buffer 都是 eager-only 临时 tensor，不挂到每层长期保存。这样不会在
Qwen3-32B 的每个 layer 上保留一份最大上下文大小的临时 K/V。

### 11.11 `_prefill_requests()`

mixed batch 中 decode token 位于前面，prefill request 位于后面。该函数根据累计 query
boundary 切出每个 prefill request 的 Q/K/V/output slice，然后逐请求调用
`_continuation_prefill()`。

### 11.12 `forward()`

这是 attention 的总路由函数：

```text
attn_metadata is None
  -> output 清零

PrefillNoCache
  -> 当前 raw K/V + FIA

DecodeOnly 或 SpecDecoding，且没有 prefill
  -> packed decode

Mixed batch
  -> 先处理 decode token range
  -> 再处理 prefill request range
```

函数要求 runner 提供 output buffer，并显式拒绝 fused output quantization。首次 prefill
或 continuation prefill 缺少当前 raw K/V 时会立即报错。

## 12. Store Triton kernel

文件：`vllm_ascend/ops/triton/turboquant_store.py`

### 12.1 `_store_quantized_value()`

每个 program 处理一个 token 的一个 KV head value vector：

1. 读取 value 并计算 `minimum`、`maximum`；
2. 根据 levels 计算 scale：

   ```text
   levels = 2^bits - 1
   scale  = max((maximum - minimum) / levels, 1e-8)
   index  = round((value - minimum) / scale)
   ```

3. 将 index clamp 到合法范围；
4. 4-bit 模式每两个 index 打包成一个 byte；
5. 3-bit 模式每八个 index 打包成 24 bit，即三个 byte；
6. 在 value payload 尾部写 FP16 scale 和 minimum。

`cache_u8_ptr` 与 `cache_f16_ptr` 指向同一 storage。前者访问 packed bytes，后者用于
偶数字节对齐的 FP16 metadata。

### 12.2 `_turboquant_store_kernel()`

kernel grid 是 `[num_tokens * num_kv_heads]`。每个 program 通过 program id 得到
`token_index` 和 `head_index`。

主要代码块如下：

1. **处理 padding**：`slot < 0` 立即返回，防止 dummy/graph padding 写坏 cache；
2. **计算物理地址**：把 slot 分成 block index 和 block offset，再叠加 head stride；
3. **读取旋转 key**：按 `BLOCK_D` 读取，超出真实 head dimension 的 lane 用 mask 屏蔽；
4. **centroid 搜索**：使用 midpoint 做固定轮数二分查找，轮数等于 key bit 数；
5. **key packing**：4-bit 两个一组，3-bit 八个一组；
6. **保存 key norm**：写在 key index payload 之后；
7. **量化 value**：调用 `_store_quantized_value()` 写 value payload 和 metadata。

固定迭代次数和 `tl.constexpr` 参数使 Triton 能在编译期展开 bit-width 分支。

### 12.3 `triton_turboquant_store()`

Python launcher 负责 kernel 外的向量运算和参数校验：

1. 检查 bit 数、K/V shape、cache dtype 和 head dimension；
2. 将 key 转为 FP32，计算每个 vector 的 L2 norm；
3. 归一化 key；
4. 乘 Hadamard，得到连续的 rotated key；
5. 让 value 变为连续内存；
6. 计算 packed payload byte offset，并检查 FP16 metadata 对齐；
7. 以每个 token、每个 KV head 一个 program 的 grid 启动 kernel。

这部分仍使用 dense Hadamard matrix multiplication，是后续 FWT 优化的主要入口。

## 13. Decode Triton kernel

文件：`vllm_ascend/ops/triton/turboquant_decode.py`

### 13.1 `_tanh()`

使用 sigmoid 实现 Triton 内的 tanh：

```text
tanh(x) = 2 * sigmoid(2x) - 1
```

仅在启用 logits soft cap 时使用。

### 13.2 `_turboquant_decode_stage1()` 的 program 划分

grid 是：

```text
(batch_size, num_query_heads, num_kv_splits)
```

每个 program 负责一个 request、一个 query head、一个 KV split。GQA 通过：

```text
kv_head = query_head // KV_GROUP_SIZE
```

把多个 query head 映射到同一个 KV head。

### 13.3 Stage 1 的 page 定位

当前 split 的 token position 先转成 logical page 和 page offset，再通过 block table 取得
physical block：

```text
logical_page  = position // block_size
page_offset   = position % block_size
physical_page = block_table[request, logical_page]
```

最终 `slot_base` 指向 packed cache 中一个 token、一个 KV head 的首字节。

### 13.4 Stage 1 的 key unpack 和 QK

对于每个 head dimension：

1. 根据 bit offset 读取相邻 byte；
2. shift 和 mask 得到 3-bit/4-bit centroid index；
3. gather centroid value；
4. norm correction 模式下，将 centroid vector 重新归一化；
5. 与旋转后的 query 做 dot product；
6. 乘 cache 中保存的原 key norm 和 attention scale。

query 和 key 使用同一个正交 Hadamard，因此旋转空间内积等价于原空间内积。

随后按顺序应用 soft cap 和 ALiBi。decode query 位于当前 sequence 末尾，ALiBi 相对位置为：

```text
position - seq_len + 1
```

### 13.5 Stage 1 的 online softmax

kernel 不保存完整 attention score，而是为每个 split 维护：

- `running_max`；
- `running_sum`；
- FP32 `accumulator`。

每读取一个 KV tile，就根据新 max 重缩放已有 sum 和 accumulator。这是数值稳定的 online
softmax，可将中间内存从与 sequence length 成正比降为与 head dimension 成正比。

### 13.6 Stage 1 的 value unpack

- 3-bit value 使用相邻 byte 拼接后 shift/mask；
- 4-bit value 从一个 byte 的低半字节或高半字节读取；
- 从 payload 末尾读取 FP16 scale 和 minimum；
- 使用 `value = index * scale + minimum` 恢复 FP32 value。

反量化 value 立即乘当前 softmax weight 并累加，不物化完整历史 value tensor。

每个 split 最终保存归一化 partial output 和 `log(sum(exp(score)))`。

### 13.7 `_turboquant_decode_stage2()`

Stage 2 的 grid 是 `(batch_size, num_query_heads)`。每个 program 读取该 head 的所有有效
split，通过各 split 的 LSE 重新缩放 partial output，然后合并得到最终 attention output。

这相当于在 split 级别再次执行 online softmax，保证 split-KV 与未切分 softmax 的数学语义
一致。

### 13.8 `_turboquant_full_dequant_kernel()`

该 kernel 只用于长 continuation fallback。grid 是：

```text
(max_seq_len, batch_size * num_kv_heads)
```

每个 program 解压一个历史 position、一个 KV head：

1. 通过 block table 定位 slot；
2. 解包 key index、gather centroid、执行可选 norm correction；
3. 乘原始 key norm，输出旋转空间 key；
4. 解包 value 并应用 scale/minimum；
5. 根据 `seq_start_locs` 写入紧凑 TND output。

它不执行 inverse Hadamard；调用方在需要原 key 空间时统一完成该步骤。

### 13.9 `_layout()`

集中返回三个 kernel launch 参数：key index bytes、value index bytes 和向上取 2 的幂的
`BLOCK_D`。store 和 decode 使用同样的 byte 公式是正确性的硬约束。

### 13.10 `_get_workspace()`

优先复用 vLLM layer 上预分配的 FP32 workspace，并检查：

- dtype；
- device；
- rank；
- 每一维容量。

如果任何一项不满足，分配精确尺寸的 eager-only 临时 tensor。该 fallback 主要用于短
continuation 将 query token 展开为 synthetic decode row 时，避免 Triton 按超出 workspace
容量的 grid 写越界。

ACLGraph 的 uniform decode 使用预分配 layer workspace，不依赖动态图内临时分配。

### 13.11 `triton_turboquant_decode_attention()`

Python launcher 的工作分为四块：

1. **输入校验**：检查 soft cap、batch、query row、block-table row、split 数和 GQA head；
2. **query rotation**：将 query 转为 FP32 并乘 Hadamard；
3. **workspace 准备**：取得 partial output、output 和 LSE；
4. **两阶段 launch**：先执行 split-KV Stage 1，再执行 Stage 2 reduction。

最终将 FP32 workspace 中的结果转换回 query dtype。decode 过程中没有完整 FP16/BF16
历史 K/V tensor。

### 13.12 `triton_turboquant_dequant_paged_cache()`

这是完整反量化 kernel 的 Python launcher。它从 output shape 得到 KV head 和 head
dimension，计算布局参数，并根据最大历史长度启动二维 grid。

`seq_lens` 让不同 request 只处理各自有效历史，`seq_start_locs` 将 ragged request
写到连续 TND buffer。

## 14. ACLGraph 数据流

TurboQuant graph capture 的稳定对象包括：

- query、key、value 和 output tensor 地址；
- packed KV Cache storage；
- block table、slot mapping 和 device-side `seq_lens` storage；
- layer-owned split workspace；
- 固定 split 数和固定 uniform query length。

replay 前可以原地更新 tensor 内容，但不能随意改变 capture shape。uniform multi-token
路径使用静态 token-step 循环，避免 workspace 乘 query length。

`tests/ut/ops/test_turboquant_triton.py` 中的 graph 测试会在 replay 前修改 Q/K/V、
block table、slot mapping 和 `seq_lens`，以确认 graph 使用的是动态内容而不是 capture
期间的旧值。

## 15. 测试代码在验证什么

### 15.1 `tests/ut/test_platform.py`

- cache dtype 为 TurboQuant 时选择正确 backend；
- metadata builder 声明 uniform batch graph support。

### 15.2 `tests/ut/test_turboquant_kv_cache.py`

测试按职责分为：

- 三种 preset 的 slot shape；
- 不支持的 dtype、head dimension 和功能组合会明确失败；
- model runner 分配和 reshape 后 shape、dtype、storage 正确；
- packed page 的 copy/swap 和重叠复制；
- graph capture metadata 使用 device-side sequence length；
- cumulative query boundary 解析；
- uniform、non-uniform、speculative 和 mixed batch 路由；
- 短 continuation 使用 packed decode；
- 长 continuation 只反量化历史且不持久保留上下文 buffer；
- ALiBi、soft cap、causal mask 和 query tile workspace。

### 15.3 `tests/ut/ops/test_turboquant_triton.py`

这是需要 NPU 的直接算子测试：

- 负 `slot_mapping` 不写 cache；
- store、完整反量化和 packed decode 与 PyTorch reference 对齐；
- FP16/BF16、GQA、不同 head dimension、三种 preset、norm correction、ALiBi 和 soft cap；
- 零 key、常量 value 等数值边界保持 finite；
- workspace 容量不足时不发生越界；
- ACLGraph replay 与 eager 结果一致。

## 16. 脚本代码在做什么

目录：`scripts/turboquant_triton/`

### 16.1 `check_environment.py`

启动前检查：

- vLLM 是兼容的 `0.20.2`/`0.20.2+empty` 系列；
- `torch_npu` 能看到 NPU；
- `TQFullAttentionSpec.page_size_bytes` 与 packed layout 一致；
- vLLM `Attention` 包含三个必需 workspace；
- 打印实际加载的 vLLM 路径和关键依赖版本。

这用于尽早发现“vLLM core 与 vLLM Ascend 分支不匹配”。

### 16.2 `run_kernel_smoke.sh`

执行环境检查和聚焦的 TurboQuant kernel 测试，适合第一次在 910B4 上验证。

### 16.3 `run_aclgraph_smoke.sh`

执行 metadata builder 和 NPUGraph capture/replay 测试，验证 eager 正确后再运行。

### 16.4 `serve_qwen3_32b.sh`

封装 Qwen3-32B 启动参数。默认使用较保守的 context、batch 和 eager 配置，可通过环境变量
覆盖 TP size、cache dtype、graph mode 等参数。

### 16.5 `smoke_request.sh`

向已经启动的 OpenAI-compatible server 发送固定请求，用于确认模型能够完成 prefill 和
连续 decode。

### 16.6 `profile_kernels.sh`

这是 profiling 的 shell 入口，负责切换到仓库根目录并把命令行参数原样传给
`profile_kernels.py`。它不实现计时逻辑，便于从任意工作目录使用统一命令启动测试。

### 16.7 `profile_kernels.py`

分别构造 store、packed decode、native NPU paged attention 和 full dequant benchmark：

- warmup 后使用 NPU event 测量 P50/P90/P99；
- 记录 allocated memory 和 operation peak memory；
- 可采集 torch-npu profiler trace；
- 计算 TurboQuant decode 相对 native paged attention 的实测 speedup；
- 记录理论 cache compression ratio。

### 16.8 `profile_matrix.sh`

批量遍历 preset、activation dtype、head dimension 和 split 数。它用于寻找正确性或性能
只在某个形状出现的回归。

### 16.9 `run_diagnostic_suite.sh`

依次运行环境、源码版本、backend 测试、kernel 测试、ACLGraph 测试和短 profiling。
单个阶段失败后继续收集后续信息，最后把日志目录打包，便于离线 debug。

## 17. Qwen3-32B 的形状示例

假设 TP=2，每个 rank 有 32 个 query heads、4 个 KV heads，`D=128`，block size 为 128，
使用 `turboquant_4bit_nc`。

单个 KV Cache page 的字节数是：

```text
128 tokens * 4 KV heads * 134 bytes = 68,608 bytes
```

一个 decode step 的主要形状为：

```text
query:       [batch, 32, 128]
kv_cache:    [num_blocks, 128, 4, 134]
block_table: [batch, max_blocks_per_request]
seq_lens:    [batch]
partial:     [batch, 32, num_splits, 129]
output:      [batch, 32, 128]
```

每个 query head 通过 GQA group 映射到 4 个 KV heads 之一。

## 18. 修改代码时必须保持的闭环

### 18.1 新增 cache preset

需要同时检查：

1. vLLM `TQ_PRESETS` 和 `CacheDType`；
2. Ascend supported dtype 列表；
3. layout validation；
4. store packing；
5. decode unpacking；
6. `TQFullAttentionSpec.page_size_bytes`；
7. shape、round-trip、decode reference 和 profiling matrix 测试。

只修改配置名称而不修改 store/decode 是不完整实现。

### 18.2 支持新的 head dimension

需要验证 Hadamard 构造、bit packing 对齐、Triton `BLOCK_D`、centroid、cache shape 和
910B4 性能。不能只删除 `validate_turboquant_layout()` 中的限制。

### 18.3 支持新的模型

普通 dense decoder 不应新增模型名称分支。应根据该模型真实使用的 attention 能力补齐：

- head dimension 和 GQA；
- ALiBi、soft cap 或 sliding window；
- prefix cache、spec decode 或 graph mode；
- TP 下每 rank 的 head 数；
- mixed native/TurboQuant boundary layer。

### 18.4 优化性能

优先 profiling 以下部分：

1. dense Hadamard matrix multiplication；
2. store 中 key normalize/rotate 的独立 PyTorch op；
3. decode 的 `BLOCK_KV`、split 数和 warp 数；
4. 长 continuation full dequant fallback；
5. FP32 workspace 和 dtype conversion。

每项优化都必须保留 packed reference、eager/graph 一致性和 native baseline 对比。

## 19. 快速定位表

| 想查的问题 | 首先阅读 |
| --- | --- |
| 为什么 backend 没选中 | `NPUPlatform.get_attn_backend_cls()` |
| 为什么启动时拒绝某功能 | `validate_turboquant_backend()` |
| 为什么 cache shape 不对 | `get_turboquant_kv_cache_shape()`、`TQFullAttentionSpec` |
| 为什么 cache 分配失败 | `_allocate_kv_cache_tensors()`、`_reshape_kv_cache_tensors()` |
| K/V 在哪里量化 | `triton_turboquant_store()` |
| key centroid index 在哪里算 | `_turboquant_store_kernel()` |
| value scale/minimum 在哪里算 | `_store_quantized_value()` |
| decode 在哪里直接读取 packed cache | `_turboquant_decode_stage1()` |
| split attention 在哪里合并 | `_turboquant_decode_stage2()` |
| 长 continuation 在哪里反量化 | `_turboquant_full_dequant_kernel()` |
| prefill/decode 如何路由 | `AscendTurboQuantAttentionImpl.forward()` |
| ACLGraph metadata 在哪里准备 | `build_for_cudagraph_capture()` |
| 如何收集完整日志 | `run_diagnostic_suite.sh` |
| 如何判断是否真正加速 | `profile_kernels.py --native-baseline` |

## 20. 相关文档

- [TurboQuant 在 vLLM Ascend 中的实现逻辑](turboquant_implementation_zh.md)
- [TurboQuant Ascend 当前功能限制](turboquant_limitations_zh.md)
- `scripts/turboquant_triton/README.md`
