# TurboQuantPagedDequant AscendC 算子实现详解

## 1. 文档范围

本文对应 `turboquant-triton-v0.20.2rc` 分支中的
`TurboQuantPagedDequant` 自定义算子，说明它在 vLLM Ascend 中的定位、数据契约、host
侧 tiling、AIV kernel、PyTorch binding、attention backend 集成、构建注册和测试方法。

这里的 TurboQuant 量化对象是推理过程中动态生成的 KV Cache，不是模型权重。因此启用
TurboQuant 不需要先通过 ModelSlim 生成一份新模型；模型权重仍按原来的 BF16、FP16 或其他
权重量化配置加载。

需要先区分三个执行组件：

1. `turboquant_store.py` 使用 Triton-Ascend 将新生成的 K/V 量化并写入 packed KV Cache；
2. `turboquant_decode.py` 可以直接读取 packed Cache，在 Triton kernel 内完成 attention；
3. `TurboQuantPagedDequant` 使用 AscendC 将 paged packed Cache 展开成 dense BNSD K/V，随后调用
   CANN FIA 完成 attention。

本文重点解释第三个组件。它融合的是 paged gather、bit unpack 和 K/V 反量化，不是把反量化、
QK、softmax 和 PV 全部融合成一个 AICore kernel。当前 `ascend_fused` 路径准确地说是：

```text
AscendC paged-dequant -> dense BNSD K/V -> CANN FIA
```

## 2. 设计动机

默认 TurboQuant decode 直接从 packed Cache 计算 attention，避免了 dense K/V 的 HBM 写回，理论上
更符合压缩 Cache 的带宽目标。但是当前 Triton-Ascend 实现在 910B4 高并发场景中包含较多 bit
unpack、paged addressing、标量控制和 split-KV 调度，高并发吞吐可能明显下降。

`TurboQuantPagedDequant` 提供另一条可测量路径：

1. 使用 AscendC AIV kernel 并行展开 packed K/V；
2. 将 attention 主计算交给成熟的 `npu_fused_infer_attention_score`；
3. 比较这条路径和 packed Triton decode 的正确性、延迟、吞吐和显存；
4. 为后续真正融合的 Ascend TurboQuant attention 积累 layout、tiling 和算子注册基础。

该设计是性能实验路径，不应预设它一定更快。它减少了 Triton packed attention 的控制开销，
但引入了 dense K/V 工作区和 HBM 写回；当前实现会复用工作区，不能消除 dense 带宽成本。

## 3. 端到端调用链

单 token decode 使用 `ascend_fused` 时，主要调用链如下：

```text
AscendTurboQuantAttentionImpl._decode_attention()
  -> _run_ascend_fused_decode()
     -> turboquant_paged_dequant_out(caller-owned dense K/V)
        -> torch.ops._C_ascend.npu_turboquant_paged_dequant_out()
           -> ACLNN TurboQuantPagedDequant
              -> AscendC turbo_quant_paged_dequant kernel
     -> torch.matmul(..., out=query_workspace)
     -> torch_npu.npu_fused_infer_attention_score.out(..., out=vLLM output)
```

Cache 写入不经过该 AscendC 算子：

```text
do_kv_cache_update()
  -> triton_turboquant_store()
     -> rotation + key quantization + value quantization + bit packing
     -> uint8 paged KV Cache
```

因此 store 和 dequant 必须共享完全相同的 byte layout。两侧只要有一个 offset、bit order 或 metadata
定义不同，模型就会产生系统性精度错误。

## 4. 代码结构

| 文件 | 职责 |
| --- | --- |
| `csrc/attention/turbo_quant_paged_dequant/op_host/turbo_quant_paged_dequant_def.cpp` | 定义输入、输出、属性、dtype、format 和目标 SoC |
| `csrc/attention/turbo_quant_paged_dequant/op_host/turbo_quant_paged_dequant_proto.cpp` | 推导两个 dense BNSD 输出的 shape 和 dtype |
| `csrc/attention/turbo_quant_paged_dequant/op_host/turbo_quant_paged_dequant_tiling.cpp` | 校验 layout，生成 tiling data 和 block dimension |
| `csrc/attention/turbo_quant_paged_dequant/op_host/turbo_quant_paged_dequant_tiling.h` | 定义 host 与 kernel 共享的 tiling data |
| `csrc/attention/turbo_quant_paged_dequant/op_kernel/turbo_quant_paged_dequant.cpp` | AIV kernel：paged gather、解包、反量化和 BNSD 写出 |
| `csrc/torch_binding.cpp` | 注册 PyTorch schema、检查输入并执行 ACLNN |
| `csrc/torch_binding_meta.cpp` | 为 tracing/compile 提供 Meta 实现 |
| `vllm_ascend/ops/turboquant.py` | 延迟加载扩展并提供稳定 Python 调用接口 |
| `vllm_ascend/attention/turboquant.py` | decode 路由、Q rotation 和 FIA 调用 |
| `scripts/turboquant_operators/` | 独立算子正确性、性能和 profiler 测试 |

## 5. 算子接口契约

### 5.1 输入

| 输入 | Shape | Dtype | 含义 |
| --- | --- | --- | --- |
| `query` | `[B, Nq, D]` | FP16/BF16 | 提供 batch、head dimension 和输出 dtype；kernel 不读取数值 |
| `kv_cache` | `[num_blocks, block_size, Nkv, slot_size]` | UINT8 | 持久化 packed TurboQuant Cache |
| `block_table` | `[B, max_pages]` | INT32 | logical page 到 physical block 的映射 |
| `seq_lens` | `[B]` | INT32 | 每个请求当前有效 KV 长度 |
| `page_table` | `[active_pages, 2]` | INT32 | 紧凑的 `(request, logical_page)` 调度表 |
| `centroids` | `[C]` | FP32 | Key Lloyd-Max codebook，至少包含 `2 ** key_bits` 个值 |

属性：

| 属性 | 含义 |
| --- | --- |
| `max_seq_len` | 本次 dense 输出的 S 维长度，必须覆盖全部有效 sequence length |
| `key_bits` | Key centroid index 位宽，当前为 3 或 4 |
| `key_packed_size` | Key packed data 加 FP16 norm 后的总字节数 |
| `value_bits` | Value uniform quantization 位宽，当前为 3 或 4 |
| `norm_correction` | 是否对 centroid 重建向量进行 norm correction |

### 5.2 输出

```text
key:   [B, Nkv, max_seq_len, D]
value: [B, Nkv, max_seq_len, D]
```

两个输出都采用 BNSD 布局，dtype 与 `query` 相同。只有 `position < seq_lens[b]` 的位置会被
kernel 写入；尾部空间不应被下游读取。FIA 通过 `actual_seq_lengths_kv` 屏蔽这些尾部位置。

### 5.3 当前 head dimension 契约

当前 Ascend store、decode、rotation 和 AscendC 路径统一要求：

```text
D in {32, 64, 128, 256}
```

也就是 `[32, 256]` 内的 2 的幂。不能只检查“32 的倍数”，否则 96、160、192、224 会通过
AscendC host 校验，却与上层 Hadamard/layout 契约不一致。

## 6. Packed slot 数据布局

每个 physical token、每个 KV head 对应一个 UINT8 slot：

```text
+----------------------+------------------+----------------------+--------------+------------------+
| packed key indices   | original norm    | packed value indices | value scale  | value minimum    |
| ceil(D*Kb/8) bytes   | FP16, 2 bytes    | ceil(D*Vb/8) bytes   | FP16, 2 bytes| FP16, 2 bytes    |
+----------------------+------------------+----------------------+--------------+------------------+
```

定义：

```text
key_data_bytes   = ceil(D * key_bits / 8)
key_packed_size  = key_data_bytes + 2
value_data_bytes = ceil(D * value_bits / 8)
payload_size     = key_packed_size + value_data_bytes + 4
slot_size        = align_up(payload_size, 2)
```

向 2 字节对齐是为了让 norm、scale 和 minimum 可以通过 FP16 view 访问。若 payload 末尾存在
padding byte，store 和 decode 都必须忽略它。

当 `D=128` 时：

| Preset | Key bits | Value bits | Key bytes | Value bytes | Slot bytes | 相对 FP16 K/V 理论压缩率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `turboquant_4bit_nc` | 4 | 4 | 66 | 68 | 134 | 3.82x |
| `turboquant_k3v4_nc` | 3 | 4 | 50 | 68 | 118 | 4.34x |
| `turboquant_3bit_nc` | 3 | 3 | 50 | 52 | 102 | 5.02x |

FP16 baseline 每个 token/head 需要 `2 * D * 2 = 4D` 字节，表中的压缩率为 `4D / slot_size`。
这些 preset 的 `nc` 表示启用 norm correction。

### 6.1 Bit packing 顺序

维度 `d` 的量化 index 从以下 bit 位置开始：

```text
bit_offset = d * bits
byte_index = bit_offset // 8
shift      = bit_offset % 8
```

4-bit 模式下每两个 index 放进一个 byte；3-bit 模式下每八个 index 连续占 24 bit，部分 index
会跨 byte。kernel 不再为每个维度重复读取 packed byte，而是分组解包：

```text
4-bit: 1 byte  -> 2 indices
3-bit: 3 bytes -> 8 indices
```

分组后 bit order 与上述公式完全一致，但 4-bit 的 scalar byte load 数约减半，3-bit 从每维最多
两次 load 降为每八维三次 load。

## 7. Host 侧实现

### 7.1 OpDef

`turbo_quant_paged_dequant_def.cpp` 完成 CANN 算子声明：

- `query` 和输出支持 FP16/BF16；
- Cache 为 UINT8；
- block table、sequence length 和紧凑 page table 为 INT32；
- centroid table 为 FP32；
- 所有输入使用 ND format 和 `AutoContiguous()`；
- AICore config 注册到 `ascend910b` 和 `ascend910_93` 产品族。

910B4 构建时顶层仍设置：

```bash
SOC_VERSION=ascend910b4
```

`build_aclnn.sh` 会把它映射为 CANN 自定义算子的 `ascend910b` 产品族。这不表示把 910B4 错误地
编译成 910B1，而是 CANN 自定义算子的产品族命名方式。

### 7.2 Shape 和 dtype inference

`turbo_quant_paged_dequant_proto.cpp` 只使用 `query`、`kv_cache` 和 `max_seq_len` 推导输出：

```text
B    = query.shape[0]
Nkv  = kv_cache.shape[2]
S    = max_seq_len
D    = query.shape[2]
out  = [B, Nkv, S, D]
```

它会拒绝空 context、错误 rank 和非正 `max_seq_len`。输出 dtype 直接继承 `query`。

### 7.3 Tiling 校验

`TilingFunc()` 在生成 kernel 参数之前建立完整的运行时防线：

1. 所有 shape、attribute、platform 和 workspace 指针必须存在；
2. 输入 rank 必须分别为 3、4、2、1、2、1；
3. batch、block、head 和 slot 维度必须为正；
4. query、block table 和 sequence lengths 的 batch 必须相等，page table 必须为 `[P, 2]`；
5. `D` 必须是 `[32, 256]` 内的 2 的幂；
6. Key/Value 只接受 3-bit 或 4-bit；
7. `max_seq_len` 不能超过 block table 的 page 容量；
8. `key_packed_size` 必须与 `D`、`key_bits` 精确匹配；
9. Cache 的 `slot_size` 必须与完整 payload 精确匹配；
10. centroid 数量必须覆盖全部 key index；
11. 所有写入 `uint32_t` tiling data 的维度和 task count 都要先检查溢出。

这些校验不是性能路径中的冗余逻辑。packed layout 只要错一个 byte，kernel 读取 FP16 metadata 时就
可能把量化数据解释为 norm/scale，导致输出完全错误。

### 7.4 Tiling data

`TurboQuantPagedDequantTilingData` 向 kernel 传递：

```text
batchSize, maxSeqLen, maxPages,
numBlocks, blockSize, numKvHeads, slotSize,
activePageCount, totalTasks
```

`head_dim`、Key/Value 位宽和 NC 不再作为热路径中的普通 tiling field 读取，它们已编码进 tiling
key，并成为 kernel 模板参数；packed offset 和 centroid 数量由模板在编译期计算。

其中：

```text
active_page_count = sum(ceil(seq_lens[b] / block_size) for b in requests)
total_tasks       = active_page_count * Nkv
block_dim    = min(total_tasks, available_aiv_cores)
```

每个 task 对应一个 `(batch, logical_page, kv_head)`，而不是一个 token。task 内部顺序处理该 page
上的有效 token。page table 由 metadata builder 使用已有 CPU sequence lengths 构造，一次 H2D 后被
所有层共享；只要各请求的 page count 没有跨过下一个 block 边界，后续 token 继续复用同一张表，
不再 H2D。长短请求混合时也不再为短请求遍历全局 `max_seq_len` 对应的空 page。
page table 的 pinned host staging buffer 由 NPU event 保护；动态批次改变、需要重写 staging buffer
时，builder 会先确认上一笔 non-blocking H2D 已完成，避免高并发请求进出时发生异步覆盖。

tiling key 同时编码 `key_bits`、`value_bits`、`norm_correction` 和 `head_dim`。kernel 入口据此分派
编译期模板，避免 slot 热循环中的位宽、NC 和维度分支。

### 7.5 Platform API 兼容性

CANN 9.0 的 `PlatformAscendC` 构造函数接收可写的 `fe::PlatFormInfos*`。因此实现必须保留
`GetPlatformInfo()` 的 mutable pointer：

```cpp
auto* platformInfo = const_cast<fe::PlatFormInfos*>(context->GetPlatformInfo());
```

不能先声明为 `const auto*`，否则会在 host 编译阶段触发 const conversion error。

## 8. AscendC AIV Kernel

### 8.1 Kernel 入口

kernel 入口使用 `DTYPE_QUERY` 实例化 FP16 或 BF16 输出类型：

```text
GET_TILING_DATA -> Init -> Process
```

`query` 指针本身不参与 AICore 计算，它只通过 OpDef 和 binding 决定 batch、D 和输出 dtype。
`workspace` 当前也未使用，因为该 kernel 的临时数据全部放在每个 AIV core 的 UB 中。

### 8.2 UB buffer 规划

每个 AIV core 初始化以下 `TBuf<VECCALC>`：

| Buffer | 类型 | 用途 |
| --- | --- | --- |
| `slotBuffer_` | UINT8 | 当前 packed slot |
| `centroidBuffer_` | FP32 | 当前 codebook |
| `indexBuffer_` | INT32 | Key gather byte offset或 Value integer index |
| `keyFloatBuffer_` | FP32 | Key centroid 重建结果 |
| `valueFloatBuffer_` | FP32 | Value 反量化结果 |
| `squaredBuffer_` | FP32 | Key norm correction 的平方值 |
| `reduceBuffer_` | FP32 | `ReduceSum` 临时空间 |
| `normBuffer_` | FP32 | norm reduction 结果 |
| `keyOutputBuffer_` | FP16/BF16 | 写回前的 Key |
| `valueOutputBuffer_` | FP16/BF16 | 写回前的 Value |

`DataCopyPad` 可能按完整 32-byte data block 写 UB，因此所有 buffer 使用独立的
`AlignUbBytes()` 向 32 字节对齐。这个 helper 使用了项目私有名称，避免与 CANN 9.0 已提供的
全局 `AlignUp()` 发生重载歧义。

### 8.3 Centroid 加载

每个 AIV core 在 `Process()` 开始时只加载一次 centroid table：

```text
GM centroids -> DataCopyPad -> UB centroidBuffer
```

3-bit Key 需要 8 个 FP32 centroid，即 32 bytes；4-bit Key 需要 16 个，即 64 bytes。后续该 core
处理的所有 page、head 和 token 都复用这份 UB codebook。

### 8.4 Task 到 page/head 的映射

线性 task index 按以下方式还原：

```text
head_index       = task % Nkv
active_page_index = task // Nkv
batch_index, page_index = page_table[active_page_index]
page_start       = page_index * block_size
```

kernel 会再次校验 page table 中的 batch/page 范围。若该 page 已经超过 `seq_lens[batch]`，task
直接返回。否则读取：

```text
physical_block = block_table[batch, page_index]
```

负 block ID 或超出 `num_blocks` 的 ID 不会访问 Cache。

### 8.5 Cache 和输出寻址

对于 page 内的 `page_offset`：

```text
slot_index = ((physical_block * block_size + page_offset) * Nkv + head_index)
cache_byte_offset = slot_index * slot_size
```

输出 token position 为：

```text
token_position = page_start + page_offset
output_element_offset = (((batch * Nkv + head) * max_seq_len + token_position) * D)
```

这正好对应 contiguous BNSD。physical block 可以不连续，输出仍按 logical token 顺序连续排列。

### 8.6 Slot 搬入和 metadata 读取

`DequantizeSlot()` 首先把一个 slot 从 GM 搬到 UB：

```text
kv_cache GM -> DataCopyPad -> slotLocal UB
```

等待 MTE2 完成后，将同一 byte buffer reinterpret 为 FP16，读取：

```text
original_norm = fp16_at(key_data_bytes)
value_scale   = fp16_at(key_packed_size + value_data_bytes)
value_minimum = next_fp16
```

host 和 Torch binding 已经保证这些 offset 为偶数且位于 slot 内。

### 8.7 Key 解包、查表和 norm correction

`UnpackIndices<KEY_BITS>()` 按 4-bit byte group 或 3-bit 24-bit group 生成 centroid index。
AscendC `Gather()` 的 index 表示 byte
offset，不是元素下标，因此 kernel 写入：

```text
gather_offset[d] = centroid_index[d] * sizeof(float)
```

随后执行：

```text
key_float[d] = centroids[centroid_index[d]]
```

当 `norm_correction=True` 时：

```text
reconstructed_norm = sqrt(sum(key_float ** 2) + 1e-16)
key_scale = original_norm / reconstructed_norm
key_float *= key_scale
```

关闭 norm correction 时则直接：

```text
key_float *= original_norm
```

nc preset 会先把 centroid 重建向量重新归一化，减小 scalar quantization 对 Key norm 的扰动。

### 8.8 Value 解包和反量化

Value index 使用相同的 bit unpack，但不需要 centroid lookup。index 先写入 INT32 buffer，再转成
FP32：

```text
value_float = cast(value_index, fp32)
value_float = value_float * value_scale + value_minimum
```

这里的 scale 和 minimum 是 store 时对每个 token、每个 KV head 单独计算并保存的 FP16 metadata。

### 8.9 Cast 和写回

K/V 全部在 FP32 UB buffer 中完成反量化，然后根据 `DTYPE_QUERY` 转成 FP16 或 BF16：

- BF16 使用 `CAST_RINT`；
- FP16 使用 `CAST_NONE`；
- 最后通过 `DataCopyPad` 写入 dense BNSD GM tensor。

输出的 head dimension 至少为 32，且 FP16/BF16 每个元素为 2 bytes，因此每个输出 vector 的
写回长度天然是 32 bytes 的整数倍。

### 8.10 手动流水同步

该算子编译时设置 `--cce-auto-sync=off`，同步必须由代码显式保证：

| 同步 | 目的 |
| --- | --- |
| `MTE2_V` | centroid GM->UB 完成后才能由 Vector 读取 |
| `MTE2_S` | slot GM->UB 完成后才能由 Scalar 读取 metadata/packed bytes |
| `S_V` | Scalar 写完 index 或 scale 后 Vector 才能执行 gather/cast/mul |
| `V_S` | Vector reduction 完成后 Scalar 才能读取 norm；复用 index buffer 前等待旧 Vector 操作 |
| `V_MTE3` | Vector cast 完成后才能写回 GM |
| `MTE3_V` | 写回完成后才能在下一个 slot 中复用输出 UB buffer |

同一 Vector pipe 内的连续算子之间使用 `PipeBarrier<PIPE_V>()`。修改 kernel 时不能只检查数值
公式，还必须重新审核每个 UB buffer 的生产者、消费者和复用点。

## 9. PyTorch Binding

### 9.1 Schema 和输入检查

`csrc/torch_binding.cpp` 注册：

```text
_C_ascend::npu_turboquant_paged_dequant(...) -> (Tensor, Tensor)
_C_ascend::npu_turboquant_paged_dequant_out(..., Tensor(a!) key, Tensor(b!) value)
    -> (Tensor(a!), Tensor(b!))
```

进入 ACLNN 前，binding 会检查：

- rank、dtype、正维度和 batch 一致性；
- 所有 Tensor 位于同一 device；
- `max_seq_len` 能被 block table 容纳；
- bit width 为 3 或 4；
- head dimension 为支持的 2 的幂；
- `key_packed_size`、slot size 和 centroid count 精确匹配。

这些检查与 host tiling 有意重复。binding 提供清晰的 Python 异常，host tiling 则保护所有可能绕过
PyTorch wrapper 的 ACLNN 调用。

### 9.2 输出分配和 ACLNN 调用

返回式 schema 为独立算子测试保留。服务热路径使用 out-style schema，检查调用方提供的输出
shape、dtype、device 和 contiguous layout 后直接调用：

```text
EXEC_NPU_CMD(aclnnTurboQuantPagedDequant, ..., key, value)
```

`torch_binding_meta.cpp` 同时注册返回式和 alias-preserving out-style Meta 实现。这样 PyTorch
tracing、compile 和 fake tensor 流程可以知道输出结构及写入语义。

### 9.3 自定义扩展的延迟加载

vLLM Ascend 不会在 Python import 时立即加载所有 CANN 自定义算子，因为过早加载可能触发 RTS
初始化。`has_turboquant_paged_dequant()` 必须先调用 `enable_custom_op()`，再检查
`torch.ops._C_ascend`。

如果只执行 `hasattr(torch.ops._C_ascend, ...)`，全新的测试进程会把“扩展尚未 import”误报成
“TurboQuant schema 没有编译”。

## 10. Attention Backend 集成

### 10.1 路径选择

通过集中定义的环境变量选择实现：

```bash
export VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=ascend_fused
```

Attention 初始化时会校验：

- 值必须为 `auto`、`ascend_fused`、`grouped_gqa` 或 `reference`；
- `ascend_fused` 不支持 ALiBi 和 logits soft cap；
- head dimension 满足 AscendC 契约；
- 扩展同时包含返回式和 out-style paged-dequant schema，否则在模型初始化阶段直接失败。

`ascend_fused` 只在所有请求都是单 token decode 时进入
`_run_ascend_fused_decode()`。multi-token decode、continuation prefill 等其他场景会把实现名转换为
`auto`，回退到 packed Triton decode。

### 10.2 Query rotation

Store 将归一化 Key 旋转到 randomized Hadamard 坐标。AscendC 输出的 dense Key 仍在这个旋转
坐标中，因此 backend 不执行 inverse rotation，而是旋转 Query：

```text
query_rotated = query @ compute_rotation
```

若 `R` 为正交矩阵：

```text
(Q R) (K R)^T = Q R R^T K^T = Q K^T
```

所以在相同旋转空间计算 QK 不改变未量化 attention 的数学结果，也避免了对完整 dense K 再做
一次矩阵乘。

### 10.3 FIA 调用

AscendC 返回 K/V 后，backend 调用：

```text
npu_fused_infer_attention_score.out(
    query=query_rotated.unsqueeze(2),
    key=key_rotated,
    value=value,
    input_layout="BNSD",
    actual_seq_lengths_kv=seq_lens_list,
    num_key_value_heads=Nkv,
    num_heads=Nq,
    scale=attention_scale,
    sparse_mode=0,
    workspace=fia_workspace,
    out=[vllm_output.unsqueeze(2), softmax_lse_workspace],
)
```

`actual_seq_lengths_kv` 是正确性关键。AscendC 只写有效 token，FIA 必须忽略每个 batch row 中
`seq_len` 之后的未初始化尾部。

## 11. 数学语义

### 11.1 Key

设原始 Key 为 `k`，随机符号 Hadamard 正交矩阵为 `R`：

```text
n = ||k||_2
z = (k R) / n
i_d = LloydMaxIndex(z_d)
c_d = centroid[i_d]
```

开启 norm correction 后，重建旋转空间 Key：

```text
k_hat_rotated = n * c / sqrt(sum(c ** 2) + epsilon)
```

Attention 使用 `qR` 与 `k_hat_rotated` 计算 score。

### 11.2 Value

对每个 value vector：

```text
levels = 2 ** value_bits - 1
scale  = max((v_max - v_min) / levels, 1e-8)
i_d    = clamp(round((v_d - v_min) / scale), 0, levels)
v_hat  = i_d * scale + v_min
```

Key 使用非均匀 centroid quantization，Value 使用 uniform affine quantization。两者不能共用同一套
反量化公式。

## 12. 内存和性能模型

### 12.1 持久 KV Cache

持久 Cache 始终保持 UINT8 packed layout，所以 `ascend_fused` 不改变 Cache capacity 和理论压缩率。

### 12.2 临时 dense K/V

单 token decode 需要以下共享工作区容量：

```text
temporary_bytes = 2 * B * Nkv * max_seq_len * D * activation_element_size
```

前面的 `2` 表示 Key 和 Value。例如 Qwen3-32B TP4 的本地 `Nkv=2`，当 `B=16`、
`S=16384`、`D=128`、BF16 时，临时 K/V 约为 256 MiB。

metadata builder 持有一组 flat dense K/V buffer，并按历史最大需求扩容；同一 model step 的所有层
按 stream 顺序复用，后续 step 继续复用。因此不会为每层永久保存一份，也不会在每层 forward
重复申请。Q rotation、FIA workspace 和 softmax placeholder 使用同一策略。观察 `npu-smi` 时，
工作区按历史峰值保留属于预期行为。底层 flat storage 和 FIA workspace 查询按 1024-token 容量
档位增长并受 block-table 容量限制，避免 decode 每增加一个 token 就重新申请；实际
paged-dequant/FIA tensor view 仍使用精确 `max_seq_len`，不会把 attention 计算长度补到容量档位。

### 12.3 可能更快的条件

该路径可能获益于：

- AscendC 比当前 Triton 更高效的 page/bit 处理；
- FIA 对 GQA、QK、softmax 和 PV 的成熟实现；
- 高并发下减少大量 split program 和标量调度。

也可能受限于：

- 每步展开全部历史 K/V；
- dense K/V 的 HBM 写回和再次读取；
- dense 工作区的峰值显存常驻；
- 上下文越长，dequant 工作量线性增长。

因此性能结论必须分别报告 `fused_dequant` 和 `fused_dequant + FIA`，不能只看整个服务的 token/s。

## 13. 构建和注册

该功能同时修改 AscendC/ACLNN 和 PyTorch C++ extension。拉取代码后必须重新编译：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910b4
export MAX_JOBS=8

rm -rf build csrc/build
rm -f vllm_ascend/vllm_ascend_C*.so
pip install -v -e . 2>&1 | tee build_910b4.log
```

构建包含两个必要产物：

1. `build_aclnn.sh` 构建并安装 `TurboQuantPagedDequant` vendor custom-op；
2. CMake 重新构建 `vllm_ascend_C`，使 PyTorch schema 和 ACLNN wrapper 可见。

只成功构建其中一个仍然不能运行。编译完成后验证：

```bash
python3 - <<'PY'
import torch
from vllm_ascend.utils import enable_custom_op

print("extension loaded:", enable_custom_op())
print(
    "TurboQuant registered:",
    hasattr(torch.ops._C_ascend, "npu_turboquant_paged_dequant")
    and hasattr(torch.ops._C_ascend, "npu_turboquant_paged_dequant_out"),
)
print(torch._C._dispatch_find_schema_or_throw(
    "_C_ascend::npu_turboquant_paged_dequant", ""
))
print(torch._C._dispatch_find_schema_or_throw(
    "_C_ascend::npu_turboquant_paged_dequant_out", ""
))
PY
```

构建其他算子时出现的 protobuf deprecated、unused variable、non-virtual destructor 或
`sparse_flash_attention` 日志格式 warning 不表示 TurboQuant 构建失败。应定位日志中的第一个
`error:`、`fatal error:` 或最终非零退出。

## 14. 正确性保护

当前实现包含以下防线：

- store 遇到 `slot_mapping < 0` 不写 Cache；
- AscendC 遇到无效 sequence/page/physical block 不越界读取；
- Python、Torch binding 和 ACLNN tiling 使用一致的 head/layout 校验；
- 所有地址乘法在关键位置升级为 64 bit；
- page task 只覆盖实际有效 page，并在 kernel 内再次检查 batch/page/physical block；
- tiling 写入 `uint32_t` 前检查范围和 task count 溢出；
- UB buffer 按 32 bytes 对齐；
- CANN helper 使用私有名称，避免 `AlignUp` 等 API 冲突；
- `GetPlatformInfo()` 显式去除 CANN 9.0 API 返回值的 const 限定；
- 自定义算子探测先触发延迟加载，避免 fresh process 误报未注册。

仍需注意：AscendC 和 Triton dequant 读取同一份 store 结果，只能证明两个 reader 与当前 writer
一致。若 writer 和 reader 共享同一个错误 layout，它们可能同时给出相同但错误的结果。因此还要
比较未量化 K/V、最终 attention output 和模型级生成质量。

## 15. 算子测试

### 15.1 冒烟测试

```bash
SOC_VERSION=ascend910b4 DEVICE=0 \
bash scripts/turboquant_operators/run_smoke.sh
```

覆盖：

- 三个 3/4-bit preset；
- Triton store 和负 `slot_mapping`；
- AscendC dequant 对 Triton dequant；
- 四个 head dimension tiling key 和 out-style storage alias；
- AscendC dequant + FIA 对 packed Triton decode；
- 一组短延迟 benchmark。

### 15.2 性能矩阵

```bash
CACHE_DTYPES="turboquant_4bit_nc" \
ACTIVATION_DTYPES="bfloat16" \
BATCH_SIZES="1 4 16" \
SEQUENCE_LENGTHS="512 2048 16384" \
WARMUP=10 ITERATIONS=50 DEVICE=0 \
bash scripts/turboquant_operators/run_matrix.sh
```

建议先从 `B=1/4` 和 `S=512/2048` 开始，确认正确性和显存，再扩大到高并发长上下文。

### 15.3 Profiler

```bash
BATCH_SIZE=4 SEQUENCE_LENGTH=4096 \
WARMUP=10 ITERATIONS=50 PROFILE_ITERATIONS=5 DEVICE=0 \
bash scripts/turboquant_operators/run_profile.sh
```

主要结果：

| 文件/指标 | 判定内容 |
| --- | --- |
| `accuracy.json` 的 K/V implementation error | AscendC 与 Triton dequant 是否一致 |
| `attention agreement` | AscendC+FIA 与 packed decode 是否一致 |
| K/V MSE、NMSE、cosine | TurboQuant 相对原始 K/V 的量化误差 |
| `fused_dequant` latency | AscendC paged-dequant 本身性能 |
| `fused_decode` latency | AscendC dequant + FIA 总性能 |
| packed/native speedup | 直接 packed decode 相对普通 paged attention |
| peak memory | dense K/V 是否符合内存公式 |

## 16. 当前支持范围

当前支持：

- `turboquant_4bit_nc`、`turboquant_k3v4_nc`、`turboquant_3bit_nc`；
- FP16/BF16 activation；
- `D=32/64/128/256`；
- MHA/GQA 的 dense decoder self-attention；
- 非连续 physical block table 和不同 request sequence length；
- V1 runner eager single-token decode；
- 910B 产品族和 `ascend910_93` 产品族的构建注册，当前重点验证 910B4。

`ascend_fused` 当前不支持：

- ACLGraph，metadata builder 对该模式返回 `AttentionCGSupport.NEVER`；
- multi-token/spec decode 直接走 AscendC 路径，它们回退到 packed Triton；
- ALiBi、logits soft cap、sliding window 和 attention sinks；
- MLA、稀疏 attention、CP、310P 和 V2 model runner；
- `turboquant_k8v4` FP8 Key；
- QJL residual/outlier 等论文完整路径。

这些边界不依赖 Qwen3 模型名称。Qwen3 是首个验证对象，但算子契约只依赖 attention shape、
Cache preset 和 backend capability，后续普通 dense decoder 可以复用。

## 17. 推荐阅读顺序

1. `vllm_ascend/kv_cache/turboquant.py`：理解 preset、slot shape 和全局限制；
2. `vllm_ascend/ops/triton/turboquant_store.py`：确认 packed data 是如何产生的；
3. `turbo_quant_paged_dequant_tiling.cpp`：理解 host 校验和 task 划分；
4. `turbo_quant_paged_dequant.cpp`：按 `Init -> ProcessPageHead -> DequantizeSlot` 阅读；
5. `csrc/torch_binding.cpp`：理解 schema、输入防线和输出分配；
6. `vllm_ascend/ops/turboquant.py`：理解自定义扩展延迟加载；
7. `vllm_ascend/attention/turboquant.py`：理解 AscendC、Q rotation 和 FIA 的组合；
8. `scripts/turboquant_operators/operator_accuracy.py`：对照测试理解每个数值门禁。

## 18. 后续优化方向

如果实机结果证明 AscendC dequant 很快，但 `dequant + FIA` 仍受 dense K/V 写回限制，下一阶段不应
继续微调当前全量展开循环，而应实现真正的 fused attention：

1. 每次只把一个 KV tile 的 packed bytes 搬入 UB；
2. 在 UB 中解包并反量化当前 tile；
3. 使用 Cube/Vector pipeline 计算 QK；
4. 设备侧维护 online softmax 的 max、sum 和 accumulator；
5. 直接完成 PV，不把完整 dense K/V 写回 HBM。

当前算子仍有独立价值：它提供了稳定的 packed layout oracle、AscendC bit unpack 基础、paged
addressing、host tiling、PyTorch/ACLNN 注册链和可量化的性能基线。
