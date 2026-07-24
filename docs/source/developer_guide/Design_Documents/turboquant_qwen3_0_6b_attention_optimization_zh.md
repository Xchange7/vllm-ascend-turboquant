# Qwen3-0.6B TurboQuant Decode Attention 优化技术报告

## 1. 范围与版本

本文记录 `fix/turboquant-qwen32-decode-a2-a3` 分支上 TurboQuant decode
attention 的实现、性能和优化过程。稳定版本为：

```text
cb7cea96 test(turboquant): add 910B4 TP4 performance runner
de0e8940 perf(turboquant): process 384-token attention tiles
83d5b988 perf(turboquant): batch cache loads with head-major layout
c4b52955 feat(turboquant): add pipelined AscendC paged attention
```

本文不计入尚未完成精度和性能验证的 int8 PV 实验代码。

Qwen3-0.6B 的模型配置为：

| 参数 | 数值 |
| --- | ---: |
| Attention heads | 16 |
| KV heads | 8 |
| Head dimension | 128 |
| GQA group size | 2 |
| TP2 每卡 attention heads | 8 |
| TP2 每卡 KV heads | 4 |

当前 `TurboQuantPagedAttention` 流水化算子专用实现要求：

```text
head_dim = 128
num_query_heads / num_kv_heads = 8
key_bits = 4
value_bits = 4
norm_correction = true
```

因此，Qwen3-0.6B 当前实际使用的融合路径是：

```text
TurboQuant packed KV Cache
  -> AscendC TurboQuantPagedDequant
  -> dense BF16 K/V workspace
  -> CANN FusedInferAttentionScore
```

完整流水化 `TurboQuantPagedAttention` 当前用于 GQA group=8 的专项性能测试。
将该算子用于 Qwen3-0.6B 前，需要把固定 group=8 的 tiling、query-head 映射、
AIV 分工和 split reduction 泛化到 group=2。

## 2. 测试环境与口径

### 2.1 Qwen3-0.6B 实际 shape 基线

| 项目 | 配置 |
| --- | --- |
| Device | Ascend910_9382 |
| Cache | `turboquant_4bit_nc` |
| Activation | BF16 |
| TP shape | Hq=8，Hkv=4，D=128 |
| Batch | 1 |
| Sequence length | 512 |
| Block size | 128 |
| KV splits | adaptive，最大 8 |
| Warmup/iterations | 2/5 |
| 测试版本 | `21fe9cac` |

该组数据是引入完整流水化算子前的短算子验证结果，不是模型端到端 serving
结果。Cache layout V2 合入后尚未对该 group=2 shape 做同配置复测，因此不能把该组
数值作为 `cb7cea96` 的重新测量结果。

### 2.2 流水化算子高并发 shape

| 项目 | 配置 |
| --- | --- |
| Device | Ascend910_9382 |
| Cache | `turboquant_4bit_nc` |
| Activation | BF16 |
| Per-rank shape | Hq=16，Hkv=2，D=128 |
| Batch | 16 |
| Sequence length | 16384 |
| Block size | 128 |
| KV splits | 8 |
| Warmup/iterations | 20/200 |

该 shape 的 GQA group size 为 8，与当前流水化专用 kernel 一致。它用于测量单层
decode attention 算子，不代表 Qwen3-0.6B 模型端到端性能。

## 3. 当前性能

### 3.1 Qwen3-0.6B/TP2 已有实测基线

| 实现 | 平均延迟 | 相对上一实现节省 | 加速比 |
| --- | ---: | ---: | ---: |
| Packed Triton decode | 1.558948 ms | — | 1.00x |
| AscendC paged dequant | 0.289376 ms | 0.516372 ms，相对 Triton dequant | 2.784x |
| AscendC dequant + FIA decode | 0.299652 ms | 1.259296 ms | 5.203x |
| Native BF16 FIA | 0.087356 ms | — | — |

`AscendC dequant + FIA` 相对 packed Triton decode 的延迟降低为：

```text
(1.558948 - 0.299652) / 1.558948 = 80.779%
```

TurboQuant dequant 子阶段从 0.805748 ms 降到 0.289376 ms，减少
0.516372 ms，延迟降低 64.086%。

### 3.2 完整流水化融合算子专项结果

| 实现 | 平均延迟 | 相对当前融合算子 |
| --- | ---: | ---: |
| naive Triton/grouped GQA | 700.35 ms | 265.9x slower |
| reference 实现 | 1023.25 ms | 388.5x slower |
| 当前稳定流水化融合算子 | 2.63 ms | 1.00x |

当前流水化融合算子相对 naive grouped 实现：

```text
延迟节省 = 700.346008 - 2.633578 = 697.712430 ms
加速比   = 700.346008 / 2.633578 = 265.930x
降幅     = 99.624%
```

相对 reference 实现：

```text
加速比 = 1023.254736 / 2.633578 = 388.542x
降幅   = 99.743%
```

当前稳定 kernel 的重复测量区间约为 2.63–2.65 ms。单次 500-iteration
测试得到 2.570755 ms，但该结果受设备频率变化影响，不作为保守主结果。

## 4. TurboQuant 数据表示

TurboQuant 量化对象是运行时 KV Cache，不是模型权重。

### 4.1 Key

Key 在写 Cache 前执行固定随机符号 Hadamard rotation。每个旋转后的元素使用
16 项 Lloyd-Max centroid codebook 编码为 4-bit index。每个 Key 向量额外保存一个
FP16 norm-correction scale。

解码时：

```text
K[d] = centroid[key_code[d]] * key_scale
```

Q 使用同一旋转矩阵变换。由于旋转矩阵正交，QK 内积保持不变，不需要把 K
逆旋转回原坐标系。

### 4.2 Value

Value 使用逐 token、逐 KV head 的 4-bit affine quantization：

```text
V[d] = value_code[d] * value_scale + value_minimum
```

每个 Value 向量保存 FP16 `value_scale` 和 FP16 `value_minimum`。

### 4.3 Packed slot

D=128、K4V4、norm correction 时，每个 token/head 的 slot 为：

```text
64-byte key codes
+ 2-byte key scale
+ 64-byte value codes
+ 2-byte value scale
+ 2-byte value minimum
= 134 bytes
```

原生 BF16 K/V 需要 512 bytes，因此 KV Cache 理论容量压缩率为：

```text
512 / 134 = 3.82x
```

## 5. Qwen3-0.6B 融合路径

### 5.1 原始 packed Triton decode

原始 decode 直接读取 packed Cache，在 Triton kernel 中完成：

```text
paged addressing
-> 4-bit unpack
-> K centroid lookup
-> V affine dequant
-> QK
-> softmax
-> PV
-> split reduction
```

该路径避免 dense K/V workspace，但在 Ascend 上存在大量 split program、
paged 标量寻址、bit unpack 和小粒度调度。Qwen3-0.6B/TP2、B1/S512
实测为 1.558948 ms。

### 5.2 AscendC paged dequant

`TurboQuantPagedDequant` 将以下工作融合到一个 AscendC AIV kernel：

1. 根据 `block_table` 把 logical page 映射到 physical block；
2. 连续读取 packed slot；
3. 解包 4-bit Key/Value code；
4. Key 执行 centroid lookup 和 norm correction；
5. Value 执行 affine reconstruction；
6. 直接写入调用方复用的 dense BNSD K/V buffer。

该实现将 dequant 从 0.805748 ms 降到 0.289376 ms，节省 0.516372 ms。

### 5.3 FIA 接管 attention

AscendC 输出 dense K/V 后，调用
`torch_npu.npu_fused_infer_attention_score.out` 完成 QK、softmax 和 PV。
输出、dense K/V、query rotation 和 FIA workspace 均由 layer/metadata
持有并复用，避免每层、每 token 重复分配。

完整 decode 从 1.558948 ms 降到 0.299652 ms，节省 1.259296 ms。

## 6. 完整流水化 AscendC Attention

### 6.1 调用链

```text
AscendTurboQuantAttentionImpl._decode_attention()
  -> _run_ascend_pipelined_decode()
     -> Q rotation
     -> turboquant_paged_attention_out()
        -> torch.ops._C_ascend.npu_turboquant_paged_attention_out()
           -> ACLNN
              -> TurboQuantPagedAttention AscendC kernel
```

Python 层只保留 Q rotation 和一次 ACLNN launch。paged gather、code 解码、
QK、online softmax、PV 和 split reduction 全部进入同一个 AscendC mixed kernel。

### 6.2 AIC/AIV 分工

算子使用 `KERNEL_TYPE_MIX_AIC_1_2`：

| 单元 | 工作 |
| --- | --- |
| AIV0/AIV1 | 各处理一半 token 和四个 query row；读取 Cache、解包 K/V、处理 metadata |
| AIC | QK Cube matmul、PV Cube matmul |
| AIV0/AIV1 | FP32 online softmax、V affine folding、tile accumulator |
| AIV | 跨 split 合并并写最终 BF16/FP16 输出 |

单个 task 对应：

```text
(batch index, KV head index, split index)
```

每个 KV head 对应 8 个 query heads。AIC 使用 `CUBE_M=16`，有效 8 行，
其余行作为 padding。两个 sibling AIV 各负责 4 行。

### 6.3 流水阶段

每个外层 token tile 依次经过：

```text
AIV: paged load + K/V code decode
  -> SYNC_DEQUANT_READY
AIC: QK Mmad + Fixpipe
  -> SYNC_SCORES_READY
AIV: key scale + online softmax + V affine folding
  -> SYNC_PROBABILITY_READY
AIC: PV Mmad + Fixpipe
  -> SYNC_TILE_OUTPUT_READY
AIV: rescale old accumulator + accumulate current tile
```

跨核同步使用 `CrossCoreSetFlag` 和 `CrossCoreWaitFlag`。相邻 tile 使用不同
event ID，避免 producer 重新设置 flag 时 consumer 尚未完成上一轮。

### 6.4 Online softmax

每个 split 保存：

```text
running_max
running_sum
FP32 output accumulator
```

`SoftmaxFlashV2` 计算当前 tile，并返回旧 accumulator 的 rescale 系数。
更新过程为：

```text
accumulator = accumulator * old_weight + tile_output
```

所有 split 完成后，AIV 使用各 split 的 max/sum 对 partial output 做稳定合并。
因此 kernel 不需要保存完整 `[query_head, sequence]` FP32 probability。

### 6.5 将 Value affine dequant 融入 attention

Value 定义为：

```text
V[t, d] = code[t, d] * scale[t] + minimum[t]
```

PV 可以改写为：

```text
sum_t(P[t] * V[t, d])
= sum_t((P[t] * scale[t]) * code[t, d])
  + sum_t(P[t] * minimum[t])
```

实现中：

1. AIV 对 probability 乘 `value_scale`；
2. AIC 使用缩放后的 probability 和 Value code 执行 PV；
3. AIV 对原 probability 与 `value_minimum` 做 ReduceSum；
4. 将该标量广播加到 128 个输出维度。

这样避免对每个 token 的 128 个 Value 元素执行完整 affine dequant。Key 的
norm correction 同样不写回到每个 Key 元素，而是在 QK scores 上乘
`key_scale`。

当前稳定实现仍把 BF16 K centroid 值和 BF16 Value code tile 写入 GM workspace，
再由 AIC 搬入 L1；该 GM 往返尚未消除。

## 7. 分阶段性能优化

### 7.1 汇总

| 阶段 | 提交/实现 | 平均延迟 | 相对上一阶段节省 | 相对上一阶段降幅 |
| --- | --- | ---: | ---: | ---: |
| Reference | FP32/reference decode | 1023.254736 ms | — | — |
| Naive grouped | Triton grouped-GQA | 700.346008 ms | 322.908728 ms | 31.557% |
| 初版流水化 | `c4b52955` | 3.680818 ms | 696.665190 ms | 99.474% |
| Cache layout V2 + batch load | `83d5b988` | 2.642060 ms | 1.038758 ms | 28.221% |
| 384-token outer tile | `de0e8940` | 2.633578 ms | 0.017085 ms | 0.645% |

384-token 优化使用同轮 256-token 基线 2.650662 ms 计算增量，避免把设备频率变化
误算为 tile 收益。各阶段使用相同的 B16/S16K、Hq=16/Hkv=2、D=128 主 shape；
Reference/naive 的短轮次数据只用于数量级对照。

### 7.2 Naive grouped-GQA

Grouped-GQA 将相同 KV head 的 query heads 合并处理，避免复制 K/V，并把 reference
从 1023.254736 ms 降到 700.346008 ms：

```text
节省 322.908728 ms
加速 1.461x
```

该实现仍由 Triton split kernels、partial buffers 和 reduction kernels 组成，
长上下文高并发时 kernel 数量和标量控制开销仍然过高。

### 7.3 初版 AscendC 流水化算子

`c4b52955` 新增 `TurboQuantPagedAttention`，主要变化为：

1. 以 `(batch, KV head, split)` 为 AIC task；
2. 一个 mixed kernel 内完成 Cache 解码、QK、softmax、PV 和 split partial；
3. AIC 使用 Cube `Mmad`，AIV 使用 vector softmax 和 reduction；
4. 使用 cross-core flag 串联 AIV 和 AIC；
5. 使用 FP32 running max/sum/accumulator 保持 tile 和 split 合并稳定；
6. 通过 out-style PyTorch schema 写入调用方 output，避免返回式临时张量。

延迟从 700.346008 ms 降到 3.680818 ms：

```text
节省 696.665190 ms
加速 190.269x
```

主要收益来自消除大量 Triton program launch、paged 标量寻址和独立 reduction
kernel，而不是单条 Cube 指令本身。

### 7.4 Cache layout V2 和连续 batch DMA

`83d5b988` 将 Cache 的物理布局从：

```text
[block, token, kv_head, packed_slot]
```

改为：

```text
[block, kv_head, token, packed_slot]
```

公共 tensor shape 保持不变，store、Triton decode、Triton dequant 和 AscendC
kernel 显式使用 V2 physical stride。

优化前，同一 KV head 的相邻 token 在 GM 中相隔多个 KV head，AIV 只能按 token
执行带 stride 的小搬运。V2 后，同一 head 的 token 连续，`LoadSlotBatch` 每次使用
一次 `DataCopyPad` 连续搬运最多 64 个 slot，再在 UB 内逐 slot 解码。

延迟从 3.680818 ms 降到 2.642060 ms：

```text
节省 1.038758 ms
加速 1.393x
降幅 28.221%
```

该阶段是初版流水化 kernel 之后最大的单项优化。

### 7.5 384-token outer tile 和 256-token Cube sub-tile

`de0e8940` 将外层 AIV/softmax tile 从 256 token 扩大到 384 token，同时保持：

```text
CUBE_TILE_TOKENS = 256
```

一个 384-token outer tile 在 Cube 内拆为：

```text
QK: 256 + 128
PV: 256 + 128，第二块继续累加 L0C
```

这样不扩大 L0A/L0B 的 256-token 容量，但将外层 tile 数量从：

```text
ceil(S / 256)
```

降到：

```text
ceil(S / 384)
```

减少约三分之一的 online-softmax 状态更新、cross-core flag 和 tile 启停。
同轮 A/B 从 2.650662 ms 降到 2.633578 ms：

```text
节省 0.017085 ms
加速 1.006x
降幅 0.645%
```

该优化的收益小于理论上的同步次数降幅，说明当前主要时间已转移到 Cache
读取、code 解码和 Cube 数据搬运。

### 7.6 Split 调度

Host tiling 先计算填满 AIC 所需的最小 split 数，再在最多 8 个 split 内搜索
归一化 critical-path work：

```text
waves(candidate) / candidate
```

调度不只追求 core occupancy，还避免尾部 straggler wave。长上下文场景中，
过少 split 会让部分 core 处理多轮 task，过多 split 会增加 partial state、
同步和最终 reduction。

B16/S16K 专项测试中：

| Splits | 平均延迟 |
| ---: | ---: |
| 1 | 3.708234 ms |
| 2 | 2.914093 ms |
| 8/自适应主配置 | 约 2.63–2.65 ms |

## 8. Workspace 与数据流

每个 AIC core 的稳定实现 workspace 包含：

| 区域 | Dtype | 用途 |
| --- | --- | --- |
| Key tile | BF16/FP16 | centroid lookup 后的 K |
| Value tile | BF16/FP16 | Value code |
| Key scale | FP32 | QK 后处理 |
| Value scale | FP32 | probability 预缩放 |
| Value minimum | FP32 | PV affine 常数项 |
| Scores | FP32 | QK 和 softmax |
| Probability | BF16/FP16 | Cube PV 输入 |
| Tile output | FP32 | PV 输出 |
| Partial accum/sum/max | FP32 | split 合并 |

当前数据流为：

```text
packed Cache GM
  -> AIV UB
  -> K/Value tile GM
  -> AIC L1/L0
  -> scores/tile output GM
  -> AIV UB
  -> partial state GM
  -> final AIV reduction
```

已经消除了完整 sequence dense K/V workspace，但 tile 级 AIV→GM→AIC 往返仍存在。

## 9. 代码组织

| 文件 | 职责 |
| --- | --- |
| `csrc/attention/turbo_quant_paged_attention/op_kernel/turbo_quant_paged_attention.cpp` | AIV/AIC mixed kernel、online softmax、QK/PV、split reduction |
| `csrc/attention/turbo_quant_paged_attention/op_host/turbo_quant_paged_attention_tiling.cpp` | shape/layout 校验、split 搜索、workspace 计算、block dimension |
| `csrc/attention/turbo_quant_paged_attention/op_host/turbo_quant_paged_attention_def.cpp` | CANN OpDef、dtype/format 和 SoC 注册 |
| `csrc/attention/turbo_quant_paged_attention/op_host/turbo_quant_paged_attention_proto.cpp` | 输出 shape/dtype inference |
| `csrc/torch_binding.cpp` | `_C_ascend` schema、输入检查、ACLNN 调用 |
| `vllm_ascend/ops/turboquant.py` | Python operator wrapper 和 schema availability 检查 |
| `vllm_ascend/attention/turboquant.py` | backend dispatch、Q rotation、workspace 复用、fallback |
| `vllm_ascend/ops/triton/turboquant_store.py` | packed Cache 写入和 V2 physical stride |
| `vllm_ascend/ops/triton/turboquant_decode.py` | Triton fallback 和 V2 physical stride |

## 10. 框架集成与 dispatch

`AscendTurboQuantAttentionImpl` 在初始化时检查：

```text
D=128
GQA group=8
K4V4
norm correction
无 ALiBi
无 logits soft cap
TurboQuantPagedAttention schema 可用
```

满足条件时：

```text
eager single-token decode -> ascend_pipelined
```

不满足流水化 shape、但 `TurboQuantPagedDequant` 可用时：

```text
eager single-token decode -> ascend_dequant_fia
```

两个 AscendC 算子均不可用时：

```text
decode -> packed_triton
```

Qwen3-0.6B 的 GQA group=2，因此当前选择 `ascend_dequant_fia`。强制
`VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=ascend_fused` 只要求至少一个
Ascend fused path 可用，不会绕过 shape 检查把 group=2 输入错误送入 group=8 kernel。

## 11. 当前结论

1. Qwen3-0.6B/TP2 当前可用优化路径是 `AscendC paged dequant + FIA`；已有
   `21fe9cac` B1/S512 实测从 1.558948 ms 降到 0.299652 ms，节省
   1.259296 ms。
2. 完整流水化 AscendC attention 在 B16/S16K、GQA group=8 专项 shape 上稳定为
   2.63–2.65 ms，相比 naive grouped Triton 的 700.346008 ms 加速 265.930x。
3. 初版流水化之后，Cache layout V2 和连续 64-slot DMA 贡献 1.038758 ms，
   是最大的后续单项收益。
4. 384-token outer tile 贡献 0.017085 ms；收益受 Cache decode 和数据搬运占比限制。
5. 当前流水化结果是单层算子数据，不是 Qwen3-0.6B 或 Qwen3-32B 的端到端
   serving 吞吐。
6. Qwen3-0.6B 要使用完整流水化算子，必须实现并验证 GQA group=2 specialization；
   Cache layout V2 合入后还需要重新测量真实 TP1/TP2、并发 1/16 和长上下文结果。

## 12. 测试记录

| 数据 | 路径 |
| --- | --- |
| Qwen3-0.6B/TP2 short profile | `logs/turboquant/debug_validation_9382_20260722/profile/benchmark.json` |
| Naive/reference profile | `logs/turboquant/full_910b4_20260722_151912/operator_profile/` |
| 初版流水化 | `profiles/turboquant_20260724_012533/benchmark.json` |
| Cache layout V2 | `profiles/turboquant_20260724_014545/benchmark.json` |
| 256-token baseline | `profiles/turboquant_longtile_baseline_20260724/benchmark.json` |
| 384-token tile | `profiles/turboquant_longtile384_20260724/benchmark.json` |
| 当前稳定重复测量 | `profiles/int8_pv_baseline_b16_s16384/benchmark.json` |
