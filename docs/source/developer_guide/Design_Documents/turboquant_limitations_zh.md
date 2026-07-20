# TurboQuant Ascend 当前功能限制与支持上限

## 1. 文档范围

本文记录基于 vLLM `0.20.2`/`0.20.2+empty` 的 Ascend TurboQuant 实现当前能够保证的能力、
实验性能力、明确拒绝的组合和后续解阻条件。这里的“支持”要求同时满足配置可启动、结果
正确、不会破坏其他 backend，并具备与风险相匹配的 NPU 测试。
同时，本文给出每项能力在当前分支、910B4 和 Triton-Ascend 技术栈下的实际工程上限，避免
把“当前未实现”“需要新 kernel”和“受上游/硬件阻塞”混为一谈。
代码调用链、packed layout 和算子计算过程见
[TurboQuant 在 vLLM Ascend 中的实现逻辑](turboquant_implementation_zh.md)。

## 2. 当前能力与工程上限

下表中的“工程上限”限定为当前 vLLM core 接口、910B4 和 Triton-Ascend 技术栈下，保持
backend 通用化和社区可维护性时能够达到的范围。它不是当前支持声明；只有完成对应 NPU
正确性、E2E 和性能门禁后，才能把能力从实验性升级为支持。

工作量定义：S 表示测试或局部适配，M 表示 metadata/单机 kernel 改造，L 表示新增主要执行
路径，XL 表示跨 runner、跨进程协议或分布式算法。

| 能力 | 当前状态 | 当前分支的实际工程上限 | 工作量/主要阻塞 |
| --- | --- | --- | --- |
| vLLM V1 model runner | 支持 | 稳定社区支持 | 代码回归已补；剩余 910B4 E2E |
| Dense causal self-attention | 支持 | RoPE + MHA/GQA，常见 `head_dim=64/128/256` | NPU correctness/profile 矩阵已补；剩余实机执行 |
| FP16/BF16 activation | 支持 | FP16/BF16 均稳定支持，packed cache 保持 `uint8` | BF16、零 key、常量 value 测试已补 |
| MSE key/value presets | 支持 | 稳定支持 4-bit、K3V4 和 3-bit 三种 preset | 三 preset 的 1/2/4/8-token 算子矩阵已补；剩余模型精度 |
| Eager decode | 支持 | 稳定支持单 token、mixed request batch 和非连续 block table | 跨 page/非连续 page 测试已补；剩余 preemption E2E |
| 首次 prefill | 支持 | 稳定支持首次 prefill，并支持同批 decode + prefill | mixed metadata 和输出切分已实现 |
| Continuation/chunked prefill | 分层支持 | 小 chunk 复用 packed multi-query decode；大 chunk 仅反量化历史再调用 FIA | workspace 容量检查和非持久 history scratch 已实现；剩余 910B4 性能门禁 |
| Prefix cache | 实验性 | 同一 engine 内稳定支持命中、page copy、eviction、preemption/resume | overlap-safe page copy/swap 已实现；剩余 scheduler E2E |
| Qwen3-32B | 首个验证目标 | 三种 preset 的 eager、ACLGraph、prefix 和 chunked prefill 支持 | 代码路径已具备；仅剩 910B4 精度与性能数据 |
| 其他 dense decoder | 架构可扩展 | 覆盖无特殊 mask 的 Qwen/Llama 类 RoPE MHA/GQA 模型 | 能力按 attention contract 校验；剩余逐模型认证 |
| ACLGraph | 实验性支持 | 支持 uniform single/multi-token decode | metadata、padding、direct replay 已覆盖；剩余完整模型 E2E |
| Speculative decoding | 实验性实现 | 标准 target-model draft verify；parallel drafting 仍拒绝 | 1/2/4/8-token kernel 和 uniform graph 已覆盖；剩余拒绝采样 E2E |
| Context parallel | 不支持 | Dense TurboQuant 的 PCP prefill 和 DCP decode 均可实现 | XL：本地 softmax `(out,lse)` 合并、cache sharding 和多卡测试 |
| KV transfer/PD | 不支持 | 先支持一个 opaque-page connector、同 TP/同 preset 的 producer/consumer | L/XL：layout 握手、per-layer page bytes、混合 cache group |
| FP8 key preset | 不支持 | 910B4 可做软件 E4M3 功能路径；不承诺原生 FP8 加速 | L：FP8 编解码语义和精度；原生性能受硬件限制 |
| ALiBi/logits soft cap | 实验性实现 | decode 在 packed kernel 内执行；prefill 使用 query-tiled 精确 fallback | kernel/reference 和 workspace 上限测试已补；剩余对应模型 E2E |
| Sliding window/sinks | 不支持 | Sliding window 需新 TQ cache spec；sinks 需独立 softmax 语义 | L：涉及 vLLM core scheduler/cache spec，不只是 backend |
| MLA、稀疏/压缩 attention | 不支持 | 不纳入当前 TurboQuant backend | 独立项目：cache 表示和 attention 数学均不同 |
| Batch invariant | 不支持 | FWT 和固定归约落地后可支持 | M/L：dense GEMM 可能随 batch shape 选择不同算法 |
| V2 model runner | 不支持 | 可达到与 V1 同等的单机 dense 能力 | L：独立 allocator、metadata 和 runner 生命周期适配 |
| Ascend 310P | 不支持 | 最多提供功能 fallback，不作为推理加速目标 | XL：算子、内存容量和运行时能力均需重新验证 |

### 2.1 支持层级

基于收益、风险和依赖关系，能力应分三层推进：

1. **近期可稳定化**：现有三种 preset、Qwen3-32B、更多 RoPE dense decoder、Prefix Cache、
   uniform multi-token ACLGraph、标准 spec decode、FP16/BF16、mixed batch 和 continuation prefill 优化；
2. **需要新核心路径**：parallel drafting、batch invariant、指定 KV connector 和 V2 runner；
3. **需要跨模块设计**：DCP/PCP、sliding-window cache spec、通用 PD connector、310P，以及
   与 MLA/稀疏 attention 的组合。

### 2.2 改动归属

| 范围 | 可以推进的能力 | 额外条件 |
| --- | --- | --- |
| 仅改当前 vLLM-Ascend 仓库 | mixed batch、continuation prefill、ALiBi、soft cap、标准 spec target path 已实现；FWT/fused store 待定 | 需要 910B4 做 kernel 和 E2E 验证 |
| 当前代码已具备，主要补测试 | Prefix Cache、uniform multi-token ACLGraph、标准 spec target path、更多 MHA/GQA dense 模型、FP16/BF16 | 需要真实 scheduler/模型负载 |
| 需要同时改 vLLM core | TQ sliding-window spec、完整 V2 生命周期、可能的通用 spec metadata | 需要固定 core/plugin commit 契约 |
| 需要 connector 协议改造 | PD、LMCache/offload、跨 engine Prefix Cache | 需要 layout version 和 page-byte handshake |
| 需要多卡算法与环境 | PCP/DCP、CP 与 graph/spec 的组合 | 单机 910B4 无法完成验收 |
| 受硬件能力约束 | 原生 FP8 加速、310P 高性能路径 | 软件 fallback 不等于推理加速 |

### 2.3 本轮 S/M 项闭环结果

本轮已经在代码或测试层闭环以下项目：

- mixed decode/prefill 显式拆分，decode 和 prefill 输出按 token boundary 写回；
- continuation prefill 分层执行，阈值以内直接读取 packed cache，阈值以上只反量化历史；
- small continuation 对 layer workspace 做容量检查，不足时使用 eager 临时 workspace；
- large continuation 使用非持久 history scratch，避免每层永久保留 context-sized buffer；
- ALiBi 和 logits soft cap 进入 packed decode score 计算，prefill 提供 query-tiled 精确 fallback；
- 三 preset 的 1/2/4/8-token、FP16/BF16、`head_dim=64/128/256` correctness case；
- 跨 128-token page、非连续 physical block table、负 slot、零 key 和常量 value；
- Prefix Cache 所需 opaque page copy/swap，并显式保证重叠 source/destination 的快照语义；
- uniform multi-token ACLGraph、graph padding 和 non-uniform eager fallback；
- `profile_matrix.sh` 覆盖三 preset、两种 activation dtype、三个 head dimension、三档 split，
  并与 native Ascend paged attention 比较。

仍需要 910B4 才能关闭的是完整模型、scheduler 和性能门禁，不是 Python/Triton 接口缺失。
Qwen3-32B、Prefix Cache hit/preemption、拒绝采样后的 spec sequence 修正和 full-model ACLGraph
都属于这一类。当前开发机没有 `torch_npu`，因此文档不会把这些测试写成“已通过”。

本轮有意不处理 MLA/稀疏 attention、310P、V2 model runner；同时继续保留 CP、KV transfer、
sliding window/sinks 和 batch invariant 的 fail-fast，因为它们不是局部 S/M 改动。

## 3. ACLGraph

### 3.1 当前支持范围

当前 decode 路径不仅包含两个 Triton kernels，还包含：

- query 到 FP32 的转换；
- query 与 Hadamard matrix 的 GEMM；
- split workspace 的切片；
- FP32 output 到 query dtype 的转换。

这些操作与上游 CUDA TurboQuant 一样，在 capture 阶段由 graph memory pool 管理。split 数量
固定为 `tq_max_kv_splits_for_cuda_graph`，layer 上的 split output、最终 FP32 output 和 LSE
workspace 在模型加载阶段预分配。metadata builder 声明：

```text
AttentionCGSupport.UNIFORM_BATCH
```

捕获时将 device-side `seq_lens` 设为每个请求的 `query_len`，即该 query shape 的最短合法
上下文；replay 前由 model runner 在相同地址更新真实 `seq_lens`、block table、slot mapping
和 query 内容。负数 slot mapping 由 store kernel 跳过，虚拟 padding 请求产生的有效 KV 长度
在设备侧截断到 0，因此 graph padding 不会写坏 packed cache，也不会把负长度传给 split kernel。

当前不把支持级别扩大为 `ALWAYS`。Continuation prefill 会根据本轮 `sum(seq_lens)` 动态分配
dense K/V tensor，混合 prefill/decode 也不适合进入 full graph。

Uniform multi-token 不会把 query tokens 展平后一次性扩大 split workspace。实现按 token step
执行静态循环，每一步仍以 request 数作为 decode batch，并复用同一组 layer workspace。对第
`j` 个 query token 使用 `final_seq_len - query_len + j + 1` 作为有效 KV 长度，因此 draft tokens
之间保持 causal relation。固定 query length 的循环会完整进入 ACLGraph capture。

### 3.2 仍需在 NPU 验证的风险

当前已有 metadata 单测和直接捕获/replay Triton store + decode 的 NPU 测试。direct replay case
会同时修改 sequence length、block table、slot mapping 和 Q/K/V；完整模型仍有以下风险：

- replay 时继续使用 capture 阶段的 tensor 地址；
- cache update 被编译器错误重排或消除；
- Triton kernel 在 capture 内首次 JIT；
- graph capture 是否始终命中预分配 workspace；eager fallback 已有容量检查。

910B4 验证顺序为：

1. 运行 `scripts/turboquant_triton/correctness/run_aclgraph_smoke.sh`，确保 kernel 在 capture 前完成 JIT；
2. 先使用 eager server 建立确定性正确率基线；
3. 设置 `ENFORCE_EAGER=0` 启动 decode-only ACLGraph；
4. 覆盖不同 batch size、sequence length、preemption 和 prefix hit；
5. 比较 graph replay 前后的 packed cache、token 输出和长文本稳定性；
6. 对比 eager/ACLGraph 的 TTFT、TPOT、吞吐和 graph pool 显存。

### 3.3 可以支持到什么程度

当前分支合理的最高目标是：

- 普通 autoregressive decode 使用 `FULL_DECODE_ONLY` ACLGraph；
- 不同 capture batch size 分别建图，真实 sequence length 在相同地址更新；
- Prefix Cache、preemption 和混合 native/TurboQuant cache group 可以参与 decode graph；
- uniform multi-token/spec decode 使用 `UNIFORM_BATCH`，捕获固定 query length；
- prefill 和 mixed prefill/decode 保持 eager 或 piecewise，不声明 `ALWAYS`。

最后一点是有意的边界，不是遗漏。动态 continuation prefill 的临时 K/V 长度取决于历史长度，
强行放入 full graph 会制造大量 capture shape 和 graph pool 显存，收益通常低于复杂度。

## 4. Prefix Cache

### 4.1 当前不是完全禁用

Prefix cache 的 scheduler 语义是复用已经计算过的完整 KV pages。scheduler 不需要理解 page
内部是 FP16 K/V 还是 packed TurboQuant bytes，因此数据模型原则上兼容。

当前 backend 已提供：

- 整个 packed page 的 `copy_blocks()`；
- 跨 cache tensor 的 `swap_blocks()`；
- 按 block table 读取非连续 physical pages；
- prefix hit 后 continuation prefill 的完整反量化 fallback。

### 4.2 为什么仍标记为实验性

现有单元测试已经确认 byte page copy/swap 不改变内容，并覆盖重叠 source/destination 的快照语义。
还没有在 910B4 上覆盖：

- partial prefix hit 和 full prefix hit；
- eviction 后重新分配同一个 physical block；
- preemption 和 request resume；
- 首尾 native cache layer 与中间 TurboQuant layer 的多 cache group；
- prefix hit 后 chunked prefill 的 causal attention；
- 多请求共享 prefix 时的精度一致性。

这些场景通过前，不能将 prefix cache 标记为稳定支持。当前实现没有在平台配置中主动拒绝
`enable_prefix_caching`。

### 4.3 可以支持到什么程度

Prefix Cache 是最接近稳定支持的扩展项。`TQFullAttentionSpec` 已向 scheduler 提供准确的
packed page bytes，scheduler 的 block hash、引用计数、eviction 和 preemption 不解析 page
内部 K/V 格式。近期可以达到：

- 同一 engine 内 partial/full prefix hit；
- 多请求共享同一 packed prefix；
- page copy、copy-on-write、eviction 和 request resume；
- 首尾 native cache layer 与中间 TurboQuant layer 共存；
- prefix hit 后继续 prefill，再进入 packed decode。

这里不自动包含 LMCache、CPU offload 或 PD disaggregation；它们属于 KV Transfer 的独立
layout/connector 契约。完成上述 E2E 后，可以把“不使用外部 KV connector 的 Prefix Cache”
升级为正式支持。

## 5. Speculative Decoding

### 5.1 当前实现

底层 packed kernel 仍保持“一行 query 对应一个有效 seq_len 和一行 block table”的简单契约，
attention backend 在其上增加 multi-token 调度：

- uniform query length：按 token step 对所有 requests 批处理；
- non-uniform query length：eager 模式按 request/token 依次执行；
- 每个 step 复用 request 级 split/output/LSE workspace；
- `query_start_loc` 的累计边界用于恢复每个 request 的 query length；
- `SpecDecoding` 和普通 `DecodeOnly` 共用 packed decode 数学路径。

### 5.2 Spec decode 改变了什么

Spec decode 一轮会为每个 request 验证多个 draft tokens。需要处理：

- 每个 request 不同的 query token 数；
- draft tokens 之间的 causal relation；
- 多个 token 的 slot mapping 和 cache 写入顺序；
- accepted/rejected tokens 对 sequence length 的影响；
- padded drafter batch；
- `query_start_loc` 到 packed decode grid 的映射。

当前平台已放开标准 speculative decoding，但继续在初始化阶段拒绝 parallel drafting。后者的
query/causal 结构不是简单的顺序 draft chain，不能套用当前 effective sequence length 公式。

### 5.3 可以支持到什么程度

标准 target-model speculative verification 已在不完全重写 packed kernel 的前提下接入。
逻辑等价于把每个 request 的多个 query token 展开成 synthetic decode rows：

1. 由 `query_start_loc` 展开 query token；
2. 为同一 request 复制 block-table row；
3. 为第 `j` 个 draft query 生成 `base_seq_len + j + 1` 的有效长度；
4. 让现有 packed decode 对每一行执行 causal attention；
5. 将结果按原 token 顺序写回 target logits buffer。

同一轮新 K/V 在 attention 前已经写入 cache。被拒绝 token 留在物理 slot 中并不构成错误，
只要下一轮 scheduler 的逻辑 sequence length 和 slot mapping 不再引用这些位置。

当前代码可处理固定配置下的 uniform multi-token graph，也保留 non-uniform eager fallback。
算子 correctness 已覆盖 1/2/4/8-token query length；支持声明仍需在 910B4 覆盖常规 draft
model、EAGLE target verification、拒绝 token 和下一轮 sequence length 修正。Parallel
drafting、spec + CP，以及 drafter 自身也使用 TurboQuant，应放到后续组合实现。ACLGraph 已
提升到 `UNIFORM_BATCH`，但不能提升为 `ALWAYS`。

## 6. Context Parallel

### 6.1 当前假设

当前 decode kernel 假设每个 TP rank 持有该 request 的完整 KV sequence，并可通过本地
block table 访问从位置 0 到 `seq_len - 1` 的所有 tokens。

### 6.2 DCP/PCP 的影响

DCP 或 PCP 会将 sequence/cache 分布到多个 rank。除了修改地址，还需要处理分布式 attention
归约：

1. 每个 rank 计算本地 score max；
2. all-reduce 得到全局 max；
3. 重新缩放并归约本地 softmax sum；
4. 归约 value accumulator；
5. 处理 local seq_lens、interleave 和 rank-specific block table；
6. 适配 cache update 的 local slot mapping。

只让每个 rank 独立运行当前 kernel 会得到只关注本地 KV shard 的错误结果。因此平台根据
`decode_context_parallel_size * prefill_context_parallel_size` fail fast。

### 6.3 可以支持到什么程度

当前 packed Stage 1 已经为每个 split 产生 normalized output 和 LSE，这与 vLLM-Ascend CP
代码中的 `_update_out_and_lse` 合并接口相匹配。因此 DCP decode 不需要传输或反量化完整
KV，可以按以下方式实现：

1. 每个 DCP rank 只读取本地 packed pages；
2. 输出本地 `(attention_output, lse)`；
3. 复用现有 CP all-to-all/all-gather 通信；
4. 用全局 LSE 合并各 rank output；
5. 使用 CP model runner 已计算的 local slot mapping 和 local sequence metadata。

PCP 首次 prefill 可以复用现有 raw QKV gather/FIA 路径；chunked prefill 则需要让历史 packed
cache 的本地 shard 参与 context output 合并。工程上可以达到 dense TurboQuant 的 PCP、DCP
及 PCP+DCP，但这是 XL 级工作，必须使用多卡测试覆盖空 shard、uneven length、prefix hit 和
preemption。第一阶段不应同时打开 CP + spec decode + ACLGraph，组合能力应逐层验证。

## 7. Continuation Prefill 性能限制

当前实现不再对每个 continuation request 全量反量化“历史 + 当前 chunk”。它采用与上游一致
的分层策略：

- `query_len <= 128`：为 query token 生成递增有效 `seq_lens`，直接复用 packed decode；
- `query_len > 128`：只反量化历史 prefix，当前 chunk 保持 raw K/V，再调用 NPU FIA；
- history key/value 使用本次调用的非持久 scratch，不挂在 attention layer 上；
- mixed batch 中 decode requests 先走 packed path，prefill requests 再逐 request 选择上述策略。

大 continuation 仍会创建 inverse-Hadamard 结果和拼接后的 dense K/V view，其临时显存与历史
长度成正比，但不会随已经执行过的 layer 数量累积持久 buffer。后续最高性能方向仍是分块
反量化并融合 attention，或实现直接消费 packed cache 的 varlen prefill kernel。

当前实现的主要加速目标是 autoregressive decode，不应根据短 prompt smoke test 推断长上下文
prefill 性能。

### 7.1 可以支持到什么程度

上游 vLLM TurboQuant 已提供一条可移植的分层策略：

- 首次 prefill：直接使用当前 raw K/V；
- 小 continuation chunk：把每个 query token 展开为 synthetic decode row，直接读取 packed
  cache；
- 大 continuation chunk：只反量化历史 prefix，当前 chunk 继续使用 raw K/V，再调用 FIA；
- 大 continuation 使用 eager-only scratch，并依赖 NPU caching allocator 回收复用。

上述策略已经移植。它仍不会达到原生 FIA prefill 的吞吐上限，但已避免短 continuation
chunk 的重复全量 dequant。进一步的最高性能方案是 fused packed-prefill kernel，不过其
工作量接近重新实现一套 varlen causal attention，不应作为首个社区版本的合并前置条件。

## 8. 算子性能限制

### 8.1 Hadamard 实现

当前 store 和 decode 使用 dense `D x D` Hadamard GEMM。对于 `D=128` 功能正确且实现简单，
但其计算复杂度是 `O(D^2)`。Fast Walsh-Hadamard Transform 可以将其降低到 `O(D log D)`，
并减少临时 FP32 tensor。

在 910B4 上合理上限是实现 per-vector Triton-Ascend FWT，并让 store 和 decode 共用同一
transform 契约。这样不仅降低复杂度，也避免 dense GEMM 根据 batch shape 改变算法，是支持
batch invariant 的前置条件之一。

### 8.2 Store 临时空间

当前实现已经把 FP32 key norm 和 normalization 融入 store Triton kernel，并使用
activation-dtype rotation GEMM，不再物化完整的 FP32 normalized key。长 prefill 仍会产生
一个 activation-dtype rotated-key 临时 tensor，需要继续评估其显存和带宽。

可以达到的优化上限是 fused FWT + centroid lookup + bit-pack store。Value 的 min/max、
scale 和 pack 已在同一 program 完成。长 prefill 后续可使用分块 grid，避免为全部 tokens
保留 rotated key。

### 8.3 Decode 固定 splits

`num_kv_splits` 当前来自 vLLM attention config。短序列使用过多 splits 会增加空 program 和
Stage 2 开销，长序列 splits 太少则限制并行度。需要基于 910B4 profiling 建立 batch size、
sequence length 与 split 数量的调优表或启发式规则。

ACLGraph replay 期间 grid 需要固定，因此不能在同一张图中按实时 sequence length 改变 split
数量。可采用“每个 capture bucket 固定 split 配置”或保持固定最大 grid、让空 split 提前
返回。近期应先 profile `8/16/32` 三档；只有收益明确时再引入多 graph split bucket。

`scripts/turboquant_triton/performance/profile_matrix.sh` 已提供 `8/16/32` splits、FP16/BF16、三 preset
和 `head_dim=64/128/256` 的无 trace 批量基线，并在相同 batch/context/head shape 下调用
native Ascend paged attention。报告直接给出 decode speedup 和理论 cache 压缩比；它不会在
没有 910B4 数据时写死启发式规则。

### 8.4 Batch invariant

packed decode 的每个 `(request, head)` 独立计算，Stage 2 的 split 合并顺序也是固定的，这两点
有利于 batch invariant。但是当前 query/key Hadamard rotation 使用 dense GEMM，底层可能根据
矩阵第一维，也就是 batch/tokens × heads，选择不同 tiling 或归约实现，量化边界附近的微小
差异可能改变 centroid index。

因此不能只覆盖 `supports_batch_invariance()` 就解除平台检查。最高可支持程度是：先以
per-vector FWT 替换 shape-sensitive GEMM，再固定 centroid search、online softmax 和 Stage 2
归约顺序，最后运行相同 request 在不同 batch 位置、batch size、ACLGraph capture size 下的
bitwise cache 和 token 输出测试。完成这些门禁后，可以把 TurboQuant 纳入 Ascend batch
invariant 模式。

## 9. 其他明确限制

### 9.1 FP8 key

`turboquant_k8v4` 需要确定 910B4/CANN 上的 FP8 encoding、bitcast 和 Triton load/store 语义。
当前 kernel 只实现 MSE 3-bit/4-bit key，因此在 backend selection 阶段拒绝 FP8 key。

上游 CUDA 使用 Triton `float8e4nv/float8e4b15` cast 和 bitcast。910B4 不应假设具备等价的
原生 FP8 算术路径。可以实现软件 E4M3 encode/decode，将 key 仍按一 byte 存入 packed cache，
从而获得功能和容量支持；但 decode 时的转换成本可能抵消带宽收益。只有 profiler 证明相对
4-bit MSE 路径有价值后，才应在 910B4 上把 `turboquant_k8v4` 标记为性能支持。原生 FP8
加速应留给明确提供对应数据类型和算子的后续 Ascend 硬件。

### 9.2 非标准 attention

以下功能会改变 cache layout、mask 或 score 语义，当前不支持：

- MLA；
- sparse/compressed attention；
- sliding window；
- attention sinks；
- multimodal prefix attention；
- non-causal attention；
- fused output quantization。

新增支持时应按能力拆分 backend 或 kernel 参数，不应写模型名称分支。

ALiBi 和 logits soft cap 已不在上述拒绝列表。packed decode 在 softmax 前执行 soft cap，并
按 query head 加入相对位置 bias；首次/continuation prefill 使用 query-tiled 精确 PyTorch
fallback。该路径不会构造完整 `[H,Q,K]` score tensor，也不会把 GQA K/V 复制到全部 query
heads；但它仍以功能正确为主，长 prompt 吞吐需在 910B4 验证，因此当前标记为实验性实现。

其余能力的可支持程度不同：

- **sliding window**：kernel 可从 `max(0, seq_len-window)` 开始读取，但当前 vLLM core 会优先
  返回普通 `SlidingWindowSpec`，其 page bytes 与 TurboQuant 不一致，因此必须新增
  TQ-aware sliding-window spec 和 scheduler 管理逻辑；
- **attention sinks**：需要在窗口淘汰时保留 sink blocks，并在分块 softmax 中保持其语义；
- **multimodal prefix/non-causal**：改变 mask 和 cache 生命周期，不纳入首个 dense decoder
  backend；
- **fused output quantization**：可在 Stage 2 后增加 NPU quantize 路径，但应与模型量化方法
  联合设计，而不是固定在 TurboQuant preset 中。

### 9.3 KV Transfer

PD disaggregation 和 KV connector 通常假定普通 K/V tensor 或已注册的 backend-specific layout。
TurboQuant 是单一 interleaved byte tensor，发送端、接收端、page size、layer grouping 和
序列化协议都需要显式适配。因此当前在 platform 和 model runner 两处拒绝 KV transfer。

packed page 本身可以作为 opaque bytes 传输，不需要在 producer 端解压。近期可选择一个
Ascend connector 做窄范围支持，要求 producer/consumer 的 vLLM commit、TurboQuant preset、
block size、TP 切分和 layer layout 完全一致，并在握手中携带 layout version 与每层 page
bytes。通用支持还要处理首尾 native cache group 与中间 TQ group 的不同 tensor shape；不能
通过删除当前 fail-fast 就宣称所有 connector 可用。

### 9.4 Hardware 和版本

当前目标是 Ascend 910B 系列、torch/torch-npu `2.10.0`、triton-ascend `3.2.1` 和 vLLM
`0.20.2` core 接口。不能假设代码可直接运行在 310P、V2 model runner 或不包含
`TQFullAttentionSpec` 的其他 vLLM commit。

V2 runner 没有算法层面的障碍，但需要独立实现 `TQFullAttentionSpec` allocation/reshape、
metadata 生命周期、cache binding 和 graph capture，不能复用 V1 model runner patch 后直接
放开。310P 则需要重新验证 Triton-Ascend、uint8 packed load/store、显存容量和 attention
fallback；即使功能可做，也不应作为本项目的推理加速目标。

### 9.5 模型支持边界

不应以“是否叫 Qwen3”判断支持，而应按 attention contract 分类：

- 第一层：RoPE、causal、MHA/GQA、`head_dim=64/128/256`，没有 window/sink/softcap；可以在
  Qwen3 验证后扩展到同类 Qwen/Llama 模型；
- 第二层：ALiBi 或 logits soft cap；增加 kernel 参数并逐模型认证；
- 第三层：sliding window、attention sinks、multimodal prefix；需要 cache spec 或 mask
  生命周期改造；
- 独立体系：MLA、DSA/SFA、其他 sparse/compressed attention；不复用当前 full-K/V packed
  backend。

模型权重量化与 TurboQuant KV Cache 正交。AWQ、GPTQ、W8A8 等组合能否使用，取决于模型
权重量化 backend 是否改变 attention 输入 dtype/layout，应按组合做 E2E，而不是在 TurboQuant
代码中增加模型或量化方法名称判断。

## 10. 精度和性能验收

社区合并前建议至少完成以下门槛。

### 10.1 算子正确性

- 三个支持的 cache dtype 均完成 store/dequant round trip；
- packed decode 与反量化 reference attention 对齐；
- 覆盖 GQA、跨 block boundary、负 slot 和非连续 block table；
- 覆盖 FP16 和 BF16 query/K/V；
- 检查 NaN、Inf、零向量和极端 value range。

上述算子 case 已写入 `tests/ut/ops/test_turboquant_triton.py`。其中三 preset、1/2/4/8-token、
FP16/BF16、三种常用 head dimension、跨 page、ALiBi/soft cap、零 key 和常量 value 已形成
参数化矩阵；是否通过必须以 910B4 pytest 结果为准。

### 10.2 模型正确性

- Qwen3-32B 短 prompt 和长 prompt；
- 单请求和 mixed batch；
- prefix hit、preemption 和 chunked prefill；
- 与 native KV Cache 比较 logits、生成结果和任务精度；
- 分别评估 4-bit、K3V4 和 3-bit 的质量下降。

### 10.3 性能

- prefill store latency；
- decode P50/P90/P99；
- 1K、4K、8K、16K 及更长 sequence sweep；
- batch size sweep；
- 不同 `num_kv_splits`；
- KV Cache 容量和峰值临时显存；
- 与 native Ascend attention 和 vLLM CUDA TurboQuant 的趋势比较。

可以使用：

```bash
bash scripts/turboquant_triton/performance/profile_kernels.sh --operation all
```

输出的 `benchmark.json` 会记录参数、软件版本、延迟分位数、吞吐、显存变化、native decode
基线和 TurboQuant speedup，profiler 目录用于定位 Hadamard GEMM、store kernel、decode
Stage 1/2 和 dequant kernel 的耗时。

## 11. 建议的实现优先级

### 11.1 第一阶段：把已有路径变成社区可合并能力

1. 在 910B4 执行现有三个 MSE preset 的 store/dequant/decode correctness 矩阵；
2. 完成 Qwen3-32B eager 与 uniform single/multi-token ACLGraph 的精度、稳定性和性能基线；
3. 执行 Prefix Cache、preemption、mixed native/TQ cache group E2E；跨 page 算子 case 已补；
4. 执行现有 FP16/BF16、MHA/GQA 和 `head_dim=64/128/256` case；
5. 认证至少一个 Qwen3 之外的同类 dense RoPE 模型。

### 11.2 第二阶段：扩展单机推理能力

1. 对已实现的 small-chunk packed decode 与 large-chunk history-only dequant 做性能门禁；
2. 对已实现的 mixed decode/prefill batch 做完整模型回归；
3. 使用 `profile_matrix.sh` 建立 910B4 split 数据，再决定是否实现 FWT/fused store；
4. 运行已有 1/2/4/8-token spec case，并完成拒绝 token 的完整模型回归；
5. 评估 multi-token ACLGraph 的图大小、capture 时间和 TPOT 收益；
6. 验证已实现的 ALiBi/logits soft cap；batch invariant 继续等待 FWT 和 bitwise 门禁。

### 11.3 第三阶段：跨模块能力

1. 为一个 opaque-page Ascend connector 增加 TurboQuant layout handshake；
2. 设计 DCP local packed attention + global LSE/output merge；
3. 接入 PCP prefill 和 chunked context；
4. 评估 TQ-aware sliding-window spec 是否应贡献到 vLLM core；
5. 最后再评估 V2 runner、软件 FP8 和 310P fallback。

不建议把 MLA/DSA/SFA 集成、通用 connector、CP + spec + ACLGraph 的组合，或 310P 支持放进
首个社区 PR。这些能力会显著扩大审查面，也无法用单机 910B4 测试证明正确。
