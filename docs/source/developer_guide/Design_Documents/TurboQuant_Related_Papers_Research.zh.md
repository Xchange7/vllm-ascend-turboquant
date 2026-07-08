# TurboQuant 相关论文调研：HIGGS 与 AQUA-KV

本文面向 vLLM Ascend 的 TurboQuant / KV cache 压缩实现，调研以下两篇论文：

- `docs/2411.17525v1.pdf`
  Pushing the Limits of Large Language Model Quantization via the Linearity
  Theorem
- `docs/2501.19392v4.pdf`
  Cache Me If You Must: Adaptive Key-Value Quantization for Large Language
  Models

## 总体结论

这两篇论文对应两个层次：

1. `2411.17525v1` 主要是权重量化论文，提出 HIGGS：Hadamard incoherence +
   Gaussian MSE-optimal grids。它不是 KV cache 系统本身，但它的量化器可以作
   为 TurboQuant / AQUA-KV 中 key、value、residual 的 backbone quantizer。
2. `2501.19392v4` 是 KV cache 压缩论文，提出 AQUA-KV：用相邻层 KV 的可预测
   性训练 compact predictors，只存 residual 的量化结果。它更像 TurboQuant 的
   下一阶段能力：不仅量化 KV，还利用跨层和 K/V 之间的冗余。

对 vLLM Ascend 来说，合理路线不是直接把两篇论文混成一个大 patch，而是分三
层做：

1. 先稳定 upstream-compatible TurboQuant packed cache layout。
2. 再把 HIGGS-style quantizer 做成可替换的 NPU store/dequant/decode kernel。
3. 最后引入 AQUA-KV 的 predictor + residual cache，把普通 KV quantization 扩
   展成 adaptive KV compression。

## HIGGS 对 vLLM Ascend 的意义

HIGGS 的核心是：

- 对待量化向量先做 Hadamard / randomized Hadamard transform，让分布更接近高
  斯。
- 用 MSE-optimal grid / centroids 做低 bit vector quantization。
- 存储 packed indices 和 scale/norm。
- 解码时可通过 lookup table + inverse transform 恢复，或者在 kernel 内融合
  dequant 和计算。

论文主目标是 model weight quantization，但有三点可以直接迁移到 KV cache：

- `key` 更适合使用 Hadamard + centroid / vector quantization，因为 attention
  score 对 key 误差敏感。
- `value` 可以使用更简单的 per-token uniform quantization；如果要更低 bit，
  也可以切换到 HIGGS grid。
- packed bit layout、centroid lookup、scale/norm metadata 是硬件 kernel 的主
  要接口。

当前 `vllm_ascend/kv_cache/turboquant.py` 的 reference 实现已经按这个方向做
了初步建模，但还不是合格的生产实现：

- Hadamard matrix 和 centroids 在 Python 热路径构造，不能用于性能评估。
- 3-bit pack/unpack 仍有 Python loop，不适合 NPU graph 或高吞吐。
- decode 会 materialize 全量 BF16/FP16 KV cache，抵消显存收益。

因此 HIGGS 在 vLLM Ascend 中应落为低层 quantizer kernel，而不是放在 Python
attention backend 中长期执行。

## AQUA-KV 对 vLLM Ascend 的意义

AQUA-KV 的核心思想是：

- 第 `i` 层的 key 可以由第 `i-1` 层重构 key 预测。
- 第 `i` 层的 value 可以由第 `i-1` 层重构 value 加当前层重构 key 预测。
- cache 中不直接存完整 KV，而是存 `KV - predictor(input)` 的 residual 量化。
- 推理时按层顺序 decode：先恢复上一层 KV，再预测当前层 KV，再加 residual。

这对 vLLM Ascend 的影响比 HIGGS 大得多，因为它改变了 attention cache 的语义：

- cache 不再只是每层独立的 packed key/value。
- 当前层 decode 依赖上一层的 reconstructed KV。
- backend 需要 layer-order aware 的 decode 流程。
- predictor 权重需要作为模型附加参数加载、切分、搬运和可能量化。

AQUA-KV 论文还给出几个实现约束：

- 第一层没有上一层 KV，通常单独量化或保留高精度。
- 前几个 attention sink token 建议保留未压缩。
- 最近 token buffer 建议保留未压缩，论文实验中使用 up to 128 tokens。
- predictor / quantizer 可在 pre-RoPE 或 post-RoPE 位置工作，论文倾向
  pre-RoPE predictor。
- residual 可使用 HIGGS 或 uniform quantizer；AQUA-KV 本身和具体 quantizer
  解耦。

## 与当前 vLLM Ascend 架构的映射

### 已有接入点

当前 TurboQuant 初版已经触达这些位置：

- `vllm_ascend/platform.py`
  根据 `--kv-cache-dtype turboquant_*` 路由到 Ascend TurboQuant backend。
- `vllm_ascend/worker/model_runner_v1.py`
  对 `TQFullAttentionSpec` 使用单 packed tensor 分配和 reshape。
- `vllm_ascend/attention/turboquant.py`
  提供 Ascend TurboQuant attention backend。
- `vllm_ascend/kv_cache/turboquant.py`
  提供 dtype/config helper 和 reference packed store/dequant。

这些接入点适合承载 HIGGS-style TurboQuant，但还不足以承载完整 AQUA-KV。

### HIGGS / TurboQuant 模块拆分

当前已经把 HIGGS reference quantizer 从 TurboQuant packed cache 编排里拆出：

```text
vllm_ascend/kv_cache/turboquant.py
  - dtype / preset / layout / shape / validation
  - packed cache store/dequant orchestration

vllm_ascend/kv_cache/higgs.py
  - normalized Hadamard transform
  - cached Lloyd-Max centroid solving
  - HIGGS vector quant/dequant reference path

vllm_ascend/ops/turboquant.py
  - Python custom-op wrapper（后续新增）
  - torch.ops._C_ascend dispatch（后续新增）

vllm_ascend/attention/turboquant.py
  - backend selection
  - metadata handling
  - call reference store/dequant now, call NPU kernels later
```

这样 HIGGS 既可以被 TurboQuant key 量化复用，也可以作为后续 AQUA-KV residual
quantizer 或独立 HIGGS NPU kernel 的算法边界。后续替换 CANN / AscendC kernel
时，不需要重写 platform 和 runner。

### AQUA-KV 建议新增模块

AQUA-KV 需要独立配置和模型附加参数管理：

```text
vllm_ascend/kv_cache/aquakv.py
  - AQUA-KV config
  - predictor metadata
  - residual cache layout
  - sink/recent-buffer policy

vllm_ascend/attention/aquakv.py
  - layer-order aware decode
  - current layer depends on previous reconstructed KV

vllm_ascend/model_loader/aquakv_loader.py
  - load predictor weights
  - tensor-parallel shard predictor weights
  - optional predictor quantization

tools/aquakv_calibrate.py
  - offline one-shot calibration
  - export predictor checkpoint
```

不要把 AQUA-KV 写成普通 `--quantization` 权重量化方法。更自然的用户入口是 KV
cache dtype / KV compression config，例如：

```bash
--kv-cache-dtype aquakv_higgs_2bit
--kv-cache-extra-config '{"recent_window":128,"sink_tokens":4}'
```

具体参数名需要和 upstream vLLM 的 cache config 兼容后再定。

## 推荐实现路线

### 阶段 0：稳定当前 TurboQuant smoke 路径

目标：先让 Qwen3 dense V1 runner 的最小 smoke 跑通。

必须补齐：

- 显式拒绝 V2 runner，或补 `vllm_ascend/worker/v2/attn_utils.py` packed cache。
- `slot_mapping` 过滤负数 padding slot，避免写坏最后一个 block。
- 不要静默 fallback 到非 upstream centroids。
- 明确禁用 ACLGraph 性能路径，直到 kernel 稳定。

### 阶段 1：实现 HIGGS-style NPU quantizer kernel

目标：把 Python reference store/dequant 换成 NPU kernel。

建议 kernel 分成三类：

1. `turboquant_store`
   输入当前 token 的 K/V、slot mapping、centroid/grid 参数，写 packed cache。
2. `turboquant_dequant_blocks`
   给定 block table 和 seq lens，只恢复当前请求需要的 block，而不是全 cache。
3. `turboquant_decode_attention`
   最终目标是 fused decode：直接从 packed cache 算 attention，不 materialize full
   KV。

阶段 1 可以先做 `store + block-table dequant + existing FIA`，阶段 2 再融合
attention。

### 阶段 2：block-table aware decode

目标：避免当前 reference 版本的全 cache dequant OOM 问题。

需要让 dequant 只处理：

- 当前 batch 的 `block_tables`
- 每个请求的有效 `seq_lens`
- 当前 attention layer 的 packed cache

输出可以是临时 K/V workspace，然后接 `npu_fused_infer_attention_score`。这个阶
段仍不是最终最优性能，但已经能保持实际显存收益。

### 阶段 3：AQUA-KV calibration 和 predictor runtime

目标：支持论文 `2501.19392v4` 的 adaptive residual cache。

需要做两条链路：

1. 离线 calibration：
   - 读取模型和 calibration data。
   - 按层导出 K/V。
   - 用上一层 reconstructed KV 训练当前层 predictor。
   - 保存 predictor checkpoint 和 compression config。
2. 在线 inference：
   - 第一层按普通 quantizer 处理。
   - 第 `i` 层 decode 前恢复第 `i-1` 层 KV。
   - 用 predictor 预测当前层 KV。
   - 从 residual cache dequant residual。
   - `reconstructed = prediction + residual` 后进入 attention。

这会改变 attention backend 的执行顺序。最保守做法是先在 `--enforce-eager`
下实现，不急于接 ACLGraph。

### 阶段 4：策略层能力

后续可以加入：

- attention sink token 不压缩。
- recent token buffer 不压缩。
- layer-wise bitwidth / skip layer。
- 与 pruning / H2O 类 token eviction 组合。
- predictor 权重量化。
- V2 runner、KV transfer、prefix caching、spec decode 支持。

## 对 NPU 的实现关注点

### 1. 不要在热路径做 Python loop

HIGGS / TurboQuant 的 3-bit pack/unpack、Hadamard、centroid lookup 必须进
kernel。Python 版只能做单元测试和语义对齐。

### 2. 不要全 cache dequant

decode 必须以 block table 为边界，只恢复当前请求实际访问的 blocks。否则长上
下文下会把压缩 cache 又膨胀成完整 BF16/FP16 cache。

### 3. Hadamard 不应显式构造大矩阵

应使用 FWHT / butterfly 形式，或者把 Hadamard 融进 kernel。显式矩阵乘法只适
合作为 correctness reference。

### 4. Predictor 需要 TP-aware sharding

AQUA-KV predictor 作用在 `num_kv_heads * head_dim` 维度。对于 GQA 模型，这个
维度小于 query hidden size，但仍需要按 tensor parallel 的 KV head 切分方式对
齐。

### 5. RoPE 位置必须提前决定

AQUA-KV 倾向 pre-RoPE predictor。vLLM Ascend 当前 attention 层拿到的 K 通常
已经经过 rotary embedding。若要 pre-RoPE，需要在模型 attention layer 或 rope
custom op 附近增加 hook，不能只在 backend 末端处理。

## 验证建议

### 功能 smoke

- 模型：Qwen3 dense，小上下文。
- 参数：`VLLM_USE_V2_MODEL_RUNNER=0`、`--enforce-eager`、`max_model_len=2048`、
  `max_num_seqs=1`。
- 先 dummy，再 real weight。
- 验证 `/v1/models` 和一次 chat completion。

### 正确性对比

- 比较 BF16 KV、TurboQuant/HIGGS KV、AQUA-KV residual KV 的短输出一致性。
- 用 WikiText-2 小样本做 PPL smoke。
- 对 LongBench 子集做长上下文回归。

### 性能和显存

- 阶段 0 Python reference 不做性能结论。
- 阶段 1 看 packed cache allocation 是否降低常驻 KV 显存。
- 阶段 2 开始看 decode 峰值显存和 tokens/s。
- 阶段 3 评估 predictor overhead。

## 建议优先级

1. 先把当前 TurboQuant backend 改成不会误用的安全 smoke 版本。
2. 做 HIGGS/TurboQuant NPU store kernel。
3. 做 block-table aware dequant，消除全 cache dequant。
4. 再做 fused packed-cache attention。
5. 最后做 AQUA-KV predictor calibration/runtime。

HIGGS 是低层 quantizer 能力，AQUA-KV 是更高层的 cache compression 策略。二者
可以组合，但不应在第一版里耦合死。
