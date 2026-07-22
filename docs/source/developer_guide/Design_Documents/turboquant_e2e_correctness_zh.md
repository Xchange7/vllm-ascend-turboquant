# TurboQuant 端到端正确性测试

## 1. 测试目标

算子测试中 `AscendC == Triton` 只能说明两个实现使用了相同的 Cache 布局。如果 Triton 的旋转、
量化或解包逻辑本身存在错误，两个实现仍可能一起通过。因此端到端测试不使用 Triton 作为正确答案，
而是把相同权重、相同输入和相同采样参数下的原生 KV Cache 服务作为行为基线。

该测试回答以下问题：

1. 启用 TurboQuant 后，固定目标 token 的概率是否发生不可接受的漂移；
2. 首 token 的 top-k 概率分布和 top-1 决策是否稳定；
3. 确定性生成是否出现明显分叉；
4. 原生模型能够答对的问题，TurboQuant 是否答错；
5. reference Triton、生产 Triton 和 AscendC 融合路径之间从哪里开始产生差异。

端到端测试能够发现“模型已经不能正常推理”一类问题，但它不能单独证明每个 packed byte 都符合论文。
bit packing、Lloyd-Max 分桶和元数据布局仍需独立 PyTorch/CPU oracle 覆盖。

## 2. 五种运行模式

脚本依次启动以下服务，所有服务加载同一模型权重：

| 模式 | KV Cache | Decode 路径 | 目的 |
| --- | --- | --- | --- |
| `native` | 原生 BF16/FP16 | CANN 原生 attention | 行为基线 |
| `native_repeat` | 原生 BF16/FP16 | 与 `native` 相同 | 测量运行时和调度噪声底线 |
| `tq_reference` | TurboQuant | FP32 旋转和 reference Triton | 隔离基础 TurboQuant/Triton 算法 |
| `tq_auto` | TurboQuant | 算子可用时关闭图捕获，单 token 优先 AscendC+CANN FIA；不支持的 shape 回退 grouped-GQA | 验证生产自动分发 |
| `tq_ascend_fused` | TurboQuant | 强制 AscendC 反量化和 CANN FIA | 验证融合路径和算子注册 |

重点看三类比较：

- `native -> native_repeat`：没有量化时的正常运行波动；
- `native -> tq_reference`：基础 TurboQuant/Triton 相对原生模型的漂移；
- `tq_reference -> tq_auto`：生产自动路径相对 reference 的整体漂移；
- `tq_auto -> tq_ascend_fused`：eager 单 token 下自动分发与显式融合路径是否一致。

如果 `native -> tq_reference` 已经严重失败，应该先检查 Triton 量化和 decode，不应继续归因于融合算子。

## 3. Teacher forcing

自由生成一旦在第一个 token 分叉，后续上下文也会不同，无法继续进行逐位置比较。Teacher forcing 使用完全
相同的长文本作为输入，并通过 vLLM 的 `prompt_logprobs` 返回每个实际目标 token 的 logprob。

测试只统计每个样例末尾的固定 token 窗口。短、中、长上下文均有覆盖，长上下文用于放大 KV Cache
读写错误和量化误差积累。

脚本默认设置 `MAX_NUM_BATCHED_TOKENS=512`，强制长输入经过 chunked/continuation prefill。
第一个 chunk 之后的计算需要读取已经写入的量化 KV Cache。若整个 prompt 在一个 prefill 中完成，
prefill attention 可能直接使用当前 K/V，prompt logprob 会低估 TurboQuant 路径的误差。

主要指标：

| 指标 | 含义 |
| --- | --- |
| Mean/P95 absolute logprob difference | 同一目标 token 的概率漂移 |
| Native/TurboQuant mean NLL | 固定目标序列的平均负对数似然 |
| Mean NLL delta | TurboQuant NLL 减去原生 NLL，正值表示退化 |
| Perplexity ratio | `exp(NLL delta)`，大于 1 表示困惑度上升 |
| Prompt top-1 match | 每个 teacher-forced 位置的 top-1 token 一致率 |

Teacher forcing 不依赖 TurboQuant 的反量化结果作为参考，因此不是 Triton 与 AscendC 之间的循环比较。

这里不把 MSE 作为唯一指标。对 KV tensor 做 MSE 适合算子级量化误差分析，但无法直接回答模型决策是否
改变。OpenAI 服务接口也不会返回完整 raw logits。目标 token NLL、top-1 决策和 top-k logprob
更接近最终推理行为。由于 top-k 是截断分布，测试不会把缺失概率人为补零后声称得到了精确 KL/JSD。

## 4. 首 token 与完整生成

每个请求还会生成 token，并返回首 token 的 top-20 分布：

- `First-token top-1 match` 判断最终决策是否一致；
- `Top-k overlap` 使用两个 top-k token 集合的 Jaccard 重合率；
- common-token logprob difference 比较两个集合中共同 token 的 logprob。

质量用例包含事实、算术、推理、上下文检索、长上下文干扰、拒绝幻觉、指令遵循和 JSON 输出。
原生结果正确而 TurboQuant 结果错误会记为 `quality_regression`。

## 5. 运行方法

Qwen3-0.6B 单卡首轮验证：

```bash
cd /opt/x50060420/Quant/vllm-ascend-turboquant
source /usr/local/Ascend/ascend-toolkit/set_env.sh

MODEL=/run/test_llm/Qwen3-0.6B-hf \
DEVICE_IDS=0 \
TP_SIZE=1 \
MAX_MODEL_LEN=16384 \
MAX_NUM_BATCHED_TOKENS=512 \
bash scripts/turboquant_triton/correctness/run_e2e_correctness.sh
```

Qwen3-32B 四卡验证：

```bash
MODEL=/run/test_llm/Qwen3-32B \
DEVICE_IDS=0,1,2,3 \
TP_SIZE=4 \
MAX_MODEL_LEN=16384 \
GPU_MEMORY_UTILIZATION=0.85 \
bash scripts/turboquant_triton/correctness/run_e2e_correctness.sh
```

只运行基础 Triton 路径以缩短首次定位时间：

```bash
MODES="native native_repeat tq_reference" \
bash scripts/turboquant_triton/correctness/run_e2e_correctness.sh
```

只比较生产 Triton，不要求已编译 AscendC 算子：

```bash
MODES="native tq_reference tq_auto" \
bash scripts/turboquant_triton/correctness/run_e2e_correctness.sh
```

## 6. 默认阈值

当前默认阈值是用于发现严重实现错误的 screening gate，并不是论文质量结论：

| 环境变量 | 默认值 |
| --- | ---: |
| `MAX_PROMPT_MEAN_ABS_LOGPROB_DIFF` | 0.30 |
| `MAX_PROMPT_P95_ABS_LOGPROB_DIFF` | 0.75 |
| `MAX_PROMPT_ABS_MEAN_NLL_DELTA` | 0.10 |
| `MIN_FIRST_TOKEN_TOP1_MATCH_RATE` | 0.85 |
| `MIN_FIRST_TOKEN_TOPK_OVERLAP` | 0.50 |
| `MAX_QUALITY_REGRESSIONS` | 0 |

应先在 Qwen3-0.6B 和 Qwen3-32B 的 `turboquant_4bit_nc` 上获得稳定结果，再为 3-bit 配置单独校准阈值。
不应通过放宽阈值掩盖单个样例中数量级异常的 logprob 漂移。

## 7. 输出文件

结果目录为 `logs/turboquant/e2e_correctness_<timestamp>/`：

```text
runs/<mode>/server.log
runs/<mode>/teacher_forcing.json
runs/<mode>/teacher_forcing_answers.txt
runs/<mode>/quality.json
runs/<mode>/quality_answers.txt
comparisons/<base>_vs_<candidate>/teacher_forcing.json
comparisons/<base>_vs_<candidate>/teacher_forcing.md
comparisons/<base>_vs_<candidate>/quality.json
comparisons/<base>_vs_<candidate>/quality.md
summary.json
summary.md
```

调试时应同时提供整个 `.tar.gz`。仅查看 `summary.md` 无法定位首次发生漂移的 token 位置，原始 JSON
保留了请求、完整响应、prompt token IDs、每个目标 token 的 logprob 和 rank。
