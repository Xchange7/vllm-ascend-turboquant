# TurboQuant 910B4 正确性、精度与性能验证手册

## 1. 验证目标

本文给出 TurboQuant 当前前三个优先级的 NPU 验证流程：

1. 确认 Python 接口、KV Cache、Triton kernel、ACLGraph 和模型服务可以正确运行；
2. 比较 native KV Cache 与 TurboQuant KV Cache 的确定性输出和 logprob 偏差；
3. 比较 packed TurboQuant decode 与原生 NPU paged attention 的延迟。

测试脚本不会修改模型权重。TurboQuant 只压缩推理期间动态生成的 KV Cache。

## 2. 测试入口

| 脚本 | 用途 |
| --- | --- |
| `run_npu_validation.sh` | 按正确性、精度、性能顺序执行全部测试 |
| `run_diagnostic_suite.sh` | 环境、backend、KV Cache、kernel 和 ACLGraph 单测 |
| `run_910b4_retest.sh` | store 对齐修复后的隔离复测和 Qwen3-0.6B 模型 smoke test |
| `run_model_startup_debug.sh` | 单独启动 TurboQuant 服务并周期采集进程和 NPU 状态 |
| `run_accuracy_comparison.sh` | 顺序启动两组服务并比较输出和 logprobs |
| `run_quality_comparison.sh` | 独立判分 native/TurboQuant，并检查幻觉和质量回归 |
| `run_serving_benchmark.sh` | 关闭 prefix cache，对比 KV 容量、TTFT 和 TPOT |
| `run_performance_validation.sh` | 运行 batch/context/split 性能矩阵 |
| `accuracy_eval.py` | 采集 OpenAI completion 响应并生成比较报告 |
| `serving_benchmark.py` | 采集流式请求计时并生成 native/TurboQuant 对比报告 |
| `summarize_profiles.py` | 汇总所有 `benchmark.json` |

所有脚本都位于 `scripts/turboquant_triton/`。

## 3. 环境准备

### 3.1 软件和代码

在 910B4 服务器上确认：

- 已安装匹配 CANN 的 `torch-npu` 和 Triton-Ascend；
- vLLM core 是与该分支匹配的 `0.20.2`/`0.20.2+empty`；
- 当前 vLLM Ascend checkout 已安装为 editable package；
- `which vllm` 和 `python3 -c 'import vllm; print(vllm.__file__)'` 指向预期环境；
- Qwen3-0.6B 或 Qwen3-32B 权重路径可读。

推荐执行：

```bash
pip install -e .
python3 scripts/turboquant_triton/check_environment.py
```

### 3.2 NPU 和模型参数

以下示例按 TP=2 编写：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MODEL=/path/to/Qwen3-32B
export TP_SIZE=2
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=4
```

Qwen3-32B BF16 权重通常不适合在单张 64 GiB NPU 上完成有意义的 KV Cache 测试。

首轮建议在单张 910B4 上使用 Qwen3-0.6B：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL=/run/test_llm/Qwen3-0.6B-hf \
TP_SIZE=1 \
bash scripts/turboquant_triton/run_910b4_retest.sh
```

该脚本默认使用 `MAX_MODEL_LEN=2048`、`MAX_NUM_SEQS=2` 和
`turboquant_4bit_nc`，服务端口默认为 `18000`。它会让四种 store 对齐 case 分别运行在
独立 pytest 进程，再运行
完整 kernel、ACLGraph 和 native-vs-TurboQuant eager 模型对照，日志最终打包到
`logs/turboquant/910b4_retest_<timestamp>.tar.gz`。

`DEBUG_SYNC=1` 默认只用于 direct kernel 阶段。模型阶段会移除
`ASCEND_LAUNCH_BLOCKING`，避免同步执行显著拖慢服务启动和逐 token 推理；只有定位模型级
异步异常时才设置 `MODEL_DEBUG_SYNC=1`。等待服务期间脚本每 30 秒打印一次最新日志行。

如果服务进程一直存活但 NPU 显存没有变化，使用 `run_model_startup_debug.sh` 绕过 native
精度基线。该脚本先在 60 秒超时内离线读取模型 config，再只启动
`turboquant_4bit_nc` eager 服务；等待期间周期打印主进程状态、子进程树、server log 尾部和
`npu-smi info`。由此可以区分模型文件 I/O、API frontend、EngineCore spawn 和 NPU worker
初始化阶段的阻塞。

测试脚本默认把 `GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME` 和
`HCCL_SOCKET_IFNAME` 统一设置为 `eth0`。vLLM-Ascend 即使在 world size 为 1 时也会创建
多个 Gloo CPU group，显式网卡可避免每个 group 重复进行错误的接口选择。诊断脚本会在
启动服务前执行单 rank Gloo probe，30 秒内不能完成则直接失败。可通过
`NETWORK_IFNAME=<interface>` 为其他服务器选择网卡。

### 3.3 端口和磁盘

默认服务端口是 8000，确保没有其他进程占用。测试会在
`logs/turboquant/npu_validation_<timestamp>/` 保存服务日志、JSON、TSV、Markdown 和
压缩包，应预留足够磁盘空间。

## 4. 一条命令运行全部测试

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=2 \
bash scripts/turboquant_triton/run_npu_validation.sh
```

默认执行：

1. Python/backend/kernel/ACLGraph correctness；
2. TurboQuant eager 与 TurboQuant ACLGraph 的模型级对照；
3. native eager 与 TurboQuant eager 的精度对照；
4. 24 组默认 decode 性能 case；
5. 一组 store/decode/dequant component profile；
6. 汇总并打包全部日志。

总脚本即使某个阶段失败，也会继续执行其他阶段。最终退出码非零表示至少一个阶段失败。

## 5. 第一优先级：正确性验证

### 5.1 独立运行 kernel 和 backend 测试

```bash
RUN_PROFILE=0 \
bash scripts/turboquant_triton/run_diagnostic_suite.sh
```

阶段含义如下：

| 阶段 | 验证内容 |
| --- | --- |
| `00_system` | NPU、软件包、环境变量 |
| `01_source` | branch、commit、dirty files |
| `02_environment` | vLLM core API、workspace、page-size contract |
| `03_platform` | backend 选择和 graph capability |
| `04_backend` | cache shape、mixed batch、continuation 和 feature fallback |
| `05_triton_kernels` | store、dequant、decode 与 reference |
| `06_aclgraph` | 动态 metadata capture/replay |

正确性通过标准：

- 所有阶段为 `PASS`；
- 不出现 NPU illegal memory access、Triton compilation error 或 NaN；
- 负 `slot_mapping` 不改变 cache；
- FP16/BF16 和配置的 head dimension 都与 reference 在测试 tolerance 内一致；
- graph replay 更新 Q/K/V、block table、slot mapping 和 `seq_lens` 后仍与 eager 一致。

### 5.2 模型级 eager 与 ACLGraph 对照

总入口会复用 `run_accuracy_comparison.sh`，依次启动：

1. `turboquant_4bit_nc` eager server；
2. `turboquant_4bit_nc` ACLGraph server。

两组服务使用完全相同的 deterministic prompts。每个 prompt 会先发送一次 warm request，
再记录第二次请求，以同时覆盖重复 prefix 的 cache reuse。

单独执行该测试：

```bash
MODEL=/path/to/Qwen3-32B \
BASE_LABEL=turboquant_eager \
BASE_CACHE_DTYPE=turboquant_4bit_nc \
BASE_ENFORCE_EAGER=1 \
TEST_LABEL=turboquant_aclgraph \
TEST_CACHE_DTYPE=turboquant_4bit_nc \
TEST_ENFORCE_EAGER=0 \
OUTPUT_DIR=logs/turboquant/graph_e2e \
bash scripts/turboquant_triton/run_accuracy_comparison.sh
```

重点检查两个 server log 中 backend、graph capture 和 replay 信息，确认请求不是全部回退到
普通 backend 或 eager 路径。

## 6. 第二优先级：精度验证

### 6.1 默认比较方式

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=2 \
OUTPUT_DIR=logs/turboquant/accuracy_manual \
bash scripts/turboquant_triton/run_accuracy_comparison.sh
```

默认顺序启动 native eager 和 `turboquant_4bit_nc` eager server。两个服务不会同时占用
NPU；脚本会记录 PID，并在成功、失败或中断时停止本次启动的服务。

测试 prompt 位于 `accuracy_prompts.jsonl`，覆盖：

- 英文和中文生成；
- 算术与代码；
- 重复 prefix；
- 较长上下文中的末尾信息检索。

长上下文用例默认按 `MAX_MODEL_LEN=2048` 留出生成空间。采集器会使用模型
tokenizer 在发送请求前检查 `prompt_tokens + max_tokens`；任一用例超限时会在
发出 completion 请求前列出用例 ID 和 token 预算，避免把测试配置错误误判为
TurboQuant 运行错误。实际 token 数和总预算也会记录在 `base.json` 和
`test.json` 中。

每个请求使用：

- `temperature=0`；
- 固定 seed；
- completion logprobs；
- 相同 prompt 和 `max_tokens`。

### 6.2 精度输出

主要文件包括：

| 文件 | 内容 |
| --- | --- |
| `base.json` | native 请求和完整响应 |
| `test.json` | TurboQuant 请求和完整响应 |
| `origin_answers.txt` | baseline/native 各题 prompt、期望值和未经规范化的原始输出 |
| `turboquant_answers.txt` | TurboQuant 各题 prompt、期望值和未经规范化的原始输出 |
| `comparison.json` | 每个 case 的文本、token prefix 和 logprob 差异 |
| `summary.md` | 可直接阅读的汇总 |
| `native_server.log` | native server 启动与运行日志 |
| `turboquant_server.log` | TurboQuant server 启动与运行日志 |

### 6.3 指标解释

- `exact_text_match_rate`：完整生成文本相同的 case 比例；
- `token_prefix_rate`：两种 cache 输出保持相同 token prefix 的比例；
- `mean_logprob_diff`：共同 prefix token 的平均绝对 logprob 差；
- `max_logprob_diff`：共同 prefix token 观察到的最大绝对 logprob 差。

量化可能改变接近决策边界的 token，因此文本不完全一致不自动等于 kernel 错误。应先确认
kernel reference 测试通过，再结合任务准确率和长上下文数据判断量化精度。

当前默认只把请求失败作为硬失败，不预设未经 910B4 数据验证的精度阈值。获得第一轮基线后
可以启用门限：

```bash
MIN_EXACT_MATCH_RATE=0.80 \
MIN_TOKEN_PREFIX_RATE=0.90 \
MAX_MEAN_LOGPROB_DIFF=0.30 \
bash scripts/turboquant_triton/run_accuracy_comparison.sh
```

这些数值只是命令示例，不是当前项目承诺的验收标准。

### 6.4 自定义 prompt

JSONL 每行格式为：

```json
{"id":"case_name","prompt":"input text","max_tokens":64}
```

长重复上下文可以使用：

```json
{"id":"long_case","prompt":"repeated text ","repeat":100,"suffix":"final question","max_tokens":32}
```

运行时设置：

```bash
PROMPTS=/path/to/custom_prompts.jsonl \
bash scripts/turboquant_triton/run_accuracy_comparison.sh
```

### 6.5 客观质量与抗幻觉测试

仅比较两种输出是否相同不能判断模型是否乱说，因为 native 也可能回答错误。
`run_quality_comparison.sh` 使用带标准答案的 chat 用例分别判分，然后统计量化回归：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=4 \
bash scripts/turboquant_triton/run_quality_comparison.sh
```

默认使用 `chat_template_kwargs={"enable_thinking": false}` 和贪心解码，覆盖：

- 稳定事实、中文事实、算术和逻辑推理；
- 直接检索、多跳检索、表格检索和长上下文末尾信息；
- 资料缺失时输出 `INSUFFICIENT_INFORMATION`，用于发现无依据编造；
- 过期值干扰、上下文 prompt injection、严格复制和 JSON 输出。

报告中的核心指标是：

- `native_accuracy`：baseline 相对标准答案的正确率；
- `turboquant_accuracy`：TurboQuant 相对标准答案的正确率；
- `quality_regressions`：native 正确、TurboQuant 错误的 case 数；
- `both_wrong`：两者都错，通常属于模型能力或测试提示问题；
- `category_accuracy`：按事实、算术、检索和抗幻觉等类别拆分的正确率。

默认 `MAX_QUALITY_REGRESSIONS=0`，任何量化回归都会让脚本返回非零，但仍会保存
完整报告和两边原始回答。可在审核首轮结果后设置
`MIN_TURBOQUANT_ACCURACY` 和新的回归门限。该固定题集用于发现明显回归，不能替代
业务数据集、长文本事实一致性评测或人工审核。

### 6.6 服务性能与 KV Cache 压缩率

使用相同模型配置顺序启动 native 和 TurboQuant 服务，并明确关闭 prefix cache：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL=/run/test_llm/Qwen3-0.6B-hf \
TP_SIZE=1 \
bash scripts/turboquant_triton/run_serving_benchmark.sh
```

默认 workload 为 1024 个输入 token、128 个输出 token、2 次预热和 10 次正式
串行请求。每次请求都使用流式 completion 和 `ignore_eos=true`，以固定输出长度并
分别记录 TTFT、TPOT、端到端延迟和 output token/s。TTFT 是请求发出到收到首个
输出 token 的时间，TPOT 按
`(末 token 时间 - 首 token 时间) / (输出 token 数 - 1)` 计算；SSE 尾包只计入
端到端延迟，不计入 TPOT。

脚本给两个服务都传递 `--no-enable-prefix-caching`，因此重复 prompt 不会获得
prefix 命中。最终 `summary.md` 从两份 server log 的
`GPU/NPU KV cache size: ... tokens` 计算同等内存预算下的容量倍率、估算
bytes/token 比率和 KV 内存减少百分比。原始逐请求数据保存在 `native.json` 和
`turboquant.json`，服务日志和 `comparison.json` 会一并归档。

可用以下变量扩大重复次数或调整上下文，但必须满足
`INPUT_TOKENS + OUTPUT_TOKENS <= MAX_MODEL_LEN`：

```bash
INPUT_TOKENS=1536 OUTPUT_TOKENS=256 \
WARMUP_REQUESTS=3 MEASURE_REQUESTS=20 \
MAX_MODEL_LEN=2048 \
bash scripts/turboquant_triton/run_serving_benchmark.sh
```

Qwen3-32B 可使用 4 卡 TP，并把客户端和服务端并发同时设为 16。建议至少运行一轮
16 请求并发预热和四轮正式请求：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=4 CONCURRENCY=16 MAX_NUM_SEQS=16 \
WARMUP_REQUESTS=16 MEASURE_REQUESTS=64 \
MAX_MODEL_LEN=2048 INPUT_TOKENS=1024 OUTPUT_TOKENS=128 \
bash scripts/turboquant_triton/run_serving_benchmark.sh
```

`CONCURRENCY` 控制客户端同时在途的流式请求数；未显式设置 `MAX_NUM_SEQS` 时，
服务端会默认使用相同值。报告除逐请求 TTFT/TPOT 外，还会给出 request/s 和
aggregate output token/s。

## 7. 第三优先级：性能验证

### 7.1 默认矩阵

```bash
bash scripts/turboquant_triton/run_performance_validation.sh
```

默认遍历：

- cache dtype：`turboquant_4bit_nc`；
- activation dtype：FP16、BF16；
- batch：1、4；
- context：1024、4096、8192；
- KV splits：8、16；
- Qwen3-32B TP=2 单 rank heads：32 query heads、4 KV heads；
- head dimension：128。

每个 case 都在同一进程中测量 packed TurboQuant decode 和 native NPU paged attention。
`decode_speedup_vs_native > 1.0` 表示 TurboQuant decode 更快。

### 7.2 快速测试

第一次只确认脚本和算子可运行：

```bash
ACTIVATION_DTYPES=float16 \
BATCH_SIZES=1 \
SEQUENCE_LENGTHS=1024 \
NUM_KV_SPLITS_LIST=8 \
ITERATIONS=5 \
RUN_COMPONENT_PROFILE=0 \
bash scripts/turboquant_triton/run_performance_validation.sh
```

### 7.3 完整 preset 矩阵

```bash
CACHE_DTYPES='turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc' \
BATCH_SIZES='1 4 8' \
SEQUENCE_LENGTHS='1024 4096 8192 16384' \
NUM_KV_SPLITS_LIST='8 16 32' \
ITERATIONS=50 \
bash scripts/turboquant_triton/run_performance_validation.sh
```

性能测试时不要设置 `ASCEND_LAUNCH_BLOCKING=1`，否则计时没有代表性。

### 7.4 性能输出

| 文件 | 内容 |
| --- | --- |
| `cases/*/benchmark.json` | 每个 case 的原始计时、显存和软件版本 |
| `performance_summary.json` | 所有成功 case 的结构化汇总 |
| `performance_summary.csv` | 便于表格分析 |
| `performance_summary.md` | 可读的延迟和 speedup 表 |
| `component_profile/` | store、decode、dequant 和 native baseline |

不要只观察理论 cache compression ratio。当前实现是否加速必须以
`decode_speedup_vs_native` 和服务级吞吐数据为准。

## 8. 控制总测试范围

只运行一个或两个优先级：

```bash
RUN_CORRECTNESS=1 RUN_ACCURACY=0 RUN_PERFORMANCE=0 \
bash scripts/turboquant_triton/run_npu_validation.sh
```

```bash
RUN_CORRECTNESS=0 RUN_ACCURACY=1 RUN_PERFORMANCE=1 \
MODEL=/path/to/Qwen3-32B \
bash scripts/turboquant_triton/run_npu_validation.sh
```

常用变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MODEL` | 无 | 模型路径，精度测试必需 |
| `TP_SIZE` | 2 | tensor parallel size |
| `TQ_CACHE_DTYPE` | `turboquant_4bit_nc` | 模型服务使用的 TQ preset |
| `MAX_MODEL_LEN` | 4096 | 模型服务上下文上限 |
| `PORT` | 8000 | 顺序启动服务使用的端口 |
| `WARM_PREFIX` | 1 | 记录前重复请求以覆盖 prefix reuse |
| `CACHE_DTYPES` | `turboquant_4bit_nc` | 性能矩阵 preset 列表 |
| `SEQUENCE_LENGTHS` | `1024 4096 8192` | kernel profile context 列表 |
| `COLLECT_TRACE` | 0 | component profile 是否采集 profiler trace |

## 9. 失败后的处理

### 9.1 环境阶段失败

首先检查 `environment.log` 中的 vLLM 路径、版本、workspace 和
`TQFullAttentionSpec.page_size_bytes`。不要在 core 契约不匹配时继续分析 Triton 数值。

### 9.2 Kernel 测试失败

仅在定位异步报错时运行：

```bash
DEBUG_SYNC=1 RUN_PROFILE=0 \
bash scripts/turboquant_triton/run_diagnostic_suite.sh
```

性能测试前必须取消 `ASCEND_LAUNCH_BLOCKING`。

如果日志包含错误码 `507035` 和 `The UB address accessed by the VEC instruction is not
aligned`，这是 Ascend Vector Core 的 UB 对齐错误。按以下顺序处理：

1. 找到第一个使用非负 `slot_mapping` 的 `triton_turboquant_store()`；负 slot case 只会
   提前返回，不能证明写 cache 路径正常；
2. 在 store 后立即执行 `torch.npu.synchronize()`，不要依据随后任意一次 tensor 分配或
   `IsFinite` 抛出的异步异常判断故障算子；
3. 检查 Triton 中小于 32 字节的 vector operand、长度很短的 shift/reduction vector，
   以及 3-bit byte packing 的内部 lane 数；
4. store 单测通过并重启 pytest 进程后，再分别运行 dequant、packed decode 和 ACLGraph。

NPU stream 在 device exception 后可能处于错误状态。同一进程里的后续失败通常只是连带
结果，不能当作独立 bug；修改 kernel 后应启动新进程复测。

### 9.3 服务启动失败

查看对应 `*_server.log` 的最后 100 行。常见原因包括：

- vLLM core 不是目标版本；
- NPU 数量与 `TP_SIZE` 不一致；
- 权重加 KV Cache 超出显存；
- 8000 端口已占用；
- graph capture shape 超出当前配置。

### 9.4 精度明显异常

按以下顺序缩小范围：

1. 确认 direct kernel reference 全部通过；
2. 比较 TurboQuant eager 与 ACLGraph；
3. 将 context 和 `max_tokens` 降低；
4. 分别测试三种 preset；
5. 检查首尾 boundary layer 是否仍使用 native cache；
6. 查看第一次不同 token 前的 logprob 差异。

### 9.5 性能低于 native

查看不同 split 数是否一致偏慢。如果短 context 偏慢而长 context 改善，通常表示固定开销
较高；如果所有 context 都偏慢，应重点 profile Hadamard rotation、key unpack、centroid
gather、value unpack 和 Stage 2 reduction。

## 10. 需要提供的日志

完成测试后，优先提供总入口打印的：

```text
logs/turboquant/npu_validation_<timestamp>.tar.gz
```

该压缩包包含：

- source revision 和环境信息；
- pytest 与 Triton 错误栈；
- eager、ACLGraph、native 和 TurboQuant server log；
- 原始请求响应及精度比较；
- 每个性能 case 的 `benchmark.json`；
- 汇总后的 Markdown、CSV 和 JSON。

在日志中保留模型路径和软件版本，但提交到公开 issue 前应检查是否包含内部路径或其他敏感信息。
