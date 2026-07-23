# TurboQuant NPU 迁移与调试技术报告

## 1. 报告目的

本文面向将当前 TurboQuant 实现迁移到 Ascend 910B4 服务器并进行现场调试的开发者，目标是：

- 固化可复现的软件、源码和算子二进制环境；
- 按层定位编译、安装、KV Cache、数值正确性、端到端生成和性能问题；
- 避免把 vLLM core 不匹配、CANN OPP 残留或分布式初始化问题误判为 TurboQuant 算法问题；
- 为 NPU 机器上的日志采集、问题复现和结果回传提供统一流程；
- 明确当前实现的能力边界，不把尚未验证的功能作为可用能力。

本文基于 `turboquant-triton-v0.20.2rc` 分支的当前实现编写。报告生成时的
vLLM Ascend 提交为 `21fe9cac333b96f02bc2916cf80c2164edd51321`。后续若代码发生变化，现场报告
必须同时记录新的提交、dirty diff 和实际加载路径。

本文只讨论 TurboQuant，不讨论 HIGGS。算法和代码块的详细讲解见：

- [TurboQuant 在 vLLM Ascend 中的实现逻辑](source/developer_guide/Design_Documents/turboquant_implementation_zh.md)
- [TurboQuant 代码阅读指南](source/developer_guide/Design_Documents/turboquant_code_walkthrough_zh.md)
- [TurboQuant Ascend paged-dequant 算子](source/developer_guide/Design_Documents/turboquant_ascend_fused_operator_zh.md)
- [TurboQuant 端到端正确性验证](source/developer_guide/Design_Documents/turboquant_e2e_correctness_zh.md)
- [TurboQuant Ascend 当前功能限制](source/developer_guide/Design_Documents/turboquant_limitations_zh.md)

## 2. 当前实现和调试边界

### 2.1 TurboQuant 在本项目中的作用

当前功能是运行时 KV Cache 量化。模型权重仍按原始格式加载，新产生的 K/V 在写入 KV Cache
时动态量化，在 attention decode 时从 packed Cache 读取或反量化。

因此：

- 不需要先用 ModelSlim 离线量化模型；
- `--kv-cache-dtype turboquant_4bit_nc` 只改变 KV Cache 和 attention backend；
- 权重量化和 KV Cache 量化是两条独立链路；
- baseline 使用 `--kv-cache-dtype auto`，即没有开启 TurboQuant KV Cache。

当前支持的 preset 是：

| Preset | Key | Value | 每个 KV head slot | 理论压缩率 |
| --- | ---: | ---: | ---: | ---: |
| `turboquant_4bit_nc` | 4 bit | 4 bit | 134 Bytes | 3.82x |
| `turboquant_k3v4_nc` | 3 bit | 4 bit | 118 Bytes | 4.34x |
| `turboquant_3bit_nc` | 3 bit | 3 bit | 102 Bytes | 5.02x |

表中的压缩率是相对 FP16/BF16 K/V payload 的布局理论值，不包含模型权重、workspace、block table
和运行时临时张量。不能用整卡显存下降比例直接代替 KV Cache 压缩率。

### 2.2 三条 decode 路径

`VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION` 在 `vllm_ascend/envs.py` 中集中定义，可取：

| 值 | 用途 | 主要限制 |
| --- | --- | --- |
| `auto` | 生产候选；支持时选择 AscendC paged dequant + CANN FIA | 算子可用时会影响 graph 选择 |
| `ascend_fused` | 强制校验 AscendC 路径 | 算子缺失、不支持的 head dim、ALiBi 或 soft cap 会直接失败 |
| `grouped_gqa` | Triton packed decode 快路径 | 适合验证 ACLGraph 和 grouped-GQA |
| `reference` | 每 query head、FP32 rotation 的诊断路径 | 慢，只用于正确性归因 |

当前名为 `ascend_fused` 的路径不是“量化 Cache 到最终 attention 输出”的单个全融合 AICore
kernel。它先用 AscendC 将 paged packed Cache 反量化到 caller-owned dense K/V workspace，再调用
CANN FIA。这个路径减少了 Triton 逐元素解包开销，但长上下文和高并发下仍会产生
`O(batch * sequence * kv_heads * head_dim)` 的临时读写。

### 2.3 本轮迁移目标

优先验证：

- Ascend 910B4；
- vLLM v1 model runner；
- Qwen3 dense attention 模型；
- `turboquant_4bit_nc`；
- eager、TP=1 和 TP=4；
- 并发 1 和 16；
- 最长 16K 的性能测试。

MLA、310P、V2 model runner、Context Parallel 等能力不应在本轮作为验收项。完整边界以
[TurboQuant Ascend 当前功能限制](source/developer_guide/Design_Documents/turboquant_limitations_zh.md) 为准。

## 3. 最重要的调试原则

### 3.1 不跨过失败层继续测试

建议使用以下门禁顺序：

1. 源码和 Python 环境一致；
2. vLLM core API contract 一致；
3. PyTorch schema 注册成功；
4. ACLNN kernel 能够真实启动；
5. store、dequant 和 attention 数值门禁通过；
6. 小模型单 token 和多 token 生成通过；
7. 32B TP4 eager 正确性通过；
8. 并发和长上下文性能通过；
9. 最后测试 ACLGraph、prefix cache 和 speculative decode。

如果第 4 层仍然段错误，不应启动 Qwen3-32B。如果单请求输出已经乱码，不应先分析并发性能。

### 3.2 版本字符串不是兼容性证明

`vllm.__version__ == 0.20.2` 或 `0.20.2+empty` 只能证明包版本，不能证明当前 TurboQuant 所需的
core patch 已经存在。必须同时验证：

- `TQFullAttentionSpec` 和 page-size contract；
- `Attention._init_turboquant_buffers()`；
- TurboQuant workspace；
- `set_forward_context(..., slot_mapping=...)`；
- 对应 attention backend 和 metadata contract；
- 实际导入的 `vllm` 源码路径及 Git 提交。

仓库中的 `scripts/turboquant_triton/common/check_environment.py` 会检查这些关键接口。缺少任意
一项都属于 core/plugin 不匹配，不能通过在 vLLM Ascend 侧增加 `hasattr` 来掩盖。

### 3.3 schema 注册不等于算子可运行

下面的检查只证明 PyTorch extension 注册了名字：

```python
hasattr(torch.ops._C_ascend, "npu_turboquant_paged_dequant_out")
```

它不能证明：

- CANN OPP 中存在匹配的 kernel binary；
- tiling key 能映射到正确的 function entry；
- `vllm_ascend_C`、`libopapi` 和 `libcust_opapi` 来自同一次构建；
- return-style 和 out-style 的 ABI 一致。

必须执行 `operator_runtime_probe.py` 的真实 ACLNN launch 后，才能进入数值测试。

### 3.4 NPU 异步错误可能在后续 API 才暴露

NPU operator 默认异步执行。真正出错的 kernel 可能在下一次 `torch.npu.synchronize()`、张量创建
或另一个算子调用时才报告。因此：

- 正确性定位子进程可以设置 `ASCEND_LAUNCH_BLOCKING=1`；
- 性能、模型服务和正式基准必须移除该变量；
- 不要仅按 Python traceback 最后一行判断根因，要查看日志中的 `current working operator name`；
- 发生段错误后应退出该 Python 进程，不要继续复用已经异常的 NPU context。

## 4. 迁移前环境清单

### 4.1 建议目录

服务器上建议保持两个独立源码目录：

```text
/vllm-workspace/vllm                # 匹配的 vLLM core
/opt/.../vllm-ascend-turboquant     # 本分支 vLLM Ascend
```

不要把两个项目的 Python 文件混装到同一源码目录。editable install 可以指向源码，但切换 worktree
或 branch 后必须重新确认导入路径。

### 4.2 固化环境指纹

迁移后首先执行：

```bash
cd /opt/.../vllm-ascend-turboquant

git status --short --branch
git rev-parse HEAD
git diff --binary > /tmp/vllm_ascend_dirty.patch

git -C /vllm-workspace/vllm status --short --branch
git -C /vllm-workspace/vllm rev-parse HEAD
git -C /vllm-workspace/vllm diff --binary > /tmp/vllm_core_dirty.patch

which python3
which vllm
python3 -m pip show vllm vllm-ascend torch torch-npu triton-ascend
python3 - <<'PY'
import vllm
import vllm_ascend

print("vLLM version:", vllm.__version__)
print("vLLM source:", vllm.__file__)
print("vLLM Ascend source:", vllm_ascend.__file__)
PY
```

还应保存：

```bash
npu-smi info
env | sort | grep -E '^(ASCEND|HCCL|GLOO|TP_|VLLM|PYTHON|LD_LIBRARY_PATH)='
```

如果机器禁止上传大日志，至少回传提交号、`summary.txt`、`results.tsv`、首个失败 stage 日志和
对应的 `npu-smi` 日志。

### 4.3 CANN 和网络环境

以下是 910B4 四卡测试的推荐基础环境，`HCCL_IF_IP` 必须改为服务器实际 IP：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export SOC_VERSION=ascend910b4
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_IF_IP=110.7.102.125
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export HCCL_CONNECT_TIMEOUT=1800

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

`SOC_VERSION=ascend910b4` 是顶层构建目标。CANN 自定义算子日志中出现
`OPS_PRODUCT_NAME="ascend910b;"` 或 `SOC_ARG=ascend910b` 是 910B 产品族映射，不表示构建成了
910B1。

## 5. 构建和安装

### 5.1 第一次完整构建

第一次迁移或 CANN/ABI 发生变化时，应做完整 clean build：

```bash
cd /opt/.../vllm-ascend-turboquant
source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf build csrc/build
unset VLLM_ASCEND_BUILD_CUSTOM_OPS
unset VLLM_ASCEND_ACLNN_INCREMENTAL_BUILD
SOC_VERSION=ascend910b4 MAX_JOBS=8 \
python3 -m pip install --no-build-isolation -v -e . \
  2>&1 | tee build_910b4_full.log
```

使用 `--no-build-isolation` 的前提是当前 Python 环境已经安装项目要求的构建依赖。这样可以避免
pip 在 `/tmp/pip-build-env-*` 中重新下载 `triton-ascend==3.2.1`，也能避免内部镜像只有
3.2.0 时错误地报告“本机没有 3.2.1”。

构建结束不能只看 pip 的最后几行。验收条件是：

- pip editable build 返回 0；
- `vllm_ascend_C` 已更新；
- 完整 custom OPP package 安装成功；
- runtime probe 能启动真实 kernel；
- native Qwen3 仍能找到其余 vLLM Ascend 自定义算子。

### 5.2 日常单算子开发构建

完成过一次完整构建后，修改 TurboQuant AscendC kernel、tiling、OpDef 或 binding 时使用：

```bash
SOC_VERSION=ascend910b4 MAX_JOBS=8 \
bash scripts/turboquant_operators/build_dev.sh
```

如果怀疑生成文件或 tiling key 残留：

```bash
SOC_VERSION=ascend910b4 MAX_JOBS=8 \
bash scripts/turboquant_operators/build_dev.sh --clean
```

单算子构建会备份完整 custom-op package，并临时安装只包含 TurboQuant 的 package。它适合运行
底层算子测试，但其他 vLLM Ascend 自定义算子可能暂时不可用。启动完整模型前可恢复：

```bash
bash scripts/turboquant_operators/build_dev.sh --restore-full
```

如果开发版本的 TurboQuant operator 也必须用于模型端到端测试，则应完成一次全量构建，使
TurboQuant 和其他算子同时存在于完整 OPP package 中。曾经出现的
`aclnnAddRmsNormBias ... not in libopapi.so` 就是单算子 package 替换完整 package 后的典型症状，
不是 Qwen3 权重或 TurboQuant KV 算法错误。

### 5.3 构建变慢的原因

完整构建可能持续几十分钟到一小时，主要时间消耗在：

- ascend protobuf 第三方库；
- 所有 ACLNN 自定义算子的 host、tiling 和 device binary；
- 多个 dtype、shape 和 tiling key 变体；
- pip build isolation 重建依赖环境；
- 清除 `csrc/build` 后丢失增量产物；
- CANN `op_build` 内部并行度和机器 CPU/磁盘性能。

日常 kernel 调试应使用 `build_dev.sh`，只在 ABI、CANN、完整 package 或发布验证时做全量 clean
build。不要为了一个 Python 逻辑改动删除 `csrc/build`。

## 6. 编译 warning 和 error 的处理规则

### 6.1 通常可暂时忽略的 warning

在编译继续运行且最终返回 0 的前提下，以下 warning 通常不是 TurboQuant 阻塞项：

- protobuf `ByteSize()` deprecated；
- protobuf `always_inline function might not be inlinable`；
- sparse flash attention 中未使用变量；
- sparse flash attention 日志格式 `%d` 与 `long int` 不匹配；
- 非虚析构 warning；
- 测试中的 `Cannot create tensor with internal format ... base format`；
- `torch.jit`、SWIG 和 image processor deprecation warning。

这些 warning 应在社区提交前逐步清理，但不能用它们解释 TurboQuant kernel 段错误或乱码。

### 6.2 必须停止并修复的 error

以下错误表示产物不可用：

| 错误 | 含义 | 处理 |
| --- | --- | --- |
| `AlignUp is ambiguous` | 本地 helper 与 CANN 9.0 API 重名 | 删除或重命名本地 helper，clean rebuild |
| `const fe::PlatFormInfos*` 转换失败 | tiling host API const contract 不兼容 | 按当前 CANN API 修正类型，clean rebuild |
| autogen `.cpp` 不存在 | `op_build` 生成目标竞争或生成失败 | 检查第一个 op_build error，串行化依赖后 clean rebuild |
| `coreDim > 65535` | Triton launch grid 超过运行时上限 | 分块 launch；确认加载了包含分块修复的提交 |
| `BinaryGetFunctionByEntry failed` | tiling key 与 binary function entry 不匹配 | 清理旧 OPP、`csrc/build` 和进程，重新全量安装 |
| `npu_turboquant_paged_dequant is not registered` | Python extension 未包含 schema | 重新编译安装 extension，确认导入路径 |
| ACLNN 调用段错误 | extension、opapi、OPP 或 kernel ABI 不一致 | 运行隔离 probe，采集已加载库 SHA，clean full rebuild |

`build_aclnn.sh returned non-zero exit status 2` 只是 pip 包装层结果。必须从完整日志向上查找第一个
`error:`、`failed` 或 `not generated`，后面的 missing object 通常只是级联错误。

## 7. 逐级验证流程

### 7.1 Gate 0：环境和 core contract

```bash
python3 scripts/turboquant_triton/common/check_environment.py
```

必须确认输出中的：

- vLLM 版本为 `0.20.2` 或 `0.20.2+empty`；
- vLLM source 指向预期 `/vllm-workspace/vllm`；
- vLLM Ascend source 指向当前 TurboQuant checkout；
- 两边 commit 可追溯；
- `TQ slot mapping: compatible`；
- `TQ core API: compatible`；
- NPU 名称为 `Ascend910B4`。

典型不匹配症状包括：

- `AttentionBackendEnum["TURBOQUANT_ASCEND"]` 不存在；
- `_use_layer_aware_fia_graph_replay` 未定义；
- `_allocate_int8_cache_tensor` 未定义；
- `TQFullAttentionSpec` 缺失；
- `set_forward_context` 不接受 `slot_mapping`。

这些错误说明 vLLM core 和 plugin patch 不成套。正确修复是 checkout 匹配 core 或补齐明确 patch，
而不是在调用点继续增加兼容分支。

### 7.2 Gate 1：真实 ACLNN runtime probe

```bash
mkdir -p /tmp/tq_probe
ASCEND_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1 \
python3 -u scripts/turboquant_operators/operator_runtime_probe.py \
  --device 0 \
  --cache-dtype turboquant_4bit_nc \
  --activation-dtype bfloat16 \
  --output /tmp/tq_probe/runtime_probe.json \
  2>&1 | tee /tmp/tq_probe/runtime_probe.log
```

该测试会：

- 检查 return-style 和 out-style schema；
- 启动一次真实 `npu_turboquant_paged_dequant_out`；
- 检查 caller-owned output alias；
- 检查零 Cache 反量化为零且结果 finite；
- 保存进程实际加载的 extension、`libopapi` 和 `libcust_opapi` 路径及 SHA256。

如果这里发生段错误，不要运行 `operator_accuracy.py`。先对比实际库路径、清理残留进程和 OPP，
再做完整 clean build。

### 7.3 Gate 2：算子正确性

```bash
SOC_VERSION=ascend910b4 DEVICE=0 \
bash scripts/turboquant_operators/run_smoke.sh
```

该入口覆盖三个 preset、CPU reference、Triton、AscendC、attention 输出、量化误差和短性能 case。
需要重点区分两类指标：

1. 实现一致性：AscendC 与 Triton/CPU 解包是否一致；
2. 算法误差：反量化 K/V 与量化前原始 K/V 的 NMSE 和余弦相似度。

仅有第 1 类通过不够。Triton、AscendC 和 CPU reference 可能共同正确地解码一份错误的 packed
Cache，形成循环验证。`operator_accuracy.py` 当前会对原始 K/V 额外施加质量门禁。

已有 910B4 筛查结果可作为数量级参考，而不是论文精度承诺：

| Preset | Key NMSE | Value NMSE |
| --- | ---: | ---: |
| `turboquant_4bit_nc` | 约 0.009 | 约 0.010 |
| `turboquant_k3v4_nc` | 约 0.034 | 约 0.010 |
| `turboquant_3bit_nc` | 约 0.034 | 约 0.047 |

若同一 seed、shape 和 dtype 下突然高出一个数量级，应优先检查 rotation、centroid 顺序、bit packing、
norm correction 和 slot layout，不应先放宽 tolerance。

### 7.4 Gate 3：正确性定向矩阵

当生成从第二个 token 开始分叉，执行：

```bash
SOC_VERSION=ascend910b4 DEVICE=0 \
bash scripts/turboquant_operators/run_correctness_debug.sh
```

该矩阵用于区分：

- `Hkv=2` 与 `Hkv=8` 的 GQA 映射；
- `splits=1` 与 `splits=4` 的 reduce；
- `reference` 与 `grouped_gqa/auto` 的实现差异；
- FP32 rotation 与优化 rotation 的误差；
- BF16 和 FP16 路径。

### 7.5 Gate 4：小模型端到端

首轮不要使用 32B TP4。建议先用：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL=/run/test_llm/Qwen3-0.6B-hf \
TP_SIZE=1 \
bash scripts/turboquant_triton/diagnostics/run_910b4_retest.sh
```

端到端定位应固定：

- eager；
- prefix cache 关闭；
- speculative decode 关闭；
- 并发 1；
- `temperature=0` 和固定 seed；
- `max_model_len <= 2048`；
- 相同 prompt 和 tokenizer。

至少对比以下 case：

| Case | Cache | Decode | 目的 |
| --- | --- | --- | --- |
| native | `auto` | `auto` | 模型和 NPU baseline |
| all-skip | 首末层规则扩展为全跳过 | 无量化 | 验证 backend 接线本身 |
| TQ reference | `turboquant_4bit_nc` | `reference` | 隔离量化算法和布局 |
| TQ grouped | `turboquant_4bit_nc` | `grouped_gqa` | 隔离 Triton 优化误差 |
| TQ Ascend | `turboquant_4bit_nc` | `ascend_fused` | 隔离 ACLNN/FIA 路径 |

生成长度按 1、2、4、16 token 递增。如果第一个 token top-1 匹配率很高，但第二个 token 后乱码，
重点检查本轮 token 的 KV 写入和下一轮读取，而不是模型加载和首轮 prefill。

### 7.6 Gate 5：32B TP4 快速全链路

小模型通过后再运行：

如果需要先手工观察服务日志，可使用仓库内统一 server wrapper。首轮固定 eager、关闭 prefix
cache，并显式记录 decode implementation：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=auto

MODEL=/run/test_llm/Qwen3-32B \
TP_SIZE=4 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=4 \
KV_CACHE_DTYPE=turboquant_4bit_nc \
ENFORCE_EAGER=1 \
NETWORK_IFNAME=eth0 \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh \
  --no-enable-prefix-caching \
  2>&1 | tee /tmp/qwen3_32b_turboquant_eager.log
```

需要做实现归因时，保持其他参数不变，只将环境变量依次改为 `reference`、`grouped_gqa` 和
`ascend_fused`。不要同时改变 graph、prefix cache、上下文和 decode implementation。

快速全链路命令如下：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_IF_IP=110.7.102.125
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

QUICK=1 \
MODEL=/run/test_llm/Qwen3-32B \
VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
NETWORK_IFNAME=eth0 \
SOC_VERSION=ascend910b4 \
bash scripts/turboquant_operators/run_full_validation.sh
```

快速模式保留 ABI、真实算子 launch、算子正确性、kernel、长上下文 profile、模型精度和 eager
serving，跳过更耗时的质量、ACLGraph/spec 和 graph serving。

### 7.7 Gate 6：完整 910B4 验证

```bash
MODEL=/run/test_llm/Qwen3-32B \
VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
NETWORK_IFNAME=eth0 \
SOC_VERSION=ascend910b4 \
bash scripts/turboquant_operators/run_full_validation.sh
```

默认 serving 矩阵使用并发 1 和 16、12K 输入和 256 输出，并复用同一个 native 或 TurboQuant
服务，避免重复加载模型。可用下列变量调整：

```bash
PERF_CONCURRENCY_LEVELS="1 4 16" \
PERF_MAX_MODEL_LEN=16384 \
PERF_INPUT_TOKENS=12288 \
PERF_OUTPUT_TOKENS=256 \
PERF_WARMUP_REQUESTS=16 \
PERF_MEASURE_REQUESTS=32 \
bash scripts/turboquant_operators/run_full_validation.sh
```

`PERF_MEASURE_REQUESTS` 必须大于等于最高并发，否则脚本会拒绝运行。

## 8. KV Cache 正确性高风险点

### 8.1 `slot_mapping`

`slot_mapping` 决定每个新 token 写入哪个物理 block 和 block offset，是最容易导致“服务启动正常、
首 token 正常、后续乱码”的位置。

必须验证：

- dtype 与 kernel 参数一致；
- slot 以 token slot 计数，而不是 byte offset；
- `physical_block * block_size + block_offset` 使用同一 block size；
- padded token 的负 slot 不写 Cache；
- chunked prefill、mixed prefill/decode 和 TP rank 使用各自正确 mapping；
- `ForwardContext.slot_mapping` 与 Ascend metadata 中的每层 mapping 一致；
- graph replay 后 mapping 指向本轮请求，而不是 capture 时的旧张量内容。

`slot_mapping < 0` 不能转换成无符号下标，也不能默认写 slot 0。负值表示该 token 不应写 Cache。

### 8.2 packed layout

需要在 store 和 decode 两侧保持完全一致：

- key/value packed size；
- slot alignment；
- centroid index bit order；
- norm correction 的存储位置和 dtype；
- page、token、KV head、byte 的维度顺序；
- block table 的 logical-to-physical 映射；
- head dim 对应的 tiling key。

任何一处不一致都可能产生有限但无意义的 K/V，因此不一定触发 NaN 或越界错误。

### 8.3 layer skip

TurboQuant 通常跳过模型首两层和末两层，以降低量化误差对输入特征形成和最终 logits 的放大。
调试时必须确认：

- skip 规则使用全局 layer index，而不是 TP rank 的局部 index；
- skip layer 分配普通 Cache，TQ layer 分配 packed Cache；
- attention backend 和 Cache spec 对同一层的判断一致；
- all-skip 诊断不会残留 TQ workspace 或错误 backend。

跳过层是精度策略，不是修复错误量化布局的方法。如果 all-skip 正常而任意 TQ layer 开启就乱码，
说明应继续检查量化和 Cache 数据链路。

## 9. 端到端精度问题的判定

### 9.1 不能只看最终答题正确率

Qwen3-0.6B 的 native 绝对正确率可能不高，因此重点应是 TurboQuant 相对 native 的回退：

- first-token top-1 match；
- first-token top-k overlap；
- 逐 token prefix match；
- teacher-forcing 下相同 token 的 logprob 差异；
- native correct 但 TQ wrong 的 regression 数；
- 原始文本，不只看 grader 结果。

如果报告中的 common-token logprob difference 全为 0，同时文本从第二 token 起完全不同，应确认
比较器是否只统计了第一个公共 token，不能把 0 解释为 logits 完全一致。

### 9.2 乱码的典型归因

| 现象 | 优先怀疑 |
| --- | --- |
| native 也乱码 | 模型、tokenizer、采样或基础 vLLM Ascend |
| 首 token 就不同 | prefill Cache、量化算法、layer skip、attention 输出 |
| 首 token 相同，第二 token 起错 | decode 前一 token 的 KV 写入、slot mapping、paged 读取 |
| `reference` 正常，`grouped_gqa` 错 | grouped head 映射、split/reduce 或低精度优化 |
| Triton 正常，`ascend_fused` 错 | ACLNN dequant、dense workspace、FIA 参数 |
| eager 正常，ACLGraph 错 | capture/replay metadata、地址稳定性或动态 shape |
| 单请求正常，重复 prefix 错 | prefix block reuse、Cache 生命周期或 block table |

不要用增大误差阈值、改变 prompt 或缩短输出掩盖乱码。乱码属于功能性失败。

## 10. 高并发和性能问题

### 10.1 需要分别测量的指标

性能不能只看 `Avg generation throughput`。至少记录：

- TTFT：请求到第一个 token；
- TPOT：第一个 token 后每个输出 token 的平均时间；
- aggregate output throughput；
- 每请求输出 token/s；
- batch size、实际并发和排队时间；
- KV Cache token capacity；
- NPU 利用率和 HBM；
- store、packed decode、Ascend dequant、FIA 的 component latency。

当前完整脚本的回归筛查门禁为：

- TTFT 不超过 baseline 2 倍；
- TPOT 不超过 baseline 2 倍；
- aggregate throughput 不低于 baseline 50%；
- KV token capacity 至少为 baseline 2 倍。

这些是发现明显回退的工程门禁，不代表最终社区性能目标。

### 10.2 并发 1 已经只有 0.5 token/s

如果 prefill 后单请求 decode 就极慢，应优先检查：

1. 是否误设 `ASCEND_LAUNCH_BLOCKING=1`；
2. 是否实际使用 `reference` decode；
3. `auto` 是否因算子缺失回退到慢路径；
4. 每 token 是否重新分配或清零最大长度 dense workspace；
5. AscendC dequant 是否反量化了远超实际 `seq_len` 的 Cache；
6. split 数是否与当前长度匹配；
7. Python hot path 是否存在 NPU tensor `.item()`；
8. 是否每层/每 token 触发 host-to-device metadata copy；
9. profiler 中 dequant、FIA 或 store 哪一项占主导。

`reference` 本来就不是性能实现。性能结论必须报告 decode implementation 和服务日志中的实际
分发路径。

### 10.3 并发 16 卡住或吞吐崩溃

高并发会放大：

- dense K/V workspace 的容量和内存带宽；
- 每请求 block table/page table 处理；
- split attention partial buffer；
- 动态 tensor 分配和显存碎片；
- host/NPU 同步；
- launch 数量和小 kernel 调度开销；
- HCCL/TP 等待中的慢 rank；
- scheduler 中 `max_num_seqs`、`max_num_batched_tokens` 的限制。

`ascend_fused` 当前先完整反量化再 FIA，高并发长上下文时不一定优于直接 packed attention。应分别
对比 `grouped_gqa` 和 `ascend_fused`，并以 profiler 证明瓶颈，不能仅凭名称假设融合路径更快。

### 10.4 Triton launch grid 上限

长序列和大 batch 可能使 store grid 超过 65535，运行时表现为：

```text
KernelLaunch failed because value ... for parameter coreDim is invalid
```

当前分支包含分块 launch 修复。如果再次出现：

- 检查实际导入的 `turboquant_store.py` 路径和提交；
- 清理 Triton cache 后重试；
- 记录 batch、sequence、num heads 和计算出的 launch range；
- 运行 B16/S16K operator profile，不要先启动模型。

## 11. 分布式初始化和服务启动

以下日志本身不表示失败：

```text
world_size=1 rank=0 local_rank=0 ... backend=hccl
Rank 0 is connected to 0 peer ranks. Expected number ... 0
```

world size 为 1 时连接 0 个 peer 是正常的。但若每次 Gloo/HCCL 初始化等待数分钟，应检查：

- 三个 `*_SOCKET_IFNAME` 是否指向存在的同一网卡；
- `HCCL_IF_IP` 是否是该网卡地址；
- hostname、DNS 和 `/etc/hosts`；
- 防火墙和临时 TCP 端口；
- 残留 EngineCore 进程；
- TP=4 时四张卡是否都可见；
- server timeout 是否覆盖公司网络环境中的初始化延迟。

如果日志一直停在 distributed init 且 NPU 显存不增长，尚未进入 TurboQuant attention，不能把该
等待归因于 KV Cache 量化。

## 12. ACLGraph、prefix cache 和 spec decode

### 12.1 ACLGraph

ACLGraph 与 CUDA Graph 的目标相同，都是复用已捕获的固定执行图，但底层 runtime、支持的算子和
内存约束不同。TurboQuant graph 正确性要求：

- capture/replay shape 稳定；
- query、slot mapping、block table、seq lens 等 metadata 地址稳定；
- replay 前内容能够更新；
- hot path 没有 `.item()` 或 CPU 条件分支；
- 没有 replay 期间动态分配输出 workspace；
- custom operator 支持 capture。

需要 graph 时显式使用 `grouped_gqa` 更容易保持 packed decode 的 graph-capable 路径。`auto` 在
AscendC fast path 可用时可能禁用或绕开相应 capture。必须从 server log 证明发生了 capture 和
replay，不能只因未报错就判为 graph 通过。

### 12.2 Prefix cache

性能 A/B 默认关闭 prefix cache，避免 baseline 与 TQ 的 block reuse 命中率差异污染 TTFT。正确性
通过后再单独开启并测试：

- 相同 prefix 的第二次请求；
- prefix block refcount；
- TQ layer 与 skip layer 的 block 生命周期；
- block eviction 后重新分配；
- 相同 prompt 的输出一致性。

### 12.3 Speculative decode

spec decode 会产生 uniform multi-token 或 mixed token 数，扩大 metadata 和 slot mapping 状态空间。
只有 eager 单 token、多 token 和 ACLGraph 分别通过后才能测试。问题定位时比较：

- TQ eager + spec；
- TQ ACLGraph + spec；
- 相同 accepted token 数；
- draft rejection 后的 Cache 写入位置。

## 13. 常见症状快速索引

| 症状 | 所在层 | 最短确认方式 | 处理方向 |
| --- | --- | --- | --- |
| backend enum 找不到 | core contract | `check_environment.py` | 切换匹配 vLLM core |
| `_use_layer_aware...` 缺失 | core/plugin mismatch | 打印两边 source/commit | 成套安装，禁止混用 |
| `_allocate_int8_cache_tensor` 缺失 | model runner 移植不完整 | 检查当前类源码路径 | 补齐 allocation contract 和测试 |
| schema 未注册 | extension | `hasattr(torch.ops...)` | 重新编译 editable extension |
| `BinaryGetFunctionByEntry` | OPP/kernel | runtime probe | 清理并全量重建 |
| ACLNN 处段错误 | ABI/OPP | probe 的库路径和 SHA | 确保 extension 与 OPP 同构建 |
| `AddRmsNormBias` 缺失 | dev-only OPP package | 检查 `.turboquant_dev_only` | 恢复或全量构建 |
| `PagedAttentionOperation setup failed` | native ATB baseline | 同步隔离 native case | 检查测试 shape/ATB，不先归因 TQ |
| 第一个 token 对，后续乱码 | Cache write/read | 1/2/4 token A/B | slot mapping 和 packed layout |
| 所有实现都一致但模型错 | 算法共同错误 | 原始 K/V NMSE/cosine | rotation、centroid、packing |
| eager 对、graph 错 | graph metadata | capture 后修改所有输入再 replay | 稳定地址和更新内容 |
| 并发 1 极慢 | decode hot path | component profiler | sync、workspace、错误回退 |
| 并发 16 卡住 | 内存/调度/慢 rank | C1/C4/C16 + NPU monitor | workspace、splits、scheduler、HCCL |
| 输入长度超限 | 测试配置 | tokenizer 预算 | 保证 input + output <= max len |

## 14. 日志和结果目录

完整入口结果位于：

```text
logs/turboquant/full_910b4_<timestamp>/
```

优先查看：

| 文件 | 用途 |
| --- | --- |
| `summary.txt` | 全部 stage 的 PASS/FAIL |
| `results.tsv` | 机器可读 stage 状态 |
| `configuration.env` | 本次参数快照 |
| `01_source.log` | 提交、checksum、package metadata |
| `02_environment.log` | vLLM core contract |
| `02a_runtime_probe.log` | 真实 ACLNN launch |
| `operator_runtime_probe.json` | 加载库路径和 SHA |
| `operator_correctness/summary.md` | 算子与量化质量 |
| `05_accuracy_native_auto.log` | native/TQ 文本和 logprob 对比 |
| `serving_eager/summary_c1.md` | 并发 1 TTFT/TPOT/吞吐 |
| `serving_eager/summary_c16.md` | 并发 16 TTFT/TPOT/吞吐 |
| `*_npu.log` | 各阶段 NPU 快照 |
| `combined.log` | 合并日志 |

脚本会同时生成：

```text
logs/turboquant/full_910b4_<timestamp>.tar.gz
```

如果压缩包无法上传，建议先执行：

```bash
grep -RniE 'FAIL|Traceback|RuntimeError|Segmentation|error:|ERR[0-9]+' \
  logs/turboquant/full_910b4_<timestamp> \
  > /tmp/turboquant_failures.txt
```

同时回传 `summary.txt`、`results.tsv`、`configuration.env`、`operator_runtime_probe.json` 和首个失败
stage 的完整日志。

## 15. 建议的现场调试决策树

```text
check_environment.py 失败？
  是 -> 修复 vLLM core / editable source / 版本 contract
  否
  |
runtime_probe 失败或段错误？
  是 -> 检查 extension、libopapi、OPP、tiling key，执行 clean full build
  否
  |
operator_accuracy 的量化质量失败？
  是 -> 检查 rotation、centroid、packing、norm correction
  否
  |
小模型 max_tokens=1 失败？
  是 -> 检查 prefill、layer skip、首轮 attention
  否
  |
max_tokens=2/4/16 失败？
  是 -> 检查 slot mapping、Cache 写入、paged 读取
  否
  |
reference 正常但 auto 失败？
  是 -> 检查 grouped-GQA、低精度优化或 AscendC/FIA 分发
  否
  |
TP1 正常但 TP4 失败？
  是 -> 检查 local heads、rank mapping、HCCL 和每 rank metadata
  否
  |
C1 正常但 C16 性能崩溃？
  是 -> profile workspace、memory bandwidth、splits、scheduler 和慢 rank
  否
  |
最后验证 ACLGraph、prefix cache 和 spec decode
```

## 16. 验收标准

### 16.1 功能验收

- 环境和 core contract 检查通过；
- runtime probe 无段错误，return/out schema 均能真实执行；
- 三个 preset 的 CPU、Triton、AscendC 一致性和量化质量门禁通过；
- 负 `slot_mapping` 不改变 Cache；
- Qwen3-0.6B native、TQ reference、TQ auto 无乱码；
- 1、2、4、16 token 的分叉可解释且满足预设精度门禁；
- Qwen3-32B TP4 eager 正确性通过；
- 不出现 NaN、illegal memory access 或 silent fallback。

### 16.2 性能验收

- benchmark 未设置 `ASCEND_LAUNCH_BLOCKING`；
- native 和 TQ 使用相同 prompt、采样、上下文、并发和输出长度；
- warmup 和测量请求均实际完成；
- 并发 1 和 16 都报告 TTFT、TPOT、aggregate throughput；
- KV capacity 由引擎日志或 Cache 配置计算，而不是整卡显存目测；
- component profile 能解释端到端 TPOT；
- 没有因请求数小于并发而形成伪并发；
- 没有把服务启动和 HCCL 初始化时间计入请求时延。

### 16.3 社区提交前验收

- 完整 custom-op package clean build 通过；
- `bash format.sh ci` 通过；
- Python UT、NPU operator test 和端到端测试均有记录；
- 新环境变量只在 `vllm_ascend/envs.py` 定义并有文档；
- model runner 修改有必要性说明和回归测试；
- hot path 中的 NPU tensor `.item()` 已审查；
- 已知限制和 fallback 行为与文档一致；
- 提交使用 Conventional Commits 并带 Signed-off-by。

## 17. 结论

TurboQuant 迁移到 910B4 的主要风险不只在量化公式。当前工程链路同时依赖 vLLM core cache
contract、vLLM Ascend metadata、Triton store/decode、AscendC paged dequant、CANN FIA、OPP
安装包和分布式运行环境。任何一层不匹配，都可能表现为相同的“服务起不来、段错误、乱码或
吞吐很低”。

现场调试最有效的方法是严格执行逐级门禁：先证明源码和 ABI，再证明真实 kernel，再证明量化
误差和多 token Cache 生命周期，最后才测试并发和 graph。完整验证脚本已经把这一顺序自动化，
建议所有 NPU 实验都保留其 `configuration.env`、stage 状态和加载库指纹，以便跨机器复现。
