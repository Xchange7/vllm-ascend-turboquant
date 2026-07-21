# 使用 turboquant-triton-v0.20.2rc 定位 TurboQuant 精度问题

## 1. 文档目标

本文说明如何使用 `turboquant-triton-v0.20.2rc` 分支定位以下问题：

- TurboQuant 服务能够启动，但生成内容明显错误；
- native 输出正常，而 TurboQuant 没有一道测试题答对；
- `turboquant-high-perfermance` 的吞吐和精度同时退化；
- kernel 单测通过，但模型端到端结果仍然错误。

本文优先定位正确性，不讨论 HIGGS，也不先优化性能。所有模型级测试应先使用 eager、关闭
prefix cache、单请求和确定性采样。ACLGraph、spec decode、并发和长上下文只能在 eager
正确性通过后测试。

## 2. 当前基线和初步结论

### 2.1 分支关系

当前已知分支关系如下：

| 项目 | 值 |
| --- | --- |
| reference 分支 | `turboquant-triton-v0.20.2rc` |
| reference 提交 | `25a9705032fe47066bfdc3550bb7360e694e00d6` |
| 性能实验分支 | `turboquant-high-perfermance` |
| 两个分支的共同起点 | `25a9705032fe47066bfdc3550bb7360e694e00d6` |

目前性能分支相对 reference 提交的主要变化还在本地工作树中，包括 grouped-GQA decode、
低精度 Hadamard 矩阵乘和低精度 PV。`turboquant-triton-v0.20.2rc` 因此适合充当性能改动前
的对照组。

但是，该分支只能作为**实现版本基线**，不能作为**算法正确性金标准**。如果 reference
分支同样出现精度崩溃，问题就在性能优化之前。

### 2.2 当前最可能的问题

根据现有代码审查，问题按优先级排列如下：

1. 当前实现并不是论文中的完整 TurboQuant。vLLM core 中的配置明确省略了 QJL residual，
   Ascend 侧也没有 QJL 和 outlier path。
2. 静态检查已确认旧实现只使用固定 Sylvester Hadamard。CPU oracle 显示常量、分块同号和
   线性结构 key 的相对重建误差可达到约 48%、48% 和 57%。当前工作树已改为固定 seed 的
   `diag(random_sign) @ H`，对应误差降至约 9%、9% 和 10%，仍需在 910B4 上做模型级验证。
3. 现有 NPU kernel 测试主要验证实现内部自洽：使用同一个 store kernel 写 cache，再用同
   一套布局进行 dequant/decode。共同的 layout 或量化错误可能同时存在于两侧而不被发现。
4. 高性能工作树另外把 Hadamard 和 PV 的部分计算从 FP32 降到了 FP16/BF16，并默认选择
   grouped-GQA kernel。这些改动可能进一步扩大误差，但不能解释 reference 分支已有的问题。
5. vLLM core 是 TurboQuant contract 的一部分。只检查版本字符串为 `0.20.2+empty` 不足以
   证明兼容；cache spec、layer skip、Attention workspace 和 backend API 必须来自同一组
   patch。

因此，当前不应先调整 tolerance，也不应直接在 32B、并发 16 和 16K context 上定位。应先
完成下文的最小 A/B 矩阵。

## 3. 定位原则

一次只改变一个变量：

| 维度 | 定位阶段固定值 |
| --- | --- |
| 模型 | `/run/test_llm/Qwen3-0.6B-hf` |
| TP | 1 |
| graph | eager |
| prefix cache | disabled |
| speculative decoding | disabled |
| 并发 | 1 |
| context | 不超过 2048 |
| sampling | `temperature=0`、固定 seed |
| cache dtype | `auto` 或 `turboquant_4bit_nc` |

不要同时修改 branch、vLLM core、模型、TP、graph 和请求格式。否则结果无法归因。

## 4. 创建隔离的 reference worktree

当前性能分支有未提交改动，不要 stash、checkout 或覆盖它。使用独立 worktree：

```bash
cd /path/to/vllm-ascend
git fetch turboquant-private

export TQ_REF_COMMIT=25a9705032fe47066bfdc3550bb7360e694e00d6
export TQ_REF_ROOT=/path/to/vllm-ascend-tq-v0202rc

git worktree add --detach "${TQ_REF_ROOT}" "${TQ_REF_COMMIT}"
git -C "${TQ_REF_ROOT}" status --short --branch
git -C "${TQ_REF_ROOT}" rev-parse HEAD
```

输出必须满足：

- HEAD 等于 `25a9705032fe47066bfdc3550bb7360e694e00d6`；
- worktree 没有修改文件；
- 测试和服务日志中的 `vllm_ascend.__file__` 指向 `${TQ_REF_ROOT}`。

在服务器的测试虚拟环境中切换 editable install：

```bash
python3 -m pip install -e "${TQ_REF_ROOT}"
python3 - <<'PY'
import vllm
import vllm_ascend

print("vllm:", vllm.__version__, vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
PY
```

`pip install -e .` 后，普通 Python 文件更新不需要重复安装；切换到另一个 worktree 时需要重新
执行 editable install，否则 `vllm` 命令仍可能加载上一个源码目录。

## 5. Gate 0：证明 vLLM core contract 匹配

`turboquant-triton-v0.20.2rc` 不是独立于 vLLM core 的完整实现。它至少依赖以下 core 能力：

- `TQFullAttentionSpec` 及 `tq_slot_size` page-size 计算；
- `Attention._init_turboquant_buffers()`；
- `_tq_mid_o_buf`、`_tq_output_buf` 和 `_tq_lse_buf`；
- `kv_cache_dtype_skip_layers`；
- TurboQuant cache dtype 的解析和逐层 backend 选择；
- 自动跳过首两层和末两层的 patch。

先运行：

```bash
cd "${TQ_REF_ROOT}"
python3 scripts/turboquant_triton/common/check_environment.py \
  2>&1 | tee logs/tq_ref_environment.log
```

再记录实际加载的两个仓库状态：

```bash
python3 - <<'PY'
from pathlib import Path
import subprocess
import vllm
import vllm_ascend

for name, module in (("vllm", vllm), ("vllm_ascend", vllm_ascend)):
    root = Path(module.__file__).resolve().parents[1]
    print(f"\n{name}: file={module.__file__} root={root}")
    for args in (("rev-parse", "HEAD"), ("status", "--short")):
        result = subprocess.run(
            ("git", "-C", str(root), *args),
            text=True,
            capture_output=True,
            check=False,
        )
        print(result.stdout, end="")
PY
```

### 判定

- 缺少 `TQFullAttentionSpec`：vLLM core 没有应用 cache spec patch。
- 缺少 `_tq_*` workspace：vLLM Attention patch 不完整。
- 服务日志没有 `TQ: skipping layers [...]`：自动 boundary skip patch 没有加载。
- `vllm.__version__` 正确但 `vllm.__file__` 指向其他目录：Python 环境混装。
- core 工作树为 dirty：必须保存其 diff；仅记录 commit 不能复现实验。

在 Gate 0 失败时，不运行模型精度测试。此时结果只说明安装或 contract 错误。

## 6. Gate 1：先证明 native 模型和评测输入正常

在 reference worktree 上先只运行 native：

```bash
cd "${TQ_REF_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL=/run/test_llm/Qwen3-0.6B-hf
export NETWORK_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

MODEL="${MODEL}" \
TP_SIZE=1 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
KV_CACHE_DTYPE=auto \
ENFORCE_EAGER=1 \
PORT=18010 \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh \
  --no-enable-prefix-caching \
  > logs/tq_ref_native_server.log 2>&1 &
echo $! > logs/tq_ref_native_server.pid
```

服务健康后，用 completion API 发送一个确定性短请求。定位阶段不要使用 chat 模式，因为当前
chat 采集路径没有输出 token logprobs，无法判断第一个分歧 token。

```bash
curl --noproxy '*' -s http://127.0.0.1:18010/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"2 + 3 =\",\"max_tokens\":4,\"temperature\":0,\"seed\":0,\"logprobs\":5}" \
  | tee logs/tq_ref_native_4tokens.json
```

### 判定

- native 也答错或为空：先检查模型、chat template、prompt、生成长度和评测器，不能归因于
  TurboQuant。
- native 正确且稳定：进入 Gate 2。
- 多次相同请求不稳定：检查 sampling 参数和环境，不继续比较 token prefix。

测试完成后只停止本次记录的 PID：

```bash
kill -TERM "$(cat logs/tq_ref_native_server.pid)"
```

## 7. Gate 2：使用“全部跳过”隔离 TurboQuant 集成

Qwen3-0.6B 有多少层必须从模型配置读取，不能硬编码：

```bash
export NUM_LAYERS="$(${PYTHON:-python3} - <<'PY'
import json
import os
from pathlib import Path

config = json.loads((Path(os.environ["MODEL"]) / "config.json").read_text())
print(config["num_hidden_layers"])
PY
)"
```

保持 `--kv-cache-dtype turboquant_4bit_nc`，但把所有层加入 skip list：

```bash
ALL_LAYERS="$(seq 0 $((NUM_LAYERS - 1)))"

MODEL="${MODEL}" \
TP_SIZE=1 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
KV_CACHE_DTYPE=turboquant_4bit_nc \
ENFORCE_EAGER=1 \
PORT=18011 \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh \
  --no-enable-prefix-caching \
  --kv-cache-dtype-skip-layers ${ALL_LAYERS} \
  > logs/tq_ref_all_skipped_server.log 2>&1 &
echo $! > logs/tq_ref_all_skipped_server.pid
```

用与 Gate 1 完全相同的请求比较结果。

### 判定

| 结果 | 结论 |
| --- | --- |
| all-skip 与 native 一致 | 模型、runner 和请求正常；错误由实际 TurboQuant layer path 引入 |
| all-skip 仍错误 | backend 选择、core patch、测试环境或请求不一致，不是量化误差 |
| 日志仍为中间层选择 TurboQuant backend | skip pattern 没有生效或加载了错误 core |

Gate 2 是最重要的集成断点。它比“kernel 单测通过”更接近实际模型。

## 8. Gate 3：用首 token 判断 prefill 还是 KV Cache decode

第一次完整 prefill 使用当前请求的原始 K/V 计算 attention，压缩 cache 从下一个生成 token
开始被读取。因此在关闭 prefix cache 的条件下：

- 第 1 个生成 token 主要验证 prefill 和 backend 集成；
- 第 2 个及后续 token 才验证 packed KV cache 的 store/decode 路径。

分别对 native、all-skip 和正常 TurboQuant 发送 `max_tokens=1`、`2`、`4` 和 `16` 的相同
completion 请求，并保存完整 JSON。不要只保存最终文本。

### 判定

| 首次分歧位置 | 优先检查 |
| --- | --- |
| token 1 | prefill 路由、混合 backend、layer skip、输入模板、加载的代码版本 |
| token 2 | `do_kv_cache_update()`、slot mapping、store packing、首次 decode |
| 跨过 block 边界后 | block table、page offset、cache copy/eviction |
| 仅 continuation prefill | history dequant、Hadamard 逆变换、causal length |
| 仅 ACLGraph | metadata replay、workspace shape、capture 时的 `seq_lens` |
| 单请求正确、并发错误 | batch stride、block table、workspace 复用或 request metadata |

如果 token 1 已经不同，不应先修改 decode Triton kernel；它尚未参与第一个 token 的
attention 读取。

## 9. Gate 4：对量化层做二分

Qwen3 默认自动跳过首两层和末两层。对其余层使用
`--kv-cache-dtype-skip-layers` 做二分，可以区分：

- 某个层的数据分布特别敏感；
- 所有量化层都会逐步累积误差；
- 某个 TP rank 或层的 cache layout 错误。

以 28 层模型为例，自动 skip 为 `0 1 26 27`：

| 实验 | 额外 skip | 实际量化层 |
| --- | --- | --- |
| A | `14 ... 25` | `2 ... 13` |
| B | `2 ... 13` | `14 ... 25` |
| A1 | `8 ... 25` | `2 ... 7` |
| A2 | `2 ... 7 14 ... 25` | `8 ... 13` |

每次只比较同一 prompt 的 1、2、4、16 token prefix。若量化任意单层都会立即导致明显错误，
优先怀疑算法或统一 layout；若只在特定层段发生，导出该层的 Q/K/V 做离线比较。

不要把“多跳过几层后准确率变好”直接视为修复。它只能证明误差与层敏感度相关。

## 10. Gate 5：独立验证 store、dequant 和 attention

现有测试 `tests/ut/ops/test_turboquant_triton.py` 必须运行，但它不是独立 oracle：

```bash
cd "${TQ_REF_ROOT}"
ASCEND_LAUNCH_BLOCKING=1 pytest -sv \
  tests/ut/ops/test_turboquant_triton.py -k 'not aclgraph'
```

当前测试中，cache 通常由 `triton_turboquant_store()` 写入，再由
`triton_turboquant_dequant_paged_cache()` 或 packed decode 读取。若二者共享同一个 byte
offset 错误，测试仍可能通过。

要得到有效结论，需要新增或临时执行三个独立 oracle：

1. CPU FP32 packer：只根据 layout spec 产生期望字节，不调用 Ascend store/dequant 代码；
2. raw attention oracle：用原始 FP32 Q/K/V 计算 attention，并直接和压缩 attention 比较；
3. structured vectors：至少覆盖常量、半块正负、单 outlier、重复坐标和真实模型层 dump。

建议记录以下指标，而不是只用 `assert_close`：

```text
key_relative_rmse
value_relative_rmse
attention_relative_rmse
attention_cosine_similarity
max_attention_logit_error
top1_token_changed
```

### 已修复：固定 Hadamard

旧 reference 实现在 `vllm_ascend/attention/turboquant.py` 的 `_build_hadamard()` 中构造固定
Sylvester Hadamard。本轮静态 debug 已完成以下 A/B：

```text
H                       # 旧实现
diag(random_sign) @ H   # 当前实现；对行向量 x，随机符号在 Hadamard 混合前生效
```

对 Q 和 K 必须使用匹配的变换，且 continuation prefill 的 inverse path 必须同步修改。随机
符号不能只加在旋转之后，因为那不会改变能量集中的位置。

结果符合该判断：structured K 在固定 Hadamard 下误差显著高于 signed Hadamard，而随机高斯
用例差距很小。该结果证明旧旋转设计存在高风险，但不能单独排除 NPU bit packing 问题；后者
仍需要独立 CPU packer 和真实模型 layer dump 验证。

## 11. Gate 6：reference 与高性能工作树做严格 A/B

只有 reference 分支达到可接受精度后，才比较高性能工作树。两次测试必须使用相同：

- vLLM core checkout 和 dirty patch；
- CANN、torch-npu 和 Triton-Ascend；
- 模型、设备、TP、cache dtype；
- prompt、seed、生成长度；
- eager、prefix cache disabled、concurrency 1。

每次切换 worktree 后重新执行：

```bash
python3 -m pip install -e /path/to/selected/vllm-ascend
python3 -c 'import vllm_ascend; print(vllm_ascend.__file__)'
```

### 判定

| reference | high performance | 结论 |
| --- | --- | --- |
| 正确 | 错误 | grouped-GQA 或低精度 math 引入回归 |
| 错误 | 错误 | 根因在 reference 算法、layout、core contract 或评测路径 |
| 正确 | 正确但慢 | 只处理性能，不再改量化算法 |
| 两者 token 1 不同 | 不可比较 | 环境、prefill 或代码加载不一致 |

高性能分支当前应强制使用 FP32 reference decode 作为 accuracy mode。`auto` dispatch 不能在
reference 通过之前成为 serving 默认值。

## 12. Gate 7：最后恢复复杂运行模式

按以下顺序逐项恢复，每次只增加一个变量：

1. eager，单请求，短 context；
2. eager，并发 2、4、8、16；
3. eager，2K、4K、8K、16K context；
4. ACLGraph，单 token decode；
5. ACLGraph，uniform multi-token；
6. speculative decode；
7. TP=4 和 Qwen3-32B。

任一阶段失败就回到上一个通过点。不能用 32B 并发测试来定位基础量化正确性，因为通信、
内存、调度和 kernel 问题会混在一起。

## 13. 日志检查清单

每次实验至少保存：

```text
vllm version, source path, commit, dirty diff
vllm-ascend source path, branch, commit, dirty diff
pip show vllm vllm-ascend torch torch-npu triton triton-ascend
npu-smi info
all VLLM/HCCL/GLOO/TP/ASCEND related environment variables
complete server command
complete server log
complete request JSON
complete response JSON including token logprobs
origin_answers.txt and turboquant_answers.txt
first differing token index
whether "TQ: skipping layers [...]" appeared
```

服务启动慢且停在 `world_size=1 rank=0 ... backend=hccl` 时，先确认：

```bash
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
```

`Rank 0 is connected to 0 peer ranks` 对 world size 1 是正常信息，不表示 TurboQuant kernel 已经
执行。此阶段 NPU 显存没有变化时，应先定位 collective 或 worker 初始化，不应分析量化精度。

## 14. 最短决策树

```text
check_environment.py 是否通过？
  否 -> 修复 vLLM core / editable install / TQ cache contract
  是 -> native 是否正确？
          否 -> 修复模型、请求模板或评测器
          是 -> TQ + all layers skipped 是否正确？
                  否 -> 修复 backend 选择、layer skip 或 core 集成
                  是 -> 正常 TQ 的 token 1 是否与 native 一致？
                          否 -> 定位 prefill、混合 backend 和加载版本
                          是 -> token 2 是否开始分歧？
                                  是 -> 定位 store/layout/decode
                                  否 -> 找到首次跨页、长上下文或并发分歧点
```

## 15. 当前建议的修复顺序

### 已确认根因：Ascend ForwardContext 丢失 slot mapping

端到端实验出现“首 token 基本正常，从第二个 token 开始乱码”的直接原因已经确认。TurboQuant
backend 设置了 `forward_includes_kv_cache_update = False`，因此 vLLM 会在 attention forward 前
调用 `unified_kv_cache_update()` 写入压缩 cache。该 custom op 不读取 `AscendMetadata.slot_mapping`，
而是读取 upstream `ForwardContext.slot_mapping`。

此前 `set_ascend_forward_context()` 没有把 Ascend runner 已经按 cache group 构建好的 per-layer
mapping 传给 upstream `set_forward_context()`。结果是 `unified_kv_cache_update()` 得到 `None` 后
静默跳过 store：首次 prefill 仍使用原始 K/V，所以首 token 正常；后续 decode 开始读取没有写入
有效数据的 TurboQuant cache，因此输出迅速失真。独立 store/decode 算子测试无法发现该问题，
因为这些测试会直接调用 store kernel。

当前修复在 Ascend forward-context 边界提取每层 metadata 中的 `slot_mapping`，并原样传给 vLLM。
不同 KV cache group 的 mapping 不会合并或复用。`check_environment.py` 同时检查当前 vLLM core 的
`set_forward_context()` 是否支持 `slot_mapping` 参数，避免 core 版本不匹配时再次静默失败。

修复后必须重新运行 eager 端到端对照。这个修复证明了 cache 未写入问题，但 NPU 实测通过前，
不能据此宣称量化精度已经达标。

1. 先在 `turboquant-triton-v0.20.2rc` 完成 native、all-skip、1/2/4/16-token 四组实验；
2. 给 accuracy collector 增加 chat token logprobs、原始答案和首个分歧 token输出；
3. 增加独立 CPU packer 和 raw FP32 attention oracle；
4. 使用真实 Qwen3 layer dump 验证 deterministic signed Hadamard，并与旧固定 Hadamard A/B；
5. 明确产品目标是简化的 `TurboQuant_mse`，还是包含 QJL/outlier path 的完整 TurboQuant；
6. reference 精度门禁通过后，再启用 grouped-GQA 和低精度计算；
7. 最后测试 ACLGraph、并发 16、16K context 和 Qwen3-32B TP=4。

在完成第 3 至第 5 项之前，kernel UT 全部通过只能证明当前实现内部一致，不能证明模型精度
正确，也不能证明已经复现论文中的完整 TurboQuant。
