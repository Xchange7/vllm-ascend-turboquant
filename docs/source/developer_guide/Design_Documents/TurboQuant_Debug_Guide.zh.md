# TurboQuant / HIGGS Debug 手册

本文面向 vLLM Ascend 的 TurboQuant KV cache 与 HIGGS key quantizer
调试。当前实现仍是 reference store/dequant 路径，主要用于验证 packed cache
layout、backend 路由和短请求正确性；它还不是最终推理加速 kernel。

## 当前支持边界

默认优先验证以下组合：

- 模型：Qwen3 dense decoder 模型，例如 `Qwen3-32B`
- Engine：vLLM V1 model runner
- Attention：标准 dense decoder attention
- dtype：FP16/BF16 权重与激活
- block size：优先 `128`
- TurboQuant preset：
  - `turboquant_4bit_nc`
  - `turboquant_k3v4_nc`
  - `turboquant_3bit_nc`
  - `turboquant_k8v4` 仅用于 FP8 key 路径，不属于 HIGGS key quantizer

当前不建议在第一轮 debug 中启用：

- ACLGraph
- V2 model runner
- speculative decoding
- prefix caching
- KV transfer / disaggregated prefill
- sliding window attention
- MLA、sparse attention、compressed attention
- 长上下文或高并发性能压测

## 最小 smoke 流程

### 1. 确认代码路径

在 NPU 容器内先确认导入的是当前修改后的 vLLM Ascend：

```bash
python - <<'PY'
import vllm_ascend
print(vllm_ascend.__file__)
PY
```

如果输出不是预期路径，先重新安装或同步代码。

### 2. HIGGS-backed TurboQuant 启动

HIGGS 当前通过 TurboQuant 的非 FP8 key preset 进入：

```bash
MODEL_PATH=/models/Qwen3-32B \
TP_SIZE=8 \
scripts/higgs/start_server.sh
```

默认参数：

- `KV_CACHE_DTYPE=turboquant_4bit_nc`
- `LOAD_FORMAT=dummy`
- `MAX_MODEL_LEN=2048`
- `MAX_NUM_SEQS=1`
- `VLLM_USE_V2_MODEL_RUNNER=0`
- `--enforce-eager`

真实权重 smoke：

```bash
MODEL_PATH=/models/Qwen3-32B \
TP_SIZE=8 \
LOAD_FORMAT=real \
scripts/higgs/start_server.sh
```

只打印命令、不启动：

```bash
DRY_RUN=1 MODEL_PATH=/models/Qwen3-32B TP_SIZE=8 scripts/higgs/start_server.sh
```

切换 HIGGS preset：

```bash
KV_CACHE_DTYPE=turboquant_k3v4_nc scripts/higgs/start_server.sh /models/Qwen3-32B
KV_CACHE_DTYPE=turboquant_3bit_nc scripts/higgs/start_server.sh /models/Qwen3-32B
```

### 3. 发送请求

另开一个 shell：

```bash
SERVED_MODEL_NAME=higgs-smoke scripts/turboquant/smoke_request.sh
```

通过标准：

1. `/v1/models` 返回 200。
2. `/v1/chat/completions` 返回 200。
3. 返回 JSON 中有非空 `choices[0].message.content`。
4. 服务日志中没有 fatal error、cache shape error、NPU op unsupported error。

## Debug 决策树

### 启动前失败

#### 症状：`Unknown TurboQuant cache dtype`

常见原因：

- `KV_CACHE_DTYPE` 拼写错误。
- 本地 vLLM 版本没有 upstream TurboQuant config。
- fallback config 没有覆盖该 preset。

处理：

```bash
KV_CACHE_DTYPE=turboquant_4bit_nc \
DRY_RUN=1 \
scripts/higgs/start_server.sh /models/Qwen3-32B
```

确认脚本打印出的 `--kv-cache-dtype` 是合法 preset。HIGGS smoke 只使用
`turboquant_4bit_nc`、`turboquant_k3v4_nc`、`turboquant_3bit_nc`。

#### 症状：`TurboQuant KV cache currently supports dense decoder attention only`

常见原因：

- 模型或配置触发 MLA、sparse、compression backend。
- 当前 TurboQuant backend 被错误路由到非 dense attention。

处理：

1. 换 Qwen3 dense 模型做 baseline。
2. 禁用 compression/sparse 相关配置。
3. 确认日志中有 `Using Ascend TurboQuant attention backend.`。

#### 症状：`TurboQuant KV cache is not implemented for Ascend 310P`

当前实现没有接 310P runner 和 310P attention backend。先在 Ascend 910B/C
环境验证；310P 需要单独实现 cache allocation 和 attention kernel。

### 服务启动成功，但 `/v1/models` 等不到

常见原因：

- 权重加载慢或真实权重路径错误。
- TP size 和设备数量不匹配。
- torch_npu 初始化失败。
- 端口被占用。

处理顺序：

```bash
DRY_RUN=1 MODEL_PATH=/models/Qwen3-32B TP_SIZE=8 scripts/higgs/start_server.sh
```

确认命令正确后先跑 dummy：

```bash
MODEL_PATH=/models/Qwen3-32B \
LOAD_FORMAT=dummy \
TP_SIZE=8 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
scripts/higgs/start_server.sh
```

dummy 通过后再切到真实权重：

```bash
MODEL_PATH=/models/Qwen3-32B \
LOAD_FORMAT=real \
TP_SIZE=8 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
scripts/higgs/start_server.sh
```

不要一开始就把 `MAX_MODEL_LEN` 拉到生产值。

### 首个 chat 请求失败

#### 症状：`HIGGS Hadamard transform requires a power-of-two head_dim`

HIGGS reference quantizer 当前要求最后一维是 power-of-two。Qwen3-32B 的
`head_dim=128` 应该通过。

处理：

1. 检查模型 `config.json` 的 attention head 维度。
2. 如果模型 head_dim 不是 power-of-two，当前不要用 HIGGS preset。
3. 后续应实现 group-wise HIGGS，而不是在 TurboQuant 中写模型名特判。

#### 症状：cache shape 或 view 报错

常见原因：

- packed cache allocation 没走 `TQFullAttentionSpec` 路径。
- `get_kv_cache_shape()` 的 `slot_size_aligned` 和实际 raw allocation 不一致。
- 使用了 V2 model runner。

处理：

1. 确认 `VLLM_USE_V2_MODEL_RUNNER=0`。
2. 确认 backend 日志是 `Ascend TurboQuant attention backend`。
3. 用 `DRY_RUN=1` 确认 `--kv-cache-dtype turboquant_4bit_nc` 已传入。
4. 降到 `MAX_MODEL_LEN=2048`、`MAX_NUM_SEQS=1` 重跑。

#### 症状：`turboquant_k8v4 requires torch.float8_e4m3fn support`

`turboquant_k8v4` 是 FP8 key 路径，不是 HIGGS key quantizer。当前 NPU smoke
优先不要使用它。

处理：

```bash
KV_CACHE_DTYPE=turboquant_4bit_nc scripts/higgs/start_server.sh /models/Qwen3-32B
```

如果必须验证 k8v4，需要单独确认当前 torch_npu 对 `float8_e4m3fn` cast 的支持。

### decode 很慢或显存上涨明显

这是当前 reference backend 的预期限制，不一定是 bug。

当前 decode 路径会把 packed TurboQuant cache 全量反量化成标准 K/V cache，然后
调用现有 Ascend attention。它能验证功能，但不会体现最终 TurboQuant 加速收益。

处理：

1. smoke 阶段保持短上下文：

   ```bash
   MAX_MODEL_LEN=2048 MAX_NUM_SEQS=1 scripts/higgs/start_server.sh /models/Qwen3-32B
   ```

2. 不要用当前 reference backend 做吞吐或长上下文结论。
3. 真正加速需要实现 packed-cache NPU decode attention kernel，避免全量
   `turboquant_dequant_cache()`。

### ACLGraph 相关失败

常见症状：

- graph capture 期间 shape 不稳定。
- replay 后输出异常。
- 日志中出现 graph 参数或 weak ref 相关错误。

原因：

当前 reference decode 每步会 materialize 新的 FP16/BF16 K/V tensor，这和
ACLGraph 需要稳定 tensor 地址与 shape 的假设冲突。

处理：

1. smoke 阶段保留 `--enforce-eager`。
2. 不要在当前 reference backend 上强行打开 ACLGraph。
3. 等 packed-cache decode kernel 落地后，再重新评估 ACLGraph。

### dummy 通过，真实权重失败

常见原因：

- 权重路径或 safetensors index 问题。
- TP size 与 checkpoint shard 不匹配。
- 模型 remote code 或 tokenizer 配置问题。
- 真实权重触发了 dummy 不覆盖的 dtype/cast/shape。

处理：

1. 先用普通 BF16 KV cache 验证模型能否在 vLLM Ascend 启动。
2. 再打开 `--kv-cache-dtype turboquant_4bit_nc`。
3. 保持 `MAX_MODEL_LEN=2048`、`MAX_NUM_SEQS=1`。
4. 如果 BF16 能过、TurboQuant 失败，重点看 cache shape、HIGGS head_dim、
   FP8 cast、slot_mapping。

## 日志检查点

启动日志应包含：

```text
Using Ascend TurboQuant attention backend.
```

脚本应打印：

```text
HIGGS-backed KV cache dtype: turboquant_4bit_nc
VLLM_USE_V2_MODEL_RUNNER=0
ENFORCE_EAGER=1
```

如果没有看到这些信息，先不要继续做性能或真实权重验证。

## 建议的排查命令

查看 vLLM Ascend 导入路径：

```bash
python - <<'PY'
import vllm_ascend
print(vllm_ascend.__file__)
PY
```

打印启动命令：

```bash
DRY_RUN=1 MODEL_PATH=/models/Qwen3-32B TP_SIZE=8 scripts/higgs/start_server.sh
```

最小 HIGGS dummy：

```bash
MODEL_PATH=/models/Qwen3-32B \
LOAD_FORMAT=dummy \
TP_SIZE=8 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
scripts/higgs/start_server.sh
```

最小真实权重：

```bash
MODEL_PATH=/models/Qwen3-32B \
LOAD_FORMAT=real \
TP_SIZE=8 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
scripts/higgs/start_server.sh
```

请求 smoke：

```bash
SERVED_MODEL_NAME=higgs-smoke scripts/turboquant/smoke_request.sh
```

## 修复优先级

如果目标是推理加速，建议按以下顺序修：

1. 加强能力门控：启动阶段拒绝 ACLGraph、V2 runner、sliding window、
   speculative decoding、KV transfer、prefix caching。
2. 为 HIGGS/TurboQuant 增加 backend routing 和 model runner allocation UT。
3. 实现 `ascend_turboquant_store` NPU custom op。
4. 实现 packed-cache `ascend_turboquant_decode_attention`，避免全量 dequant。
5. decode kernel 稳定后，再打开 ACLGraph 和长上下文性能验证。

## 上报 bug 时需要的信息

请至少提供：

- 完整启动命令或 `DRY_RUN=1` 输出。
- `MODEL_PATH` 和模型 `config.json` 中 attention 相关字段。
- `KV_CACHE_DTYPE`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`TP_SIZE`。
- 是否 `LOAD_FORMAT=dummy`。
- 是否 `--enforce-eager`。
- `/v1/models` 是否成功。
- 首个 `/v1/chat/completions` 的错误栈。
- 日志中是否出现 `Using Ascend TurboQuant attention backend.`。
- 如果是 HIGGS preset，确认不是 `turboquant_k8v4`。
