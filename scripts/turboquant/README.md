# TurboQuant Smoke Test

这组脚本用于在 Ascend NPU 环境上做 TurboQuant 的最小功能验证。目标是先确认
`--kv-cache-dtype turboquant_*` 能完成服务启动、`/v1/models` readiness 和一次
OpenAI chat 请求。

当前 TurboQuant backend 还是 reference store/dequant 实现，不适合直接做长上下文
性能评估。第一次 smoke test 建议使用短上下文、单并发、eager 模式。

## 1. 准备环境

在官方 vllm-ascend 容器内，确认运行时导入的是你要测试的 vllm-ascend 代码：

```bash
python - <<'PY'
import vllm_ascend
print(vllm_ascend.__file__)
PY
```

如果输出不是你已更新的 vllm-ascend 路径，需要先安装或同步代码到运行环境。
不要用 ModelSlim。TurboQuant 是 KV cache 运行时量化，入口是
`--kv-cache-dtype turboquant_*`。

## 2. 启动服务

推荐先跑 dummy smoke。它能快速验证模型结构、attention backend、KV cache
allocation/reshape 和 TurboQuant cache update 路径，但不等价于真实权重验证。

```bash
MODEL_PATH=/models/Qwen3-32B \
TP_SIZE=8 \
scripts/turboquant/start_server.sh
```

默认参数：

- `LOAD_FORMAT=dummy`
- `KV_CACHE_DTYPE=turboquant_4bit_nc`
- `MAX_MODEL_LEN=2048`
- `MAX_NUM_SEQS=1`
- `PORT=8000`
- `VLLM_USE_V2_MODEL_RUNNER=0`
- `--enforce-eager`

真实权重 smoke：

```bash
MODEL_PATH=/models/Qwen3-32B \
LOAD_FORMAT=real \
TP_SIZE=8 \
scripts/turboquant/start_server.sh
```

只打印启动命令、不实际启动：

```bash
DRY_RUN=1 MODEL_PATH=/models/Qwen3-32B TP_SIZE=8 scripts/turboquant/start_server.sh
```

切换 TurboQuant preset：

```bash
KV_CACHE_DTYPE=turboquant_k8v4 scripts/turboquant/start_server.sh /models/Qwen3-32B
KV_CACHE_DTYPE=turboquant_k3v4_nc scripts/turboquant/start_server.sh /models/Qwen3-32B
KV_CACHE_DTYPE=turboquant_3bit_nc scripts/turboquant/start_server.sh /models/Qwen3-32B
```

## 3. 发送 smoke 请求

服务启动后，另开一个 shell：

```bash
scripts/turboquant/smoke_request.sh
```

指定端口和模型名：

```bash
PORT=8000 SERVED_MODEL_NAME=turboquant-smoke scripts/turboquant/smoke_request.sh
```

通过标准：

1. `/v1/models` 返回 200。
2. `/v1/chat/completions` 返回 200。
3. 响应里包含非空 `choices`。
4. 服务日志没有 fatal error、cache shape error、NPU op unsupported error。

## 4. 推荐排查顺序

如果启动失败：

1. 保持 `MAX_MODEL_LEN=2048`、`MAX_NUM_SEQS=1`、`--enforce-eager` 重跑一次。
2. 确认 `VLLM_USE_V2_MODEL_RUNNER=0` 生效。当前 TurboQuant 只接了 V1 runner。
3. 先用 `LOAD_FORMAT=dummy` 排除权重加载问题。
4. dummy 通过后，再用 `LOAD_FORMAT=real` 验证真实权重。
5. 真实权重通过后，再逐步增大 `MAX_MODEL_LEN`。

不要在第一轮就开 ACLGraph、V2 runner、长上下文或高并发。当前版本主要验证功能
通路，不代表最终 TurboQuant 性能。
