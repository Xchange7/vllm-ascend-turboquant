# TurboQuant 算子测算脚本

本目录只测试 TurboQuant 的底层算子，不负责启动模型服务。测试覆盖：

- Triton 量化写入，包括负 `slot_mapping` 不写 Cache；
- AscendC paged-dequant 与 Triton dequant 的数值一致性；
- AscendC dequant + FIA 与直接读取压缩 Cache 的 Triton decode 一致性；
- Key/Value 量化误差，包括 MSE、NMSE、MAE、最大绝对误差和余弦相似度；
- store、decode、dequant、dequant + FIA 和原生 paged attention 的时延、吞吐及峰值显存；
- CANN/PyTorch NPU profiler trace。

## 运行前准备

910B4 构建时应明确设置：

```bash
export SOC_VERSION=ascend910b4
source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf build csrc/build
pip install -v -e . 2>&1 | tee build_910b4.log
```

`build_aclnn.sh` 中出现 `SOC_ARG=ascend910b` 和
`OPS_PRODUCT_NAME="ascend910b;"` 是正常现象。CANN 自定义算子按 910B 产品族构建，
而顶层 `SOC_VERSION` 仍然是 `ascend910b4`。

脚本默认使用单张 NPU 0。算子微基准不包含 TP/HCCL 通信，单卡测试能更准确地定位
算子本身。设置了 `ASCEND_RT_VISIBLE_DEVICES=2` 时，进程内可见设备通常会重新编号为
0，此时仍使用 `DEVICE=0`。

## 快速冒烟

```bash
SOC_VERSION=ascend910b4 DEVICE=0 \
bash scripts/turboquant_operators/run_smoke.sh
```

该入口测试三个 Cache preset 的 FP16 正确性，并运行一个 B=2、S=512 的短性能用例。

## 完整性能矩阵

默认使用 Qwen3-32B TP4 对应的本地 head 配置：Q heads=16、KV heads=2、D=128，
测试 B=1/4/16 和 S=512/2048/16384。

```bash
SOC_VERSION=ascend910b4 DEVICE=0 \
bash scripts/turboquant_operators/run_matrix.sh
```

可通过环境变量缩小或扩大矩阵：

```bash
CACHE_DTYPES="turboquant_4bit_nc turboquant_3bit_nc" \
ACTIVATION_DTYPES="float16 bfloat16" \
BATCH_SIZES="1 4 16" \
SEQUENCE_LENGTHS="2048 8192 16384" \
WARMUP=10 ITERATIONS=100 DEVICE=0 \
bash scripts/turboquant_operators/run_matrix.sh
```

如果只测性能，不重复正确性测试：

```bash
RUN_ACCURACY=0 bash scripts/turboquant_operators/run_matrix.sh
```

## Profiler

下面的命令会为每个算子生成 CPU/NPU profiler 数据：

```bash
BATCH_SIZE=4 SEQUENCE_LENGTH=4096 PROFILE_ITERATIONS=5 DEVICE=0 \
bash scripts/turboquant_operators/run_profile.sh
```

长上下文、高并发的 `fused_dequant` 会展开稠密 K/V，可能产生明显峰值显存。
建议先运行默认 case，再逐步增加到 B=16、S=16384。

## 结果解释

结果保存在 `logs/turboquant/operators_*_<timestamp>/`，并自动生成 `.tar.gz`：

- `environment.txt`：commit、SoC、软件版本和算子注册状态；
- `accuracy.json`：实现一致性和量化误差；
- `benchmark.json`：每个算子的 mean/P50/P90/P99、吞吐和峰值显存；
- `summary.md`、`summary.json`、`summary_*.csv`：聚合结果；
- `results.tsv`：矩阵中每个 case 的 PASS/FAIL；
- `profile/`：`torch_npu.profiler` 原始 trace。

主要指标含义：

- `K/V impl max` 应处于脚本给定容差内，它验证 AscendC 和 Triton 解码同一份 Cache；
- `Attention max` 验证 AscendC + FIA 和 packed Triton decode 的最终输出；
- `K/V NMSE` 衡量量化算法误差，不要求接近零；
- `Packed/native > 1` 表示压缩 Cache decode 比原生 paged attention 更快；
- `Ascend/Triton dequant > 1` 表示 AscendC dequant 比 Triton full-dequant 更快；
- `Ascend/packed decode > 1` 表示 dequant + FIA 比直接 packed decode 更快；
- `Compression` 是相同物理 token capacity 下的理论 KV Cache 压缩率。

把整个 `.tar.gz` 日志包回传即可进行后续定位。
