# TurboQuant Ascend 融合算子设计与验证

## 1. 目标与范围

当前 TurboQuant decode 由 Triton-Ascend kernel 直接读取 packed KV cache，并在一个 kernel 内完成
解包、反量化、QK、softmax 和 PV。该路径保持了压缩数据的低带宽优势，但在 910B4 高并发场景
出现严重的标量地址计算和调度开销。

本轮新增 `TurboQuantPagedDequant` AscendC 算子，将下列操作融合到一次设备 kernel 中：

1. 按 `block_table` 将 logical page 映射到 physical page；
2. 按 TurboQuant slot layout 读取 packed K/V；
3. 解包 3-bit 或 4-bit K centroid index；
4. 执行 Lloyd-Max centroid lookup 和 key norm correction；
5. 解包 3-bit 或 4-bit V，并应用 FP16 scale/minimum；
6. 直接输出 CANN FIA 使用的 dense `BNSD` K/V。

随后 attention backend 将 Q 旋转到与 packed K 相同的 randomized Hadamard 坐标，并调用
`npu_fused_infer_attention_score`。正交变换满足

```text
(Q R) (K R)^T = Q K^T
```

因此不需要先将 dense K 逆旋转回原坐标，也不会改变 attention 数学定义。

该实现是 **AscendC 融合反量化 + CANN FIA** 两段式路径，不是最终的单 kernel TurboQuant
attention。它首先用于验证高并发瓶颈是否来自现有 Triton packed attention。

## 2. 代码结构

| 文件 | 职责 |
| --- | --- |
| `csrc/attention/turboquant_paged_dequant/op_kernel/turboquant_paged_dequant.cpp` | AIV kernel：paged gather、bit unpack、centroid lookup、norm correction 和 V 反量化 |
| `csrc/attention/turboquant_paged_dequant/op_host/` | OpDef、shape/dtype inference 和 tiling |
| `csrc/torch_binding.cpp` | 注册 `_C_ascend.npu_turboquant_paged_dequant` 并分配 BNSD 输出 |
| `csrc/torch_binding_meta.cpp` | PyTorch Meta 实现 |
| `vllm_ascend/ops/turboquant.py` | Python 可用性检查和稳定调用接口 |
| `vllm_ascend/attention/turboquant.py` | 单 token decode 路由、Q rotation 和 FIA 调用 |
| `tests/ut/ops/test_turboquant_triton.py` | AscendC 对 Triton dequant、AscendC+FIA 对 packed decode 的数值门禁 |
| `scripts/turboquant_triton/diagnostics/run_ascend_fused_validation.sh` | 910B4 correctness、性能矩阵、日志和归档入口 |

## 3. 算子契约

输入：

- `query`: `[B, Nq, D]`，仅用于确定输出 dtype、batch 和 head dimension；
- `kv_cache`: `[num_blocks, block_size, Nkv, slot_size]`，dtype 为 `uint8`；
- `block_table`: `[B, max_pages]`，dtype 为 `int32`；
- `seq_lens`: `[B]`，dtype 为 `int32`；
- `centroids`: `[2 ** key_bits]`，dtype 为 `float32`；
- attributes: `max_seq_len`、`key_bits`、`key_packed_size`、`value_bits` 和
  `norm_correction`。

输出：

- `key`: `[B, Nkv, max_seq_len, D]`；
- `value`: `[B, Nkv, max_seq_len, D]`；
- dtype 与 `query` 相同，布局为 `BNSD`。

kernel 以 `(batch, logical_page, kv_head)` 为 task。每个 AIV core 以 stride 方式处理若干 page-head，
在 UB 中复用 slot、centroid、index、FP32 dequant 和输出 buffer。无效 page 和负 physical block
会被跳过。

## 4. 当前支持边界

当前支持：

- Ascend 910B/910B4 和 910C 的构建注册；
- `turboquant_4bit_nc`、`turboquant_k3v4_nc` 和 `turboquant_3bit_nc`；
- FP16/BF16 activation；
- `head_dim` 为 32 的倍数且位于 `[32, 256]`；
- MHA/GQA、非连续 physical pages 和不同 request sequence length；
- V1 runner 的 eager、uniform single-token decode。

当前不进入该路径：

- ACLGraph；选择 `ascend_fused` 时 metadata builder 声明 `AttentionCGSupport.NEVER`；
- multi-token/spec decode 和 continuation prefill；它们自动回退到 `auto` Triton 路径；
- ALiBi 和 logits soft cap；显式选择融合路径时会 fail fast；
- sliding window、sinks、MLA、CP、310P 和 V2 model runner。

该路径不会改变持久 KV cache 的压缩率，但会为每层 attention 临时展开 dense K/V。临时空间为：

```text
2 * B * Nkv * max_seq_len * D * activation_element_size
```

例如 Qwen3-32B TP4、并发 16、16K context、`Nkv=2`、`D=128`、BF16 时约为 256 MiB。
临时 tensor 不保存在 layer 上，由 NPU caching allocator 跨层复用。

## 5. 构建与启动

该改动包含 C++/AscendC 源码，拉取代码后必须重新构建，不能只依赖之前的 editable install：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -e .
```

确认 operator 已注册：

```bash
python3 - <<'PY'
from vllm_ascend.ops.turboquant import has_turboquant_paged_dequant
assert has_turboquant_paged_dequant()
PY
```

启动 eager server：

```bash
export VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=ascend_fused
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

MODEL=/run/test_llm/Qwen3-32B \
TP_SIZE=4 MAX_MODEL_LEN=16384 MAX_NUM_SEQS=16 ENFORCE_EAGER=1 \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh
```

不要在该模式下设置 `ENFORCE_EAGER=0`。ACLGraph 继续使用默认 `auto` Triton 路径。

## 6. 910B4 验证

完整算子验证：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
BATCH_SIZES="1 4 16" SEQUENCE_LENGTHS="2048 16384" \
bash scripts/turboquant_triton/diagnostics/run_ascend_fused_validation.sh
```

默认输出到 `logs/turboquant/ascend_fused_<timestamp>/`，并生成同名 `.tar.gz`。主要检查：

1. 三种 preset 的 AscendC dense K/V 与 Triton reference 是否在 FP16/BF16 容差内一致；
2. AscendC+FIA 输出与 packed decode 是否一致；
3. `fused_dequant` 相对 `dequant` 的 kernel 时间；
4. `fused_decode` 相对 `decode` 的 mean、P50、P90、P99 和吞吐；
5. dense 临时 K/V 的峰值显存是否符合公式。

端到端比较：

```bash
VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=ascend_fused \
ENFORCE_EAGER=1 MODEL=/run/test_llm/Qwen3-32B \
TP_SIZE=4 CONCURRENCY=16 MAX_NUM_SEQS=16 MAX_MODEL_LEN=16384 \
bash scripts/turboquant_triton/performance/run_serving_benchmark.sh
```

## 7. 验收与后续优化

该路径必须同时满足以下条件，才能成为 `auto` 的候选：

- 所有 fused correctness case 通过；
- Qwen3-0.6B 和 Qwen3-32B 的固定 prompt 输出不出现新增精度回归；
- 并发 1/4/16 的 TPOT 和吞吐均优于当前 packed Triton decode；
- 16K context 不出现 OOM 或 allocator 持续增长；
- profiling 中热点位于预期的 AscendC dequant 和 FIA，而不是 host sync 或重复内存分配。

如果 FIA 路径仍受 dense K/V HBM 写回限制，下一阶段应基于现有
`sparse_flash_attention` 的 Cube/Vector pipeline 实现真正的 TurboQuant attention：page tile 在 UB
中解包并反量化，Cube 计算 QK/PV，Vector 执行 online softmax，不将完整 dense K/V 写回 HBM。
在该单 kernel 版本通过实机数值与性能门禁前，`ascend_fused` 保持显式 opt-in。
