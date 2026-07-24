# TurboQuant Triton-Ascend smoke test

This directory validates the Ascend TurboQuant implementation in increasing
order of cost. The implementation requires vLLM `0.20.2` or
`0.20.2+empty`, which already contains the TurboQuant cache dtype, cache spec,
centroids, and layer workspaces.

## Prerequisites

- Ascend 910B-series NPU with a matching CANN runtime.
- Python environment with this branch of vLLM Ascend installed.
- `torch==2.10.0`, `torch-npu==2.10.0`, and `triton-ascend==3.2.1`.
- vLLM `0.20.2` or `0.20.2+empty` from the same environment used to launch
  the server.
- Qwen3-0.6B or Qwen3-32B weights accessible through `MODEL`.

Install this checkout after installing the matching vLLM core:

```bash
pip install -e .
```

TurboQuant compresses only the runtime KV cache. It does not require a
ModelSlim calibration artifact and does not quantize model weights.

The experimental AscendC paged-dequant + CANN FIA decode path requires a
rebuild and eager execution. Its design, memory tradeoff, and 910B4 validation
procedure are documented in
`docs/source/developer_guide/Design_Documents/turboquant_ascend_fused_operator_zh.md`.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -e .
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/turboquant_triton/diagnostics/run_ascend_fused_validation.sh
```

## Directory layout

| Directory | Purpose |
| --- | --- |
| `common/` | Environment checks, model server launcher, and request helper |
| `correctness/` | Kernel, ACLGraph, output-quality, and accuracy comparisons |
| `performance/` | Operator profiling, cache capacity, TTFT, TPOT, and throughput |
| `diagnostics/` | Startup debugging, 910B4 retest, and combined validation suites |

All shell entrypoints resolve the repository root through `common/paths.sh`,
so they can be launched from any working directory.

For one exhaustive Ascend 910B4 run across devices 0-3, use the full
orchestrator. It validates the loaded editable checkout, runs isolated kernel
and ACLGraph tests, profiles the Qwen3-32B TP4 operator shape, compares native,
reference, and grouped-auto quality, and measures eager/ACLGraph serving:

```bash
MODEL=/run/test_llm/Qwen3-32B \
VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
bash scripts/turboquant_triton/diagnostics/run_full_910b4_validation.sh
```

The complete matrix starts several model servers and can take hours. Set
`QUICK=1` for kernel, operator, basic accuracy, and eager serving only. Every
stage has an individual `RUN_*` switch, and all logs, periodic `npu-smi`
snapshots, raw answers, benchmark JSON, summaries, and server logs are packed
under `logs/turboquant/full_910b4_<timestamp>.tar.gz`.

## 1. Check environment and kernels

Run from the repository root:

```bash
bash scripts/turboquant_triton/correctness/run_kernel_smoke.sh
```

This checks the exact vLLM interface, compiles the Triton-Ascend kernels,
verifies negative slot mappings do not write cache, and compares packed decode
against a dequantized attention reference.

For a complete diagnostic run with logs suitable for offline debugging, use:

```bash
bash scripts/turboquant_triton/diagnostics/run_diagnostic_suite.sh
```

The suite continues after individual failures and stores the environment,
source revision, backend tests, Triton tests, ACLGraph replay test, and a short
profile under `logs/turboquant/diagnostic_<timestamp>/`. It also creates a
`.tar.gz` archive next to that directory. Set `RUN_PROFILE=0` to skip the short
profile, `DEBUG_SYNC=1` to diagnose an asynchronous kernel failure, or
`COLLECT_PROFILE_TRACE=1` to include torch-npu profiler traces.

After a store-kernel alignment failure, use the isolated 910B4 retest. Each
positive-slot store preset runs in a fresh pytest process before the full
kernel, ACLGraph, and model checks:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL=/run/test_llm/Qwen3-0.6B-hf \
TP_SIZE=1 \
bash scripts/turboquant_triton/diagnostics/run_910b4_retest.sh
```

The retest uses port `18000` by default and refuses to start if any process is
already listening there. Override it with `PORT=<free-port>` when necessary.
Kernel stages use synchronous launches for precise error attribution, while
the model stage removes `ASCEND_LAUNCH_BLOCKING` by default. Set
`MODEL_DEBUG_SYNC=1` only when diagnosing an asynchronous model-level error.

If the model stage stays alive without allocating NPU memory, bypass the
native comparison and run one TurboQuant server with a startup watchdog:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL=/run/test_llm/Qwen3-0.6B-hf \
bash scripts/turboquant_triton/diagnostics/run_model_startup_debug.sh
```

The watchdog prints the process state, child-process tree, latest server log,
and NPU state every 30 seconds. Text logs are not excluded by the repository's
`*.log` ignore rule. It defaults `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and
`HCCL_SOCKET_IFNAME` to `eth0`, then requires a single-rank Gloo probe to
finish within 30 seconds before starting vLLM. Set
`NETWORK_IFNAME=<interface>` to select another NIC.

To run correctness, eager/ACLGraph model comparison, native/TurboQuant
accuracy comparison, and the kernel performance matrix in one command:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=2 \
bash scripts/turboquant_triton/diagnostics/run_npu_validation.sh
```

The combined suite writes one archive under `logs/turboquant/`. See
`docs/source/developer_guide/Design_Documents/turboquant_npu_validation_zh.md`
for stage definitions, report interpretation, and reduced test matrices.

To compare end-to-end KV-cache capacity, TTFT, and TPOT with prefix caching
disabled, run the serving benchmark:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL=/run/test_llm/Qwen3-0.6B-hf \
TP_SIZE=1 \
bash scripts/turboquant_triton/performance/run_serving_benchmark.sh
```

It starts native and TurboQuant servers one at a time, performs two warmup and
ten measured streaming requests by default, and writes raw samples, server
logs, `comparison.json`, and `summary.md` under `logs/turboquant/`. The summary
derives the effective KV-cache compression ratio from the cache token capacity
reported by each server. Override `INPUT_TOKENS`, `OUTPUT_TOKENS`,
`WARMUP_REQUESTS`, or `MEASURE_REQUESTS` to change the workload. Keep
`INPUT_TOKENS + OUTPUT_TOKENS <= MAX_MODEL_LEN`.

For Qwen3-32B on four NPUs with 16 concurrent sequences, use at least one
full concurrent warmup wave and several measured waves:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=4 CONCURRENCY=16 MAX_NUM_SEQS=16 \
WARMUP_REQUESTS=16 MEASURE_REQUESTS=64 \
bash scripts/turboquant_triton/performance/run_serving_benchmark.sh
```

On a four-card Ascend 910B4 server, the dedicated TP4 entrypoint validates all
four device models, profiles the TP4 per-rank attention shape, and then runs an
actual Qwen3-32B TP4 native/TurboQuant serving A/B:

```bash
MODEL=/path/to/Qwen3-32B \
bash scripts/turboquant_triton/performance/run_qwen3_32b_tp4_910b4.sh
```

It defaults to 16 concurrent sequences, 12,288 input tokens, 256 generated
tokens, an 8,192-token chunked-prefill budget, one full warmup wave, and 64
measured requests. It records TTFT, TPOT, request and token throughput,
KV-cache capacity, server logs, operator reports, and periodic `npu-smi`
snapshots under
`logs/turboquant/qwen3_32b_tp4_910b4_<timestamp>/`. To collect a concurrency
curve instead, set `CONCURRENCY_LEVELS="1 4 8 16"`; `MAX_NUM_SEQS` is inferred
from the largest level. Set `RUN_OPERATOR_PROFILE=0` or `RUN_SERVING=0` to run
only one part. `DRY_RUN=1` validates arguments and prints the commands without
requiring model weights or NPU access. The script clears an inherited
`HCCL_IF_IP` so it cannot conflict with `NETWORK_IFNAME`; set
`HCCL_IF_IP_OVERRIDE=<address>` only when the deployment requires an explicit
interface address.

To compare ground-truth quality and hallucination resistance, run the chat
quality suite. It grades native and TurboQuant independently, then treats a
native-correct/TurboQuant-wrong case as a quantization regression:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=4 \
bash scripts/turboquant_triton/correctness/run_quality_comparison.sh
```

The cases cover factual recall, arithmetic, reasoning, context retrieval,
long-context distractors, missing-information refusal, prompt injection,
instruction following, and structured output. Reports are written under
`logs/turboquant/accuracy_*`. The default fails when any quality regression is
observed; set `MAX_QUALITY_REGRESSIONS` only after manually reviewing the raw
native and TurboQuant answers.

For the full non-circular end-to-end correctness comparison, run five real
servers sequentially. This combines teacher-forced target-token NLL,
first-token top-k distributions, deterministic generation, and ground-truth
grading:

```bash
MODEL=/run/test_llm/Qwen3-0.6B-hf \
DEVICE_IDS=0 TP_SIZE=1 \
bash scripts/turboquant_triton/correctness/run_e2e_correctness.sh
```

The default modes are `native`, `native_repeat`, `tq_reference`, `tq_auto`,
and `tq_ascend_fused`. The key comparisons are:

- `native -> native_repeat`: runtime and scheduling noise floor.
- `native -> tq_reference`: TurboQuant cache format and reference Triton path.
- `tq_reference -> tq_auto`: drift from the production automatic path. With
  the custom operator installed, auto disables graph capture so single-token
  decode selects AscendC paged dequantization plus CANN FIA. Unsupported shapes
  use grouped GQA with activation-dtype rotation; select `grouped_gqa` explicitly
  when validating ACLGraph.
- `tq_auto -> tq_ascend_fused`: verifies that explicit fused mode matches the
  automatic eager dispatch and fails fast when the operator is unavailable.

Prefix caching and ACLGraph are disabled so the first run isolates KV-cache
quantization. Long teacher-forcing inputs use 512-token chunked prefill so
later chunks consume previously compressed cache pages. Raw requests,
responses, answers, server logs, pairwise reports, and one top-level
`summary.md` are archived under
`logs/turboquant/e2e_correctness_<timestamp>.tar.gz`. See
`docs/source/developer_guide/Design_Documents/turboquant_e2e_correctness_zh.md`
for metric definitions and threshold guidance.

## 2. Start Qwen3-32B

The default uses tensor parallel size 2 because BF16 Qwen3-32B weights usually
do not leave useful KV-cache capacity on one 64 GB device:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh
```

Use a small initial context while validating correctness. Parameters can be
overridden through the environment:

```bash
TP_SIZE=4 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=8 \
KV_CACHE_DTYPE=turboquant_4bit_nc \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh
```

The initial implementation supports `turboquant_4bit_nc`,
`turboquant_k3v4_nc`, and `turboquant_3bit_nc`. FP8-key TurboQuant is rejected
explicitly. The default remains eager mode for the first correctness run.

Prefix caching is not disabled: packed pages support ordinary block-table
reuse and the backend implements whole-page copies. Treat it as experimental
until prefix-hit, eviction, preemption, and mixed native/TurboQuant boundary
layers have passed end-to-end NPU coverage.

## 3. Validate and enable ACLGraph

First run the focused metadata and kernel capture/replay checks:

```bash
bash scripts/turboquant_triton/correctness/run_aclgraph_smoke.sh
```

Then start the server without `--enforce-eager`:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
ENFORCE_EAGER=0 \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh
```

When `ENFORCE_EAGER=0`, the script explicitly selects
`FULL_DECODE_ONLY`. Override `COMPILATION_CONFIG` to test another graph mode
or a custom set of capture sizes.

Exercise a four-token uniform verification step with n-gram speculation:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
ENFORCE_EAGER=0 \
SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":1,"prompt_lookup_max":4}' \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh
```

Use a prompt containing repeated token sequences so the n-gram proposer can
produce drafts. Server logs and profiler traces should show graph replay at a
uniform query length of four; requests with no usable draft fall back to the
runtime-selected eager or piecewise path.

ACLGraph support is advertised as uniform single-token or multi-token decode.
The multi-token path reuses request-sized split workspaces one token step at a
time, so graph memory does not scale by the draft length. Standard speculative
verification can use this path; parallel drafting remains unsupported.
Prefill, mixed prefill/decode batches, and context parallel execution are not
captured by the TurboQuant backend. Compare deterministic eager and graph
responses before collecting performance data.

## 4. Send a request

In a second shell:

```bash
MODEL=/path/to/Qwen3-32B \
bash scripts/turboquant_triton/common/smoke_request.sh
```

For the first NPU run, set `ASCEND_LAUNCH_BLOCKING=1` only when diagnosing a
kernel failure. It makes the failing launch easier to locate but should not be
used for performance measurements.

## 5. Profile the Triton kernels

Profile store, packed decode, and the full-dequant fallback with Qwen3-style
head dimensions:

```bash
bash scripts/turboquant_triton/performance/profile_kernels.sh \
    --operation all \
    --cache-dtype turboquant_4bit_nc \
    --batch-size 4 \
    --sequence-length 4096 \
    --num-query-heads 32 \
    --num-kv-heads 4 \
    --head-dim 128
```

`--operation decode` measures packed attention only. Use
`--operation decode_step` to include the current-token TurboQuant cache write,
which is the closer approximation of one attention layer during generation.
Select `--decode-implementation reference` to compare the original per-query-
head kernel with the default grouped-GQA path.

For the Qwen3-32B TP4 production shape (16 local query heads, 2 local KV
heads, batch 16, and 16K context), run:

```bash
bash scripts/turboquant_triton/performance/profile_qwen3_32b_tp4.sh
```

This records pure decode and full decode-step timings for grouped `BLOCK_KV`
16/32 and the reference implementation. Native paged attention is included in
the pure-decode cases. The script writes all reports and logs under
`logs/turboquant/qwen3_32b_tp4_<timestamp>/`.

When throughput collapses only at higher concurrency, sweep batch size and
split count independently:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/turboquant_triton/performance/profile_concurrency_splits.sh
```

The default matrix uses the Qwen3-32B TP4 per-rank shape at 16K context and
tests `B=1/2/4/8/16` with fixed split counts `1/2/4/8/16/32`. It also runs the
production adaptive policy once per batch. Results and per-case
`benchmark.json` files are archived under
`logs/turboquant/concurrency_splits_<timestamp>.tar.gz`. Use the split with the
lowest `decode.mean_ms` at each batch as the hardware evidence for tuning the
policy; a large gap between every TurboQuant case and `native_decode` points
to the packed paged-load kernel rather than split selection.

Production serving selects grouped GQA and activation-dtype rotation
automatically. To isolate either optimization without changing code, start the
server with `reference`; this restores the per-query-head kernel and FP32
key/query rotation:

```bash
VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION=reference \
bash scripts/turboquant_triton/common/serve_qwen3_32b.sh
```

Run the compact compatibility/performance matrix for all presets, FP16/BF16,
head dimensions 64/128/256, and split counts 8/16/32:

```bash
bash scripts/turboquant_triton/performance/profile_matrix.sh
```

Set `PROFILE_ROOT`, `ITERATIONS`, or `SEQUENCE_LENGTH` to override the matrix
defaults. Each case writes an independent `benchmark.json` without collecting
large profiler traces.

Head counts are per TP rank. For Qwen3-32B, use `64/8` query/KV heads for
TP1, `32/4` for TP2, or `16/2` for TP4.

The script performs unprofiled warmup, measures 50 iterations with NPU events,
and captures five separate profiler iterations. Results are written under
`profiles/turboquant_<timestamp>/`:

- `benchmark.json` contains mean/P50/P90/P99 latency, throughput, parameters,
  the allocated-memory baseline, the peak increase during each operation,
  native paged-attention comparison, and the measured decode speedup.
- `store/`, `decode/`, and `dequant/` contain `torch_npu.profiler` traces.

Run only packed decode when comparing sequence lengths:

```bash
for length in 1024 4096 8192 16384; do
    bash scripts/turboquant_triton/performance/profile_kernels.sh \
        --operation decode \
        --sequence-length "${length}" \
        --trace-dir "profiles/tq_decode_${length}"
done
```

Use `--no-trace` for lower-overhead timing sweeps. Do not set
`ASCEND_LAUNCH_BLOCKING=1` while collecting performance numbers.
