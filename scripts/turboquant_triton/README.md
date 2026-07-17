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

## 1. Check environment and kernels

Run from the repository root:

```bash
bash scripts/turboquant_triton/run_kernel_smoke.sh
```

This checks the exact vLLM interface, compiles the Triton-Ascend kernels,
verifies negative slot mappings do not write cache, and compares packed decode
against a dequantized attention reference.

For a complete diagnostic run with logs suitable for offline debugging, use:

```bash
bash scripts/turboquant_triton/run_diagnostic_suite.sh
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
bash scripts/turboquant_triton/run_910b4_retest.sh
```

The retest uses port `18000` by default and refuses to start if any process is
already listening there. Override it with `PORT=<free-port>` when necessary.
Kernel stages use synchronous launches for precise error attribution, while
the model stage removes `ASCEND_LAUNCH_BLOCKING` by default. Set
`MODEL_DEBUG_SYNC=1` only when diagnosing an asynchronous model-level error.

To run correctness, eager/ACLGraph model comparison, native/TurboQuant
accuracy comparison, and the kernel performance matrix in one command:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
TP_SIZE=2 \
bash scripts/turboquant_triton/run_npu_validation.sh
```

The combined suite writes one archive under `logs/turboquant/`. See
`docs/source/developer_guide/Design_Documents/turboquant_npu_validation_zh.md`
for stage definitions, report interpretation, and reduced test matrices.

## 2. Start Qwen3-32B

The default uses tensor parallel size 2 because BF16 Qwen3-32B weights usually
do not leave useful KV-cache capacity on one 64 GB device:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
bash scripts/turboquant_triton/serve_qwen3_32b.sh
```

Use a small initial context while validating correctness. Parameters can be
overridden through the environment:

```bash
TP_SIZE=4 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=8 \
KV_CACHE_DTYPE=turboquant_4bit_nc \
bash scripts/turboquant_triton/serve_qwen3_32b.sh
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
bash scripts/turboquant_triton/run_aclgraph_smoke.sh
```

Then start the server without `--enforce-eager`:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL=/path/to/Qwen3-32B \
ENFORCE_EAGER=0 \
bash scripts/turboquant_triton/serve_qwen3_32b.sh
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
bash scripts/turboquant_triton/serve_qwen3_32b.sh
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
bash scripts/turboquant_triton/smoke_request.sh
```

For the first NPU run, set `ASCEND_LAUNCH_BLOCKING=1` only when diagnosing a
kernel failure. It makes the failing launch easier to locate but should not be
used for performance measurements.

## 5. Profile the Triton kernels

Profile store, packed decode, and the full-dequant fallback with Qwen3-style
head dimensions:

```bash
bash scripts/turboquant_triton/profile_kernels.sh \
    --operation all \
    --cache-dtype turboquant_4bit_nc \
    --batch-size 4 \
    --sequence-length 4096 \
    --num-query-heads 32 \
    --num-kv-heads 4 \
    --head-dim 128
```

Run the compact compatibility/performance matrix for all presets, FP16/BF16,
head dimensions 64/128/256, and split counts 8/16/32:

```bash
bash scripts/turboquant_triton/profile_matrix.sh
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
    bash scripts/turboquant_triton/profile_kernels.sh \
        --operation decode \
        --sequence-length "${length}" \
        --trace-dir "profiles/tq_decode_${length}"
done
```

Use `--no-trace` for lower-overhead timing sweeps. Do not set
`ASCEND_LAUNCH_BLOCKING=1` while collecting performance numbers.
