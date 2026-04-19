# C Micro-Bench

This directory contains focused INT8 micro-bench designs used to probe the
current stable C accel flow.

Numeric policy for INT8-producing stages:
- accumulate into wide `int32_t` temporaries
- keep GEMM / conv contraction loops numerically pure
- perform narrowing only in separate epilogue loops
- use a uniform inline requantization step in those epilogues:
  `value = (value + rounding) >> shift;` followed by clamp
- keep softmax-specific integer normalization as an allowed exception

Current validation note:
- this policy gives deterministic narrowing behavior and keeps simple FC/direct-conv
  cases mappable in the current repo-local accel flow
- current repo-local validation also restores the expected mapped GEMM counts for
  the composite `cnn5`, `mlp3`, and `mha2h` designs under the same policy

Uniform requant constants for the non-softmax INT8 epilogues:
- `shift = 7`
- `rounding = 1 << (shift - 1)`

## Designs

- `fc64x64_b8_int8.c`
  - FC / GEMM-equivalent micro-bench
  - logical shape: `M=64`, `K=64`, `N=8`
- `conv2d_10x10x8_k3_oc8_int8.c`
  - direct valid convolution micro-bench
  - top-level interface uses shaped arrays
  - input: `1x8x10x10`
  - weights: `8x8x3x3`
  - output: `1x8x8x8`
  - equivalent GEMM shape after conv-to-GEMM lowering: `M=8`, `K=72`, `N=64`
- `attention32x64_int8.c`
  - single-head attention micro-bench with projected `Q`, `K`, and `V` inputs
  - fixed shape: `SEQ_LEN=32`, `EMBED_DIM=64`
  - widened top-level output: `int32_t O[32][64]`
  - `QK^T` GEMM-equivalent shape: `M=32`, `K=64`, `N=32`
  - `softmax * V` GEMM-equivalent shape: `M=32`, `K=32`, `N=64`
  - uses a row-wise integer softmax approximation with a small exponent LUT and Q0.7 attention weights
- `mha2h_8x24_ffn32_int8.c`
  - compact 2-head MHA + FFN micro-bench
  - fixed shape: `SEQ_LEN=8`, `MODEL_DIM=24`, `NUM_HEADS=2`, `HEAD_DIM=12`
  - stage sequence:
    - per-head Q / K / V linear projections
    - per-head `QK^T -> softmax -> A * V`
    - head concatenation
    - output projection with residual add
    - FFN `24 -> 32 -> 24` with ReLU and residual add
  - GEMM-equivalent stages:
    - 3x projections: `M=24`, `K=24`, `N=8`
    - 2x score GEMMs: `M=8`, `K=12`, `N=8`
    - 2x value GEMMs: `M=8`, `K=8`, `N=12`
    - 1x output projection: `M=24`, `K=24`, `N=8`
    - 1x FFN layer 1: `M=32`, `K=24`, `N=8`
    - 1x FFN layer 2: `M=24`, `K=32`, `N=8`
  - keeps softmax, concatenation, bias, and residual logic in separate loop nests
- `mlp3_16x24x32x16_b8_int8.c`
  - fixed 3-layer MLP micro-bench
  - shape chain: `16 -> 24 -> 32 -> 16`, batch size `8`
  - GEMM-equivalent FC stages:
    - layer 1: `M=24`, `K=16`, `N=8`
    - layer 2: `M=32`, `K=24`, `N=8`
    - layer 3: `M=16`, `K=32`, `N=8`
  - uses per-layer INT8 bias vectors
  - applies ReLU after the first two layers in separate elementwise loops
  - uses two internal INT8 activation buffers between layers
- `cnn5_6x6x8_oc16_int8.c`
  - fixed CNN micro-bench with conv/pool/dense staging
  - top-level interface uses shaped arrays
  - layer sequence:
    - `1x1 conv -> ReLU`
    - `3x3 valid conv -> ReLU`
    - `2x2 maxpool`
    - `1x1 conv -> ReLU`
    - `flatten`
    - `dense`
  - input: `6x6x8`
  - intermediate activations: `6x6x8`, `4x4x8`, pooled `2x2x8`, then `2x2x16`
  - flattened feature size: `64`
  - output: `16x1`
  - the three conv stages are written as natural direct-convolution loop nests
  - equivalent GEMM shapes:
    - conv 1: `M=8`, `K=8`, `N=36`
    - conv 2: `M=8`, `K=72`, `N=16`
    - conv 3: `M=16`, `K=8`, `N=4`
    - dense-equivalent stage: `M=16`, `K=64`, `N=1`
  - uses separate ReLU loop nests between the conv stages

## Default Runner

Use:

```bash
python3 c-designs-ml/micro-bench/run_micro_bench.py
```

The runner:

- discovers `.c` and `.cpp` files directly under this directory
- runs them through `tools/scalehls-c-to-cpp.py`
- uses the GEMM-only accel pipeline
- clears each per-design artifact directory before rerunning
- writes artifacts under `c-designs-ml/micro-bench/results/`

## Current ABI Expectations

For the current stable C flow:

- the top-level design signatures remain source-shaped
- generated accelerator helpers use the stable current pointer ABI
- current GEMM-family helper boundaries are intentionally 2D:
  - `A[M][K]`
  - `B[K][N]`
  - `C[M][N]`

So:

- the FC micro-bench lowers to a 2D GEMM helper directly
- the conv micro-bench lowers through explicit im2col-style packing and then
  calls a 2D GEMM helper
- the attention micro-bench is intentionally composite: it contains two 2D
  GEMM-style stages separated by explicit row-wise softmax normalization
- the MHA micro-bench is intentionally composite: it contains projection GEMMs,
  per-head attention GEMMs, an output projection GEMM, and a two-layer FFN,
  with softmax, concatenation, bias, and residual logic kept separate
- the MLP micro-bench is intentionally composite: it contains three back-to-back
  2D GEMM-style FC stages with separate bias/ReLU elementwise stages
- the CNN micro-bench is intentionally composite: it contains three conv
  stages written as direct conv loop nests plus separate maxpool, flatten, and
  dense stages
- non-softmax INT8-producing stages use a
  wide-accumulate-plus-separate-inline-requantization policy; current
  repo-local validation shows the full micro-bench suite reaches the expected
  mapped GEMM counts under this policy

## Ideal Mapping Estimates

- `fc64x64_b8_int8.c`
  - expected mapped GEMMs: `1`
  - expected GEMM shapes:
    - `M=64`, `K=64`, `N=8`
- `conv2d_10x10x8_k3_oc8_int8.c`
  - expected mapped GEMMs: `1`
  - expected GEMM shapes:
    - `M=8`, `K=72`, `N=64`
- `attention32x64_int8.c`
  - expected mapped GEMMs: `2`
  - expected GEMM shapes:
    - `M=32`, `K=64`, `N=32`
    - `M=32`, `K=32`, `N=64`
- `mha2h_8x24_ffn32_int8.c`
  - expected mapped GEMMs: `13`
  - expected GEMM shapes:
    - 6x `M=12`, `K=24`, `N=8`
    - 2x `M=8`, `K=12`, `N=8`
    - 2x `M=8`, `K=8`, `N=12`
    - 1x `M=24`, `K=24`, `N=8`
    - 1x `M=32`, `K=24`, `N=8`
    - 1x `M=24`, `K=32`, `N=8`
  - current repo-local validation: expected mapped GEMM count restored
- `mlp3_16x24x32x16_b8_int8.c`
  - expected mapped GEMMs: `3`
  - expected GEMM shapes:
    - `M=24`, `K=16`, `N=8`
    - `M=32`, `K=24`, `N=8`
    - `M=16`, `K=32`, `N=8`
  - current repo-local validation: expected mapped GEMM count restored
- `cnn5_6x6x8_oc16_int8.c`
  - expected mapped GEMMs: `3`
  - expected GEMM shapes:
    - `M=8`, `K=8`, `N=36`
    - `M=8`, `K=72`, `N=16`
    - `M=16`, `K=8`, `N=4`
  - current observed non-mapped GEMM-like stage:
    - dense head `M=16`, `K=64`, `N=1`
  - current repo-local validation: expected mapped GEMM count restored for the
    three conv stages

The conv case is still expected to contain explicit packing loops in emitted
HLS C++, but it should not expose a fake `1xMxK` / `1xKxN` / `1xMxN` helper ABI.

## Source Design Guidance

For source designs in this directory, keep surrounding non-GEMM logic clearly
separated from the GEMM loop nests when accelerator mapping is the intended
behavior.

Prefer:

- one loop nest for the GEMM itself
- separate loop nests for bias, ReLU, masking, softmax, scaling, packing, or
  output conversion
- separate epilogue loops for any `int32_t -> int8_t` narrowing, with inline
  requantization and clamp rather than helper-function casts

Do not assume the current tool will robustly extract a GEMM when operations like
the following are fused directly into the contraction loop nest:

- requantization
- clamp / saturation
- truncation / narrowing
- explicit output scaling or divide-based requant logic

If those behaviors are needed in a micro-bench, place them in separate
surrounding loops unless the point of the design is to document a currently
unsupported fused pattern.
