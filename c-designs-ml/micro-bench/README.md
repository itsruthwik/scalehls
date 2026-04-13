# C Micro-Bench

This directory contains focused INT8 micro-bench designs used to probe the
current stable C accel flow.

## Designs

- `fc64x64_b8_int8.cpp`
  - FC / GEMM-equivalent micro-bench
  - logical shape: `M=64`, `K=64`, `N=8`
- `conv2d_10x10x8_k3_oc8_int8.cpp`
  - direct valid convolution micro-bench
  - input: `1x8x10x10`
  - weights: `8x8x3x3`
  - output: `1x8x8x8`
  - equivalent GEMM shape in `gemm-only` mode: `M=8`, `K=72`, `N=64`

## Default Runner

Use:

```bash
python3 c-designs-ml/micro-bench/run_micro_bench.py
```

The runner:

- discovers `.cpp` files directly under this directory
- runs them through `tools/scalehls-c-to-cpp.py`
- uses the accel pipeline in `--gemm-only` mode
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

The conv case is still expected to contain explicit packing loops in emitted
HLS C++, but it should not expose a fake `1xMxK` / `1xKxN` / `1xMxN` helper ABI.
