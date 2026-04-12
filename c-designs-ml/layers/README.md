# C Layer Designs

This directory contains the active canonical C layer designs used for frontend and mapper exploration.

## Naming

Files follow this convention:

- `<layer>_<precision>.cpp`

The exported function name matches the filename stem exactly.

Examples:

- `gemmv_fp32.cpp` -> `gemmv_fp32(...)`
- `gemm_int8.cpp` -> `gemm_int8(...)`
- `conv2d_fp16.cpp` -> `conv2d_fp16(...)`

## Precision Variants

The current layer set includes these precision variants where available:

- `fp32`
- `fp16`
- `int16`
- `int8`

These files are intended to make precision-specific frontend behavior explicit and easy to test.

## Canonical Accelerator-Family Layers

These are the primary canonical layerwise designs for the shared `accel` flow:

- `gemmv_*`
- `gemm_*`
- `conv2d_*`

They are written to match the current accelerator-family contract as directly as possible.

In practice:

- `gemmv_*` is the canonical rank-2 matmul/GEMMV-style layer
- `gemm_*` is the canonical batched GEMM-style layer
- `conv2d_*` is the canonical valid 2D convolution layer

These are the files the current C frontend validation scripts use by default for stable layerwise coverage.

## Composite Exploratory Layers

Some layers in this directory are intentionally more complex and are not canonical single-family kernels.

Current example:

- `attention_*`

The attention designs contain multiple stages:

- `QK^T`
- row-wise softmax
- `softmax * V`

These are useful for broader frontend exploration, but they should not be interpreted as direct single-IP canonical mapper inputs in the same way as `gemmv_*`, `gemm_*`, or `conv2d_*`.

## Pointer-Top-Level Experiment

This directory also includes a focused pointer-style top-level variant:

- `gemmv_fp32_ptr.cpp`

This file exists to test how the C frontend behaves when the source-level top
function uses raw pointers instead of shaped array parameters. The currently
supported case is a contiguous linearized GEMMV buffer layout with known static
problem sizes. The frontend rewrites the rank-1 dynamic memrefs into static 2D
views before mapping.

## Archive

Older and exploratory C designs that are not part of the active canonical layer set are kept under:

- `c-designs-ml/archive/`

The archive exists only to preserve previous experiments temporarily. The intention is that it can be cleaned up later without affecting the active layerwise design set.
