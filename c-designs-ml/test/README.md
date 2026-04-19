# C Test Designs

This directory contains focused INT8 test designs organized by:

- logical source pattern: rank-2 matmul-style, batched GEMM-style, and convolution-style designs
- top-level interface style:
  - `array`: full static array arguments
  - `ptr`: full raw-pointer arguments with linearized indexing
  - `tiled_ptr`: raw-pointer arguments with explicit blocked loop nests

The function name always matches the filename stem exactly.

These files are the active C design set for frontend and mapper experiments.
They also contain the canonical FP32 layerwise designs used by the reusable
C frontend validation flow.

Additional focused assets:

- `gemmv_fp32.cpp`, `gemm_fp32.cpp`, `conv2d_fp32.cpp`: canonical FP32 array
  designs used by the reusable C frontend mapper validation
- `gemmv_fp32_ptr.cpp`, `gemmv_fp32_tiled_ptr.cpp`: canonical FP32 pointer-top
  experiments for contiguous and tiled pointer frontends
- `gemm_tiling_fp32.cpp`: larger canonical GEMM used to validate fixed-IP
  GEMM tiling configurations such as `8x8x8` and `16x16x16`
- `gemm_duplicate_fp32.cpp`: two same-shape GEMMs in one parent function used
  to validate unique `gemm_<MxKxN>_call<N>` helper naming
- `validate_gemm_ip_tiling.py`: end-to-end C-side validation for spatial and
  serial GEMM IP tiling
- `validate_unique_gemm_calls.py`: end-to-end validation for unique helper
  names and manifest port-role labeling
