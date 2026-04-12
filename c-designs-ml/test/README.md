# C Test Designs

This directory contains focused INT8 test designs organized by:

- accelerator family: `gemmv`, `gemm`, `conv2d`
- top-level interface style:
  - `array`: full static array arguments
  - `ptr`: full raw-pointer arguments with linearized indexing
  - `tiled_ptr`: raw-pointer arguments with explicit blocked loop nests

The function name always matches the filename stem exactly.

These files are intended for frontend and mapper experiments. They are a
separate test matrix from the stable canonical layer set under
`c-designs-ml/layers/`.
