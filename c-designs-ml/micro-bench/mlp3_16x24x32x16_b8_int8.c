/*
 * Author: Ruthwik Reddy Sunketa
 */

#include <stdint.h>

#define INPUT_DIM 16
#define HIDDEN_DIM1 24
#define HIDDEN_DIM2 32
#define OUTPUT_DIM 16
#define BATCH 8

// Equivalent GEMMs:
//   Layer 1: M = 24, K = 16, N = 8
//   Layer 2: M = 32, K = 24, N = 8
//   Layer 3: M = 16, K = 32, N = 8
void mlp3_16x24x32x16_b8_int8(int8_t O[OUTPUT_DIM][BATCH],
                              int8_t W1[HIDDEN_DIM1][INPUT_DIM],
                              int8_t B1[HIDDEN_DIM1],
                              int8_t W2[HIDDEN_DIM2][HIDDEN_DIM1],
                              int8_t B2[HIDDEN_DIM2],
                              int8_t W3[OUTPUT_DIM][HIDDEN_DIM2],
                              int8_t B3[OUTPUT_DIM],
                              int8_t X[INPUT_DIM][BATCH]) {
  int32_t H1_GEMM[HIDDEN_DIM1][BATCH];
  int8_t H1[HIDDEN_DIM1][BATCH];
  int32_t H2_GEMM[HIDDEN_DIM2][BATCH];
  int8_t H2[HIDDEN_DIM2][BATCH];
  int32_t O_GEMM[OUTPUT_DIM][BATCH];

  // Layer 1 GEMM: H1_GEMM = W1 * X.
  for (int m = 0; m < HIDDEN_DIM1; ++m) {
    for (int n = 0; n < BATCH; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < INPUT_DIM; ++k)
        acc += (int32_t)W1[m][k] * (int32_t)X[k][n];
      H1_GEMM[m][n] = acc;
    }
  }

  // Layer 1 epilogue: add B1 and apply ReLU.
  for (int m = 0; m < HIDDEN_DIM1; ++m) {
    for (int n = 0; n < BATCH; ++n) {
      int32_t value = (int32_t)H1_GEMM[m][n] + (int32_t)B1[m];
      value = (value + 64) >> 7;
      if (value < 0)
        value = 0;
      if (value > 127)
        value = 127;
      H1[m][n] = (int8_t)value;
    }
  }

  // Layer 2 GEMM: H2_GEMM = W2 * H1.
  for (int m = 0; m < HIDDEN_DIM2; ++m) {
    for (int n = 0; n < BATCH; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < HIDDEN_DIM1; ++k)
        acc += (int32_t)W2[m][k] * (int32_t)H1[k][n];
      H2_GEMM[m][n] = acc;
    }
  }

  // Layer 2 epilogue: add B2 and apply ReLU.
  for (int m = 0; m < HIDDEN_DIM2; ++m) {
    for (int n = 0; n < BATCH; ++n) {
      int32_t value = (int32_t)H2_GEMM[m][n] + (int32_t)B2[m];
      value = (value + 64) >> 7;
      if (value < 0)
        value = 0;
      if (value > 127)
        value = 127;
      H2[m][n] = (int8_t)value;
    }
  }

  // Layer 3 GEMM: O_GEMM = W3 * H2.
  for (int m = 0; m < OUTPUT_DIM; ++m) {
    for (int n = 0; n < BATCH; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < HIDDEN_DIM2; ++k)
        acc += (int32_t)W3[m][k] * (int32_t)H2[k][n];
      O_GEMM[m][n] = acc;
    }
  }

  // Output epilogue: add B3.
  for (int m = 0; m < OUTPUT_DIM; ++m) {
    for (int n = 0; n < BATCH; ++n) {
      int32_t value = (int32_t)O_GEMM[m][n] + (int32_t)B3[m];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      O[m][n] = (int8_t)value;
    }
  }
}
