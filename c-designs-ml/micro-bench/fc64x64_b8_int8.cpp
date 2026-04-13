/*
 * Author: Ruthwik Reddy Sunketa
 */

#include <stdint.h>

#define OUT_FEATURES 64
#define IN_FEATURES 64
#define BATCH 8

// Equivalent GEMM:
//   M = 64
//   K = 64
//   N = 8
extern "C" void fc64x64_b8_int8(int8_t C[OUT_FEATURES][BATCH],
                                 int8_t W[OUT_FEATURES][IN_FEATURES],
                                 int8_t X[IN_FEATURES][BATCH]) {
  for (int m = 0; m < OUT_FEATURES; ++m) {
    for (int n = 0; n < BATCH; ++n) {
      // int32_t acc = (int32_t)C[m][n];
      int32_t acc = 0;
      for (int k = 0; k < IN_FEATURES; ++k) {
        acc += (int32_t)W[m][k] * (int32_t)X[k][n];
      }
      C[m][n] = (int8_t)acc;
    }
  }
}
