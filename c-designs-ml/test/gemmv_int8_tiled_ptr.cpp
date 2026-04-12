#include <stdint.h>

#define M 8
#define N 8
#define K 16

#define TI 4
#define TJ 4
#define TK 8

extern "C" void gemmv_int8_tiled_ptr(int8_t *C, int8_t *A, int8_t *B) {
  for (int ii = 0; ii < M; ii += TI) {
    for (int jj = 0; jj < N; jj += TJ) {
      for (int kk = 0; kk < K; kk += TK) {
        for (int i = ii; i < ii + TI; ++i) {
          for (int j = jj; j < jj + TJ; ++j) {
            for (int k = kk; k < kk + TK; ++k) {
              int c_idx = (i * N) + j;
              int a_idx = (i * K) + k;
              int b_idx = (k * N) + j;
              int32_t a = (int32_t)A[a_idx];
              int32_t b = (int32_t)B[b_idx];
              int32_t prod = a * b;
              int32_t c = (int32_t)C[c_idx];
              int32_t sum = c + prod;
              C[c_idx] = (int8_t)sum;
            }
          }
        }
      }
    }
  }
}
