#include <stdint.h>

#define BATCH 2
#define M 4
#define N 5
#define K 6

#define TI 2
#define TJ 5
#define TK 3

extern "C" void gemm_int8_tiled_ptr(int8_t *C, int8_t *A, int8_t *B) {
  for (int b = 0; b < BATCH; ++b) {
    for (int ii = 0; ii < M; ii += TI) {
      for (int jj = 0; jj < N; jj += TJ) {
        for (int kk = 0; kk < K; kk += TK) {
          for (int i = ii; i < ii + TI; ++i) {
            for (int j = jj; j < jj + TJ; ++j) {
              for (int k = kk; k < kk + TK; ++k) {
                int c_idx = ((b * M + i) * N) + j;
                int a_idx = ((b * M + i) * K) + k;
                int b_idx = ((b * K + k) * N) + j;
                int32_t a = (int32_t)A[a_idx];
                int32_t w = (int32_t)B[b_idx];
                int32_t prod = a * w;
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
}
