#include <stdint.h>

#define BATCH 2
#define M 4
#define N 5
#define K 6

extern "C" void gemm_int8_ptr(int8_t *C, int8_t *A, int8_t *B) {
  for (int b = 0; b < BATCH; ++b) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
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
