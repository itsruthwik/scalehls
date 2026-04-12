#include <stdint.h>

#define M 8
#define N 8
#define K 16

extern "C" void gemmv_int8_ptr(int8_t *C, int8_t *A, int8_t *B) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
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
