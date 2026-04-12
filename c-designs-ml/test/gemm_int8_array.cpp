#include <stdint.h>

#define BATCH 2
#define M 4
#define N 5
#define K 6

extern "C" void gemm_int8_array(int8_t C[BATCH][M][N],
                                 int8_t A[BATCH][M][K],
                                 int8_t B[BATCH][K][N]) {
  for (int b = 0; b < BATCH; ++b) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
          int32_t a = (int32_t)A[b][i][k];
          int32_t w = (int32_t)B[b][k][j];
          int32_t prod = a * w;
          int32_t c = (int32_t)C[b][i][j];
          int32_t sum = c + prod;
          C[b][i][j] = (int8_t)sum;
        }
      }
    }
  }
}
