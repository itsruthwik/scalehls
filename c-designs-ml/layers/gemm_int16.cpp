#include <stdint.h>

#define BATCH 2
#define M 4
#define N 5
#define K 6

extern "C" void gemm_int16(int16_t C[BATCH][M][N],
                            int16_t A[BATCH][M][K],
                            int16_t B[BATCH][K][N]) {
  for (int b = 0; b < BATCH; ++b) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
          C[b][i][j] += A[b][i][k] * B[b][k][j];
        }
      }
    }
  }
}
