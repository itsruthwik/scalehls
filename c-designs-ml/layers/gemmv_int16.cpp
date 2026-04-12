#include <stdint.h>

#define M 8
#define N 8
#define K 16

extern "C" void gemmv_int16(int16_t C[M][N], int16_t A[M][K], int16_t B[K][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
