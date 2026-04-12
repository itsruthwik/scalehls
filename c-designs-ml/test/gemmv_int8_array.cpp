#include <stdint.h>

#define M 8
#define N 8
#define K 16

extern "C" void gemmv_int8_array(int8_t C[M][N],
                                  int8_t A[M][K],
                                  int8_t B[K][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        int32_t a = (int32_t)A[i][k];
        int32_t b = (int32_t)B[k][j];
        int32_t prod = a * b;
        int32_t c = (int32_t)C[i][j];
        int32_t sum = c + prod;
        C[i][j] = (int8_t)sum;
      }
    }
  }
}
