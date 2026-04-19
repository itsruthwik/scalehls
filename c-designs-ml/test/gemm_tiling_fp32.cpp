/*
 * Author: Ruthwik Reddy Sunketa
 */

#define BATCH 2
#define M 16
#define N 16
#define K 16

extern "C" void gemm_tiling_fp32(float C[BATCH][M][N],
                                 float A[BATCH][M][K],
                                 float B[BATCH][K][N]) {
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
