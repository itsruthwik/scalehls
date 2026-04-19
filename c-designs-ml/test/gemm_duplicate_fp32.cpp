/*
 * Author: Ruthwik Reddy Sunketa
 */

#define M 8
#define N 8
#define K 8

extern "C" void gemm_duplicate_fp32(float C0[M][N],
                                    float C1[M][N],
                                    float A0[M][K],
                                    float B0[K][N],
                                    float A1[M][K],
                                    float B1[K][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C0[i][j] += A0[i][k] * B0[k][j];
      }
    }
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C1[i][j] += A1[i][k] * B1[k][j];
      }
    }
  }
}
