#define BATCH 2
#define M 4
#define N 5
#define K 6

extern "C" void gemm_fp32(float C[BATCH][M][N],
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
