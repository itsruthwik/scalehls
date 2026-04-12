#define M 8
#define N 8
#define K 16

extern "C" void gemmv_fp32(float C[M][N], float A[M][K], float B[K][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
