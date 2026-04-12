#define M 8
#define N 8
#define K 16

extern "C" void gemmv_fp16(_Float16 C[M][N], _Float16 A[M][K], _Float16 B[K][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
