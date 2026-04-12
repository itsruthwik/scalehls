#define M 8
#define N 8
#define K 16

#define TI 4
#define TJ 4
#define TK 8

extern "C" void gemmv_fp32_tiled_ptr(float *C, float *A, float *B) {
  for (int ii = 0; ii < M; ii += TI) {
    for (int jj = 0; jj < N; jj += TJ) {
      for (int kk = 0; kk < K; kk += TK) {
        for (int i = ii; i < ii + TI; ++i) {
          for (int j = jj; j < jj + TJ; ++j) {
            for (int k = kk; k < kk + TK; ++k) {
              int c_idx = (i * N) + j;
              int a_idx = (i * K) + k;
              int b_idx = (k * N) + j;
              C[c_idx] += A[a_idx] * B[b_idx];
            }
          }
        }
      }
    }
  }
}
