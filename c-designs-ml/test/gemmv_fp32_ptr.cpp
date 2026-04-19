#define M 8
#define N 8
#define K 16

extern "C" void gemmv_fp32_ptr(float *C, float *A, float *B) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        int c_idx = (i * N) + j;
        int a_idx = (i * K) + k;
        int b_idx = (k * N) + j;
        C[c_idx] += A[a_idx] * B[b_idx];
      }
    }
  }
}
