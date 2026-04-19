// RUN: not scalehls-opt -verify-diagnostics %s >/dev/null

func.func @invalid_gemm_bad_result(
    %arg0: tensor<8x32xi8>,
    %arg1: tensor<32x16xi8>) -> tensor<8x32xi32> {
  // expected-error @+1 {{requires 2D GEMM shapes A[M,K], B[K,N], result[M,N]}}
  %0 = "accel.gemm"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<8x32xi8>, tensor<32x16xi8>) -> tensor<8x32xi32>
  return %0 : tensor<8x32xi32>
}

func.func @invalid_gemm_bad_existing_input(
    %arg0: tensor<8x32xi8>,
    %arg1: tensor<32x16xi8>,
    %arg2: tensor<8x16xi8>) -> tensor<8x16xi32> {
  // expected-error @+1 {{requires existing_input element type to match result element type}}
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 0, 1>} : (tensor<8x32xi8>, tensor<32x16xi8>, tensor<8x16xi8>) -> tensor<8x16xi32>
  return %0 : tensor<8x16xi32>
}
