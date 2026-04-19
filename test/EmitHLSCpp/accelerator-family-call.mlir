// RUN: scalehls-translate --scalehls-emit-hlscpp %s | FileCheck %s

module {
  // CHECK-DAG: void gemm_8x32x16_call0(
  // CHECK-DAG: float A[8][32]
  // CHECK-DAG: float B[32][16]
  // CHECK-DAG: float C[8][16]
  // CHECK-DAG: void gemm_8x32x16_call1(
  // CHECK-DAG: float ExistingInput[8][16]
  func.func private @gemm_8x32x16_call0(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> attributes {scalehls.gemm, scalehls.accelerator_has_bias = false}

  // CHECK-LABEL: void forward(
  // CHECK: gemm_8x32x16_call0(v0, v1, v2);
  func.func @forward(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
    %0 = call @gemm_8x32x16_call0(%arg0, %arg1) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }

  func.func private @gemm_8x32x16_call1(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> attributes {scalehls.gemm, scalehls.accelerator_has_bias = false, scalehls.gemm_a_arg = 0 : i64, scalehls.gemm_b_arg = 1 : i64, scalehls.gemm_existing_input_arg = 2 : i64, scalehls.gemm_c_arg = 3 : i64}

  // CHECK-LABEL: void forward_existing_input(
  // CHECK: gemm_8x32x16_call1(v3, v4, v5, v6);
  func.func @forward_existing_input(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = call @gemm_8x32x16_call1(%arg0, %arg1, %arg2) : (tensor<8x32xf32>, tensor<32x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}
