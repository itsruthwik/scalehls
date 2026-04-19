// RUN: scalehls-opt %s | FileCheck %s

func.func @gemm_ops(
    %arg0: tensor<8x32xi8>,
    %arg1: tensor<32x16xi8>,
    %arg2: tensor<16xi32>,
    %arg3: tensor<8x16xi32>) -> tensor<8x16xi32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2, %arg3) {operand_segment_sizes = array<i32: 1, 1, 1, 1>} : (tensor<8x32xi8>, tensor<32x16xi8>, tensor<16xi32>, tensor<8x16xi32>) -> tensor<8x16xi32>
  return %0 : tensor<8x16xi32>
}

// CHECK-LABEL: func.func @gemm_ops
// CHECK: "accel.gemm"
// CHECK: operand_segment_sizes = array<i32: 1, 1, 1, 1>

// -----

func.func @gemm_wide_accum(
    %arg0: tensor<8x32xi8>,
    %arg1: tensor<32x16xi8>) -> tensor<8x16xi32> {
  %0 = "accel.gemm"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<8x32xi8>, tensor<32x16xi8>) -> tensor<8x16xi32>
  return %0 : tensor<8x16xi32>
}

// CHECK-LABEL: func.func @gemm_wide_accum
// CHECK: "accel.gemm"(%arg0, %arg1)
