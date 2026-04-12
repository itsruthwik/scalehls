// RUN: scalehls-opt %s | FileCheck %s

func.func @gemm_family_ops(
    %arg0: tensor<8x32xf32>,
    %arg1: tensor<32x16xf32>,
    %arg2: tensor<16xf32>,
    %arg3: tensor<8x16xf32>,
    %arg4: tensor<1x32x10x10xf32>,
    %arg5: tensor<64x32x3x3xf32>) -> (tensor<8x16xf32>, tensor<1x64x8x8xf32>) {
  %0 = "accel.gemmv"(%arg0, %arg1, %arg2, %arg3) {operand_segment_sizes = array<i32: 1, 1, 1, 1>} : (tensor<8x32xf32>, tensor<32x16xf32>, tensor<16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "accel.conv"(%arg4, %arg5, %arg2, %arg3) {dilations = array<i64: 1, 1>, operand_segment_sizes = array<i32: 1, 1, 1, 1>, strides = array<i64: 1, 1>} : (tensor<1x32x10x10xf32>, tensor<64x32x3x3xf32>, tensor<16xf32>, tensor<8x16xf32>) -> tensor<1x64x8x8xf32>
  return %0, %1 : tensor<8x16xf32>, tensor<1x64x8x8xf32>
}

// CHECK-LABEL: func.func @gemm_family_ops
// CHECK: "accel.gemmv"
// CHECK: operand_segment_sizes = array<i32: 1, 1, 1, 1>
// CHECK: "accel.conv"
