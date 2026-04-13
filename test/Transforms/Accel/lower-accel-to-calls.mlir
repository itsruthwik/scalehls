// RUN: scalehls-opt -split-input-file -lower-accel-to-calls %s | FileCheck %s
// RUN: not scalehls-opt -split-input-file -lower-accel-to-calls="max-elements=64" %s 2>&1 | FileCheck %s --check-prefix=TILED-FAIL

// CHECK: func.func private @forward_accel_gemmv_f32_in8x32_w32x16_out8x16
// CHECK-LABEL: func.func @forward(
// CHECK-SAME: scalehls.gemm_outlined = @forward_accel_gemmv_f32_in8x32_w32x16_out8x16
// CHECK: %[[CALL:.*]] = call @forward_accel_gemmv_f32_in8x32_w32x16_out8x16(%arg0, %arg1) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK: return %[[CALL]]
func.func @forward(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = "accel.gemmv"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func private @conv_bias_accel_conv_f32_in1x32x10x10_w64x32x3x3_b64_out1x64x8x8(
// CHECK-LABEL: func.func @conv_bias(
// CHECK: call @conv_bias_accel_conv_f32_in1x32x10x10_w64x32x3x3_b64_out1x64x8x8(%arg0, %arg1, %arg2)
func.func @conv_bias(%arg0: tensor<1x32x10x10xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x8x8xf32> {
  %0 = "accel.conv"(%arg0, %arg1, %arg2) {dilations = array<i64: 1, 1>, operand_segment_sizes = array<i32: 1, 1, 1, 0>, strides = array<i64: 1, 1>} : (tensor<1x32x10x10xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>) -> tensor<1x64x8x8xf32>
  return %0 : tensor<1x64x8x8xf32>
}

// -----

// CHECK-LABEL: func.func private @batch_existing_input_accel_gemm_f32_in8x32_w32x16_ei8x16_out8x16(
// CHECK-LABEL: func.func @batch_existing_input(
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK: %[[A_SLICE:.*]] = tensor.extract_slice %arg0
// CHECK: %[[A2D:.*]] = tensor.collapse_shape %[[A_SLICE]]
// CHECK: %[[B_SLICE:.*]] = tensor.extract_slice %arg1
// CHECK: %[[B2D:.*]] = tensor.collapse_shape %[[B_SLICE]]
// CHECK: %[[C_SLICE:.*]] = tensor.extract_slice %arg4
// CHECK: %[[C2D:.*]] = tensor.collapse_shape %[[C_SLICE]]
// CHECK: %[[CALL:.*]] = func.call @batch_existing_input_accel_gemm_f32_in8x32_w32x16_ei8x16_out8x16(%[[A2D]], %[[B2D]], %[[C2D]])
// CHECK: %[[EXP:.*]] = tensor.expand_shape %[[CALL]]
// CHECK: %[[INS:.*]] = tensor.insert_slice %[[EXP]] into %arg4
// CHECK: scf.yield %[[INS]]
func.func @batch_existing_input(%arg0: tensor<4x8x32xf32>, %arg1: tensor<4x32x16xf32>, %arg2: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 0, 1>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>, tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

func.func @forward_small(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "accel.gemmv"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// TILED-FAIL: error: persistent tiled pointer ABI not implemented yet for logical output size 128 > max-elements=64
