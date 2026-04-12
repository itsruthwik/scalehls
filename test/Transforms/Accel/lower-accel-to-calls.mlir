// RUN: scalehls-opt -split-input-file -lower-accel-to-calls %s | FileCheck %s
// RUN: scalehls-opt -split-input-file -lower-accel-to-calls="abi-mode=full-data" %s | FileCheck %s --check-prefix=FULL-DATA
// RUN: not scalehls-opt -split-input-file -lower-accel-to-calls="max-elements=64" %s 2>&1 | FileCheck %s --check-prefix=TILED-FAIL
// RUN: not scalehls-opt -split-input-file -lower-accel-to-calls="abi-mode=stream" %s 2>&1 | FileCheck %s --check-prefix=ABI-FAIL

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

// CHECK-LABEL: func.func private @batch_existing_input_accel_gemm_f32_in4x8x32_w4x32x16_ei4x8x16_out4x8x16(
// CHECK-LABEL: func.func @batch_existing_input(
// CHECK: call @batch_existing_input_accel_gemm_f32_in4x8x32_w4x32x16_ei4x8x16_out4x8x16(%arg0, %arg1, %arg2)
func.func @batch_existing_input(%arg0: tensor<4x8x32xf32>, %arg1: tensor<4x32x16xf32>, %arg2: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 0, 1>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>, tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

// FULL-DATA-LABEL: func.func private @forward_small_accel_gemmv_f32_in2x2_w2x2_out2x2(
// FULL-DATA-SAME: f32, f32, f32, f32, f32, f32, f32, f32) -> (f32, f32, f32, f32)
// FULL-DATA-LABEL: func.func @forward_small(
// FULL-DATA: %[[A00:.+]] = tensor.extract %arg0[%c0{{.*}}, %c0{{.*}}] : tensor<2x2xf32>
// FULL-DATA: %[[B00:.+]] = tensor.extract %arg1[%c0{{.*}}, %c0{{.*}}] : tensor<2x2xf32>
// FULL-DATA: %[[CALL:.*]]:4 = call @forward_small_accel_gemmv_f32_in2x2_w2x2_out2x2(
// FULL-DATA: %[[TENSOR:.*]] = tensor.from_elements %[[CALL]]#0, %[[CALL]]#1, %[[CALL]]#2, %[[CALL]]#3 : tensor<2x2xf32>
// FULL-DATA: return %[[TENSOR]]
func.func @forward_small(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "accel.gemmv"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// TILED-FAIL: error: persistent tiled pointer ABI not implemented yet for logical output size 128 > max-elements=64
// ABI-FAIL: error: ABI mode '{{.*}}' is not implemented yet; supported modes are pointer and full-data
