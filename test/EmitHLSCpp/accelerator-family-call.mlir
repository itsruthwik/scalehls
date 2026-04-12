// RUN: scalehls-translate -scalehls-emit-hlscpp %s | FileCheck %s

// CHECK-DAG: void forward_accel_gemmv_f32_in8x32_w32x16_out8x16(
// CHECK-DAG: void batched_accel_gemm_f32_in1x16x64_w1x64x64_out1x16x64(
// CHECK-DAG: void conv_bias_accel_conv_f32_in1x32x10x10_w64x32x3x3_b64_out1x64x8x8(
// CHECK-LABEL: void forward(
// CHECK: forward_accel_gemmv_f32_in8x32_w32x16_out8x16(
// CHECK-LABEL: void batched(
// CHECK: batched_accel_gemm_f32_in1x16x64_w1x64x64_out1x16x64(
// CHECK-LABEL: void conv_bias(
// CHECK: conv_bias_accel_conv_f32_in1x32x10x10_w64x32x3x3_b64_out1x64x8x8(

func.func private @forward_accel_gemmv_f32_in8x32_w32x16_out8x16(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> attributes {scalehls.gemm, scalehls.gemm_contract = "tensor_family", scalehls.accelerator_family = "GEMMV"} 

func.func @forward(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = call @forward_accel_gemmv_f32_in8x32_w32x16_out8x16(%arg0, %arg1) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

func.func private @batched_accel_gemm_f32_in1x16x64_w1x64x64_out1x16x64(%arg0: tensor<1x16x64xf32>, %arg1: tensor<1x64x64xf32>) -> tensor<1x16x64xf32> attributes {scalehls.gemm, scalehls.gemm_contract = "tensor_family", scalehls.accelerator_family = "GEMM"}

func.func @batched(%arg0: tensor<1x16x64xf32>, %arg1: tensor<1x64x64xf32>) -> tensor<1x16x64xf32> {
  %0 = call @batched_accel_gemm_f32_in1x16x64_w1x64x64_out1x16x64(%arg0, %arg1) : (tensor<1x16x64xf32>, tensor<1x64x64xf32>) -> tensor<1x16x64xf32>
  return %0 : tensor<1x16x64xf32>
}

func.func private @conv_bias_accel_conv_f32_in1x32x10x10_w64x32x3x3_b64_out1x64x8x8(%arg0: tensor<1x32x10x10xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x8x8xf32> attributes {scalehls.gemm, scalehls.gemm_contract = "tensor_family", scalehls.accelerator_family = "CONV", scalehls.accelerator_has_bias = true}

func.func @conv_bias(%arg0: tensor<1x32x10x10xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x8x8xf32> {
  %0 = call @conv_bias_accel_conv_f32_in1x32x10x10_w64x32x3x3_b64_out1x64x8x8(%arg0, %arg1, %arg2) : (tensor<1x32x10x10xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>) -> tensor<1x64x8x8xf32>
  return %0 : tensor<1x64x8x8xf32>
}
