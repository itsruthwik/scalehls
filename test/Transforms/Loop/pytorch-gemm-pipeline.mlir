// RUN: scalehls-opt -split-input-file -scaleflow-pytorch-pipeline="top-func=forward debug-point=3 axi-interface=false" %s | FileCheck %s --check-prefix=DEFAULT
// RUN: scalehls-opt -split-input-file -pytorch-accel-pipeline="top-func=forward debug-point=3 axi-interface=false" %s | FileCheck %s --check-prefix=MAPPED

// DEFAULT-LABEL: func.func @forward(
// DEFAULT-NOT: scalehls.gemm_outlined
// DEFAULT-NOT: @forward_accel

// MAPPED-LABEL: func.func private @forward_accel_gemmv_f32_in8x32_w32x16_out8x16(
// MAPPED: accelerator_family = "GEMMV"
// MAPPED: gemm_contract = "tensor_family"
// MAPPED-LABEL: func.func @forward(
// MAPPED-SAME: scalehls.gemm_outlined = @forward_accel_gemmv_f32_in8x32_w32x16_out8x16
// MAPPED: call @forward_accel_gemmv_f32_in8x32_w32x16_out8x16(
// MAPPED: memref.copy

module attributes {torch.debug_module_name = "_MatmulProbe"} {
  func.func @forward(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x16xf32>
    %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>) outs(%1 : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %2 : tensor<8x16xf32>
  }
}
