// RUN: scalehls-opt -pytorch-accel-pipeline="top-func=forward debug-point=3 axi-interface=false" %s | FileCheck %s

// CHECK: module attributes {torch.debug_module_name = "_ActivationProbe"} {
// CHECK-LABEL: func.func @forward(
// CHECK-SAME: scalehls.gemm_skip_reason =
// CHECK-NOT: scalehls.gemm_outlined
// CHECK-NOT: @forward_accel_

#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "_ActivationProbe"} {
  func.func @forward(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ugt, %in, %c0 : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
    } -> tensor<1x16xf32>
    return %1 : tensor<1x16xf32>
  }
}
