// RUN: scalehls-opt -split-input-file -lower-linalg-to-accel %s | FileCheck %s

module {
  func.func @forward(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x16xf32>
    %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>) outs(%1 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg2, %2 : tensor<16xf32>, tensor<8x16xf32>) outs(%0 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %base: f32, %out: f32):
      %sum = arith.addf %in, %base : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %3 : tensor<8x16xf32>
  }
}

// CHECK-LABEL: func.func @forward
// CHECK: %[[ACCEL:.*]] = "accel.gemm"(%arg0, %arg1, %arg2)
// CHECK-SAME: operand_segment_sizes = array<i32: 1, 1, 1, 0>
// CHECK-NOT: linalg.matmul
// CHECK: return %[[ACCEL]]

// -----

module {
  func.func @wide_accum(%arg0: tensor<8x32xi8>, %arg1: tensor<32x16xi8>) -> tensor<8x16xi32> {
    %c0 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<8x16xi32>
    %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x32xi8>, tensor<32x16xi8>) outs(%1 : tensor<8x16xi32>) -> tensor<8x16xi32>
    return %2 : tensor<8x16xi32>
  }
}

// CHECK-LABEL: func.func @wide_accum
// CHECK: %[[ACCEL:.*]] = "accel.gemm"(%arg0, %arg1)
// CHECK-SAME: operand_segment_sizes = array<i32: 1, 1, 0, 0>
// CHECK-NOT: linalg.matmul
// CHECK: return %[[ACCEL]]

// -----

module {
  func.func @wide_accum_with_bias(%arg0: tensor<8x32xi8>, %arg1: tensor<32x16xi8>, %arg2: tensor<16xi32>) -> tensor<8x16xi32> {
    %c0 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<8x16xi32>
    %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x32xi8>, tensor<32x16xi8>) outs(%1 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg2, %2 : tensor<16xi32>, tensor<8x16xi32>) outs(%0 : tensor<8x16xi32>) {
    ^bb0(%in: i32, %base: i32, %out: i32):
      %sum = arith.addi %in, %base : i32
      linalg.yield %sum : i32
    } -> tensor<8x16xi32>
    return %3 : tensor<8x16xi32>
  }
}

// CHECK-LABEL: func.func @wide_accum_with_bias
// CHECK: %[[ACCEL:.*]] = "accel.gemm"(%arg0, %arg1, %arg2)
// CHECK-SAME: operand_segment_sizes = array<i32: 1, 1, 1, 0>
// CHECK-NOT: linalg.matmul
// CHECK: return %[[ACCEL]]

// -----

module {
  func.func @conv_bias(%arg0: tensor<1x32x10x10xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x8x8xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64x8x8xf32>
    %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg2 : tensor<64xf32>) outs(%1 : tensor<1x64x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x8x8xf32>
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%arg0, %arg1 : tensor<1x32x10x10xf32>, tensor<64x32x3x3xf32>)
         outs(%2 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    return %3 : tensor<1x64x8x8xf32>
  }
}

// CHECK-LABEL: func.func @conv_bias
// CHECK-SAME: scalehls.gemm_skip_reason = "unsupported linalg accel candidate: conv_2d_nchw_fchw remains tensor-side only"
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-NOT: "accel.gemm"

// -----

module {
  func.func @conv1x1_stride2(%arg0: tensor<1x32x16x16xf32>, %arg1: tensor<64x32x1x1xf32>) -> tensor<1x64x8x8xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64x8x8xf32>
    %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%arg0, %arg1 : tensor<1x32x16x16xf32>, tensor<64x32x1x1xf32>)
         outs(%1 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    return %2 : tensor<1x64x8x8xf32>
  }
}

// CHECK-LABEL: func.func @conv1x1_stride2
// CHECK-SAME: scalehls.gemm_skip_reason = "unsupported linalg accel candidate: conv_2d_nchw_fchw remains tensor-side only"
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-NOT: "accel.gemm"
// CHECK: return

// -----

module {
  func.func @batch_existing_input(%arg0: tensor<4x8x32xf32>, %arg1: tensor<4x32x16xf32>, %arg2: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    %2 = linalg.batch_matmul
         ins(%arg0, %arg1 : tensor<4x8x32xf32>, tensor<4x32x16xf32>)
         outs(%1 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%2, %arg2 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) {
    ^bb0(%base: f32, %existing: f32, %out: f32):
      %sum = arith.addf %base, %existing : f32
      linalg.yield %sum : f32
    } -> tensor<4x8x16xf32>
    return %3 : tensor<4x8x16xf32>
  }
}

// CHECK-LABEL: func.func @batch_existing_input
// CHECK: %[[ACCEL:.*]] = "accel.gemm"(%arg0, %arg1, %arg2)
// CHECK-SAME: operand_segment_sizes = array<i32: 1, 1, 0, 1>
// CHECK-NOT: linalg.batch_matmul
// CHECK: return %[[ACCEL]]

// -----

module {
  func.func @mixed_supported_and_unsupported(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<1x32x10x10xf32>, %arg3: tensor<64x32x3x3xf32>) -> (tensor<8x16xf32>, tensor<1x64x8x8xf32>) {
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x16xf32>
    %1 = linalg.fill ins(%c0_f32 : f32) outs(%0 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>) outs(%1 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %3 = tensor.empty() : tensor<1x64x8x8xf32>
    %4 = linalg.fill ins(%c0_f32 : f32) outs(%3 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %5 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%arg2, %arg3 : tensor<1x32x10x10xf32>, tensor<64x32x3x3xf32>)
         outs(%4 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    return %2, %5 : tensor<8x16xf32>, tensor<1x64x8x8xf32>
  }
}

// CHECK-LABEL: func.func @mixed_supported_and_unsupported
// CHECK-SAME: scalehls.gemm_skip_reason = "unsupported linalg accel candidate: conv_2d_nchw_fchw remains tensor-side only"
// CHECK: "accel.gemm"(%arg0, %arg1)
// CHECK: linalg.conv_2d_nchw_fchw
