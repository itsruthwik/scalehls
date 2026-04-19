// RUN: scalehls-opt -split-input-file -tile-accel-gemm="ip-size=2x2x3" %s | FileCheck %s --check-prefix=PARALLEL
// RUN: scalehls-opt -split-input-file -tile-accel-gemm="ip-size=2x2x3 serial" %s | FileCheck %s --check-prefix=SERIAL
// RUN: scalehls-opt -split-input-file -tile-accel-gemm="ip-size=3x2x4" %s | FileCheck %s --check-prefix=PAD
// RUN: scalehls-opt -split-input-file -tile-accel-gemm="ip-size=2x2x3" %s | FileCheck %s --check-prefix=BIAS1D

// PARALLEL-LABEL: func.func @batched_tiled(
// PARALLEL: %[[C00:.*]] = tensor.extract_slice %arg2[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<1x4x4xf32> to tensor<1x2x2xf32>
// PARALLEL: %[[A000:.*]] = tensor.extract_slice %arg0[0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<1x4x6xf32> to tensor<1x2x3xf32>
// PARALLEL: %[[B000:.*]] = tensor.extract_slice %arg1[0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<1x6x4xf32> to tensor<1x3x2xf32>
// PARALLEL: %[[P000:.*]] = "accel.gemm"(%[[A000]], %[[B000]]) {operand_segment_sizes = array<i32: 1, 1, 0, 0>, scalehls.accelerator_ip_tile_col = 0 : i64, scalehls.accelerator_ip_tile_k = 0 : i64, scalehls.accelerator_ip_tile_row = 0 : i64, scalehls.accelerator_ip_tile_size = "2x2x3", scalehls.accelerator_ip_tiling_mode = "parallel"} : (tensor<1x2x3xf32>, tensor<1x3x2xf32>) -> tensor<1x2x2xf32>
// PARALLEL: %[[P001:.*]] = "accel.gemm"
// PARALLEL: linalg.generic
// PARALLEL: linalg.generic
// PARALLEL: tensor.insert_slice
// PARALLEL-COUNT-8: "accel.gemm"
// PARALLEL: scalehls.accelerator_ip_tile_k = 1 : i64
// PARALLEL: scalehls.accelerator_ip_tile_row = 1 : i64
// PARALLEL: scalehls.accelerator_ip_tile_col = 1 : i64
// PARALLEL: scalehls.accelerator_ip_tiling_mode = "parallel"
// PARALLEL: return

// SERIAL-LABEL: func.func @batched_tiled(
// SERIAL: scalehls.accelerator_ip_tile_size = "2x2x3"
// SERIAL: scalehls.accelerator_ip_tiling_mode = "serial"
// SERIAL: scalehls.accelerator_ip_tile_k = 1 : i64
// SERIAL: return
func.func @batched_tiled(%arg0: tensor<1x4x6xf32>, %arg1: tensor<1x6x4xf32>, %arg2: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 0, 1>} : (tensor<1x4x6xf32>, tensor<1x6x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}

// -----

// PAD-LABEL: func.func @non_divisible(
// PAD: %[[PA:.*]] = tensor.empty() : tensor<1x6x8xf32>
// PAD: %[[PAA:.*]] = linalg.fill
// PAD: %[[PI:.*]] = tensor.insert_slice %arg0 into %[[PAA]][0, 0, 0] [1, 4, 6] [1, 1, 1] : tensor<1x4x6xf32> into tensor<1x6x8xf32>
// PAD: %[[PB:.*]] = tensor.empty() : tensor<1x8x4xf32>
// PAD: tensor.insert_slice %arg1
// PAD: %[[PC:.*]] = tensor.empty() : tensor<1x6x4xf32>
// PAD: tensor.insert_slice %arg2
// PAD-COUNT-8: "accel.gemm"
// PAD: scalehls.accelerator_ip_tile_size = "3x2x4"
// PAD: %[[SLICE:.*]] = tensor.extract_slice
// PAD-SAME: [1, 4, 4]
// PAD: return %[[SLICE]] : tensor<1x4x4xf32>
func.func @non_divisible(%arg0: tensor<1x4x6xf32>, %arg1: tensor<1x6x4xf32>, %arg2: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 0, 1>} : (tensor<1x4x6xf32>, tensor<1x6x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}

// -----

// BIAS1D-LABEL: func.func @batched_rank1_bias(
// BIAS1D: %[[BCAST:.*]] = linalg.generic
// BIAS1D-SAME: ins(%arg2 : tensor<4xf32>)
// BIAS1D: %[[BIAS00:.*]] = tensor.extract_slice %[[BCAST]][0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<1x4x4xf32> to tensor<1x2x2xf32>
// BIAS1D: ins(%[[BIAS00]], {{.*}} : tensor<1x2x2xf32>, tensor<1x2x2xf32>)
// BIAS1D: scalehls.accelerator_ip_tile_k = 1 : i64
// BIAS1D: scalehls.accelerator_ip_tile_row = 1 : i64
// BIAS1D: scalehls.accelerator_ip_tile_col = 1 : i64
func.func @batched_rank1_bias(%arg0: tensor<1x4x6xf32>, %arg1: tensor<1x6x4xf32>, %arg2: tensor<4xf32>) -> tensor<1x4x4xf32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 1, 0>} : (tensor<1x4x6xf32>, tensor<1x6x4xf32>, tensor<4xf32>) -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}
