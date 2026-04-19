// RUN: scalehls-opt %s -lower-accel-to-calls | FileCheck %s

// CHECK-LABEL: func.func private @gemm_8x32x16_call0(
// CHECK-SAME: scalehls.gemm_a_arg = 0 : i64
// CHECK-SAME: scalehls.gemm_b_arg = 1 : i64
// CHECK-LABEL: func.func @forward(
// CHECK: scalehls.gemm_outlined = @gemm_8x32x16_call0
// CHECK: %[[CALL:.*]] = call @gemm_8x32x16_call0(%arg0, %arg1) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
func.func @forward(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = "accel.gemm"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func private @gemm_8x16x16_call0(
// CHECK-LABEL: func.func @with_bias(
// CHECK: scalehls.gemm_outlined = @gemm_8x16x16_call0
// CHECK: call @gemm_8x16x16_call0(%arg0, %arg1, %arg2)
func.func @with_bias(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16xf32>) -> tensor<8x16xf32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 1, 0>} : (tensor<8x16xf32>, tensor<16x16xf32>, tensor<16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func private @gemm_8x32x16_call0{{(_[0-9]+)?}}(
// CHECK-SAME: tensor<8x32xi8>, tensor<32x16xi8>, tensor<8x16xi32>
// CHECK-SAME: ) -> tensor<8x16xi32>
// CHECK-LABEL: func.func @wide_accum(
// CHECK: scalehls.gemm_outlined = @gemm_8x32x16_call0{{(_[0-9]+)?}}
// CHECK: call @gemm_8x32x16_call0{{(_[0-9]+)?}}(%arg0, %arg1, %arg2) : (tensor<8x32xi8>, tensor<32x16xi8>, tensor<8x16xi32>) -> tensor<8x16xi32>
func.func @wide_accum(%arg0: tensor<8x32xi8>, %arg1: tensor<32x16xi8>, %arg2: tensor<8x16xi32>) -> tensor<8x16xi32> {
  %0 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 0, 1>} : (tensor<8x32xi8>, tensor<32x16xi8>, tensor<8x16xi32>) -> tensor<8x16xi32>
  return %0 : tensor<8x16xi32>
}

// -----

// CHECK-LABEL: func.func private @gemm_8x32x16_call0{{(_[0-9]+)?}}(
// CHECK-LABEL: func.func private @gemm_4x6x5_call0(
// CHECK-LABEL: func.func @multi_helper(
// CHECK-SAME: scalehls.gemm_outlined_helpers = [@gemm_8x32x16_call0{{(_[0-9]+)?}}, @gemm_4x6x5_call0]
// CHECK-NOT: scalehls.gemm_outlined =
// CHECK: %[[CALL0:.*]] = call @gemm_8x32x16_call0{{(_[0-9]+)?}}(%arg0, %arg1) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK: %[[CALL1:.*]] = call @gemm_4x6x5_call0(%arg2, %arg3) : (tensor<4x6xf32>, tensor<6x5xf32>) -> tensor<4x5xf32>
func.func @multi_helper(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<4x6xf32>, %arg3: tensor<6x5xf32>) -> (tensor<8x16xf32>, tensor<4x5xf32>) {
  %0 = "accel.gemm"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %1 = "accel.gemm"(%arg2, %arg3) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (tensor<4x6xf32>, tensor<6x5xf32>) -> tensor<4x5xf32>
  return %0, %1 : tensor<8x16xf32>, tensor<4x5xf32>
}

// -----

// CHECK-LABEL: func.func private @gemm_8x8x8_ip2x2x2(
// CHECK-SAME: tensor<8x8xf32>, tensor<8x8xf32>
// CHECK-LABEL: func.func private @gemm_8x8x8_ip2x2x2_1(
// CHECK-SAME: tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>
// CHECK-LABEL: func.func @serial_helper_contract_collision(
// CHECK-SAME: scalehls.gemm_outlined_helpers = [@gemm_8x8x8_ip2x2x2, @gemm_8x8x8_ip2x2x2_1]
// CHECK: %[[CALL0:.*]] = call @gemm_8x8x8_ip2x2x2(%arg0, %arg1) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: %[[CALL1:.*]] = call @gemm_8x8x8_ip2x2x2_1(%arg0, %arg1, %arg2) : (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<8x8xf32>
func.func @serial_helper_contract_collision(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %0 = "accel.gemm"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>, scalehls.accelerator_ip_tile_size = "2x2x2", scalehls.accelerator_ip_tiling_mode = "serial"} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = "accel.gemm"(%arg0, %arg1, %arg2) {operand_segment_sizes = array<i32: 1, 1, 1, 0>, scalehls.accelerator_ip_tile_size = "2x2x2", scalehls.accelerator_ip_tiling_mode = "serial"} : (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func private @gemm_8x8x8_ip2x2x2{{(_[0-9]+)?}}(
// CHECK-LABEL: func.func @serial_helper_cross_candidate_reuse(
// CHECK-SAME: scalehls.gemm_outlined = @gemm_8x8x8_ip2x2x2{{(_[0-9]+)?}}
// CHECK: %[[CALL0:.*]] = call @gemm_8x8x8_ip2x2x2{{(_[0-9]+)?}}(%arg0, %arg1) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: %[[CALL1:.*]] = call @gemm_8x8x8_ip2x2x2{{(_[0-9]+)?}}(%arg0, %arg1) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
func.func @serial_helper_cross_candidate_reuse(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %0 = "accel.gemm"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>, scalehls.accelerator_ip_tile_size = "2x2x2", scalehls.accelerator_ip_tiling_mode = "serial", scalehls.gemm_candidate_index = 0 : i64} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = "accel.gemm"(%arg0, %arg1) {operand_segment_sizes = array<i32: 1, 1, 0, 0>, scalehls.accelerator_ip_tile_size = "2x2x2", scalehls.accelerator_ip_tiling_mode = "serial", scalehls.gemm_candidate_index = 7 : i64} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}
