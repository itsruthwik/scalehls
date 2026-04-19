// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s
// RUN: scalehls-opt -split-input-file -c-accel-pipeline="top-func=gemmv_strict_f32" %s | FileCheck %s --check-prefix=GEMM-CALL
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s --check-prefix=TRANSPOSED-RHS
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel="max-normalized-operand-elements=64 allow-large-normalization=false" %s | FileCheck %s --check-prefix=TRANSPOSED-RHS-SKIP
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s --check-prefix=CONV-GEMM
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel="max-normalized-operand-elements=128 allow-large-normalization=false" %s | FileCheck %s --check-prefix=CONV-GEMM-SKIP
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel="max-normalized-operand-elements=128" %s | FileCheck %s --check-prefix=CONV-GEMM-ALLOW
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s --check-prefix=SEMANTIC
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel="max-normalized-operand-elements=16 allow-large-normalization=false" %s | FileCheck %s --check-prefix=SEMANTIC-SKIP
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel="max-normalized-operand-elements=16" %s | FileCheck %s --check-prefix=SEMANTIC-OVERRIDE

// CHECK-LABEL: func.func @gemmv_strict_f32(
// CHECK: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<8x16xf32>
// CHECK: %[[B:.*]] = bufferization.to_tensor %arg2 : memref<16x8xf32>
// CHECK: %[[C:.*]] = bufferization.to_tensor %arg0 : memref<8x8xf32>
// CHECK: %[[R:.*]] = "accel.gemm"(%[[A]], %[[B]], %[[C]])
// CHECK-SAME: operand_segment_sizes = array<i32: 1, 1, 0, 1>
// CHECK-SAME: scalehls.accelerator_family = "GEMM"
// CHECK: memref.copy
func.func @gemmv_strict_f32(%C: memref<8x8xf32>, %A: memref<8x16xf32>, %B: memref<16x8xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 16 {
        %a = affine.load %A[%i, %k] : memref<8x16xf32>
        %b = affine.load %B[%k, %j] : memref<16x8xf32>
        %prod = arith.mulf %a, %b : f32
        %c = affine.load %C[%i, %j] : memref<8x8xf32>
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<8x8xf32>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @gemm_wide_accum_i32(
// CHECK: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<8x16xi8>
// CHECK: %[[B:.*]] = bufferization.to_tensor %arg2 : memref<16x8xi8>
// CHECK: %[[C:.*]] = bufferization.to_tensor %arg0 : memref<8x8xi32>
// CHECK: %[[R:.*]] = "accel.gemm"(%[[A]], %[[B]], %[[C]])
// CHECK-SAME: operand_segment_sizes = array<i32: 1, 1, 0, 1>
// CHECK: memref.copy
func.func @gemm_wide_accum_i32(%C: memref<8x8xi32>, %A: memref<8x16xi8>, %B: memref<16x8xi8>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 16 {
        %a = affine.load %A[%i, %k] : memref<8x16xi8>
        %a32 = arith.extsi %a : i8 to i32
        %b = affine.load %B[%k, %j] : memref<16x8xi8>
        %b32 = arith.extsi %b : i8 to i32
        %prod = arith.muli %a32, %b32 : i32
        %c = affine.load %C[%i, %j] : memref<8x8xi32>
        %sum = arith.addi %c, %prod : i32
        affine.store %sum, %C[%i, %j] : memref<8x8xi32>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @gemmv_iter_args_f32(
// CHECK: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<8x16xf32>
// CHECK: %[[B:.*]] = bufferization.to_tensor %arg2 : memref<16x4xf32>
// CHECK: "accel.gemm"(%[[A]], %[[B]])
func.func @gemmv_iter_args_f32(%O: memref<8x4xf32>, %A: memref<8x16xf32>, %B: memref<16x4xf32>) {
  %c0 = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 4 {
      %sum = affine.for %k = 0 to 16 iter_args(%acc = %c0) -> (f32) {
        %a = affine.load %A[%i, %k] : memref<8x16xf32>
        %b = affine.load %B[%k, %j] : memref<16x4xf32>
        %prod = arith.mulf %a, %b : f32
        %next = arith.addf %acc, %prod : f32
        affine.yield %next : f32
      }
      affine.store %sum, %O[%i, %j] : memref<8x4xf32>
    }
  }
  return
}

// -----

// GEMM-CALL-LABEL: func.func private @gemm_8x16x8_call0(
// GEMM-CALL-SAME: memref<8x16xf32>, memref<16x8xf32>, memref<8x8xf32>, memref<8x8xf32>
// GEMM-CALL-LABEL: func.func @gemmv_strict_f32(
// GEMM-CALL: scalehls.gemm_outlined = @gemm_8x16x8_call0
// GEMM-CALL: %[[TMP:.*]] = memref.alloc() : memref<8x8xf32>
// GEMM-CALL: call @gemm_8x16x8_call0(%arg1, %arg2, %arg0, %[[TMP]])
// GEMM-CALL: affine.load %[[TMP]]

// -----

// TRANSPOSED-RHS-LABEL: func.func @gemmv_transposed_rhs_f32(
// TRANSPOSED-RHS: %[[BT:.*]] = memref.alloc() : memref<16x8xf32>
// TRANSPOSED-RHS: memref.load %arg2
// TRANSPOSED-RHS: memref.store
// TRANSPOSED-RHS: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<8x16xf32>
// TRANSPOSED-RHS: %[[B:.*]] = bufferization.to_tensor %[[BT]] : memref<16x8xf32>
// TRANSPOSED-RHS: %[[C:.*]] = bufferization.to_tensor %arg0 : memref<8x8xf32>
// TRANSPOSED-RHS: "accel.gemm"(%[[A]], %[[B]], %[[C]])
// TRANSPOSED-RHS-SAME: scalehls.accelerator_family = "GEMM"
// TRANSPOSED-RHS-SKIP-LABEL: func.func @gemmv_transposed_rhs_f32(
// TRANSPOSED-RHS-SKIP-SAME: scalehls.gemm_skip_reason = "transposed GEMM weight materialization exceeds max-normalized-operand-elements"
// TRANSPOSED-RHS-SKIP-NOT: "accel.gemm"
// TRANSPOSED-RHS-SKIP: return
func.func @gemmv_transposed_rhs_f32(%C: memref<8x8xf32>, %A: memref<8x16xf32>, %B: memref<8x16xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 16 {
        %a = affine.load %A[%i, %k] : memref<8x16xf32>
        %b = affine.load %B[%j, %k] : memref<8x16xf32>
        %prod = arith.mulf %a, %b : f32
        %c = affine.load %C[%i, %j] : memref<8x8xf32>
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<8x8xf32>
      }
    }
  }
  return
}

// -----

// CONV-GEMM-LABEL: func.func @conv2d_iter_args_i8(
// CONV-GEMM: memref.alloc() : memref<3x18xi8>
// CONV-GEMM: memref.alloc() : memref<18x9xi8>
// CONV-GEMM: bufferization.to_tensor {{.*}} : memref<3x18xi8>
// CONV-GEMM: bufferization.to_tensor {{.*}} : memref<18x9xi8>
// CONV-GEMM: memref.alloc() : memref<3x9xi8>
// CONV-GEMM: bufferization.to_tensor {{.*}} : memref<3x9xi8>
// CONV-GEMM: "accel.gemm"({{.*}}) {operand_segment_sizes = array<i32: 1, 1, 0, 1>, scalehls.accelerator_family = "GEMM"
// CONV-GEMM-NOT: "accel.conv"
// CONV-GEMM-SKIP-LABEL: func.func @conv2d_iter_args_i8(
// CONV-GEMM-SKIP: attributes {scalehls.gemm_skip_reason = "direct CONV packing exceeds max-normalized-operand-elements"}
// CONV-GEMM-SKIP: return
// CONV-GEMM-ALLOW-LABEL: func.func @conv2d_iter_args_i8(
// CONV-GEMM-ALLOW: "accel.gemm"
func.func @conv2d_iter_args_i8(%O: memref<1x3x3x3xi8>, %I: memref<1x2x5x5xi8>, %W: memref<3x2x3x3xi8>) {
  affine.for %oc = 0 to 3 {
    affine.for %oh = 0 to 3 {
      affine.for %ow = 0 to 3 {
        %out0 = affine.load %O[0, %oc, %oh, %ow] : memref<1x3x3x3xi8>
        %sum_ic = affine.for %ic = 0 to 2 iter_args(%acc_ic = %out0) -> (i8) {
          %sum_kh = affine.for %kh = 0 to 3 iter_args(%acc_kh = %acc_ic) -> (i8) {
            %sum_kw = affine.for %kw = 0 to 3 iter_args(%acc_kw = %acc_kh) -> (i8) {
              %in = affine.load %I[0, %ic, %oh + %kh, %ow + %kw] : memref<1x2x5x5xi8>
              %in32 = arith.extsi %in : i8 to i32
              %w = affine.load %W[%oc, %ic, %kh, %kw] : memref<3x2x3x3xi8>
              %w32 = arith.extsi %w : i8 to i32
              %prod = arith.muli %in32, %w32 : i32
              %acc32 = arith.extsi %acc_kw : i8 to i32
              %next32 = arith.addi %acc32, %prod : i32
              %next = arith.trunci %next32 : i32 to i8
              affine.yield %next : i8
            }
            affine.yield %sum_kw : i8
          }
          affine.yield %sum_kh : i8
        }
        affine.store %sum_ic, %O[0, %oc, %oh, %ow] : memref<1x3x3x3xi8>
      }
    }
  }
  return
}

// SEMANTIC-LABEL: func.func @gemm_semantic_rhs_bias_relu_i8(
// SEMANTIC: %[[TMP:.*]] = memref.alloc() : memref<8x4xi8>
// SEMANTIC: affine.for
// SEMANTIC: affine.store
// SEMANTIC: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<8x8xi8>
// SEMANTIC: %[[B:.*]] = bufferization.to_tensor %[[TMP]] : memref<8x4xi8>
// SEMANTIC: %[[R:.*]] = "accel.gemm"(%[[A]], %[[B]])
// SEMANTIC: %[[RM:.*]] = bufferization.to_memref %[[R]] : memref<8x4xi8>
// SEMANTIC: memref.copy %[[RM]], %arg0
// SEMANTIC-SKIP-LABEL: func.func @gemm_semantic_rhs_bias_relu_i8(
// SEMANTIC-SKIP-SAME: scalehls.gemm_skip_reason = "semantic GEMM normalization for rhs operand exceeds max-normalized-operand-elements"
// SEMANTIC-SKIP-NOT: "accel.gemm"
// SEMANTIC-SKIP: return
// SEMANTIC-OVERRIDE-LABEL: func.func @gemm_semantic_rhs_bias_relu_i8(
// SEMANTIC-OVERRIDE: memref.alloc() : memref<8x4xi8>
// SEMANTIC-OVERRIDE: "accel.gemm"
func.func @gemm_semantic_rhs_bias_relu_i8(%O: memref<8x4xi8>, %A: memref<8x8xi8>, %X: memref<8x4xi8>, %Bias: memref<8xi8>) {
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c7_i32 = arith.constant 7 : i32
  %c64_i32 = arith.constant 64 : i32
  %c127_i32 = arith.constant 127 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 4 {
      %sum = affine.for %k = 0 to 8 iter_args(%acc = %c0_i32) -> (i32) {
        %a = affine.load %A[%i, %k] : memref<8x8xi8>
        %a32 = arith.extsi %a : i8 to i32
        %x = affine.load %X[%k, %j] : memref<8x4xi8>
        %x32 = arith.extsi %x : i8 to i32
        %b = affine.load %Bias[%k] : memref<8xi8>
        %b32 = arith.extsi %b : i8 to i32
        %biased = arith.addi %x32, %b32 : i32
        %scaled = arith.shli %biased, %c1_i32 : i32
        %rounded = arith.addi %scaled, %c64_i32 : i32
        %requant = arith.shrsi %rounded, %c7_i32 : i32
        %is_neg = arith.cmpi slt, %requant, %c0_i32 : i32
        %is_sat_hi = scf.if %is_neg -> (i1) {
          scf.yield %false : i1
        } else {
          %hi = arith.cmpi sgt, %requant, %c127_i32 : i32
          scf.yield %hi : i1
        }
        %relu_lo = arith.select %is_neg, %c0_i32, %requant : i32
        %relu = arith.select %is_sat_hi, %c127_i32, %relu_lo : i32
        %relu8 = arith.trunci %relu : i32 to i8
        %relu32 = arith.extsi %relu8 : i8 to i32
        %prod = arith.muli %a32, %relu32 : i32
        %next = arith.addi %acc, %prod : i32
        affine.yield %next : i32
      }
      %out = arith.trunci %sum : i32 to i8
      affine.store %out, %O[%i, %j] : memref<8x4xi8>
    }
  }
  return
}

// -----

// SEMANTIC-LABEL: func.func @gemm_semantic_both_sides_i8(
// SEMANTIC: %[[LHS:.*]] = memref.alloc() : memref<4x4xi8>
// SEMANTIC: %[[RHS:.*]] = memref.alloc() : memref<4x4xi8>
// SEMANTIC: %[[A:.*]] = bufferization.to_tensor %[[LHS]] : memref<4x4xi8>
// SEMANTIC: %[[B:.*]] = bufferization.to_tensor %[[RHS]] : memref<4x4xi8>
// SEMANTIC: %[[R:.*]] = "accel.gemm"(%[[A]], %[[B]])
// SEMANTIC: %[[RM:.*]] = bufferization.to_memref %[[R]] : memref<4x4xi8>
// SEMANTIC: memref.copy %[[RM]], %arg0
func.func @gemm_semantic_both_sides_i8(%O: memref<4x4xi8>, %A: memref<4x4xi8>, %ABias: memref<4xi8>, %B: memref<4x4xi8>, %BBias: memref<4xi8>) {
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c7_i32 = arith.constant 7 : i32
  %c64_i32 = arith.constant 64 : i32
  %c127_i32 = arith.constant 127 : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %sum = affine.for %k = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
        %a = affine.load %A[%i, %k] : memref<4x4xi8>
        %a32 = arith.extsi %a : i8 to i32
        %ab = affine.load %ABias[%i] : memref<4xi8>
        %ab32 = arith.extsi %ab : i8 to i32
        %lhs_biased = arith.addi %a32, %ab32 : i32
        %lhs_scaled = arith.shli %lhs_biased, %c1_i32 : i32
        %lhs_rounded = arith.addi %lhs_scaled, %c64_i32 : i32
        %lhs_requant = arith.shrsi %lhs_rounded, %c7_i32 : i32
        %lhs_neg = arith.cmpi slt, %lhs_requant, %c0_i32 : i32
        %lhs_sat_hi = scf.if %lhs_neg -> (i1) {
          scf.yield %false : i1
        } else {
          %lhs_hi = arith.cmpi sgt, %lhs_requant, %c127_i32 : i32
          scf.yield %lhs_hi : i1
        }
        %lhs_relu_lo = arith.select %lhs_neg, %c0_i32, %lhs_requant : i32
        %lhs_relu = arith.select %lhs_sat_hi, %c127_i32, %lhs_relu_lo : i32
        %lhs8 = arith.trunci %lhs_relu : i32 to i8
        %lhs32 = arith.extsi %lhs8 : i8 to i32
        %b = affine.load %B[%k, %j] : memref<4x4xi8>
        %b32 = arith.extsi %b : i8 to i32
        %bb = affine.load %BBias[%k] : memref<4xi8>
        %bb32 = arith.extsi %bb : i8 to i32
        %rhs_biased = arith.addi %b32, %bb32 : i32
        %rhs_scaled = arith.shli %rhs_biased, %c1_i32 : i32
        %rhs_rounded = arith.addi %rhs_scaled, %c64_i32 : i32
        %rhs_requant = arith.shrsi %rhs_rounded, %c7_i32 : i32
        %rhs_neg = arith.cmpi slt, %rhs_requant, %c0_i32 : i32
        %rhs_sat_hi = scf.if %rhs_neg -> (i1) {
          scf.yield %false : i1
        } else {
          %rhs_hi = arith.cmpi sgt, %rhs_requant, %c127_i32 : i32
          scf.yield %rhs_hi : i1
        }
        %rhs_relu_lo = arith.select %rhs_neg, %c0_i32, %rhs_requant : i32
        %rhs_relu = arith.select %rhs_sat_hi, %c127_i32, %rhs_relu_lo : i32
        %rhs8 = arith.trunci %rhs_relu : i32 to i8
        %rhs32 = arith.extsi %rhs8 : i8 to i32
        %prod = arith.muli %lhs32, %rhs32 : i32
        %next = arith.addi %acc, %prod : i32
        affine.yield %next : i32
      }
      %out = arith.trunci %sum : i32 to i8
      affine.store %out, %O[%i, %j] : memref<4x4xi8>
    }
  }
  return
}

// -----

// SEMANTIC-LABEL: func.func @gemm_semantic_high_rank_weight_i8(
// SEMANTIC: %[[WV:.*]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 8], strides: [8, 1]
// SEMANTIC: %[[A:.*]] = bufferization.to_tensor %[[WV]] : memref<8x8xi8>
// SEMANTIC: %[[B:.*]] = bufferization.to_tensor %arg2 : memref<8x36xi8>
// SEMANTIC: %[[R:.*]] = "accel.gemm"(%[[A]], %[[B]])
// SEMANTIC: %[[RM:.*]] = bufferization.to_memref %[[R]] : memref<8x36xi8>
// SEMANTIC: memref.copy %[[RM]], %arg0
func.func @gemm_semantic_high_rank_weight_i8(%O: memref<8x36xi8>, %W: memref<8x8x1x1xi8>, %X: memref<8x36xi8>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 36 {
      %sum = affine.for %k = 0 to 8 iter_args(%acc = %c0_i32) -> (i32) {
        %w = affine.load %W[%i, %k, 0, 0] : memref<8x8x1x1xi8>
        %w32 = arith.extsi %w : i8 to i32
        %x = affine.load %X[%k, %j] : memref<8x36xi8>
        %x32 = arith.extsi %x : i8 to i32
        %prod = arith.muli %w32, %x32 : i32
        %next = arith.addi %acc, %prod : i32
        affine.yield %next : i32
      }
      %out = arith.trunci %sum : i32 to i8
      affine.store %out, %O[%i, %j] : memref<8x36xi8>
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @gemm_semantic_high_rank_weight_padded_i8(
// CHECK-SAME: scalehls.gemm_skip_reason = "matched affine candidate requires static ranked tensor views"
// CHECK-NOT: "accel.gemm"
func.func @gemm_semantic_high_rank_weight_padded_i8(%O: memref<8x36xi8>, %W: memref<8x8x1x1xi8, strided<[16, 2, 2, 2]>>, %X: memref<8x36xi8>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 36 {
      %sum = affine.for %k = 0 to 8 iter_args(%acc = %c0_i32) -> (i32) {
        %w = affine.load %W[%i, %k, 0, 0] : memref<8x8x1x1xi8, strided<[16, 2, 2, 2]>>
        %w32 = arith.extsi %w : i8 to i32
        %x = affine.load %X[%k, %j] : memref<8x36xi8>
        %x32 = arith.extsi %x : i8 to i32
        %prod = arith.muli %w32, %x32 : i32
        %next = arith.addi %acc, %prod : i32
        affine.yield %next : i32
      }
      %out = arith.trunci %sum : i32 to i8
      affine.store %out, %O[%i, %j] : memref<8x36xi8>
    }
  }
  return
}
