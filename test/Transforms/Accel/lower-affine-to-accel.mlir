// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s
// RUN: scalehls-opt -split-input-file -c-accel-pipeline="top-func=gemmv_strict_f32" %s | FileCheck %s --check-prefix=GEMMV-CALL
// RUN: scalehls-opt -split-input-file -c-accel-pipeline="top-func=batched_gemm_strict_f32" %s | FileCheck %s --check-prefix=GEMM-CALL
// RUN: scalehls-opt -split-input-file -c-accel-pipeline="top-func=conv2d_valid_f32" %s | FileCheck %s --check-prefix=CONV-CALL
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s --check-prefix=GEMMV-I8
// RUN: scalehls-opt -split-input-file -c-accel-pipeline="top-func=gemmv_strict_i8" %s | FileCheck %s --check-prefix=GEMMV-I8-CALL
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s --check-prefix=CONV-I8
// RUN: scalehls-opt -split-input-file -c-accel-pipeline="top-func=conv2d_valid_i8" %s | FileCheck %s --check-prefix=CONV-I8-CALL
// RUN: scalehls-opt -split-input-file -lower-affine-to-accel %s | FileCheck %s --check-prefix=GEMMV-PTR
// RUN: scalehls-opt -split-input-file -c-accel-pipeline="top-func=gemmv_fp32_ptr" %s | FileCheck %s --check-prefix=GEMMV-PTR-CALL

// GEMMV-LABEL: func.func @gemmv_strict_f32(
// GEMMV: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<8x16xf32>
// GEMMV: %[[B:.*]] = bufferization.to_tensor %arg2 : memref<16x8xf32>
// GEMMV: %[[C:.*]] = bufferization.to_tensor %arg0 : memref<8x8xf32>
// GEMMV: %[[R:.*]] = "accel.gemmv"(%[[A]], %[[B]], %[[C]])
// GEMMV-SAME: operand_segment_sizes = array<i32: 1, 1, 0, 1>
// GEMMV: scalehls.accelerator_family = "GEMMV"
// GEMMV: %[[RM:.*]] = bufferization.to_memref %[[R]] : memref<8x8xf32>
// GEMMV: memref.copy %[[RM]], %arg0 : memref<8x8xf32> to memref<8x8xf32>
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

// GEMMV-CALL-LABEL: func.func private @gemmv_strict_f32_accel_gemmv_f32_in8x16_w16x8_ei8x8_out8x8(
// GEMMV-CALL-SAME: memref<8x16xf32>, memref<16x8xf32>, memref<8x8xf32>, memref<8x8xf32>
// GEMMV-CALL: scalehls.accelerator_family = "GEMMV"
// GEMMV-CALL-LABEL: func.func @gemmv_strict_f32(
// GEMMV-CALL: %[[TMP:.*]] = memref.alloc() : memref<8x8xf32>
// GEMMV-CALL: call @gemmv_strict_f32_accel_gemmv_f32_in8x16_w16x8_ei8x8_out8x8(%arg1, %arg2, %arg0, %[[TMP]])
// GEMMV-CALL: memref.copy %[[TMP]], %arg0 : memref<8x8xf32> to memref<8x8xf32>
// GEMMV-CALL: return

// -----

// GEMM-LABEL: func.func @batched_gemm_strict_f32(
// GEMM: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<2x4x6xf32>
// GEMM: %[[B:.*]] = bufferization.to_tensor %arg2 : memref<2x6x5xf32>
// GEMM: %[[C:.*]] = bufferization.to_tensor %arg0 : memref<2x4x5xf32>
// GEMM: %[[R:.*]] = "accel.gemm"(%[[A]], %[[B]], %[[C]])
// GEMM-SAME: operand_segment_sizes = array<i32: 1, 1, 0, 1>
// GEMM: scalehls.accelerator_family = "GEMM"
// GEMM: %[[RM:.*]] = bufferization.to_memref %[[R]] : memref<2x4x5xf32>
// GEMM: memref.copy %[[RM]], %arg0 : memref<2x4x5xf32> to memref<2x4x5xf32>
func.func @batched_gemm_strict_f32(%C: memref<2x4x5xf32>, %A: memref<2x4x6xf32>, %B: memref<2x6x5xf32>) {
  affine.for %b = 0 to 2 {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 5 {
        affine.for %k = 0 to 6 {
          %a = affine.load %A[%b, %i, %k] : memref<2x4x6xf32>
          %w = affine.load %B[%b, %k, %j] : memref<2x6x5xf32>
          %prod = arith.mulf %a, %w : f32
          %c = affine.load %C[%b, %i, %j] : memref<2x4x5xf32>
          %sum = arith.addf %c, %prod : f32
          affine.store %sum, %C[%b, %i, %j] : memref<2x4x5xf32>
        }
      }
    }
  }
  return
}

// -----

// GEMM-CALL-LABEL: func.func private @batched_gemm_strict_f32_accel_gemm_f32_in2x4x6_w2x6x5_ei2x4x5_out2x4x5(
// GEMM-CALL-SAME: memref<2x4x6xf32>, memref<2x6x5xf32>, memref<2x4x5xf32>, memref<2x4x5xf32>
// GEMM-CALL: scalehls.accelerator_family = "GEMM"
// GEMM-CALL-LABEL: func.func @batched_gemm_strict_f32(
// GEMM-CALL: %[[TMP:.*]] = memref.alloc() : memref<2x4x5xf32>
// GEMM-CALL: call @batched_gemm_strict_f32_accel_gemm_f32_in2x4x6_w2x6x5_ei2x4x5_out2x4x5(%arg1, %arg2, %arg0, %[[TMP]])
// GEMM-CALL: memref.copy %[[TMP]], %arg0 : memref<2x4x5xf32> to memref<2x4x5xf32>
// GEMM-CALL: return

// -----

// CONV-LABEL: func.func @conv2d_valid_f32(
// CONV: %[[IN:.*]] = bufferization.to_tensor %arg1 : memref<1x2x5x5xf32>
// CONV: %[[W:.*]] = bufferization.to_tensor %arg2 : memref<3x2x3x3xf32>
// CONV: %[[OUT:.*]] = bufferization.to_tensor %arg0 : memref<1x3x3x3xf32>
// CONV: %[[R:.*]] = "accel.conv"(%[[IN]], %[[W]], %[[OUT]])
// CONV-SAME: dilations = array<i64: 1, 1>
// CONV-SAME: operand_segment_sizes = array<i32: 1, 1, 0, 1>
// CONV-SAME: scalehls.accelerator_family = "CONV"
// CONV-SAME: strides = array<i64: 1, 1>
// CONV: %[[RM:.*]] = bufferization.to_memref %[[R]] : memref<1x3x3x3xf32>
// CONV: memref.copy %[[RM]], %arg0 : memref<1x3x3x3xf32> to memref<1x3x3x3xf32>
func.func @conv2d_valid_f32(%O: memref<1x3x3x3xf32>, %I: memref<1x2x5x5xf32>, %W: memref<3x2x3x3xf32>) {
  affine.for %oc = 0 to 3 {
    affine.for %oh = 0 to 3 {
      affine.for %ow = 0 to 3 {
        affine.for %ic = 0 to 2 {
          affine.for %kh = 0 to 3 {
            affine.for %kw = 0 to 3 {
              %in = affine.load %I[0, %ic, %oh + %kh, %ow + %kw] : memref<1x2x5x5xf32>
              %w = affine.load %W[%oc, %ic, %kh, %kw] : memref<3x2x3x3xf32>
              %prod = arith.mulf %in, %w : f32
              %out = affine.load %O[0, %oc, %oh, %ow] : memref<1x3x3x3xf32>
              %sum = arith.addf %out, %prod : f32
              affine.store %sum, %O[0, %oc, %oh, %ow] : memref<1x3x3x3xf32>
            }
          }
        }
      }
    }
  }
  return
}

// -----

// CONV-CALL-LABEL: func.func private @conv2d_valid_f32_accel_conv_f32_in1x2x5x5_w3x2x3x3_ei1x3x3x3_out1x3x3x3(
// CONV-CALL-SAME: memref<1x2x5x5xf32>, memref<3x2x3x3xf32>, memref<1x3x3x3xf32>, memref<1x3x3x3xf32>
// CONV-CALL: scalehls.accelerator_family = "CONV"
// CONV-CALL-LABEL: func.func @conv2d_valid_f32(
// CONV-CALL: %[[TMP:.*]] = memref.alloc() : memref<1x3x3x3xf32>
// CONV-CALL: call @conv2d_valid_f32_accel_conv_f32_in1x2x5x5_w3x2x3x3_ei1x3x3x3_out1x3x3x3(%arg1, %arg2, %arg0, %[[TMP]])
// CONV-CALL: memref.copy %[[TMP]], %arg0 : memref<1x3x3x3xf32> to memref<1x3x3x3xf32>
// CONV-CALL: return

// -----

// CHECK-LABEL: func.func @test_not_family(
// CHECK: affine.for
func.func @test_not_family(%A: memref<8x8xf32>, %B: memref<8x8xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %a = affine.load %A[%i, %j] : memref<8x8xf32>
      %b = affine.load %B[%i, %j] : memref<8x8xf32>
      %sum = arith.addf %a, %b : f32
      affine.store %sum, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}

// -----

// GEMMV-I8-LABEL: func.func @gemmv_strict_i8(
// GEMMV-I8: %[[A:.*]] = bufferization.to_tensor %arg1 : memref<8x16xi8>
// GEMMV-I8: %[[B:.*]] = bufferization.to_tensor %arg2 : memref<16x8xi8>
// GEMMV-I8: %[[C:.*]] = bufferization.to_tensor %arg0 : memref<8x8xi8>
// GEMMV-I8: %[[R:.*]] = "accel.gemmv"(%[[A]], %[[B]], %[[C]])
// GEMMV-I8-SAME: scalehls.accelerator_family = "GEMMV"
// GEMMV-I8-SAME: scalehls.gemm_precision = "i"
// GEMMV-I8-CALL-LABEL: func.func private @gemmv_strict_i8_accel_gemmv_i_in8x16_w16x8_ei8x8_out8x8(
// GEMMV-I8-CALL-SAME: memref<8x16xi8>, memref<16x8xi8>, memref<8x8xi8>, memref<8x8xi8>
// GEMMV-I8-CALL: scalehls.accelerator_family = "GEMMV"
// GEMMV-I8-CALL-LABEL: func.func @gemmv_strict_i8(
// GEMMV-I8-CALL: %[[TMP:.*]] = memref.alloc() : memref<8x8xi8>
// GEMMV-I8-CALL: call @gemmv_strict_i8_accel_gemmv_i_in8x16_w16x8_ei8x8_out8x8(%arg1, %arg2, %arg0, %[[TMP]])
// GEMMV-I8-CALL: memref.copy %[[TMP]], %arg0 : memref<8x8xi8> to memref<8x8xi8>
func.func @gemmv_strict_i8(%C: memref<8x8xi8>, %A: memref<8x16xi8>, %B: memref<16x8xi8>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 16 {
        %a = affine.load %A[%i, %k] : memref<8x16xi8>
        %a32 = arith.extsi %a : i8 to i32
        %b = affine.load %B[%k, %j] : memref<16x8xi8>
        %b32 = arith.extsi %b : i8 to i32
        %prod = arith.muli %a32, %b32 : i32
        %c = affine.load %C[%i, %j] : memref<8x8xi8>
        %prod_i8 = arith.trunci %prod : i32 to i8
        %sum = arith.addi %c, %prod_i8 : i8
        affine.store %sum, %C[%i, %j] : memref<8x8xi8>
      }
    }
  }
  return
}

// -----

// CONV-I8-LABEL: func.func @conv2d_valid_i8(
// CONV-I8: %[[IN:.*]] = bufferization.to_tensor %arg1 : memref<1x2x5x5xi8>
// CONV-I8: %[[W:.*]] = bufferization.to_tensor %arg2 : memref<3x2x3x3xi8>
// CONV-I8: %[[OUT:.*]] = bufferization.to_tensor %arg0 : memref<1x3x3x3xi8>
// CONV-I8: %[[R:.*]] = "accel.conv"(%[[IN]], %[[W]], %[[OUT]])
// CONV-I8-SAME: scalehls.accelerator_family = "CONV"
// CONV-I8-SAME: scalehls.gemm_precision = "i"
// CONV-I8-CALL-LABEL: func.func private @conv2d_valid_i8_accel_conv_i_in1x2x5x5_w3x2x3x3_ei1x3x3x3_out1x3x3x3(
// CONV-I8-CALL-SAME: memref<1x2x5x5xi8>, memref<3x2x3x3xi8>, memref<1x3x3x3xi8>, memref<1x3x3x3xi8>
// CONV-I8-CALL: scalehls.accelerator_family = "CONV"
// CONV-I8-CALL-LABEL: func.func @conv2d_valid_i8(
// CONV-I8-CALL: %[[TMP:.*]] = memref.alloc() : memref<1x3x3x3xi8>
// CONV-I8-CALL: call @conv2d_valid_i8_accel_conv_i_in1x2x5x5_w3x2x3x3_ei1x3x3x3_out1x3x3x3(%arg1, %arg2, %arg0, %[[TMP]])
// CONV-I8-CALL: memref.copy %[[TMP]], %arg0 : memref<1x3x3x3xi8> to memref<1x3x3x3xi8>
func.func @conv2d_valid_i8(%O: memref<1x3x3x3xi8>, %I: memref<1x2x5x5xi8>, %W: memref<3x2x3x3xi8>) {
  affine.for %oc = 0 to 3 {
    affine.for %oh = 0 to 3 {
      affine.for %ow = 0 to 3 {
        affine.for %ic = 0 to 2 {
          affine.for %kh = 0 to 3 {
            affine.for %kw = 0 to 3 {
              %in = affine.load %I[0, %ic, %oh + %kh, %ow + %kw] : memref<1x2x5x5xi8>
              %in32 = arith.extsi %in : i8 to i32
              %w = affine.load %W[%oc, %ic, %kh, %kw] : memref<3x2x3x3xi8>
              %w32 = arith.extsi %w : i8 to i32
              %prod = arith.muli %in32, %w32 : i32
              %out = affine.load %O[0, %oc, %oh, %ow] : memref<1x3x3x3xi8>
              %out32 = arith.extsi %out : i8 to i32
              %sum = arith.addi %out32, %prod : i32
              %sum_i8 = arith.trunci %sum : i32 to i8
              affine.store %sum_i8, %O[0, %oc, %oh, %ow] : memref<1x3x3x3xi8>
            }
          }
        }
      }
    }
  }
  return
}

// -----

// GEMMV-PTR-LABEL: func.func @gemmv_fp32_ptr(
// GEMMV-PTR: %[[A2D:.*]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 16], strides: [16, 1]
// GEMMV-PTR: %[[B2D:.*]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 8], strides: [8, 1]
// GEMMV-PTR: %[[C2D:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 8], strides: [8, 1]
// GEMMV-PTR: %[[A:.*]] = bufferization.to_tensor %[[A2D]] : memref<8x16xf32>
// GEMMV-PTR: %[[B:.*]] = bufferization.to_tensor %[[B2D]] : memref<16x8xf32>
// GEMMV-PTR: %[[C:.*]] = bufferization.to_tensor %[[C2D]] : memref<8x8xf32>
// GEMMV-PTR: %[[R:.*]] = "accel.gemmv"(%[[A]], %[[B]], %[[C]])
// GEMMV-PTR-SAME: scalehls.accelerator_family = "GEMMV"
// GEMMV-PTR-CALL-LABEL: func.func private @gemmv_fp32_ptr_accel_gemmv_f32_in8x16_w16x8_ei8x8_out8x8(
// GEMMV-PTR-CALL-SAME: memref<8x16xf32>, memref<16x8xf32>, memref<8x8xf32>, memref<8x8xf32>
func.func @gemmv_fp32_ptr(%C: memref<?xf32>, %A: memref<?xf32>, %B: memref<?xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 16 {
        %a = affine.load %A[%k + %i * 16] : memref<?xf32>
        %b = affine.load %B[%j + %k * 8] : memref<?xf32>
        %prod = arith.mulf %a, %b : f32
        %c = affine.load %C[%j + %i * 8] : memref<?xf32>
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%j + %i * 8] : memref<?xf32>
      }
    }
  }
  return
}
