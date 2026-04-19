// RUN: scalehls-opt -lower-affine-to-accel %s | FileCheck %s

// CHECK-LABEL: func.func @direct_cnn_three_stage_i8(
// CHECK: "accel.gemm"
// CHECK-SAME: scalehls.gemm_shape = [1, 2, 16]
// CHECK: "accel.gemm"
// CHECK-SAME: scalehls.gemm_shape = [1, 2, 4]
// CHECK-NOT: "accel.conv"
func.func @direct_cnn_three_stage_i8(%O: memref<4x2x2xi8>, %I: memref<2x4x4xi8>, %W1: memref<2x2x1x1xi8>, %W2: memref<2x2x3x3xi8>, %W3: memref<4x2x1x1xi8>) {
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c7_i32 = arith.constant 7 : i32
  %c64_i32 = arith.constant 64 : i32
  %c127_i32 = arith.constant 127 : i32
  %Y1 = memref.alloc() : memref<2x4x4xi8>
  %A1 = memref.alloc() : memref<2x4x4xi8>
  %Y2 = memref.alloc() : memref<2x2x2xi8>
  %A2 = memref.alloc() : memref<2x2x2xi8>

  affine.for %oc = 0 to 2 {
    affine.for %oh = 0 to 4 {
      affine.for %ow = 0 to 4 {
        %sum = affine.for %ic = 0 to 2 iter_args(%acc = %c0_i32) -> (i32) {
          %w = affine.load %W1[%oc, %ic, 0, 0] : memref<2x2x1x1xi8>
          %w32 = arith.extsi %w : i8 to i32
          %in = affine.load %I[%ic, %oh, %ow] : memref<2x4x4xi8>
          %in32 = arith.extsi %in : i8 to i32
          %prod = arith.muli %w32, %in32 : i32
          %next = arith.addi %acc, %prod : i32
          affine.yield %next : i32
        }
        %out = arith.trunci %sum : i32 to i8
        affine.store %out, %Y1[%oc, %oh, %ow] : memref<2x4x4xi8>
      }
    }
  }

  affine.for %oc = 0 to 2 {
    affine.for %oh = 0 to 4 {
      affine.for %ow = 0 to 4 {
        %v = affine.load %Y1[%oc, %oh, %ow] : memref<2x4x4xi8>
        %v32 = arith.extsi %v : i8 to i32
        %scaled = arith.shli %v32, %c1_i32 : i32
        %rounded = arith.addi %scaled, %c64_i32 : i32
        %requant = arith.shrsi %rounded, %c7_i32 : i32
        %neg = arith.cmpi slt, %requant, %c0_i32 : i32
        %relu = arith.select %neg, %c0_i32, %requant : i32
        %relu8 = arith.trunci %relu : i32 to i8
        affine.store %relu8, %A1[%oc, %oh, %ow] : memref<2x4x4xi8>
      }
    }
  }

  affine.for %oc = 0 to 2 {
    affine.for %oh = 0 to 2 {
      affine.for %ow = 0 to 2 {
        %sum = affine.for %ic = 0 to 2 iter_args(%acc_ic = %c0_i32) -> (i32) {
          %sum_kh = affine.for %kh = 0 to 3 iter_args(%acc_kh = %acc_ic) -> (i32) {
            %sum_kw = affine.for %kw = 0 to 3 iter_args(%acc_kw = %acc_kh) -> (i32) {
              %w = affine.load %W2[%oc, %ic, %kh, %kw] : memref<2x2x3x3xi8>
              %w32 = arith.extsi %w : i8 to i32
              %in = affine.load %A1[%ic, %oh + %kh, %ow + %kw] : memref<2x4x4xi8>
              %in32 = arith.extsi %in : i8 to i32
              %scaled = arith.shli %in32, %c1_i32 : i32
              %rounded = arith.addi %scaled, %c64_i32 : i32
              %requant = arith.shrsi %rounded, %c7_i32 : i32
              %neg = arith.cmpi slt, %requant, %c0_i32 : i32
              %sat_hi = scf.if %neg -> (i1) {
                scf.yield %false : i1
              } else {
                %hi = arith.cmpi sgt, %requant, %c127_i32 : i32
                scf.yield %hi : i1
              }
              %relu_lo = arith.select %neg, %c0_i32, %requant : i32
              %relu = arith.select %sat_hi, %c127_i32, %relu_lo : i32
              %relu8 = arith.trunci %relu : i32 to i8
              %relu32 = arith.extsi %relu8 : i8 to i32
              %prod = arith.muli %w32, %relu32 : i32
              %next = arith.addi %acc_kw, %prod : i32
              affine.yield %next : i32
            }
            affine.yield %sum_kw : i32
          }
          affine.yield %sum_kh : i32
        }
        %out = arith.trunci %sum : i32 to i8
        affine.store %out, %Y2[%oc, %oh, %ow] : memref<2x2x2xi8>
      }
    }
  }

  affine.for %oc = 0 to 2 {
    affine.for %oh = 0 to 2 {
      affine.for %ow = 0 to 2 {
        %v = affine.load %Y2[%oc, %oh, %ow] : memref<2x2x2xi8>
        %v32 = arith.extsi %v : i8 to i32
        %scaled = arith.shli %v32, %c1_i32 : i32
        %rounded = arith.addi %scaled, %c64_i32 : i32
        %requant = arith.shrsi %rounded, %c7_i32 : i32
        %neg = arith.cmpi slt, %requant, %c0_i32 : i32
        %relu = arith.select %neg, %c0_i32, %requant : i32
        %relu8 = arith.trunci %relu : i32 to i8
        affine.store %relu8, %A2[%oc, %oh, %ow] : memref<2x2x2xi8>
      }
    }
  }

  affine.for %oc = 0 to 4 {
    affine.for %oh = 0 to 2 {
      affine.for %ow = 0 to 2 {
        %sum = affine.for %ic = 0 to 2 iter_args(%acc = %c0_i32) -> (i32) {
          %w = affine.load %W3[%oc, %ic, 0, 0] : memref<4x2x1x1xi8>
          %w32 = arith.extsi %w : i8 to i32
          %in = affine.load %A2[%ic, %oh, %ow] : memref<2x2x2xi8>
          %in32 = arith.extsi %in : i8 to i32
          %scaled = arith.shli %in32, %c1_i32 : i32
          %rounded = arith.addi %scaled, %c64_i32 : i32
          %requant = arith.shrsi %rounded, %c7_i32 : i32
          %neg = arith.cmpi slt, %requant, %c0_i32 : i32
          %relu = arith.select %neg, %c0_i32, %requant : i32
          %relu8 = arith.trunci %relu : i32 to i8
          %relu32 = arith.extsi %relu8 : i8 to i32
          %prod = arith.muli %w32, %relu32 : i32
          %next = arith.addi %acc, %prod : i32
          affine.yield %next : i32
        }
        %out = arith.trunci %sum : i32 to i8
        affine.store %out, %O[%oc, %oh, %ow] : memref<4x2x2xi8>
      }
    }
  }
  return
}
