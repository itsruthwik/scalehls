// RUN: scalehls-translate -scalehls-emit-hlscpp %s | FileCheck %s

func.func @test_affine_for_iter_args(%arg0: memref<8x8xf32>, %arg1: memref<8x8xi8>, %arg2: memref<8x8xi8>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %sum = affine.for %k = 0 to 8 iter_args(%acc = %cst) -> (f32) {
        %a = affine.load %arg1[%i, %k] : memref<8x8xi8>
        %af = arith.sitofp %a : i8 to f32
        %b = affine.load %arg2[%j, %k] : memref<8x8xi8>
        %bf = arith.sitofp %b : i8 to f32
        %prod = arith.mulf %af, %bf : f32
        %next = arith.addf %acc, %prod : f32
        affine.yield %next : f32
      }
      affine.store %sum, %arg0[%i, %j] : memref<8x8xf32>
    }
  }
  return
}

// CHECK-LABEL: void test_affine_for_iter_args(
// CHECK: for (int
// CHECK: float
// CHECK: = (float)0.000000;
// CHECK: for (int
// CHECK: ap_int<8>
// CHECK: float
// CHECK: float
// CHECK: = {{.*}} * {{.*}};
// CHECK: float
// CHECK: = {{.*}} + {{.*}};
// CHECK: {{.*}} = {{.*}};
// CHECK: v0
