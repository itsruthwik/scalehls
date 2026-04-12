// RUN: scalehls-translate -scalehls-emit-hlscpp %s | FileCheck %s

func.func private @fd_helper(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) attributes {scalehls.accelerator_abi_mode = "full-data"}

func.func @wrapper(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> f32 attributes {func_directive = #hls.fd<pipeline=false, targetInterval=1, dataflow=false>} {
  // CHECK-LABEL: void fd_helper(
  // CHECK: float arg0,
  // CHECK: float arg1,
  // CHECK: float arg2,
  // CHECK: float arg3,
  // CHECK: float &result0,
  // CHECK: float &result1
  // CHECK-LABEL: void wrapper(
  // CHECK: float v5;
  // CHECK: float v6;
  // CHECK: fd_helper(v0, v1, v2, v3, v5, v6);
  %0:2 = call @fd_helper(%arg0, %arg1, %arg2, %arg3) : (f32, f32, f32, f32) -> (f32, f32)
  %1 = arith.addf %0#0, %0#1 : f32
  return %1 : f32
}
