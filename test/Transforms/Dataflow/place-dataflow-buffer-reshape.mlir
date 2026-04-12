// RUN: scalehls-opt -scalehls-place-dataflow-buffer %s | FileCheck %s

// CHECK-LABEL: func.func @preserve_collapse_shape_memory_space(
// CHECK-SAME: %[[ARG0:.*]]: memref<1x512x1x1xf32, 12>, %[[ARG1:.*]]: memref<1x10xf32, 12>
// CHECK: %[[BUF:.*]] = hls.dataflow.buffer {depth = 1 : i32} : memref<1x512x1x1xf32, 7>
// CHECK: memref.copy %[[ARG0]], %[[BUF]] : memref<1x512x1x1xf32, 12> to memref<1x512x1x1xf32, 7>
// CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %[[BUF]]
// CHECK-SAME: : memref<1x512x1x1xf32, 7> into memref<1x512xf32, 7>
func.func @preserve_collapse_shape_memory_space(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x10xf32>) {
  hls.dataflow.dispatch {
    %0 = hls.dataflow.buffer {depth = 1 : i32} : memref<1x512x1x1xf32>
    memref.copy %arg0, %0 : memref<1x512x1x1xf32> to memref<1x512x1x1xf32>
    %collapsed = memref.collapse_shape %0 [[0], [1, 2, 3]] : memref<1x512x1x1xf32> into memref<1x512xf32>
    %1 = hls.dataflow.buffer {depth = 1 : i32} : memref<512x10xf32>
    %2 = hls.dataflow.buffer {depth = 1 : i32, init_value = 0.000000e+00 : f32} : memref<1x10xf32>
    linalg.matmul ins(%collapsed, %1 : memref<1x512xf32>, memref<512x10xf32>) outs(%2 : memref<1x10xf32>)
    memref.copy %2, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  }
  return
}
