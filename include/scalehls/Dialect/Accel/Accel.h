//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_DIALECT_ACCEL_ACCEL_H
#define SCALEHLS_DIALECT_ACCEL_ACCEL_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "scalehls/Dialect/Accel/AccelDialect.h.inc"

namespace mlir {
namespace scalehls {
namespace accel {
} // namespace accel
} // namespace scalehls
} // namespace mlir

#define GET_OP_CLASSES
#include "scalehls/Dialect/Accel/Accel.h.inc"

#endif // SCALEHLS_DIALECT_ACCEL_ACCEL_H
