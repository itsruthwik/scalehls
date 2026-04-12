//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "scalehls/Dialect/Accel/Accel.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace scalehls;
using namespace accel;

namespace {
static LogicalResult verifyCommonValueSemantics(Operation *op,
                                                Value input,
                                                Value weight,
                                                Value bias,
                                                Value existingInput,
                                                Value result) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !weightType || !resultType)
    return op->emitOpError("requires ranked tensor operands and result");
  if (!inputType.hasStaticShape() || !weightType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return op->emitOpError("requires static shapes");
  if (inputType.getElementType() != weightType.getElementType() ||
      inputType.getElementType() != resultType.getElementType())
    return op->emitOpError("requires a consistent element type");

  for (Value optional : {bias, existingInput}) {
    if (!optional)
      continue;
    auto optionalType = dyn_cast<RankedTensorType>(optional.getType());
    if (!optionalType || !optionalType.hasStaticShape())
      return op->emitOpError("requires optional operands to be static ranked tensors");
    if (optionalType.getElementType() != resultType.getElementType())
      return op->emitOpError("requires optional operands to match result element type");
  }
  return success();
}
} // namespace

void AccelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "scalehls/Dialect/Accel/Accel.cpp.inc"
      >();
}

LogicalResult GEMMVOp::verify() {
  return verifyCommonValueSemantics(*this, getInput(), getWeight(), getBias(),
                                    getExistingInput(), getResult());
}

LogicalResult GEMMOp::verify() {
  return verifyCommonValueSemantics(*this, getInput(), getWeight(), getBias(),
                                    getExistingInput(), getResult());
}

LogicalResult CONVOp::verify() {
  if (failed(verifyCommonValueSemantics(*this, getInput(), getWeight(), getBias(),
                                        getExistingInput(), getResult())))
    return failure();
  if (getStrides().size() != 2)
    return emitOpError("requires exactly two stride values");
  if (getDilations().size() != 2)
    return emitOpError("requires exactly two dilation values");
  return success();
}

#define GET_OP_CLASSES
#include "scalehls/Dialect/Accel/Accel.cpp.inc"

#include "scalehls/Dialect/Accel/AccelDialect.cpp.inc"
