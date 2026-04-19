//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "scalehls/Dialect/Accel/Accel.h"
using namespace mlir;
using namespace scalehls;
using namespace accel;

namespace {
static LogicalResult verifyShapeEquality(Operation *op, RankedTensorType lhs,
                                         RankedTensorType rhs,
                                         StringRef message) {
  if (lhs.getShape() != rhs.getShape())
    return op->emitOpError(message);
  return success();
}

static LogicalResult verifyBiasSemantics(Operation *op, RankedTensorType biasType,
                                         RankedTensorType resultType) {
  if (biasType.getElementType() != resultType.getElementType())
    return op->emitOpError("requires bias element type to match result element type");

  if (biasType.getRank() == 1) {
    int64_t expectedBiasExtent =
        resultType.getRank() == 4 ? resultType.getShape()[1]
                                  : resultType.getShape().back();
    if (biasType.getShape().front() != expectedBiasExtent) {
      return op->emitOpError(
          resultType.getRank() == 4
              ? "requires rank-1 bias length to match the output channel dimension"
              : "requires rank-1 bias length to match the GEMM N dimension");
    }
    return success();
  }

  if (biasType.getRank() != resultType.getRank())
    return op->emitOpError("requires bias to be rank-1 or match the result rank");

  return verifyShapeEquality(op, biasType, resultType,
                             "requires rank-matched bias to match the result shape");
}

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
  if (inputType.getRank() != weightType.getRank() ||
      inputType.getRank() != resultType.getRank())
    return op->emitOpError("requires input, weight, and result to have the same rank");
  if (inputType.getRank() != 2 && inputType.getRank() != 3 &&
      inputType.getRank() != 4)
    return op->emitOpError(
        "requires 2D, 3D, or legacy conv-lowered 4D tensors");
  if (inputType.getElementType() != weightType.getElementType())
    return op->emitOpError("requires input and weight element types to match");

  if (inputType.getRank() == 2) {
    if (inputType.getShape()[1] != weightType.getShape()[0] ||
        resultType.getShape()[0] != inputType.getShape()[0] ||
        resultType.getShape()[1] != weightType.getShape()[1]) {
      return op->emitOpError(
          "requires 2D GEMM shapes A[M,K], B[K,N], result[M,N]");
    }
  } else if (inputType.getRank() == 3) {
    if (inputType.getShape()[0] != weightType.getShape()[0] ||
        inputType.getShape()[0] != resultType.getShape()[0] ||
        inputType.getShape()[2] != weightType.getShape()[1] ||
        resultType.getShape()[1] != inputType.getShape()[1] ||
        resultType.getShape()[2] != weightType.getShape()[2]) {
      return op->emitOpError(
          "requires 3D GEMM shapes A[B,M,K], B[B,K,N], result[B,M,N]");
    }
  } else {
    if (inputType.getShape()[0] != resultType.getShape()[0] ||
        inputType.getShape()[1] != weightType.getShape()[1] ||
        resultType.getShape()[1] != weightType.getShape()[0]) {
      return op->emitOpError(
          "requires legacy conv-lowered 4D tensors input[N,IC,IH,IW], "
          "weight[OC,IC,KH,KW], result[N,OC,OH,OW]");
    }
  }

  if (bias) {
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType || !biasType.hasStaticShape())
      return op->emitOpError("requires bias to be a static ranked tensor");
    if (failed(verifyBiasSemantics(op, biasType, resultType)))
      return failure();
  }

  if (existingInput) {
    auto existingType = dyn_cast<RankedTensorType>(existingInput.getType());
    if (!existingType || !existingType.hasStaticShape())
      return op->emitOpError("requires existing_input to be a static ranked tensor");
    if (existingType.getElementType() != resultType.getElementType())
      return op->emitOpError(
          "requires existing_input element type to match result element type");
    if (failed(verifyShapeEquality(op, existingType, resultType,
                                   "requires existing_input shape to match result shape")))
      return failure();
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

LogicalResult GEMMOp::verify() {
  return verifyCommonValueSemantics(*this, getInput(), getWeight(), getBias(),
                                    getExistingInput(), getResult());
}

#define GET_OP_CLASSES
#include "scalehls/Dialect/Accel/Accel.cpp.inc"

#include "scalehls/Dialect/Accel/AccelDialect.cpp.inc"
