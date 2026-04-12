//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "scalehls/Dialect/Accel/Accel.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

namespace {
constexpr StringLiteral kDetectedAttr = "scalehls.gemm";
constexpr StringLiteral kContractAttr = "scalehls.gemm_contract";
constexpr StringLiteral kAffineContract = "affine_family";
constexpr StringLiteral kFamilyAttr = "scalehls.accelerator_family";
constexpr StringLiteral kShapeAttr = "scalehls.gemm_shape";
constexpr StringLiteral kAShapeAttr = "scalehls.gemm_a_shape";
constexpr StringLiteral kBShapeAttr = "scalehls.gemm_b_shape";
constexpr StringLiteral kCShapeAttr = "scalehls.gemm_c_shape";
constexpr StringLiteral kPrecisionAttr = "scalehls.gemm_precision";
constexpr StringLiteral kElementBitsAttr = "scalehls.gemm_element_bits";
constexpr StringLiteral kElementBytesAttr = "scalehls.gemm_element_bytes";
constexpr StringLiteral kHasBiasAttr = "scalehls.accelerator_has_bias";
constexpr StringLiteral kParentFuncAttr = "scalehls.gemm_parent_func";
constexpr StringLiteral kCandidateIndexAttr = "scalehls.gemm_candidate_index";

struct AffineCandidateInfo {
  StringRef family;
  Value inputMemref;
  Value weightMemref;
  Value outputMemref;
  SmallVector<int64_t, 4> inputShape;
  SmallVector<int64_t, 4> weightShape;
  SmallVector<int64_t, 4> outputShape;
  DenseI64ArrayAttr strides;
  DenseI64ArrayAttr dilations;
};

static SmallVector<Operation *, 8> getBodyOps(Block &block) {
  SmallVector<Operation *, 8> ops;
  for (Operation &op : block.without_terminator())
    ops.push_back(&op);
  return ops;
}

static bool isRectangularLoop(AffineForOp loop) {
  return loop.hasConstantLowerBound() && loop.getConstantLowerBound() == 0 &&
         loop.hasConstantUpperBound() && loop.getStep() == 1;
}

static bool isStaticZeroBasedLoop(AffineForOp loop) {
  return loop.hasConstantLowerBound() && loop.getConstantLowerBound() == 0 &&
         loop.hasConstantUpperBound() && loop.getStep() > 0;
}

static RankedTensorType getTensorType(Value memref,
                                      ArrayRef<int64_t> fallbackShape = {}) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType)
    return RankedTensorType();
  if (memrefType.hasStaticShape())
    return RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
  if (fallbackShape.empty())
    return RankedTensorType();
  return RankedTensorType::get(fallbackShape, memrefType.getElementType());
}

static ArrayAttr buildI64ArrayAttr(MLIRContext *context,
                                   ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(IntegerAttr::get(IntegerType::get(context, 64), value));
  return ArrayAttr::get(context, attrs);
}

static void annotateOperandShape(Operation *op, StringRef attrName,
                                 ArrayRef<int64_t> shape) {
  if (shape.empty())
    return;
  op->setAttr(attrName, buildI64ArrayAttr(op->getContext(), shape));
}

static StringRef getPrecisionString(Type type) {
  if (type.isF16())
    return StringRef("f16");
  if (type.isF32())
    return StringRef("f32");
  if (type.isF64())
    return StringRef("f64");
  if (auto integerType = dyn_cast<IntegerType>(type))
    return integerType.isUnsigned() ? StringRef("ui") : StringRef("i");
  return StringRef("unknown");
}

static void annotatePrecision(Operation *op, Type elementType) {
  unsigned bits = elementType.getIntOrFloatBitWidth();
  unsigned bytes = (bits + 7) / 8;
  op->setAttr(kPrecisionAttr,
              StringAttr::get(op->getContext(), getPrecisionString(elementType)));
  op->setAttr(kElementBitsAttr,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), bits));
  op->setAttr(kElementBytesAttr,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), bytes));
}

static void annotateCandidateAttrs(Operation *op, func::FuncOp func,
                                   const AffineCandidateInfo &candidate) {
  auto outputType = getTensorType(candidate.outputMemref, candidate.outputShape);
  if (!outputType)
    return;
  op->setAttr(kDetectedAttr, UnitAttr::get(op->getContext()));
  op->setAttr(kContractAttr, StringAttr::get(op->getContext(), kAffineContract));
  op->setAttr(kFamilyAttr, StringAttr::get(op->getContext(), candidate.family));
  op->setAttr(kHasBiasAttr, BoolAttr::get(op->getContext(), false));
  annotateOperandShape(op, kShapeAttr, candidate.outputShape);
  annotateOperandShape(op, kAShapeAttr, candidate.inputShape);
  annotateOperandShape(op, kBShapeAttr, candidate.weightShape);
  annotateOperandShape(op, kCShapeAttr, candidate.outputShape);
  annotatePrecision(op, outputType.getElementType());
  op->setAttr(kParentFuncAttr, StringAttr::get(op->getContext(), func.getName()));
  op->setAttr(kCandidateIndexAttr,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), 0));
}

static Optional<unsigned> findOperandPosition(Value value,
                                              ValueRange operands) {
  for (auto [index, operand] : llvm::enumerate(operands)) {
    if (operand == value)
      return index;
  }
  return llvm::None;
}

static bool matchesAffineExpr(AffineExpr expr, ValueRange operands,
                              Value value) {
  auto position = findOperandPosition(value, operands);
  return position.has_value() &&
         expr == getAffineDimExpr(*position, expr.getContext());
}

static bool matchesAffineExpr(AffineExpr expr, ValueRange operands, Value lhs,
                              Value rhs) {
  auto lhsPosition = findOperandPosition(lhs, operands);
  auto rhsPosition = findOperandPosition(rhs, operands);
  if (!lhsPosition.has_value() || !rhsPosition.has_value())
    return false;
  auto lhsExpr = getAffineDimExpr(*lhsPosition, expr.getContext());
  auto rhsExpr = getAffineDimExpr(*rhsPosition, expr.getContext());
  return expr == lhsExpr + rhsExpr || expr == rhsExpr + lhsExpr;
}

static bool matchesAffineExprPlusConstant(AffineExpr expr, ValueRange operands,
                                          Value value, int64_t constant) {
  auto position = findOperandPosition(value, operands);
  if (!position.has_value())
    return false;
  auto context = expr.getContext();
  auto valueExpr = getAffineDimExpr(*position, context);
  auto constExpr = getAffineConstantExpr(constant, context);
  return simplifyAffineExpr(expr, operands.size(), 0) ==
         simplifyAffineExpr(valueExpr + constExpr, operands.size(), 0);
}

static bool matchesLinearCombinationExpr(
    AffineExpr expr, ValueRange operands,
    ArrayRef<std::pair<Value, int64_t>> terms, int64_t constant = 0) {
  auto *context = expr.getContext();
  AffineExpr expected = getAffineConstantExpr(constant, context);
  for (auto [value, coefficient] : terms) {
    auto position = findOperandPosition(value, operands);
    if (!position.has_value())
      return false;
    AffineExpr term = getAffineDimExpr(*position, context);
    if (coefficient != 1)
      term = term * getAffineConstantExpr(coefficient, context);
    expected = expected + term;
  }
  return simplifyAffineExpr(expr, operands.size(), 0) ==
         simplifyAffineExpr(expected, operands.size(), 0);
}

static bool matchesLinearized2DExpr(AffineExpr expr, ValueRange operands,
                                    Value outer, int64_t innerExtent,
                                    Value inner) {
  return matchesLinearCombinationExpr(
      expr, operands, {{outer, innerExtent}, {inner, 1}});
}

static bool matchesLinearized3DExpr(AffineExpr expr, ValueRange operands,
                                    Value dim0, int64_t stride0, Value dim1,
                                    int64_t stride1, Value dim2) {
  return matchesLinearCombinationExpr(
      expr, operands, {{dim0, stride0}, {dim1, stride1}, {dim2, 1}});
}

static bool matchesAffineExpr(AffineExpr expr, int64_t constant) {
  return expr == getAffineConstantExpr(constant, expr.getContext());
}

static bool sameMemrefAccess(AffineLoadOp load, Value memref,
                             ArrayRef<function_ref<bool(AffineExpr, ValueRange)>> checks) {
  if (load.getMemRef() != memref)
    return false;
  auto map = load.getAffineMap();
  auto operands = load.getMapOperands();
  if (map.getNumResults() != static_cast<int64_t>(checks.size()))
    return false;
  for (auto [expr, check] : llvm::zip(map.getResults(), checks)) {
    if (!check(expr, operands))
      return false;
  }
  return true;
}

static bool sameMemrefAccess(AffineStoreOp store, Value memref,
                             ArrayRef<function_ref<bool(AffineExpr, ValueRange)>> checks) {
  if (store.getMemRef() != memref)
    return false;
  auto map = store.getAffineMap();
  auto operands = store.getMapOperands();
  if (map.getNumResults() != static_cast<int64_t>(checks.size()))
    return false;
  for (auto [expr, check] : llvm::zip(map.getResults(), checks)) {
    if (!check(expr, operands))
      return false;
  }
  return true;
}

static Value stripIntegerCasts(Value value) {
  while (true) {
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto trunc = value.getDefiningOp<arith::TruncIOp>()) {
      value = trunc.getIn();
      continue;
    }
    return value;
  }
}

static bool matchesMulInputs(Value lhs, Value rhs, Value inputValue,
                             Value weightValue) {
  Value strippedLhs = stripIntegerCasts(lhs);
  Value strippedRhs = stripIntegerCasts(rhs);
  return (strippedLhs == inputValue && strippedRhs == weightValue) ||
         (strippedLhs == weightValue && strippedRhs == inputValue);
}

static bool matchesReductionComputation(ArrayRef<Operation *> ops,
                                        AffineLoadOp inputLoad,
                                        AffineLoadOp weightLoad,
                                        AffineLoadOp outputLoad,
                                        AffineStoreOp store) {
  arith::MulFOp mulf;
  arith::MulIOp muli;
  arith::AddFOp addf;
  arith::AddIOp addi;

  for (Operation *op : ops) {
    if (isa<AffineLoadOp, AffineStoreOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::TruncIOp>(op))
      continue;
    if (!mulf)
      mulf = dyn_cast<arith::MulFOp>(op);
    if (!muli)
      muli = dyn_cast<arith::MulIOp>(op);
    if (!addf)
      addf = dyn_cast<arith::AddFOp>(op);
    if (!addi)
      addi = dyn_cast<arith::AddIOp>(op);
    if (!isa<arith::MulFOp, arith::MulIOp, arith::AddFOp, arith::AddIOp>(op))
      return false;
  }

  Value mulValue;
  if (mulf) {
    if (!matchesMulInputs(mulf.getLhs(), mulf.getRhs(), inputLoad.getResult(),
                          weightLoad.getResult()))
      return false;
    mulValue = mulf.getResult();
  } else if (muli) {
    if (!matchesMulInputs(muli.getLhs(), muli.getRhs(), inputLoad.getResult(),
                          weightLoad.getResult()))
      return false;
    mulValue = muli.getResult();
  } else {
    return false;
  }

  Value addValue;
  if (addf) {
    Value lhs = addf.getLhs();
    Value rhs = addf.getRhs();
    bool lhsIsOutput = stripIntegerCasts(lhs) == outputLoad.getResult();
    bool rhsIsOutput = stripIntegerCasts(rhs) == outputLoad.getResult();
    if (lhsIsOutput == rhsIsOutput)
      return false;
    Value other = lhsIsOutput ? rhs : lhs;
    if (stripIntegerCasts(other) != mulValue)
      return false;
    addValue = addf.getResult();
  } else if (addi) {
    Value lhs = addi.getLhs();
    Value rhs = addi.getRhs();
    bool lhsIsOutput = stripIntegerCasts(lhs) == outputLoad.getResult();
    bool rhsIsOutput = stripIntegerCasts(rhs) == outputLoad.getResult();
    if (lhsIsOutput == rhsIsOutput)
      return false;
    Value other = lhsIsOutput ? rhs : lhs;
    if (stripIntegerCasts(other) != mulValue)
      return false;
    addValue = addi.getResult();
  } else {
    return false;
  }

  return stripIntegerCasts(store.getValueToStore()) == addValue;
}

static bool matchStoreStyleReduction2D(ArrayRef<Operation *> ops, Value row,
                                       Value col, Value red, Value outputMemref,
                                       Value &inputMemref,
                                       Value &weightMemref) {
  SmallVector<AffineLoadOp, 3> loads;
  AffineStoreOp store;
  for (Operation *op : ops) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (!store)
      store = dyn_cast<AffineStoreOp>(op);
  }
  if (loads.size() != 3 || !store)
    return false;

  auto rowCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, row);
  };
  auto colCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, col);
  };
  auto redCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, red);
  };

  AffineLoadOp outputLoad;
  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    if (sameMemrefAccess(load, outputMemref, {rowCheck, colCheck})) {
      outputLoad = load;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(), {rowCheck, redCheck})) {
      inputLoad = load;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(), {redCheck, colCheck})) {
      weightLoad = load;
      continue;
    }
  }

  if (!outputLoad || !inputLoad || !weightLoad)
    return false;
  if (!sameMemrefAccess(store, outputMemref, {rowCheck, colCheck}))
    return false;
  if (!matchesReductionComputation(ops, inputLoad, weightLoad, outputLoad,
                                   store))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  if (inputMemref == outputMemref || weightMemref == outputMemref ||
      inputMemref == weightMemref)
    return false;
  return true;
}

static bool matchStoreStyleReduction2DLinearized(ArrayRef<Operation *> ops,
                                                 Value row, Value col, Value red,
                                                 int64_t colExtent,
                                                 int64_t redExtent,
                                                 Value outputMemref,
                                                 Value &inputMemref,
                                                 Value &weightMemref) {
  SmallVector<AffineLoadOp, 3> loads;
  AffineStoreOp store;
  for (Operation *op : ops) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (!store)
      store = dyn_cast<AffineStoreOp>(op);
  }
  if (loads.size() != 3 || !store)
    return false;

  auto outputCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearized2DExpr(expr, operands, row, colExtent, col);
  };
  auto inputCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearized2DExpr(expr, operands, row, redExtent, red);
  };
  auto weightCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearized2DExpr(expr, operands, red, colExtent, col);
  };

  AffineLoadOp outputLoad;
  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    auto map = load.getAffineMap();
    auto operands = load.getMapOperands();
    if (map.getNumResults() != 1)
      continue;
    auto expr = map.getResult(0);
    if (load.getMemRef() == outputMemref && outputCheck(expr, operands)) {
      outputLoad = load;
      continue;
    }
    if (inputCheck(expr, operands)) {
      inputLoad = load;
      continue;
    }
    if (weightCheck(expr, operands)) {
      weightLoad = load;
      continue;
    }
  }

  if (!outputLoad || !inputLoad || !weightLoad)
    return false;

  auto storeMap = store.getAffineMap();
  auto storeOperands = store.getMapOperands();
  if (store.getMemRef() != outputMemref || storeMap.getNumResults() != 1 ||
      !outputCheck(storeMap.getResult(0), storeOperands))
    return false;

  if (!matchesReductionComputation(ops, inputLoad, weightLoad, outputLoad, store))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  if (inputMemref == outputMemref || weightMemref == outputMemref ||
      inputMemref == weightMemref)
    return false;
  return true;
}

static bool matchStoreStyleReduction3D(ArrayRef<Operation *> ops, Value batch,
                                       Value row, Value col, Value red,
                                       Value outputMemref, Value &inputMemref,
                                       Value &weightMemref) {
  SmallVector<AffineLoadOp, 3> loads;
  AffineStoreOp store;
  for (Operation *op : ops) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (!store)
      store = dyn_cast<AffineStoreOp>(op);
  }
  if (loads.size() != 3 || !store)
    return false;

  auto batchCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, batch);
  };
  auto rowCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, row);
  };
  auto colCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, col);
  };
  auto redCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, red);
  };

  AffineLoadOp outputLoad;
  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    if (sameMemrefAccess(load, outputMemref, {batchCheck, rowCheck, colCheck})) {
      outputLoad = load;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(),
                         {batchCheck, rowCheck, redCheck})) {
      inputLoad = load;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(),
                         {batchCheck, redCheck, colCheck})) {
      weightLoad = load;
      continue;
    }
  }

  if (!outputLoad || !inputLoad || !weightLoad)
    return false;
  if (!sameMemrefAccess(store, outputMemref, {batchCheck, rowCheck, colCheck}))
    return false;
  if (!matchesReductionComputation(ops, inputLoad, weightLoad, outputLoad,
                                   store))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  if (inputMemref == outputMemref || weightMemref == outputMemref ||
      inputMemref == weightMemref)
    return false;
  return true;
}

static bool matchStoreStyleReduction3DLinearized(ArrayRef<Operation *> ops,
                                                 Value batch, Value row,
                                                 Value col, Value red,
                                                 int64_t rowExtent,
                                                 int64_t colExtent,
                                                 int64_t redExtent,
                                                 Value outputMemref,
                                                 Value &inputMemref,
                                                 Value &weightMemref) {
  SmallVector<AffineLoadOp, 3> loads;
  AffineStoreOp store;
  for (Operation *op : ops) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (!store)
      store = dyn_cast<AffineStoreOp>(op);
  }
  if (loads.size() != 3 || !store)
    return false;

  const int64_t inputBatchStride = rowExtent * redExtent;
  const int64_t outputBatchStride = rowExtent * colExtent;
  const int64_t weightBatchStride = redExtent * colExtent;

  auto outputCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearized3DExpr(expr, operands, batch, outputBatchStride,
                                   row, colExtent, col);
  };
  auto inputCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearized3DExpr(expr, operands, batch, inputBatchStride,
                                   row, redExtent, red);
  };
  auto weightCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearized3DExpr(expr, operands, batch, weightBatchStride,
                                   red, colExtent, col);
  };

  AffineLoadOp outputLoad;
  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    auto map = load.getAffineMap();
    auto operands = load.getMapOperands();
    if (map.getNumResults() != 1)
      continue;
    auto expr = map.getResult(0);
    if (load.getMemRef() == outputMemref && outputCheck(expr, operands)) {
      outputLoad = load;
      continue;
    }
    if (inputCheck(expr, operands)) {
      inputLoad = load;
      continue;
    }
    if (weightCheck(expr, operands)) {
      weightLoad = load;
      continue;
    }
  }

  if (!outputLoad || !inputLoad || !weightLoad)
    return false;

  auto storeMap = store.getAffineMap();
  auto storeOperands = store.getMapOperands();
  if (store.getMemRef() != outputMemref || storeMap.getNumResults() != 1 ||
      !outputCheck(storeMap.getResult(0), storeOperands))
    return false;

  if (!matchesReductionComputation(ops, inputLoad, weightLoad, outputLoad,
                                   store))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  if (inputMemref == outputMemref || weightMemref == outputMemref ||
      inputMemref == weightMemref)
    return false;
  return true;
}

static bool matchStoreStyleReductionConv(ArrayRef<Operation *> ops, Value oc,
                                         Value oh, Value ow, Value ic, Value kh,
                                         Value kw, Value outputMemref,
                                         Value &inputMemref,
                                         Value &weightMemref) {
  SmallVector<AffineLoadOp, 3> loads;
  AffineStoreOp store;
  for (Operation *op : ops) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (!store)
      store = dyn_cast<AffineStoreOp>(op);
  }
  if (loads.size() != 3 || !store)
    return false;

  auto zeroCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, 0);
  };
  auto ocCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, oc);
  };
  auto ohCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, oh);
  };
  auto owCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, ow);
  };
  auto icCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, ic);
  };
  auto khCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, kh);
  };
  auto kwCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, kw);
  };
  auto ohPlusKhCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, oh, kh);
  };
  auto owPlusKwCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, ow, kw);
  };

  AffineLoadOp outputLoad;
  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    if (sameMemrefAccess(load, outputMemref,
                         {zeroCheck, ocCheck, ohCheck, owCheck})) {
      outputLoad = load;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(),
                         {zeroCheck, icCheck, ohPlusKhCheck, owPlusKwCheck})) {
      inputLoad = load;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(),
                         {ocCheck, icCheck, khCheck, kwCheck})) {
      weightLoad = load;
      continue;
    }
  }

  if (!outputLoad || !inputLoad || !weightLoad)
    return false;
  if (!sameMemrefAccess(store, outputMemref,
                        {zeroCheck, ocCheck, ohCheck, owCheck}))
    return false;
  if (!matchesReductionComputation(ops, inputLoad, weightLoad, outputLoad,
                                   store))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  if (inputMemref == outputMemref || weightMemref == outputMemref ||
      inputMemref == weightMemref)
    return false;
  return true;
}

static bool matchStoreStyleReductionConvLinearized(
    ArrayRef<Operation *> ops, Value oc, Value oh, Value ow, Value ic, Value kh,
    Value kw, int64_t icExtent, int64_t ihExtent, int64_t iwExtent,
    int64_t ocExtent, int64_t ohExtent, int64_t owExtent, int64_t khExtent,
    int64_t kwExtent, Value outputMemref, Value &inputMemref,
    Value &weightMemref) {
  SmallVector<AffineLoadOp, 3> loads;
  AffineStoreOp store;
  for (Operation *op : ops) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (!store)
      store = dyn_cast<AffineStoreOp>(op);
  }
  if (loads.size() != 3 || !store)
    return false;

  const int64_t inputChannelStride = ihExtent * iwExtent;
  const int64_t outputChannelStride = ohExtent * owExtent;
  const int64_t weightOCStride = icExtent * khExtent * kwExtent;
  const int64_t weightICStride = khExtent * kwExtent;

  auto outputCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearCombinationExpr(
        expr, operands,
        {{oc, outputChannelStride}, {oh, owExtent}, {ow, 1}});
  };
  auto inputCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearCombinationExpr(
        expr, operands,
        {{ic, inputChannelStride}, {oh, iwExtent}, {kh, iwExtent}, {ow, 1},
         {kw, 1}});
  };
  auto weightCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesLinearCombinationExpr(
        expr, operands,
        {{oc, weightOCStride}, {ic, weightICStride}, {kh, kwExtent}, {kw, 1}});
  };

  AffineLoadOp outputLoad;
  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    auto map = load.getAffineMap();
    auto operands = load.getMapOperands();
    if (map.getNumResults() != 1)
      continue;
    auto expr = map.getResult(0);
    if (load.getMemRef() == outputMemref && outputCheck(expr, operands)) {
      outputLoad = load;
      continue;
    }
    if (inputCheck(expr, operands)) {
      inputLoad = load;
      continue;
    }
    if (weightCheck(expr, operands)) {
      weightLoad = load;
      continue;
    }
  }

  if (!outputLoad || !inputLoad || !weightLoad)
    return false;

  auto storeMap = store.getAffineMap();
  auto storeOperands = store.getMapOperands();
  if (store.getMemRef() != outputMemref || storeMap.getNumResults() != 1 ||
      !outputCheck(storeMap.getResult(0), storeOperands))
    return false;

  if (!matchesReductionComputation(ops, inputLoad, weightLoad, outputLoad,
                                   store))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  if (inputMemref == outputMemref || weightMemref == outputMemref ||
      inputMemref == weightMemref)
    return false;
  return true;
}

static bool matchStripMinedInnerLoop(AffineForOp innerLoop,
                                     AffineForOp outerLoop) {
  if (!outerLoop.hasConstantUpperBound() || outerLoop.getStep() <= 0)
    return false;
  if (innerLoop.getStep() != 1)
    return false;
  auto lowerMap = innerLoop.getLowerBoundMap();
  auto lowerOperands = innerLoop.getLowerBoundOperands();
  if (lowerMap.getNumResults() != 1 ||
      !matchesAffineExpr(lowerMap.getResult(0), lowerOperands,
                         outerLoop.getInductionVar()))
    return false;
  auto upperMap = innerLoop.getUpperBoundMap();
  auto upperOperands = innerLoop.getUpperBoundOperands();
  if (upperMap.getNumResults() != 1 ||
      !matchesAffineExprPlusConstant(upperMap.getResult(0), upperOperands,
                                     outerLoop.getInductionVar(),
                                     outerLoop.getStep()))
    return false;
  return true;
}

static LogicalResult matchStripMinedGEMMV(func::FuncOp func,
                                          AffineCandidateInfo &info) {
  auto bodyOps = getBodyOps(func.front());
  if (bodyOps.size() != 1 || !isa<AffineForOp>(bodyOps.front()))
    return failure();
  auto rowTileLoop = cast<AffineForOp>(bodyOps.front());
  if (!isStaticZeroBasedLoop(rowTileLoop))
    return failure();

  auto rowTileOps = getBodyOps(*rowTileLoop.getBody());
  if (rowTileOps.size() != 1 || !isa<AffineForOp>(rowTileOps.front()))
    return failure();
  auto colTileLoop = cast<AffineForOp>(rowTileOps.front());
  if (!isStaticZeroBasedLoop(colTileLoop))
    return failure();

  auto colTileOps = getBodyOps(*colTileLoop.getBody());
  if (colTileOps.size() != 1 || !isa<AffineForOp>(colTileOps.front()))
    return failure();
  auto redTileLoop = cast<AffineForOp>(colTileOps.front());
  if (!isStaticZeroBasedLoop(redTileLoop))
    return failure();

  auto redTileOps = getBodyOps(*redTileLoop.getBody());
  if (redTileOps.size() != 1 || !isa<AffineForOp>(redTileOps.front()))
    return failure();
  auto rowLoop = cast<AffineForOp>(redTileOps.front());
  if (!matchStripMinedInnerLoop(rowLoop, rowTileLoop))
    return failure();

  auto rowOps = getBodyOps(*rowLoop.getBody());
  if (rowOps.size() != 1 || !isa<AffineForOp>(rowOps.front()))
    return failure();
  auto colLoop = cast<AffineForOp>(rowOps.front());
  if (!matchStripMinedInnerLoop(colLoop, colTileLoop))
    return failure();

  auto colOps = getBodyOps(*colLoop.getBody());
  if (colOps.size() != 1 || !isa<AffineForOp>(colOps.front()))
    return failure();
  auto redLoop = cast<AffineForOp>(colOps.front());
  if (!matchStripMinedInnerLoop(redLoop, redTileLoop))
    return failure();

  Value inputMemref;
  Value weightMemref;
  auto redOps = getBodyOps(*redLoop.getBody());
  auto outputStore = dyn_cast<AffineStoreOp>(redOps.back());
  if (!outputStore)
    return failure();
  if (!matchStoreStyleReduction2D(redOps, rowLoop.getInductionVar(),
                                  colLoop.getInductionVar(),
                                  redLoop.getInductionVar(),
                                  outputStore.getMemRef(), inputMemref,
                                  weightMemref))
    if (!matchStoreStyleReduction2DLinearized(
            redOps, rowLoop.getInductionVar(), colLoop.getInductionVar(),
            redLoop.getInductionVar(), colTileLoop.getConstantUpperBound(),
            redTileLoop.getConstantUpperBound(), outputStore.getMemRef(),
            inputMemref, weightMemref))
      return failure();

  info.family = StringRef("GEMMV");
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputStore.getMemRef();
  info.inputShape = {rowTileLoop.getConstantUpperBound(),
                     redTileLoop.getConstantUpperBound()};
  info.weightShape = {redTileLoop.getConstantUpperBound(),
                      colTileLoop.getConstantUpperBound()};
  info.outputShape = {rowTileLoop.getConstantUpperBound(),
                      colTileLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchStripMinedGEMM(func::FuncOp func,
                                         AffineCandidateInfo &info) {
  auto bodyOps = getBodyOps(func.front());
  if (bodyOps.size() != 1 || !isa<AffineForOp>(bodyOps.front()))
    return failure();
  auto batchLoop = cast<AffineForOp>(bodyOps.front());
  if (!isRectangularLoop(batchLoop))
    return failure();

  auto batchOps = getBodyOps(*batchLoop.getBody());
  if (batchOps.size() != 1 || !isa<AffineForOp>(batchOps.front()))
    return failure();
  auto rowTileLoop = cast<AffineForOp>(batchOps.front());
  if (!isStaticZeroBasedLoop(rowTileLoop))
    return failure();

  auto rowTileOps = getBodyOps(*rowTileLoop.getBody());
  if (rowTileOps.size() != 1 || !isa<AffineForOp>(rowTileOps.front()))
    return failure();
  auto redTileLoop = cast<AffineForOp>(rowTileOps.front());
  if (!isStaticZeroBasedLoop(redTileLoop))
    return failure();

  auto redTileOps = getBodyOps(*redTileLoop.getBody());
  if (redTileOps.size() != 1 || !isa<AffineForOp>(redTileOps.front()))
    return failure();
  auto rowLoop = cast<AffineForOp>(redTileOps.front());
  if (!matchStripMinedInnerLoop(rowLoop, rowTileLoop))
    return failure();

  auto rowOps = getBodyOps(*rowLoop.getBody());
  if (rowOps.size() != 1 || !isa<AffineForOp>(rowOps.front()))
    return failure();
  auto colLoop = cast<AffineForOp>(rowOps.front());
  if (!isRectangularLoop(colLoop))
    return failure();

  auto colOps = getBodyOps(*colLoop.getBody());
  if (colOps.size() != 1 || !isa<AffineForOp>(colOps.front()))
    return failure();
  auto redLoop = cast<AffineForOp>(colOps.front());
  if (!matchStripMinedInnerLoop(redLoop, redTileLoop))
    return failure();

  Value inputMemref;
  Value weightMemref;
  auto redOps = getBodyOps(*redLoop.getBody());
  auto outputStore = dyn_cast<AffineStoreOp>(redOps.back());
  if (!outputStore)
    return failure();
  if (!matchStoreStyleReduction3D(redOps, batchLoop.getInductionVar(),
                                  rowLoop.getInductionVar(),
                                  colLoop.getInductionVar(),
                                  redLoop.getInductionVar(),
                                  outputStore.getMemRef(), inputMemref,
                                  weightMemref))
    if (!matchStoreStyleReduction3DLinearized(
            redOps, batchLoop.getInductionVar(), rowLoop.getInductionVar(),
            colLoop.getInductionVar(), redLoop.getInductionVar(),
            rowTileLoop.getConstantUpperBound(), colLoop.getConstantUpperBound(),
            redTileLoop.getConstantUpperBound(), outputStore.getMemRef(),
            inputMemref, weightMemref))
      return failure();

  info.family = StringRef("GEMM");
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputStore.getMemRef();
  info.inputShape = {batchLoop.getConstantUpperBound(),
                     rowTileLoop.getConstantUpperBound(),
                     redTileLoop.getConstantUpperBound()};
  info.weightShape = {batchLoop.getConstantUpperBound(),
                      redTileLoop.getConstantUpperBound(),
                      colLoop.getConstantUpperBound()};
  info.outputShape = {batchLoop.getConstantUpperBound(),
                      rowTileLoop.getConstantUpperBound(),
                      colLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchStripMinedCONV(func::FuncOp func,
                                         AffineCandidateInfo &info) {
  auto bodyOps = getBodyOps(func.front());
  if (bodyOps.size() != 1 || !isa<AffineForOp>(bodyOps.front()))
    return failure();
  auto ocTileLoop = cast<AffineForOp>(bodyOps.front());
  if (!isStaticZeroBasedLoop(ocTileLoop))
    return failure();

  auto ocTileOps = getBodyOps(*ocTileLoop.getBody());
  if (ocTileOps.size() != 1 || !isa<AffineForOp>(ocTileOps.front()))
    return failure();
  auto ohTileLoop = cast<AffineForOp>(ocTileOps.front());
  if (!isStaticZeroBasedLoop(ohTileLoop))
    return failure();

  auto ohTileOps = getBodyOps(*ohTileLoop.getBody());
  if (ohTileOps.size() != 1 || !isa<AffineForOp>(ohTileOps.front()))
    return failure();
  auto owTileLoop = cast<AffineForOp>(ohTileOps.front());
  if (!isStaticZeroBasedLoop(owTileLoop))
    return failure();

  auto owTileOps = getBodyOps(*owTileLoop.getBody());
  if (owTileOps.size() != 1 || !isa<AffineForOp>(owTileOps.front()))
    return failure();
  auto ocLoop = cast<AffineForOp>(owTileOps.front());
  if (!matchStripMinedInnerLoop(ocLoop, ocTileLoop))
    return failure();

  auto ocOps = getBodyOps(*ocLoop.getBody());
  if (ocOps.size() != 1 || !isa<AffineForOp>(ocOps.front()))
    return failure();
  auto ohLoop = cast<AffineForOp>(ocOps.front());
  if (!matchStripMinedInnerLoop(ohLoop, ohTileLoop))
    return failure();

  auto ohOps = getBodyOps(*ohLoop.getBody());
  if (ohOps.size() != 1 || !isa<AffineForOp>(ohOps.front()))
    return failure();
  auto owLoop = cast<AffineForOp>(ohOps.front());
  if (!matchStripMinedInnerLoop(owLoop, owTileLoop))
    return failure();

  auto owOps = getBodyOps(*owLoop.getBody());
  if (owOps.size() != 1 || !isa<AffineForOp>(owOps.front()))
    return failure();
  auto icLoop = cast<AffineForOp>(owOps.front());
  if (!isRectangularLoop(icLoop))
    return failure();

  auto icOps = getBodyOps(*icLoop.getBody());
  if (icOps.size() != 1 || !isa<AffineForOp>(icOps.front()))
    return failure();
  auto khLoop = cast<AffineForOp>(icOps.front());
  if (!isRectangularLoop(khLoop))
    return failure();

  auto khOps = getBodyOps(*khLoop.getBody());
  if (khOps.size() != 1 || !isa<AffineForOp>(khOps.front()))
    return failure();
  auto kwLoop = cast<AffineForOp>(khOps.front());
  if (!isRectangularLoop(kwLoop))
    return failure();

  Value inputMemref;
  Value weightMemref;
  auto redOps = getBodyOps(*kwLoop.getBody());
  auto outputStore = dyn_cast<AffineStoreOp>(redOps.back());
  if (!outputStore)
    return failure();
  if (!matchStoreStyleReductionConv(redOps, ocLoop.getInductionVar(),
                                    ohLoop.getInductionVar(),
                                    owLoop.getInductionVar(),
                                    icLoop.getInductionVar(),
                                    khLoop.getInductionVar(),
                                    kwLoop.getInductionVar(),
                                    outputStore.getMemRef(), inputMemref,
                                    weightMemref))
    if (!matchStoreStyleReductionConvLinearized(
            redOps, ocLoop.getInductionVar(), ohLoop.getInductionVar(),
            owLoop.getInductionVar(), icLoop.getInductionVar(),
            khLoop.getInductionVar(), kwLoop.getInductionVar(),
            icLoop.getConstantUpperBound(),
            ohTileLoop.getConstantUpperBound() + khLoop.getConstantUpperBound() -
                1,
            owTileLoop.getConstantUpperBound() + kwLoop.getConstantUpperBound() -
                1,
            ocTileLoop.getConstantUpperBound(), ohTileLoop.getConstantUpperBound(),
            owTileLoop.getConstantUpperBound(), khLoop.getConstantUpperBound(),
            kwLoop.getConstantUpperBound(), outputStore.getMemRef(),
            inputMemref, weightMemref))
      return failure();

  info.family = StringRef("CONV");
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputStore.getMemRef();
  info.inputShape = {1, icLoop.getConstantUpperBound(),
                     ohTileLoop.getConstantUpperBound() +
                         khLoop.getConstantUpperBound() - 1,
                     owTileLoop.getConstantUpperBound() +
                         kwLoop.getConstantUpperBound() - 1};
  info.weightShape = {ocTileLoop.getConstantUpperBound(),
                      icLoop.getConstantUpperBound(),
                      khLoop.getConstantUpperBound(),
                      kwLoop.getConstantUpperBound()};
  info.outputShape = {1, ocTileLoop.getConstantUpperBound(),
                      ohTileLoop.getConstantUpperBound(),
                      owTileLoop.getConstantUpperBound()};
  auto context = func.getContext();
  info.strides = DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1, 1});
  info.dilations = DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1, 1});
  return success();
}

static LogicalResult matchCanonicalGEMMV(func::FuncOp func,
                                         AffineCandidateInfo &info) {
  auto bodyOps = getBodyOps(func.front());
  if (bodyOps.size() != 1 || !isa<AffineForOp>(bodyOps.front()))
    return failure();

  auto rowLoop = cast<AffineForOp>(bodyOps.front());
  if (!isRectangularLoop(rowLoop))
    return failure();
  auto rowOps = getBodyOps(*rowLoop.getBody());
  if (rowOps.size() != 1 || !isa<AffineForOp>(rowOps.front()))
    return failure();

  auto colLoop = cast<AffineForOp>(rowOps.front());
  if (!isRectangularLoop(colLoop))
    return failure();
  auto colOps = getBodyOps(*colLoop.getBody());
  if (colOps.size() != 1 || !isa<AffineForOp>(colOps.front()))
    return failure();

  auto redLoop = cast<AffineForOp>(colOps.front());
  if (!isRectangularLoop(redLoop))
    return failure();

  Value inputMemref;
  Value weightMemref;
  auto redOps = getBodyOps(*redLoop.getBody());
  auto outputStore = dyn_cast<AffineStoreOp>(redOps.back());
  if (!outputStore)
    return failure();
  if (!matchStoreStyleReduction2D(redOps, rowLoop.getInductionVar(),
                                  colLoop.getInductionVar(),
                                  redLoop.getInductionVar(),
                                  outputStore.getMemRef(), inputMemref,
                                  weightMemref))
    if (!matchStoreStyleReduction2DLinearized(redOps, rowLoop.getInductionVar(),
                                              colLoop.getInductionVar(),
                                              redLoop.getInductionVar(),
                                              colLoop.getConstantUpperBound(),
                                              redLoop.getConstantUpperBound(),
                                              outputStore.getMemRef(),
                                              inputMemref, weightMemref))
      return failure();

  info.family = StringRef("GEMMV");
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputStore.getMemRef();
  info.inputShape = {rowLoop.getConstantUpperBound(), redLoop.getConstantUpperBound()};
  info.weightShape = {redLoop.getConstantUpperBound(), colLoop.getConstantUpperBound()};
  info.outputShape = {rowLoop.getConstantUpperBound(), colLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchCanonicalGEMM(func::FuncOp func,
                                        AffineCandidateInfo &info) {
  auto bodyOps = getBodyOps(func.front());
  if (bodyOps.size() != 1 || !isa<AffineForOp>(bodyOps.front()))
    return failure();

  auto batchLoop = cast<AffineForOp>(bodyOps.front());
  if (!isRectangularLoop(batchLoop))
    return failure();
  auto batchOps = getBodyOps(*batchLoop.getBody());
  if (batchOps.size() != 1 || !isa<AffineForOp>(batchOps.front()))
    return failure();

  auto rowLoop = cast<AffineForOp>(batchOps.front());
  if (!isRectangularLoop(rowLoop))
    return failure();
  auto rowOps = getBodyOps(*rowLoop.getBody());
  if (rowOps.size() != 1 || !isa<AffineForOp>(rowOps.front()))
    return failure();

  auto colLoop = cast<AffineForOp>(rowOps.front());
  if (!isRectangularLoop(colLoop))
    return failure();
  auto colOps = getBodyOps(*colLoop.getBody());
  if (colOps.size() != 1 || !isa<AffineForOp>(colOps.front()))
    return failure();

  auto redLoop = cast<AffineForOp>(colOps.front());
  if (!isRectangularLoop(redLoop))
    return failure();

  Value inputMemref;
  Value weightMemref;
  auto redOps = getBodyOps(*redLoop.getBody());
  auto outputStore = dyn_cast<AffineStoreOp>(redOps.back());
  if (!outputStore)
    return failure();
  if (!matchStoreStyleReduction3D(redOps, batchLoop.getInductionVar(),
                                  rowLoop.getInductionVar(),
                                  colLoop.getInductionVar(),
                                  redLoop.getInductionVar(),
                                  outputStore.getMemRef(), inputMemref,
                                  weightMemref))
    if (!matchStoreStyleReduction3DLinearized(
            redOps, batchLoop.getInductionVar(), rowLoop.getInductionVar(),
            colLoop.getInductionVar(), redLoop.getInductionVar(),
            rowLoop.getConstantUpperBound(), colLoop.getConstantUpperBound(),
            redLoop.getConstantUpperBound(), outputStore.getMemRef(),
            inputMemref, weightMemref))
    return failure();

  info.family = StringRef("GEMM");
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputStore.getMemRef();
  info.inputShape = {batchLoop.getConstantUpperBound(), rowLoop.getConstantUpperBound(),
                     redLoop.getConstantUpperBound()};
  info.weightShape = {batchLoop.getConstantUpperBound(), redLoop.getConstantUpperBound(),
                      colLoop.getConstantUpperBound()};
  info.outputShape = {batchLoop.getConstantUpperBound(), rowLoop.getConstantUpperBound(),
                      colLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchCanonicalCONV(func::FuncOp func,
                                        AffineCandidateInfo &info) {
  auto bodyOps = getBodyOps(func.front());
  if (bodyOps.size() != 1 || !isa<AffineForOp>(bodyOps.front()))
    return failure();
  auto ocLoop = cast<AffineForOp>(bodyOps.front());
  if (!isRectangularLoop(ocLoop))
    return failure();

  auto ocOps = getBodyOps(*ocLoop.getBody());
  if (ocOps.size() != 1 || !isa<AffineForOp>(ocOps.front()))
    return failure();
  auto ohLoop = cast<AffineForOp>(ocOps.front());
  if (!isRectangularLoop(ohLoop))
    return failure();

  auto ohOps = getBodyOps(*ohLoop.getBody());
  if (ohOps.size() != 1 || !isa<AffineForOp>(ohOps.front()))
    return failure();
  auto owLoop = cast<AffineForOp>(ohOps.front());
  if (!isRectangularLoop(owLoop))
    return failure();

  auto owOps = getBodyOps(*owLoop.getBody());
  if (owOps.size() != 1 || !isa<AffineForOp>(owOps.front()))
    return failure();
  auto icLoop = cast<AffineForOp>(owOps.front());
  if (!isRectangularLoop(icLoop))
    return failure();

  auto icOps = getBodyOps(*icLoop.getBody());
  if (icOps.size() != 1 || !isa<AffineForOp>(icOps.front()))
    return failure();
  auto khLoop = cast<AffineForOp>(icOps.front());
  if (!isRectangularLoop(khLoop))
    return failure();

  auto khOps = getBodyOps(*khLoop.getBody());
  if (khOps.size() != 1 || !isa<AffineForOp>(khOps.front()))
    return failure();
  auto kwLoop = cast<AffineForOp>(khOps.front());
  if (!isRectangularLoop(kwLoop))
    return failure();

  Value inputMemref;
  Value weightMemref;
  auto redOps = getBodyOps(*kwLoop.getBody());
  auto outputStore = dyn_cast<AffineStoreOp>(redOps.back());
  if (!outputStore)
    return failure();
  if (!matchStoreStyleReductionConv(redOps, ocLoop.getInductionVar(),
                                    ohLoop.getInductionVar(),
                                    owLoop.getInductionVar(),
                                    icLoop.getInductionVar(),
                                    khLoop.getInductionVar(),
                                    kwLoop.getInductionVar(),
                                    outputStore.getMemRef(), inputMemref,
                                    weightMemref))
    if (!matchStoreStyleReductionConvLinearized(
            redOps, ocLoop.getInductionVar(), ohLoop.getInductionVar(),
            owLoop.getInductionVar(), icLoop.getInductionVar(),
            khLoop.getInductionVar(), kwLoop.getInductionVar(),
            icLoop.getConstantUpperBound(),
            ohLoop.getConstantUpperBound() + khLoop.getConstantUpperBound() - 1,
            owLoop.getConstantUpperBound() + kwLoop.getConstantUpperBound() - 1,
            ocLoop.getConstantUpperBound(), ohLoop.getConstantUpperBound(),
            owLoop.getConstantUpperBound(), khLoop.getConstantUpperBound(),
            kwLoop.getConstantUpperBound(), outputStore.getMemRef(),
            inputMemref, weightMemref))
    return failure();

  info.family = StringRef("CONV");
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputStore.getMemRef();
  info.inputShape = {1, icLoop.getConstantUpperBound(),
                     ohLoop.getConstantUpperBound() + khLoop.getConstantUpperBound() - 1,
                     owLoop.getConstantUpperBound() + kwLoop.getConstantUpperBound() - 1};
  info.weightShape = {ocLoop.getConstantUpperBound(), icLoop.getConstantUpperBound(),
                      khLoop.getConstantUpperBound(), kwLoop.getConstantUpperBound()};
  info.outputShape = {1, ocLoop.getConstantUpperBound(), ohLoop.getConstantUpperBound(),
                      owLoop.getConstantUpperBound()};
  auto context = func.getContext();
  info.strides = DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1, 1});
  info.dilations = DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1, 1});
  return success();
}

static SmallVector<int64_t, 4> getContiguousStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> strides(shape.size(), 1);
  int64_t running = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = running;
    running *= shape[i];
  }
  return strides;
}

static Value materializeStaticMemrefView(OpBuilder &builder, Location loc,
                                         Value source, ArrayRef<int64_t> shape) {
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType || shape.empty())
    return source;
  if (sourceType.hasStaticShape() && sourceType.getShape().size() == shape.size() &&
      llvm::equal(sourceType.getShape(), shape))
    return source;
  if (sourceType.getRank() != 1)
    return source;

  auto resultType = MemRefType::get(shape, sourceType.getElementType());
  SmallVector<OpFoldResult, 4> sizes;
  SmallVector<OpFoldResult, 4> strides;
  auto contiguousStrides = getContiguousStrides(shape);
  for (int64_t size : shape)
    sizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : contiguousStrides)
    strides.push_back(builder.getIndexAttr(stride));
  return builder.create<memref::ReinterpretCastOp>(loc, resultType, source,
                                                   builder.getIndexAttr(0), sizes,
                                                   strides);
}

static FailureOr<AffineCandidateInfo> matchSingleAffineCandidate(func::FuncOp func) {
  AffineCandidateInfo info;
  if (succeeded(matchCanonicalCONV(func, info)))
    return info;
  if (succeeded(matchStripMinedCONV(func, info)))
    return info;
  if (succeeded(matchCanonicalGEMM(func, info)))
    return info;
  if (succeeded(matchStripMinedGEMM(func, info)))
    return info;
  if (succeeded(matchCanonicalGEMMV(func, info)))
    return info;
  if (succeeded(matchStripMinedGEMMV(func, info)))
    return info;
  return failure();
}

struct LowerAffineToAccel : public LowerAffineToAccelBase<LowerAffineToAccel> {
  void runOnOperation() override {
    auto func = getOperation();
    auto candidate = matchSingleAffineCandidate(func);
    if (failed(candidate))
      return;

    auto inputType = getTensorType(candidate->inputMemref, candidate->inputShape);
    auto weightType = getTensorType(candidate->weightMemref, candidate->weightShape);
    auto outputType = getTensorType(candidate->outputMemref, candidate->outputShape);
    if (!inputType || !weightType || !outputType)
      return;

    auto returnOp = cast<func::ReturnOp>(func.front().getTerminator());
    SmallVector<Operation *, 8> oldOps;
    for (Operation &op : func.front().without_terminator())
      oldOps.push_back(&op);

    OpBuilder builder(returnOp);
    Value shapedInput = materializeStaticMemrefView(builder, func.getLoc(),
                                                    candidate->inputMemref,
                                                    candidate->inputShape);
    Value shapedWeight = materializeStaticMemrefView(builder, func.getLoc(),
                                                     candidate->weightMemref,
                                                     candidate->weightShape);
    Value shapedOutput = materializeStaticMemrefView(builder, func.getLoc(),
                                                     candidate->outputMemref,
                                                     candidate->outputShape);
    auto input = builder.create<bufferization::ToTensorOp>(
        func.getLoc(), inputType, shapedInput);
    auto weight = builder.create<bufferization::ToTensorOp>(
        func.getLoc(), weightType, shapedWeight);
    auto existing = builder.create<bufferization::ToTensorOp>(
        func.getLoc(), outputType, shapedOutput);

    Value lowered;
    if (candidate->family == "GEMMV") {
      lowered = builder
                    .create<accel::GEMMVOp>(func.getLoc(), outputType, input,
                                            weight, Value(), existing)
                    .getResult();
    } else if (candidate->family == "GEMM") {
      lowered = builder
                    .create<accel::GEMMOp>(func.getLoc(), outputType, input,
                                           weight, Value(), existing)
                    .getResult();
    } else {
      lowered = builder
                    .create<accel::CONVOp>(func.getLoc(), outputType, input,
                                           weight, Value(), existing,
                                           candidate->strides,
                                           candidate->dilations)
                    .getResult();
    }

    annotateCandidateAttrs(lowered.getDefiningOp(), func, *candidate);
    auto resultMemref = builder.create<bufferization::ToMemrefOp>(
        func.getLoc(), shapedOutput.getType(), lowered);
    builder.create<memref::CopyOp>(func.getLoc(), resultMemref, shapedOutput);

    for (Operation *op : oldOps)
      op->erase();
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createLowerAffineToAccelPass() {
  return std::make_unique<LowerAffineToAccel>();
}
