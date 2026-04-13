//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  StringRef sourceFamily;
  Value inputMemref;
  Value weightMemref;
  Value outputMemref;
  bool hasExistingInput = false;
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

static SmallVector<Operation *, 8> getBodyOpsIgnoringConstants(Block &block) {
  SmallVector<Operation *, 8> ops;
  for (Operation &op : block.without_terminator()) {
    if (isa<arith::ConstantOp>(op))
      continue;
    ops.push_back(&op);
  }
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

static bool isZeroInitValue(Value value) {
  value = stripIntegerCasts(value);
  if (auto intConst = value.getDefiningOp<arith::ConstantIntOp>())
    return intConst.value() == 0;
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().isZero();
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue()))
      return floatAttr.getValue().isZero();
  }
  return false;
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

static bool matchesYieldedReductionComputation(ArrayRef<Operation *> ops,
                                               AffineLoadOp inputLoad,
                                               AffineLoadOp weightLoad,
                                               Value accumulator,
                                               Value yieldedValue) {
  arith::MulFOp mulf;
  arith::MulIOp muli;
  arith::AddFOp addf;
  arith::AddIOp addi;

  for (Operation *op : ops) {
    if (isa<AffineLoadOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
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
    bool lhsIsAccumulator = stripIntegerCasts(lhs) == accumulator;
    bool rhsIsAccumulator = stripIntegerCasts(rhs) == accumulator;
    if (lhsIsAccumulator == rhsIsAccumulator)
      return false;
    Value other = lhsIsAccumulator ? rhs : lhs;
    if (stripIntegerCasts(other) != mulValue)
      return false;
    addValue = addf.getResult();
  } else if (addi) {
    Value lhs = addi.getLhs();
    Value rhs = addi.getRhs();
    bool lhsIsAccumulator = stripIntegerCasts(lhs) == accumulator;
    bool rhsIsAccumulator = stripIntegerCasts(rhs) == accumulator;
    if (lhsIsAccumulator == rhsIsAccumulator)
      return false;
    Value other = lhsIsAccumulator ? rhs : lhs;
    if (stripIntegerCasts(other) != mulValue)
      return false;
    addValue = addi.getResult();
  } else {
    return false;
  }

  return stripIntegerCasts(yieldedValue) == addValue;
}

static bool matchIterArgReduction2DCanonical(ArrayRef<Operation *> ops, Value row,
                                             Value col, Value &inputMemref,
                                             Value &weightMemref,
                                             Value &outputMemref,
                                             bool &hasExistingInput) {
  if (ops.empty())
    return false;

  auto outputLoad = dyn_cast<AffineLoadOp>(ops.front());
  AffineForOp redLoop;
  for (Operation *op : ops)
    if (auto loop = dyn_cast<AffineForOp>(op))
      redLoop = loop;
  auto store = dyn_cast<AffineStoreOp>(ops.back());
  if (!outputLoad || !redLoop || !store || redLoop.getNumIterOperands() != 1 ||
      !isRectangularLoop(redLoop)) {
    outputLoad = AffineLoadOp();
    if (!redLoop || !store || redLoop.getNumIterOperands() != 1 ||
        !isRectangularLoop(redLoop))
      return false;
  }

  auto rowCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, row);
  };
  auto colCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, col);
  };
  if (outputLoad) {
    if (!sameMemrefAccess(outputLoad, outputLoad.getMemRef(), {rowCheck, colCheck}) ||
        !sameMemrefAccess(store, outputLoad.getMemRef(), {rowCheck, colCheck}))
      return false;
    if (stripIntegerCasts(redLoop.getIterOperands().front()) != outputLoad.getResult())
      return false;
    outputMemref = outputLoad.getMemRef();
    hasExistingInput = true;
  } else {
    if (!sameMemrefAccess(store, store.getMemRef(), {rowCheck, colCheck}) ||
        !isZeroInitValue(redLoop.getIterOperands().front()))
      return false;
    outputMemref = store.getMemRef();
    hasExistingInput = false;
  }
  if (stripIntegerCasts(store.getValueToStore()) != redLoop.getResult(0))
    return false;

  auto redBodyOps = getBodyOps(*redLoop.getBody());
  SmallVector<AffineLoadOp, 2> loads;
  for (Operation *op : redBodyOps)
    if (auto load = dyn_cast<AffineLoadOp>(op))
      loads.push_back(load);
  if (loads.size() != 2)
    return false;

  auto redCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, redLoop.getInductionVar());
  };

  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    if (sameMemrefAccess(load, load.getMemRef(), {rowCheck, redCheck}))
      inputLoad = load;
    else if (sameMemrefAccess(load, load.getMemRef(), {redCheck, colCheck}))
      weightLoad = load;
  }
  if (!inputLoad || !weightLoad)
    return false;

  auto yield = cast<AffineYieldOp>(redLoop.getBody()->getTerminator());
  if (!matchesYieldedReductionComputation(redBodyOps, inputLoad, weightLoad,
                                          redLoop.getRegionIterArgs()[0],
                                          yield.getOperands()[0]))
    return false;
  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  return inputMemref != weightMemref && inputMemref != outputMemref &&
         weightMemref != outputMemref;
}

static bool matchIterArgReduction3DCanonical(ArrayRef<Operation *> ops,
                                             Value batch, Value row, Value col,
                                             Value &inputMemref,
                                             Value &weightMemref,
                                             Value &outputMemref,
                                             bool &hasExistingInput) {
  if (ops.empty())
    return false;
  auto outputLoad = dyn_cast<AffineLoadOp>(ops.front());
  AffineForOp redLoop;
  for (Operation *op : ops)
    if (auto loop = dyn_cast<AffineForOp>(op))
      redLoop = loop;
  auto store = dyn_cast<AffineStoreOp>(ops.back());
  if (!outputLoad || !redLoop || !store || redLoop.getNumIterOperands() != 1 ||
      !isRectangularLoop(redLoop)) {
    outputLoad = AffineLoadOp();
    if (!redLoop || !store || redLoop.getNumIterOperands() != 1 ||
        !isRectangularLoop(redLoop))
      return false;
  }

  auto batchCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, batch);
  };
  auto rowCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, row);
  };
  auto colCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, col);
  };
  if (outputLoad) {
    if (!sameMemrefAccess(outputLoad, outputLoad.getMemRef(),
                          {batchCheck, rowCheck, colCheck}) ||
        !sameMemrefAccess(store, outputLoad.getMemRef(),
                          {batchCheck, rowCheck, colCheck}))
      return false;
    if (stripIntegerCasts(redLoop.getIterOperands().front()) != outputLoad.getResult())
      return false;
    outputMemref = outputLoad.getMemRef();
    hasExistingInput = true;
  } else {
    if (!sameMemrefAccess(store, store.getMemRef(),
                          {batchCheck, rowCheck, colCheck}) ||
        !isZeroInitValue(redLoop.getIterOperands().front()))
      return false;
    outputMemref = store.getMemRef();
    hasExistingInput = false;
  }
  if (stripIntegerCasts(store.getValueToStore()) != redLoop.getResult(0))
    return false;

  auto redBodyOps = getBodyOps(*redLoop.getBody());
  SmallVector<AffineLoadOp, 2> loads;
  for (Operation *op : redBodyOps)
    if (auto load = dyn_cast<AffineLoadOp>(op))
      loads.push_back(load);
  if (loads.size() != 2)
    return false;

  auto redCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, redLoop.getInductionVar());
  };

  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    if (sameMemrefAccess(load, load.getMemRef(),
                         {batchCheck, rowCheck, redCheck}))
      inputLoad = load;
    else if (sameMemrefAccess(load, load.getMemRef(),
                              {batchCheck, redCheck, colCheck}))
      weightLoad = load;
  }
  if (!inputLoad || !weightLoad)
    return false;

  auto yield = cast<AffineYieldOp>(redLoop.getBody()->getTerminator());
  if (!matchesYieldedReductionComputation(redBodyOps, inputLoad, weightLoad,
                                          redLoop.getRegionIterArgs()[0],
                                          yield.getOperands()[0]))
    return false;
  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  return inputMemref != weightMemref && inputMemref != outputMemref &&
         weightMemref != outputMemref;
}

static bool matchIterArgReductionConvCanonical(ArrayRef<Operation *> ops,
                                               Value oc, Value oh, Value ow,
                                               Value &inputMemref,
                                               Value &weightMemref,
                                               Value &outputMemref,
                                               bool &hasExistingInput) {
  if (ops.empty())
    return false;
  auto outputLoad = dyn_cast<AffineLoadOp>(ops.front());
  AffineForOp icLoop;
  for (Operation *op : ops)
    if (auto loop = dyn_cast<AffineForOp>(op))
      icLoop = loop;
  auto store = dyn_cast<AffineStoreOp>(ops.back());
  if (!outputLoad || !icLoop || !store || icLoop.getNumIterOperands() != 1 ||
      !isRectangularLoop(icLoop)) {
    outputLoad = AffineLoadOp();
    if (!icLoop || !store || icLoop.getNumIterOperands() != 1 ||
        !isRectangularLoop(icLoop))
      return false;
  }

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
  if (outputLoad) {
    if (!sameMemrefAccess(outputLoad, outputLoad.getMemRef(),
                          {zeroCheck, ocCheck, ohCheck, owCheck}) ||
        !sameMemrefAccess(store, outputLoad.getMemRef(),
                          {zeroCheck, ocCheck, ohCheck, owCheck}))
      return false;
    if (stripIntegerCasts(icLoop.getIterOperands().front()) != outputLoad.getResult())
      return false;
    outputMemref = outputLoad.getMemRef();
    hasExistingInput = true;
  } else {
    if (!sameMemrefAccess(store, store.getMemRef(),
                          {zeroCheck, ocCheck, ohCheck, owCheck}) ||
        !isZeroInitValue(icLoop.getIterOperands().front()))
      return false;
    outputMemref = store.getMemRef();
    hasExistingInput = false;
  }
  if (stripIntegerCasts(store.getValueToStore()) != icLoop.getResult(0))
    return false;

  auto icBodyOps = getBodyOps(*icLoop.getBody());
  if (icBodyOps.size() != 1 || !isa<AffineForOp>(icBodyOps.front()))
    return false;
  auto khLoop = cast<AffineForOp>(icBodyOps.front());
  if (khLoop.getNumIterOperands() != 1 || !isRectangularLoop(khLoop) ||
      stripIntegerCasts(khLoop.getIterOperands().front()) != icLoop.getRegionIterArgs()[0])
    return false;

  auto khBodyOps = getBodyOps(*khLoop.getBody());
  if (khBodyOps.size() != 1 || !isa<AffineForOp>(khBodyOps.front()))
    return false;
  auto kwLoop = cast<AffineForOp>(khBodyOps.front());
  if (kwLoop.getNumIterOperands() != 1 || !isRectangularLoop(kwLoop) ||
      stripIntegerCasts(kwLoop.getIterOperands().front()) != khLoop.getRegionIterArgs()[0])
    return false;

  auto kwBodyOps = getBodyOps(*kwLoop.getBody());
  SmallVector<AffineLoadOp, 2> loads;
  for (Operation *op : kwBodyOps)
    if (auto load = dyn_cast<AffineLoadOp>(op))
      loads.push_back(load);
  if (loads.size() != 2)
    return false;

  auto icCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, icLoop.getInductionVar());
  };
  auto khCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, khLoop.getInductionVar());
  };
  auto kwCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, kwLoop.getInductionVar());
  };
  auto ohPlusKhCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, oh, khLoop.getInductionVar());
  };
  auto owPlusKwCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, operands, ow, kwLoop.getInductionVar());
  };

  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    if (sameMemrefAccess(load, load.getMemRef(),
                         {zeroCheck, icCheck, ohPlusKhCheck, owPlusKwCheck}))
      inputLoad = load;
    else if (sameMemrefAccess(load, load.getMemRef(),
                              {ocCheck, icCheck, khCheck, kwCheck}))
      weightLoad = load;
  }
  if (!inputLoad || !weightLoad)
    return false;

  auto kwYield = cast<AffineYieldOp>(kwLoop.getBody()->getTerminator());
  if (!matchesYieldedReductionComputation(kwBodyOps, inputLoad, weightLoad,
                                          kwLoop.getRegionIterArgs()[0],
                                          kwYield.getOperands()[0]))
    return false;

  auto khYield = cast<AffineYieldOp>(khLoop.getBody()->getTerminator());
  auto icYield = cast<AffineYieldOp>(icLoop.getBody()->getTerminator());
  if (stripIntegerCasts(khYield.getOperands()[0]) != kwLoop.getResult(0) ||
      stripIntegerCasts(icYield.getOperands()[0]) != khLoop.getResult(0))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  return inputMemref != weightMemref && inputMemref != outputMemref &&
         weightMemref != outputMemref;
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
  info.sourceFamily = info.family;
  info.hasExistingInput = true;
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
  info.sourceFamily = info.family;
  info.hasExistingInput = true;
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
  info.sourceFamily = info.family;
  info.hasExistingInput = true;
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
  auto bodyOps = getBodyOpsIgnoringConstants(func.front());
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
  int64_t reductionExtent = -1;
  Value inputMemref;
  Value weightMemref;
  Value outputMemref;
  if (colOps.size() == 1 && isa<AffineForOp>(colOps.front())) {
    auto redLoop = cast<AffineForOp>(colOps.front());
    if (!isRectangularLoop(redLoop))
      return failure();
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
              redLoop.getInductionVar(), colLoop.getConstantUpperBound(),
              redLoop.getConstantUpperBound(), outputStore.getMemRef(),
              inputMemref, weightMemref))
        return failure();
    reductionExtent = redLoop.getConstantUpperBound();
    outputMemref = outputStore.getMemRef();
  } else if (!matchIterArgReduction2DCanonical(colOps, rowLoop.getInductionVar(),
                                               colLoop.getInductionVar(),
                                               inputMemref, weightMemref,
                                               outputMemref,
                                               info.hasExistingInput)) {
    return failure();
  }

  info.family = StringRef("GEMMV");
  info.sourceFamily = info.family;
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputMemref;
  auto inputType = cast<MemRefType>(inputMemref.getType());
  auto weightType = cast<MemRefType>(weightMemref.getType());
  int64_t kExtent = reductionExtent;
  if (kExtent < 0) {
    if (inputType.getRank() > 1 && inputType.hasStaticShape())
      kExtent = inputType.getShape().back();
    else if (weightType.getRank() > 1 && weightType.hasStaticShape())
      kExtent = weightType.getShape().front();
  }
  info.inputShape = {rowLoop.getConstantUpperBound(), kExtent};
  info.weightShape = {kExtent, colLoop.getConstantUpperBound()};
  info.outputShape = {rowLoop.getConstantUpperBound(), colLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchCanonicalGEMM(func::FuncOp func,
                                        AffineCandidateInfo &info) {
  auto bodyOps = getBodyOpsIgnoringConstants(func.front());
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
  int64_t reductionExtent = -1;
  Value inputMemref;
  Value weightMemref;
  Value outputMemref;
  if (colOps.size() == 1 && isa<AffineForOp>(colOps.front())) {
    auto redLoop = cast<AffineForOp>(colOps.front());
    if (!isRectangularLoop(redLoop))
      return failure();
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
    reductionExtent = redLoop.getConstantUpperBound();
    outputMemref = outputStore.getMemRef();
  } else if (!matchIterArgReduction3DCanonical(
                 colOps, batchLoop.getInductionVar(), rowLoop.getInductionVar(),
                 colLoop.getInductionVar(), inputMemref, weightMemref,
                 outputMemref, info.hasExistingInput)) {
    return failure();
  }

  info.family = StringRef("GEMM");
  info.sourceFamily = info.family;
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputMemref;
  auto inputType = cast<MemRefType>(inputMemref.getType());
  auto weightType = cast<MemRefType>(weightMemref.getType());
  int64_t kExtent = reductionExtent;
  if (kExtent < 0) {
    if (inputType.getRank() > 2 && inputType.hasStaticShape())
      kExtent = inputType.getShape().back();
    else if (weightType.getRank() > 2 && weightType.hasStaticShape())
      kExtent = weightType.getShape()[1];
  }
  info.inputShape = {batchLoop.getConstantUpperBound(), rowLoop.getConstantUpperBound(),
                     kExtent};
  info.weightShape = {batchLoop.getConstantUpperBound(), kExtent,
                      colLoop.getConstantUpperBound()};
  info.outputShape = {batchLoop.getConstantUpperBound(), rowLoop.getConstantUpperBound(),
                      colLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchCanonicalCONV(func::FuncOp func,
                                        AffineCandidateInfo &info) {
  auto bodyOps = getBodyOpsIgnoringConstants(func.front());
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
  Value inputMemref;
  Value weightMemref;
  Value outputMemref;
  if (owOps.size() == 1 && isa<AffineForOp>(owOps.front())) {
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
    outputMemref = outputStore.getMemRef();
  } else if (!matchIterArgReductionConvCanonical(
                 owOps, ocLoop.getInductionVar(), ohLoop.getInductionVar(),
                 owLoop.getInductionVar(), inputMemref, weightMemref,
                 outputMemref, info.hasExistingInput)) {
    return failure();
  }

  info.family = StringRef("CONV");
  info.sourceFamily = info.family;
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputMemref;
  auto inputType = cast<MemRefType>(inputMemref.getType());
  auto weightType = cast<MemRefType>(weightMemref.getType());
  info.inputShape = llvm::to_vector(inputType.getShape());
  info.weightShape = llvm::to_vector(weightType.getShape());
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

static int64_t getNumElements(ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t dim : shape)
    total *= dim;
  return total;
}

static Value materializeStaticMemrefView(OpBuilder &builder, Location loc,
                                         Value source, ArrayRef<int64_t> shape) {
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType || shape.empty())
    return source;
  if (sourceType.hasStaticShape() && sourceType.getShape().size() == shape.size() &&
      llvm::equal(sourceType.getShape(), shape))
    return source;
  if (sourceType.getRank() == 1) {
    auto resultType = MemRefType::get(shape, sourceType.getElementType());
    SmallVector<OpFoldResult, 4> sizes;
    SmallVector<OpFoldResult, 4> strides;
    auto contiguousStrides = getContiguousStrides(shape);
    for (int64_t size : shape)
      sizes.push_back(builder.getIndexAttr(size));
    for (int64_t stride : contiguousStrides)
      strides.push_back(builder.getIndexAttr(stride));
    return builder.create<memref::ReinterpretCastOp>(loc, resultType, source,
                                                     builder.getIndexAttr(0),
                                                     sizes, strides);
  }
  if (!sourceType.hasStaticShape())
    return source;
  if (getNumElements(sourceType.getShape()) != getNumElements(shape))
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

static RankedTensorType getDesiredTensorType(Value memref,
                                             ArrayRef<int64_t> shape) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType)
    return RankedTensorType();
  if (!shape.empty())
    return RankedTensorType::get(shape, memrefType.getElementType());
  if (!memrefType.hasStaticShape())
    return RankedTensorType();
  return RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
}

static Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static accel::GEMMOp lowerConvAsGemmOnly(OpBuilder &builder, Location loc,
                                         Value inputMemref, Value weightMemref,
                                         Value outputMemref,
                                         bool hasExistingInput) {
  auto inputType = cast<MemRefType>(inputMemref.getType());
  auto weightType = cast<MemRefType>(weightMemref.getType());
  auto outputType = cast<MemRefType>(outputMemref.getType());
  const int64_t oc = outputType.getShape()[1];
  const int64_t oh = outputType.getShape()[2];
  const int64_t ow = outputType.getShape()[3];
  const int64_t ic = inputType.getShape()[1];
  const int64_t kh = weightType.getShape()[2];
  const int64_t kw = weightType.getShape()[3];
  const int64_t flatK = ic * kh * kw;
  const int64_t flatN = oh * ow;

  auto gemmAType = MemRefType::get({oc, flatK}, weightType.getElementType());
  auto gemmBType = MemRefType::get({flatK, flatN}, inputType.getElementType());
  auto gemmOutType = MemRefType::get({oc, flatN}, outputType.getElementType());

  Value gemmA = builder.create<memref::AllocOp>(loc, gemmAType);
  Value gemmB = builder.create<memref::AllocOp>(loc, gemmBType);
  auto c0 = createIndexConstant(builder, loc, 0);
  auto cKhKw = createIndexConstant(builder, loc, kh * kw);
  auto cOw = createIndexConstant(builder, loc, ow);

  auto ocLoop = builder.create<AffineForOp>(loc, 0, oc, 1);
  builder.setInsertionPointToStart(ocLoop.getBody());
  auto flatKLoop = builder.create<AffineForOp>(loc, 0, flatK, 1);
  builder.setInsertionPointToStart(flatKLoop.getBody());
  Value icIdx =
      builder.create<arith::DivUIOp>(loc, flatKLoop.getInductionVar(), cKhKw);
  Value rem0 =
      builder.create<arith::RemUIOp>(loc, flatKLoop.getInductionVar(), cKhKw);
  Value khIdx = builder.create<arith::DivUIOp>(
      loc, rem0, createIndexConstant(builder, loc, kw));
  Value kwIdx = builder.create<arith::RemUIOp>(
      loc, rem0, createIndexConstant(builder, loc, kw));
  Value aElt = builder.create<memref::LoadOp>(
      loc, weightMemref,
      ValueRange{ocLoop.getInductionVar(), icIdx, khIdx, kwIdx});
  builder.create<memref::StoreOp>(
      loc, aElt, gemmA,
      ValueRange{ocLoop.getInductionVar(), flatKLoop.getInductionVar()});
  builder.setInsertionPointAfter(flatKLoop);
  builder.setInsertionPointAfter(ocLoop);

  auto flatKLoopB = builder.create<AffineForOp>(loc, 0, flatK, 1);
  builder.setInsertionPointToStart(flatKLoopB.getBody());
  auto flatNLoopB = builder.create<AffineForOp>(loc, 0, flatN, 1);
  builder.setInsertionPointToStart(flatNLoopB.getBody());
  Value icIdxB =
      builder.create<arith::DivUIOp>(loc, flatKLoopB.getInductionVar(), cKhKw);
  Value remB =
      builder.create<arith::RemUIOp>(loc, flatKLoopB.getInductionVar(), cKhKw);
  Value khIdxB = builder.create<arith::DivUIOp>(
      loc, remB, createIndexConstant(builder, loc, kw));
  Value kwIdxB = builder.create<arith::RemUIOp>(
      loc, remB, createIndexConstant(builder, loc, kw));
  Value ohIdx = builder.create<arith::DivUIOp>(loc, flatNLoopB.getInductionVar(), cOw);
  Value owIdx = builder.create<arith::RemUIOp>(loc, flatNLoopB.getInductionVar(), cOw);
  Value ihIdx = builder.create<arith::AddIOp>(loc, ohIdx, khIdxB);
  Value iwIdx = builder.create<arith::AddIOp>(loc, owIdx, kwIdxB);
  Value bElt = builder.create<memref::LoadOp>(
      loc, inputMemref, ValueRange{c0, icIdxB, ihIdx, iwIdx});
  builder.create<memref::StoreOp>(
      loc, bElt, gemmB,
      ValueRange{flatKLoopB.getInductionVar(), flatNLoopB.getInductionVar()});
  builder.setInsertionPointAfter(flatNLoopB);
  builder.setInsertionPointAfter(flatKLoopB);

  auto gemmATensor = builder.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get({oc, flatK}, weightType.getElementType()), gemmA);
  auto gemmBTensor = builder.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get({flatK, flatN}, inputType.getElementType()), gemmB);
  Value existingTensor;
  if (hasExistingInput) {
    auto existingType = MemRefType::get({oc, flatN}, outputType.getElementType());
    Value existing = builder.create<memref::AllocOp>(loc, existingType);
    auto ocLoopE = builder.create<AffineForOp>(loc, 0, oc, 1);
    builder.setInsertionPointToStart(ocLoopE.getBody());
    auto flatNLoopE = builder.create<AffineForOp>(loc, 0, flatN, 1);
    builder.setInsertionPointToStart(flatNLoopE.getBody());
    Value ohIdxE =
        builder.create<arith::DivUIOp>(loc, flatNLoopE.getInductionVar(), cOw);
    Value owIdxE =
        builder.create<arith::RemUIOp>(loc, flatNLoopE.getInductionVar(), cOw);
    Value eElt = builder.create<memref::LoadOp>(
        loc, outputMemref,
        ValueRange{c0, ocLoopE.getInductionVar(), ohIdxE, owIdxE});
    builder.create<memref::StoreOp>(
        loc, eElt, existing,
        ValueRange{ocLoopE.getInductionVar(), flatNLoopE.getInductionVar()});
    builder.setInsertionPointAfter(flatNLoopE);
    builder.setInsertionPointAfter(ocLoopE);
    existingTensor = builder.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get({oc, flatN}, outputType.getElementType()),
        existing);
  }
  auto gemm = builder.create<accel::GEMMOp>(
      loc, RankedTensorType::get({oc, flatN}, outputType.getElementType()),
      gemmATensor, gemmBTensor, Value(), existingTensor);

  auto gemmResultMemref = builder.create<bufferization::ToMemrefOp>(
      loc, gemmOutType, gemm.getResult());
  auto ocLoopR = builder.create<AffineForOp>(loc, 0, oc, 1);
  builder.setInsertionPointToStart(ocLoopR.getBody());
  auto ohLoopR = builder.create<AffineForOp>(loc, 0, oh, 1);
  builder.setInsertionPointToStart(ohLoopR.getBody());
  auto owLoopR = builder.create<AffineForOp>(loc, 0, ow, 1);
  builder.setInsertionPointToStart(owLoopR.getBody());
  Value flatNIdxR = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(loc, ohLoopR.getInductionVar(), cOw),
      owLoopR.getInductionVar());
  Value outElt = builder.create<memref::LoadOp>(
      loc, gemmResultMemref,
      ValueRange{ocLoopR.getInductionVar(), flatNIdxR});
  builder.create<memref::StoreOp>(
      loc, outElt, outputMemref,
      ValueRange{c0, ocLoopR.getInductionVar(), ohLoopR.getInductionVar(),
                 owLoopR.getInductionVar()});
  builder.setInsertionPointAfter(owLoopR);
  builder.setInsertionPointAfter(ohLoopR);
  builder.setInsertionPointAfter(ocLoopR);
  return gemm;
}

static void convertGEMMVToGEMMOnly(AffineCandidateInfo &info) {
  info.family = StringRef("GEMM");
}

static void convertCONVToGEMMOnly(AffineCandidateInfo &info) {
  auto inputShape = info.inputShape;
  auto weightShape = info.weightShape;
  auto outputShape = info.outputShape;
  const int64_t oc = outputShape[1];
  const int64_t oh = outputShape[2];
  const int64_t ow = outputShape[3];
  const int64_t ic = inputShape[1];
  const int64_t kh = weightShape[2];
  const int64_t kw = weightShape[3];
  info.family = StringRef("GEMM");
  info.inputShape = {1, oc, ic * kh * kw};
  info.weightShape = {1, ic * kh * kw, oh * ow};
  info.outputShape = {1, oc, oh * ow};
}

static FailureOr<AffineCandidateInfo> matchSingleAffineCandidate(func::FuncOp func,
                                                                 bool gemmOnly) {
  AffineCandidateInfo info;
  if (succeeded(matchCanonicalGEMM(func, info)))
    return info;
  if (succeeded(matchStripMinedGEMM(func, info)))
    return info;
  if (!gemmOnly) {
    if (succeeded(matchCanonicalCONV(func, info)))
      return info;
    if (succeeded(matchStripMinedCONV(func, info)))
      return info;
  } else {
    if (succeeded(matchCanonicalCONV(func, info))) {
      convertCONVToGEMMOnly(info);
      return info;
    }
    if (succeeded(matchStripMinedCONV(func, info))) {
      convertCONVToGEMMOnly(info);
      return info;
    }
  }
  if (!gemmOnly) {
    if (succeeded(matchCanonicalGEMMV(func, info)))
      return info;
    if (succeeded(matchStripMinedGEMMV(func, info)))
      return info;
  } else {
    if (succeeded(matchCanonicalGEMMV(func, info))) {
      convertGEMMVToGEMMOnly(info);
      return info;
    }
    if (succeeded(matchStripMinedGEMMV(func, info))) {
      convertGEMMVToGEMMOnly(info);
      return info;
    }
  }
  return failure();
}

struct LowerAffineToAccel : public LowerAffineToAccelBase<LowerAffineToAccel> {
  LowerAffineToAccel() = default;
  explicit LowerAffineToAccel(bool gemmOnlyValue) { gemmOnly = gemmOnlyValue; }

  void runOnOperation() override {
    auto func = getOperation();
    auto candidate = matchSingleAffineCandidate(func, gemmOnly);
    if (failed(candidate))
      return;

    auto inputType = getDesiredTensorType(candidate->inputMemref, candidate->inputShape);
    auto weightType = getDesiredTensorType(candidate->weightMemref, candidate->weightShape);
    auto outputType = getDesiredTensorType(candidate->outputMemref, candidate->outputShape);
    if (!inputType || !weightType || !outputType)
      return;

    auto returnOp = cast<func::ReturnOp>(func.front().getTerminator());
    SmallVector<Operation *, 8> oldOps;
    for (Operation &op : func.front().without_terminator()) {
      if (isa<arith::ConstantOp>(op))
        continue;
      oldOps.push_back(&op);
    }

    OpBuilder builder(returnOp);
    if (candidate->family == "GEMM" &&
        candidate->sourceFamily == StringRef("CONV")) {
      auto gemm = lowerConvAsGemmOnly(builder, func.getLoc(),
                                      candidate->inputMemref,
                                      candidate->weightMemref,
                                      candidate->outputMemref,
                                      candidate->hasExistingInput);
      annotateCandidateAttrs(gemm.getOperation(), func, *candidate);
      for (Operation *op : llvm::reverse(oldOps))
        op->erase();
      return;
    }

    Value shapedInput = materializeStaticMemrefView(builder, func.getLoc(),
                                                    candidate->inputMemref,
                                                    candidate->inputShape);
    Value shapedWeight = materializeStaticMemrefView(builder, func.getLoc(),
                                                     candidate->weightMemref,
                                                     candidate->weightShape);
    Value shapedOutput = materializeStaticMemrefView(builder, func.getLoc(),
                                                     candidate->outputMemref,
                                                     candidate->outputShape);
    auto input =
        builder.create<bufferization::ToTensorOp>(func.getLoc(), inputType, shapedInput);
    auto weight =
        builder.create<bufferization::ToTensorOp>(func.getLoc(), weightType, shapedWeight);
    Value existing;
    if (candidate->hasExistingInput)
      existing = builder.create<bufferization::ToTensorOp>(func.getLoc(), outputType,
                                                           shapedOutput);

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

    for (Operation *op : llvm::reverse(oldOps))
      op->erase();
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createLowerAffineToAccelPass(bool gemmOnly) {
  return std::make_unique<LowerAffineToAccel>(gemmOnly);
}
