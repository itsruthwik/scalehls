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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "scalehls/Dialect/Accel/Accel.h"
#include "scalehls/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

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
constexpr StringLiteral kSkipReasonAttr = "scalehls.gemm_skip_reason";

enum class SemanticOperandKind {
  Unknown,
  RowRed,
  RedCol,
  ConvInput,
  ConvWeight,
};

struct SemanticOperandInfo {
  bool directLoad = false;
  bool needsMaterialization = false;
  Value memref;
  Value rootValue;
  SmallVector<Operation *, 8> sliceOps;
  SemanticOperandKind kind = SemanticOperandKind::Unknown;
};

struct AffineCandidateInfo {
  StringRef family;
  StringRef sourceFamily;
  AffineForOp rootLoop;
  Value inputMemref;
  Value weightMemref;
  Value outputMemref;
  bool hasExistingInput = false;
  bool weightTransposed = false;
  SmallVector<int64_t, 4> inputShape;
  SmallVector<int64_t, 4> weightShape;
  SmallVector<int64_t, 4> outputShape;
  SmallVector<int64_t, 4> sourceInputShape;
  SmallVector<int64_t, 4> sourceWeightShape;
  SmallVector<int64_t, 4> sourceOutputShape;
  DenseI64ArrayAttr strides;
  DenseI64ArrayAttr dilations;
  Value rowIV;
  Value colIV;
  Value redIV;
  Value convOcIV;
  Value convOhIV;
  Value convOwIV;
  Value convIcIV;
  Value convKhIV;
  Value convKwIV;
  FailureOr<SemanticOperandInfo> semanticInput;
  FailureOr<SemanticOperandInfo> semanticWeight;
  std::string skipReason;
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

static AffineForOp getOwningAffineFor(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return dyn_cast_or_null<AffineForOp>(blockArg.getOwner()->getParentOp());
  return value ? value.getDefiningOp<AffineForOp>() : AffineForOp();
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
  RankedTensorType outputType =
      candidate.outputMemref ? getTensorType(candidate.outputMemref,
                                             candidate.outputShape)
                             : RankedTensorType();
  if (!outputType)
    return;
  Type inputElementType = candidate.inputMemref
                              ? cast<MemRefType>(candidate.inputMemref.getType())
                                    .getElementType()
                              : (succeeded(candidate.semanticInput)
                                     ? candidate.semanticInput->rootValue.getType()
                                     : Type());
  if (!inputElementType)
    inputElementType = outputType.getElementType();
  op->setAttr(kDetectedAttr, UnitAttr::get(op->getContext()));
  op->setAttr(kContractAttr, StringAttr::get(op->getContext(), kAffineContract));
  op->setAttr(kFamilyAttr, StringAttr::get(op->getContext(), candidate.family));
  op->setAttr(kHasBiasAttr, BoolAttr::get(op->getContext(), false));
  annotateOperandShape(op, kShapeAttr, candidate.outputShape);
  annotateOperandShape(op, kAShapeAttr, candidate.inputShape);
  annotateOperandShape(op, kBShapeAttr, candidate.weightShape);
  annotateOperandShape(op, kCShapeAttr, candidate.outputShape);
  annotatePrecision(op, inputElementType);
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

static bool sameConvInputAccess(AffineLoadOp load, Value memref,
                                function_ref<bool(AffineExpr, ValueRange)> icCheck,
                                function_ref<bool(AffineExpr, ValueRange)> hCheck,
                                function_ref<bool(AffineExpr, ValueRange)> wCheck) {
  auto zeroCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, 0);
  };
  return sameMemrefAccess(load, memref, {zeroCheck, icCheck, hCheck, wCheck}) ||
         sameMemrefAccess(load, memref, {icCheck, hCheck, wCheck});
}

static bool sameConvOutputAccess(AffineLoadOp load, Value memref,
                                 function_ref<bool(AffineExpr, ValueRange)> ocCheck,
                                 function_ref<bool(AffineExpr, ValueRange)> ohCheck,
                                 function_ref<bool(AffineExpr, ValueRange)> owCheck) {
  auto zeroCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, 0);
  };
  return sameMemrefAccess(load, memref, {zeroCheck, ocCheck, ohCheck, owCheck}) ||
         sameMemrefAccess(load, memref, {ocCheck, ohCheck, owCheck});
}

static bool sameConvOutputAccess(AffineStoreOp store, Value memref,
                                 function_ref<bool(AffineExpr, ValueRange)> ocCheck,
                                 function_ref<bool(AffineExpr, ValueRange)> ohCheck,
                                 function_ref<bool(AffineExpr, ValueRange)> owCheck) {
  auto zeroCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, 0);
  };
  return sameMemrefAccess(store, memref, {zeroCheck, ocCheck, ohCheck, owCheck}) ||
         sameMemrefAccess(store, memref, {ocCheck, ohCheck, owCheck});
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

static bool isZeroInitValue(Value value);
static Value materializeStaticMemrefView(OpBuilder &builder, Location loc,
                                         Value source, ArrayRef<int64_t> shape);

static Value stripWideningIntegerExts(Value value) {
  while (true) {
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
      value = ext.getIn();
      continue;
    }
    return value;
  }
}

static bool isSupportedSemanticProducerOp(Operation *op) {
  return isa<AffineLoadOp, arith::ConstantOp, arith::ConstantIntOp,
             arith::ConstantIndexOp, arith::ExtSIOp, arith::ExtUIOp,
             arith::TruncIOp, arith::AddIOp, arith::AddFOp, arith::SubIOp,
             arith::SubFOp, arith::MulIOp, arith::MulFOp, arith::ShLIOp,
             arith::ShRSIOp,
             arith::ShRUIOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
             scf::IfOp>(op);
}

static bool affineExprUsesOnlyAllowedOperands(
    AffineExpr expr, ValueRange operands,
    const llvm::DenseSet<Value> &allowedOperands,
    llvm::DenseSet<Value> &usedOperands) {
  bool ok = true;
  expr.walk([&](AffineExpr subExpr) {
    if (auto dim = subExpr.dyn_cast<AffineDimExpr>()) {
      if (dim.getPosition() >= operands.size()) {
        ok = false;
        return;
      }
      Value operand = operands[dim.getPosition()];
      if (!allowedOperands.contains(operand)) {
        ok = false;
        return;
      }
      usedOperands.insert(operand);
    }
  });
  return ok;
}

static FailureOr<SemanticOperandInfo>
analyzeSemanticOperand2D(Value originalRoot, Value row, Value col, Value red) {
  Value root = stripWideningIntegerExts(originalRoot);
  SemanticOperandInfo info;
  info.rootValue = root;

  llvm::DenseSet<Value> allowedIVs;
  allowedIVs.insert(row);
  allowedIVs.insert(col);
  allowedIVs.insert(red);
  llvm::DenseSet<Operation *> visitedOps;
  llvm::DenseSet<Value> usedIVs;

  std::function<LogicalResult(Value)> visitValue = [&](Value value) -> LogicalResult {
    if (!value)
      return failure();
    value = stripWideningIntegerExts(value);
    if (allowedIVs.contains(value)) {
      usedIVs.insert(value);
      return success();
    }
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return failure();
    Operation *op = value.getDefiningOp();
    if (!op)
      return failure();
    if (isa<arith::ConstantOp, arith::ConstantIntOp, arith::ConstantIndexOp>(op))
      return success();
    if (!isSupportedSemanticProducerOp(op))
      return failure();
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (!ifOp.elseBlock() || ifOp.getNumResults() != 1)
        return failure();
      if (failed(visitValue(ifOp.getCondition())))
        return failure();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yield || yield.getNumOperands() != 1)
          return failure();
        if (failed(visitValue(yield.getOperand(0))))
          return failure();
      }
      if (visitedOps.insert(op).second)
        info.sliceOps.push_back(op);
      return success();
    }
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      auto map = load.getAffineMap();
      auto operands = load.getMapOperands();
      llvm::DenseSet<Value> usedByLoad;
      for (AffineExpr expr : map.getResults()) {
        if (!affineExprUsesOnlyAllowedOperands(expr, operands, allowedIVs,
                                               usedByLoad))
          return failure();
      }
      for (Value used : usedByLoad)
        usedIVs.insert(used);
      if (visitedOps.insert(op).second)
        info.sliceOps.push_back(op);
      return success();
    }
    for (Value operand : op->getOperands())
      if (failed(visitValue(operand)))
        return failure();
    if (visitedOps.insert(op).second)
      info.sliceOps.push_back(op);
    return success();
  };

  if (failed(visitValue(root)))
    return failure();

  bool usesRow = usedIVs.contains(row);
  bool usesCol = usedIVs.contains(col);
  bool usesRed = usedIVs.contains(red);
  if (!usesRed)
    return failure();
  if (usesRow && usesCol)
    return failure();
  if (usesRow)
    info.kind = SemanticOperandKind::RowRed;
  else
    info.kind = SemanticOperandKind::RedCol;

  if (auto load = dyn_cast_or_null<AffineLoadOp>(root.getDefiningOp())) {
    info.directLoad = true;
    info.needsMaterialization = false;
    info.memref = load.getMemRef();
  } else {
    info.needsMaterialization = true;
  }
  return info;
}

static FailureOr<SemanticOperandInfo>
analyzeSemanticOperandConv(Value originalRoot, Value oc, Value oh, Value ow,
                           Value ic, Value kh, Value kw) {
  Value root = stripWideningIntegerExts(originalRoot);
  SemanticOperandInfo info;
  info.rootValue = root;

  llvm::DenseSet<Value> allowedIVs;
  for (Value iv : {oc, oh, ow, ic, kh, kw})
    allowedIVs.insert(iv);
  llvm::DenseSet<Operation *> visitedOps;
  llvm::DenseSet<Value> usedIVs;

  std::function<LogicalResult(Value)> visitValue = [&](Value value) -> LogicalResult {
    if (!value)
      return failure();
    value = stripWideningIntegerExts(value);
    if (allowedIVs.contains(value)) {
      usedIVs.insert(value);
      return success();
    }
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return failure();
    Operation *op = value.getDefiningOp();
    if (!op)
      return failure();
    if (isa<arith::ConstantOp, arith::ConstantIntOp, arith::ConstantIndexOp>(op))
      return success();
    if (!isSupportedSemanticProducerOp(op))
      return failure();
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (!ifOp.elseBlock() || ifOp.getNumResults() != 1)
        return failure();
      if (failed(visitValue(ifOp.getCondition())))
        return failure();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yield || yield.getNumOperands() != 1)
          return failure();
        if (failed(visitValue(yield.getOperand(0))))
          return failure();
      }
      if (visitedOps.insert(op).second)
        info.sliceOps.push_back(op);
      return success();
    }
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      auto map = load.getAffineMap();
      auto operands = load.getMapOperands();
      llvm::DenseSet<Value> usedByLoad;
      for (AffineExpr expr : map.getResults()) {
        if (!affineExprUsesOnlyAllowedOperands(expr, operands, allowedIVs,
                                               usedByLoad))
          return failure();
      }
      for (Value used : usedByLoad)
        usedIVs.insert(used);
      if (visitedOps.insert(op).second)
        info.sliceOps.push_back(op);
      return success();
    }
    for (Value operand : op->getOperands())
      if (failed(visitValue(operand)))
        return failure();
    if (visitedOps.insert(op).second)
      info.sliceOps.push_back(op);
    return success();
  };

  if (failed(visitValue(root)))
    return failure();

  bool usesOc = usedIVs.contains(oc);
  bool usesOh = usedIVs.contains(oh);
  bool usesOw = usedIVs.contains(ow);
  bool usesIc = usedIVs.contains(ic);
  bool usesKh = usedIVs.contains(kh);
  bool usesKw = usedIVs.contains(kw);
  if (!usesIc && !usesKh && !usesKw)
    return failure();
  if (usesOc && (usesOh || usesOw))
    return failure();
  if (usesOc)
    info.kind = SemanticOperandKind::ConvWeight;
  else if (usesOh || usesOw)
    info.kind = SemanticOperandKind::ConvInput;
  else
    return failure();

  if (auto load = dyn_cast_or_null<AffineLoadOp>(root.getDefiningOp())) {
    info.directLoad = true;
    info.needsMaterialization = false;
    info.memref = load.getMemRef();
  } else {
    info.needsMaterialization = true;
  }
  return info;
}

static bool slicesIntersectNonTrivially(const SemanticOperandInfo &lhs,
                                        const SemanticOperandInfo &rhs) {
  llvm::DenseSet<Operation *> lhsOps;
  for (Operation *op : lhs.sliceOps)
    lhsOps.insert(op);
  for (Operation *op : rhs.sliceOps) {
    if (!lhsOps.contains(op))
      continue;
    if (isa<arith::ConstantOp, arith::ConstantIntOp, arith::ConstantIndexOp>(op))
      continue;
    return true;
  }
  return false;
}

static bool matchSemanticYieldedReduction2D(ArrayRef<Operation *> ops, Value row,
                                            Value col, Value &inputMemref,
                                            Value &weightMemref,
                                            Value &outputMemref,
                                            bool &hasExistingInput,
                                            bool &weightTransposed,
                                            FailureOr<SemanticOperandInfo> &inputInfo,
                                            FailureOr<SemanticOperandInfo> &weightInfo,
                                            Value &redIV) {
  if (ops.empty())
    return false;

  auto outputLoad = dyn_cast<AffineLoadOp>(ops.front());
  AffineForOp redLoop;
  for (Operation *op : ops)
    if (auto loop = dyn_cast<AffineForOp>(op))
      redLoop = loop;
  auto store = dyn_cast<AffineStoreOp>(ops.back());
  if (!redLoop || !store || redLoop.getNumIterOperands() != 1 ||
      !isRectangularLoop(redLoop))
    return false;
  redIV = redLoop.getInductionVar();

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
  auto yield = cast<AffineYieldOp>(redLoop.getBody()->getTerminator());
  Value yielded = stripIntegerCasts(yield.getOperands()[0]);
  auto addi = yielded.getDefiningOp<arith::AddIOp>();
  auto addf = yielded.getDefiningOp<arith::AddFOp>();
  Value accumulator;
  Value mulValue;
  if (addi) {
    Value lhs = addi.getLhs();
    Value rhs = addi.getRhs();
    bool lhsAcc = stripIntegerCasts(lhs) == redLoop.getRegionIterArgs()[0];
    bool rhsAcc = stripIntegerCasts(rhs) == redLoop.getRegionIterArgs()[0];
    if (lhsAcc == rhsAcc)
      return false;
    accumulator = lhsAcc ? lhs : rhs;
    mulValue = lhsAcc ? rhs : lhs;
  } else if (addf) {
    Value lhs = addf.getLhs();
    Value rhs = addf.getRhs();
    bool lhsAcc = stripIntegerCasts(lhs) == redLoop.getRegionIterArgs()[0];
    bool rhsAcc = stripIntegerCasts(rhs) == redLoop.getRegionIterArgs()[0];
    if (lhsAcc == rhsAcc)
      return false;
    accumulator = lhsAcc ? lhs : rhs;
    mulValue = lhsAcc ? rhs : lhs;
  } else {
    return false;
  }
  if (stripIntegerCasts(accumulator) != redLoop.getRegionIterArgs()[0])
    return false;

  mulValue = stripIntegerCasts(mulValue);
  auto muli = mulValue.getDefiningOp<arith::MulIOp>();
  auto mulf = mulValue.getDefiningOp<arith::MulFOp>();
  Value mulLhs;
  Value mulRhs;
  if (muli) {
    mulLhs = muli.getLhs();
    mulRhs = muli.getRhs();
  } else if (mulf) {
    mulLhs = mulf.getLhs();
    mulRhs = mulf.getRhs();
  } else {
    return false;
  }

  auto lhsInfo = analyzeSemanticOperand2D(mulLhs, row, col, redIV);
  auto rhsInfo = analyzeSemanticOperand2D(mulRhs, row, col, redIV);
  if (failed(lhsInfo) || failed(rhsInfo))
    return false;
  if (slicesIntersectNonTrivially(*lhsInfo, *rhsInfo))
    return false;

  SemanticOperandInfo lhs = *lhsInfo;
  SemanticOperandInfo rhs = *rhsInfo;
  if (lhs.kind == SemanticOperandKind::RowRed &&
      rhs.kind == SemanticOperandKind::RedCol) {
    inputInfo = lhs;
    weightInfo = rhs;
  } else if (lhs.kind == SemanticOperandKind::RedCol &&
             rhs.kind == SemanticOperandKind::RowRed) {
    inputInfo = rhs;
    weightInfo = lhs;
  } else {
    return false;
  }

  if (inputInfo->directLoad)
    inputMemref = inputInfo->memref;
  if (weightInfo->directLoad)
    weightMemref = weightInfo->memref;
  weightTransposed = false;
  return true;
}

static Value cloneSemanticProducerValue(OpBuilder &builder, Value value,
                                        DenseMap<Value, Value> &valueMap) {
  if (auto it = valueMap.find(value); it != valueMap.end())
    return it->second;
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    auto cloned = builder.create<arith::ConstantOp>(constOp.getLoc(),
                                                    constOp.getValue());
    return cloned.getResult();
  }
  if (auto load = value.getDefiningOp<AffineLoadOp>()) {
    SmallVector<Value, 4> remappedOperands;
    remappedOperands.reserve(load.getMapOperands().size());
    for (Value operand : load.getMapOperands())
      remappedOperands.push_back(cloneSemanticProducerValue(builder, operand, valueMap));
    auto cloned = builder.create<AffineLoadOp>(load.getLoc(), load.getMemRef(),
                                               load.getAffineMap(),
                                               remappedOperands);
    valueMap[value] = cloned.getResult();
    return cloned.getResult();
  }
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    Value condition =
        cloneSemanticProducerValue(builder, ifOp.getCondition(), valueMap);
    auto cloned = builder.create<scf::IfOp>(ifOp.getLoc(), ifOp.getResultTypes(),
                                            condition, /*withElseRegion=*/true);
    auto cloneYieldRegion = [&](Region &source, Region &target) {
      auto yield = cast<scf::YieldOp>(source.front().getTerminator());
      Block &targetBlock = target.front();
      OpBuilder::InsertionGuard guard(builder);
      if (!targetBlock.empty())
        targetBlock.back().erase();
      builder.setInsertionPointToEnd(&targetBlock);
      SmallVector<Value, 1> results;
      for (Value operand : yield.getOperands())
        results.push_back(cloneSemanticProducerValue(builder, operand, valueMap));
      builder.create<scf::YieldOp>(ifOp.getLoc(), results);
    };
    cloneYieldRegion(ifOp.getThenRegion(), cloned.getThenRegion());
    cloneYieldRegion(ifOp.getElseRegion(), cloned.getElseRegion());
    for (auto [oldResult, newResult] :
         llvm::zip(ifOp.getResults(), cloned.getResults()))
      valueMap[oldResult] = newResult;
    return valueMap[value];
  }
  Operation *op = value.getDefiningOp();
  SmallVector<Value, 4> remappedOperands;
  remappedOperands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands())
    remappedOperands.push_back(cloneSemanticProducerValue(builder, operand, valueMap));
  OperationState state(op->getLoc(), op->getName().getStringRef());
  state.addOperands(remappedOperands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  Operation *cloned = builder.create(state);
  for (auto [oldResult, newResult] : llvm::zip(op->getResults(), cloned->getResults()))
    valueMap[oldResult] = newResult;
  return valueMap[value];
}

static FailureOr<Value> materializeSemanticOperand2D(
    OpBuilder &builder, Location loc, const SemanticOperandInfo &operand,
    ArrayRef<int64_t> shape, Value dim0Original, Value dim1Original,
    bool allowLargeNormalization, unsigned maxNormalizedOperandElements) {
  if (operand.directLoad)
    return materializeStaticMemrefView(builder, loc, operand.memref, shape);

  int64_t elementCount = 1;
  for (int64_t dim : shape)
    elementCount *= dim;
  if (!allowLargeNormalization && maxNormalizedOperandElements != 0 &&
      static_cast<unsigned>(elementCount) > maxNormalizedOperandElements)
    return failure();

  auto elementType = operand.rootValue.getType();
  auto memrefType = MemRefType::get(shape, elementType);
  Value result = builder.create<memref::AllocOp>(loc, memrefType);
  auto outerLoop = builder.create<AffineForOp>(loc, 0, shape[0], 1);
  builder.setInsertionPointToStart(outerLoop.getBody());
  auto innerLoop = builder.create<AffineForOp>(loc, 0, shape[1], 1);
  builder.setInsertionPointToStart(innerLoop.getBody());

  DenseMap<Value, Value> valueMap;
  valueMap[dim0Original] = outerLoop.getInductionVar();
  valueMap[dim1Original] = innerLoop.getInductionVar();
  Value materialized =
      cloneSemanticProducerValue(builder, operand.rootValue, valueMap);
  builder.create<AffineStoreOp>(loc, materialized, result,
                                ValueRange{outerLoop.getInductionVar(),
                                           innerLoop.getInductionVar()});
  builder.setInsertionPointAfter(innerLoop);
  builder.setInsertionPointAfter(outerLoop);
  return result;
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
  auto isPointwiseDerivedFrom = [&](Value value, Value root) -> bool {
    SmallPtrSet<Value, 16> visited;
    std::function<bool(Value)> recurse = [&](Value current) -> bool {
      current = stripIntegerCasts(current);
      if (current == root)
        return true;
      if (!visited.insert(current).second)
        return true;
      if (auto blockArg = dyn_cast<BlockArgument>(current))
        return blockArg == root;
      Operation *op = current.getDefiningOp();
      if (!op)
        return false;
      if (isa<arith::ConstantOp, arith::ConstantIntOp, arith::ConstantIndexOp>(op))
        return true;
      if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
               arith::ShRSIOp,
               arith::ShRUIOp, arith::CmpIOp, arith::SelectOp,
               arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
        return false;
      bool sawRoot = false;
      for (Value operand : op->getOperands()) {
        bool operandOk = recurse(operand);
        if (!operandOk)
          return false;
        if (stripIntegerCasts(operand) == root)
          sawRoot = true;
      }
      if (isa<arith::SelectOp>(op) || isa<arith::CmpIOp>(op))
        return true;
      return sawRoot || current != value;
    };
    return recurse(value);
  };

  return (isPointwiseDerivedFrom(lhs, inputValue) &&
          isPointwiseDerivedFrom(rhs, weightValue)) ||
         (isPointwiseDerivedFrom(lhs, weightValue) &&
          isPointwiseDerivedFrom(rhs, inputValue));
}

static bool isSupportedPointwiseIntegerOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
             arith::ShRSIOp,
             arith::ShRUIOp, arith::CmpIOp, arith::SelectOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::ConstantOp,
             arith::ConstantIntOp, arith::ConstantIndexOp>(op);
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
            arith::TruncIOp, arith::CmpIOp, arith::SelectOp>(op))
      continue;
    if (!mulf)
      mulf = dyn_cast<arith::MulFOp>(op);
    if (!muli)
      muli = dyn_cast<arith::MulIOp>(op);
    if (!addf)
      addf = dyn_cast<arith::AddFOp>(op);
    if (!addi)
      addi = dyn_cast<arith::AddIOp>(op);
    if (!isa<arith::MulFOp, arith::MulIOp, arith::AddFOp, arith::AddIOp>(op) &&
        !isSupportedPointwiseIntegerOp(op))
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
    if (isa<AffineLoadOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::CmpIOp, arith::SelectOp>(op))
      continue;
    if (!mulf)
      mulf = dyn_cast<arith::MulFOp>(op);
    if (!muli)
      muli = dyn_cast<arith::MulIOp>(op);
    if (!addf)
      addf = dyn_cast<arith::AddFOp>(op);
    if (!addi)
      addi = dyn_cast<arith::AddIOp>(op);
    if (!isa<arith::MulFOp, arith::MulIOp, arith::AddFOp, arith::AddIOp>(op) &&
        !isSupportedPointwiseIntegerOp(op))
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
                                             bool &hasExistingInput,
                                             bool &weightTransposed,
                                             int64_t &reductionExtent) {
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
  bool matchedTransposedWeight = false;
  for (auto load : loads) {
    if (sameMemrefAccess(load, load.getMemRef(), {rowCheck, redCheck}))
      inputLoad = load;
    else if (sameMemrefAccess(load, load.getMemRef(), {redCheck, colCheck})) {
      weightLoad = load;
      matchedTransposedWeight = false;
    } else if (sameMemrefAccess(load, load.getMemRef(),
                                {colCheck, redCheck})) {
      weightLoad = load;
      matchedTransposedWeight = true;
    }
  }
  if (!inputLoad || !weightLoad)
    return false;

  auto yield = cast<AffineYieldOp>(redLoop.getBody()->getTerminator());
  if (!matchesYieldedReductionComputation(redBodyOps, inputLoad, weightLoad,
                                          redLoop.getRegionIterArgs()[0],
                                          yield.getOperands()[0]))
    return false;
  reductionExtent = redLoop.getConstantUpperBound();
  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  weightTransposed = matchedTransposedWeight;
  return inputMemref != weightMemref && inputMemref != outputMemref &&
         weightMemref != outputMemref;
}

static bool matchIterArgReduction3DCanonical(ArrayRef<Operation *> ops,
                                             Value batch, Value row, Value col,
                                             Value &inputMemref,
                                             Value &weightMemref,
                                             Value &outputMemref,
                                             bool &hasExistingInput,
                                             bool &weightTransposed,
                                             int64_t &reductionExtent) {
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
  bool matchedTransposedWeight = false;
  for (auto load : loads) {
    if (sameMemrefAccess(load, load.getMemRef(),
                         {batchCheck, rowCheck, redCheck}))
      inputLoad = load;
    else if (sameMemrefAccess(load, load.getMemRef(),
                              {batchCheck, redCheck, colCheck})) {
      weightLoad = load;
      matchedTransposedWeight = false;
    } else if (sameMemrefAccess(load, load.getMemRef(),
                                {batchCheck, colCheck, redCheck})) {
      weightLoad = load;
      matchedTransposedWeight = true;
    }
  }
  if (!inputLoad || !weightLoad)
    return false;

  auto yield = cast<AffineYieldOp>(redLoop.getBody()->getTerminator());
  if (!matchesYieldedReductionComputation(redBodyOps, inputLoad, weightLoad,
                                          redLoop.getRegionIterArgs()[0],
                                          yield.getOperands()[0]))
    return false;
  reductionExtent = redLoop.getConstantUpperBound();
  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  weightTransposed = matchedTransposedWeight;
  return inputMemref != weightMemref && inputMemref != outputMemref &&
         weightMemref != outputMemref;
}

static bool matchIterArgReductionConvCanonical(ArrayRef<Operation *> ops,
                                               Value oc, Value oh, Value ow,
                                               Value &inputMemref,
                                               Value &weightMemref,
                                               Value &outputMemref,
                                               bool &hasExistingInput,
                                               int64_t &icExtent,
                                               int64_t &khExtent,
                                               int64_t &kwExtent) {
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
    if (!sameConvOutputAccess(outputLoad, outputLoad.getMemRef(), ocCheck,
                              ohCheck, owCheck) ||
        !sameConvOutputAccess(store, outputLoad.getMemRef(), ocCheck, ohCheck,
                              owCheck))
      return false;
    if (stripIntegerCasts(icLoop.getIterOperands().front()) != outputLoad.getResult())
      return false;
    outputMemref = outputLoad.getMemRef();
    hasExistingInput = true;
  } else {
    if (!sameConvOutputAccess(store, store.getMemRef(), ocCheck, ohCheck,
                              owCheck) ||
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

  icExtent = icLoop.getConstantUpperBound();
  khExtent = khLoop.getConstantUpperBound();
  kwExtent = kwLoop.getConstantUpperBound();

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
    if (sameConvInputAccess(load, load.getMemRef(), icCheck, ohPlusKhCheck,
                            owPlusKwCheck))
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

static bool matchSemanticIterArgReductionConvCanonical(
    ArrayRef<Operation *> ops, Value oc, Value oh, Value ow, Value &inputMemref,
    Value &weightMemref, Value &outputMemref, bool &hasExistingInput,
    int64_t &icExtent, int64_t &khExtent, int64_t &kwExtent,
    FailureOr<SemanticOperandInfo> &inputInfo,
    FailureOr<SemanticOperandInfo> &weightInfo, Value &icIV, Value &khIV,
    Value &kwIV) {
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
    if (!sameConvOutputAccess(outputLoad, outputLoad.getMemRef(), ocCheck,
                              ohCheck, owCheck) ||
        !sameConvOutputAccess(store, outputLoad.getMemRef(), ocCheck, ohCheck,
                              owCheck))
      return false;
    if (stripIntegerCasts(icLoop.getIterOperands().front()) !=
        outputLoad.getResult())
      return false;
    outputMemref = outputLoad.getMemRef();
    hasExistingInput = true;
  } else {
    if (!sameConvOutputAccess(store, store.getMemRef(), ocCheck, ohCheck,
                              owCheck) ||
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
      stripIntegerCasts(khLoop.getIterOperands().front()) !=
          icLoop.getRegionIterArgs()[0])
    return false;

  auto khBodyOps = getBodyOps(*khLoop.getBody());
  if (khBodyOps.size() != 1 || !isa<AffineForOp>(khBodyOps.front()))
    return false;
  auto kwLoop = cast<AffineForOp>(khBodyOps.front());
  if (kwLoop.getNumIterOperands() != 1 || !isRectangularLoop(kwLoop) ||
      stripIntegerCasts(kwLoop.getIterOperands().front()) !=
          khLoop.getRegionIterArgs()[0])
    return false;

  icExtent = icLoop.getConstantUpperBound();
  khExtent = khLoop.getConstantUpperBound();
  kwExtent = kwLoop.getConstantUpperBound();
  icIV = icLoop.getInductionVar();
  khIV = khLoop.getInductionVar();
  kwIV = kwLoop.getInductionVar();

  auto kwBodyOps = getBodyOps(*kwLoop.getBody());
  auto kwYield = cast<AffineYieldOp>(kwLoop.getBody()->getTerminator());
  Value yielded = stripIntegerCasts(kwYield.getOperands()[0]);
  auto addi = yielded.getDefiningOp<arith::AddIOp>();
  auto addf = yielded.getDefiningOp<arith::AddFOp>();
  Value accumulator;
  Value mulValue;
  if (addi) {
    Value lhs = addi.getLhs();
    Value rhs = addi.getRhs();
    bool lhsAcc = stripIntegerCasts(lhs) == kwLoop.getRegionIterArgs()[0];
    bool rhsAcc = stripIntegerCasts(rhs) == kwLoop.getRegionIterArgs()[0];
    if (lhsAcc == rhsAcc)
      return false;
    accumulator = lhsAcc ? lhs : rhs;
    mulValue = lhsAcc ? rhs : lhs;
  } else if (addf) {
    Value lhs = addf.getLhs();
    Value rhs = addf.getRhs();
    bool lhsAcc = stripIntegerCasts(lhs) == kwLoop.getRegionIterArgs()[0];
    bool rhsAcc = stripIntegerCasts(rhs) == kwLoop.getRegionIterArgs()[0];
    if (lhsAcc == rhsAcc)
      return false;
    accumulator = lhsAcc ? lhs : rhs;
    mulValue = lhsAcc ? rhs : lhs;
  } else {
    return false;
  }
  if (stripIntegerCasts(accumulator) != kwLoop.getRegionIterArgs()[0])
    return false;

  mulValue = stripIntegerCasts(mulValue);
  auto muli = mulValue.getDefiningOp<arith::MulIOp>();
  auto mulf = mulValue.getDefiningOp<arith::MulFOp>();
  Value mulLhs;
  Value mulRhs;
  if (muli) {
    mulLhs = muli.getLhs();
    mulRhs = muli.getRhs();
  } else if (mulf) {
    mulLhs = mulf.getLhs();
    mulRhs = mulf.getRhs();
  } else {
    return false;
  }

  auto lhsInfo =
      analyzeSemanticOperandConv(mulLhs, oc, oh, ow, icIV, khIV, kwIV);
  auto rhsInfo =
      analyzeSemanticOperandConv(mulRhs, oc, oh, ow, icIV, khIV, kwIV);
  if (failed(lhsInfo) || failed(rhsInfo))
    return false;
  if (slicesIntersectNonTrivially(*lhsInfo, *rhsInfo))
    return false;

  SemanticOperandInfo lhs = *lhsInfo;
  SemanticOperandInfo rhs = *rhsInfo;
  if (lhs.kind == SemanticOperandKind::ConvInput &&
      rhs.kind == SemanticOperandKind::ConvWeight) {
    inputInfo = lhs;
    weightInfo = rhs;
  } else if (lhs.kind == SemanticOperandKind::ConvWeight &&
             rhs.kind == SemanticOperandKind::ConvInput) {
    inputInfo = rhs;
    weightInfo = lhs;
  } else {
    return false;
  }

  auto khYield = cast<AffineYieldOp>(khLoop.getBody()->getTerminator());
  auto icYield = cast<AffineYieldOp>(icLoop.getBody()->getTerminator());
  if (stripIntegerCasts(khYield.getOperands()[0]) != kwLoop.getResult(0) ||
      stripIntegerCasts(icYield.getOperands()[0]) != khLoop.getResult(0))
    return false;

  if (inputInfo->directLoad)
    inputMemref = inputInfo->memref;
  if (weightInfo->directLoad)
    weightMemref = weightInfo->memref;
  return true;
}

static bool matchIterArgReductionConv1x1Canonical(ArrayRef<Operation *> ops,
                                                  Value oc, Value oh, Value ow,
                                                  Value &inputMemref,
                                                  Value &weightMemref,
                                                  Value &outputMemref,
                                                  bool &hasExistingInput,
                                                  int64_t &icExtent) {
  if (ops.empty())
    return false;

  auto outputLoad = dyn_cast<AffineLoadOp>(ops.front());
  AffineForOp icLoop;
  for (Operation *op : ops)
    if (auto loop = dyn_cast<AffineForOp>(op))
      icLoop = loop;
  auto store = dyn_cast<AffineStoreOp>(ops.back());
  if (!icLoop || !store || icLoop.getNumIterOperands() != 1 ||
      !isRectangularLoop(icLoop)) {
    outputLoad = AffineLoadOp();
    if (!icLoop || !store || icLoop.getNumIterOperands() != 1 ||
        !isRectangularLoop(icLoop))
      return false;
  }
  icExtent = icLoop.getConstantUpperBound();

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
    return matchesAffineExpr(expr, operands, icLoop.getInductionVar());
  };
  auto zeroCheck = [&](AffineExpr expr, ValueRange operands) {
    return matchesAffineExpr(expr, 0);
  };

  if (outputLoad) {
    if (!sameConvOutputAccess(outputLoad, outputLoad.getMemRef(), ocCheck,
                              ohCheck, owCheck) ||
        !sameConvOutputAccess(store, outputLoad.getMemRef(), ocCheck, ohCheck,
                              owCheck))
      return false;
    if (stripIntegerCasts(icLoop.getIterOperands().front()) != outputLoad.getResult())
      return false;
    outputMemref = outputLoad.getMemRef();
    hasExistingInput = true;
  } else {
    if (!sameConvOutputAccess(store, store.getMemRef(), ocCheck, ohCheck,
                              owCheck) ||
        !isZeroInitValue(icLoop.getIterOperands().front()))
      return false;
    outputMemref = store.getMemRef();
    hasExistingInput = false;
  }
  if (stripIntegerCasts(store.getValueToStore()) != icLoop.getResult(0))
    return false;

  auto icBodyOps = getBodyOps(*icLoop.getBody());
  SmallVector<AffineLoadOp, 2> loads;
  for (Operation *op : icBodyOps)
    if (auto load = dyn_cast<AffineLoadOp>(op))
      loads.push_back(load);
  if (loads.size() != 2)
    return false;

  AffineLoadOp inputLoad;
  AffineLoadOp weightLoad;
  for (auto load : loads) {
    if (sameConvInputAccess(load, load.getMemRef(), icCheck, ohCheck, owCheck)) {
      inputLoad = load;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(),
                         {ocCheck, icCheck, zeroCheck, zeroCheck})) {
      weightLoad = load;
      continue;
    }
  }
  if (!inputLoad || !weightLoad)
    return false;

  auto icYield = cast<AffineYieldOp>(icLoop.getBody()->getTerminator());
  if (!matchesYieldedReductionComputation(icBodyOps, inputLoad, weightLoad,
                                          icLoop.getRegionIterArgs()[0],
                                          icYield.getOperands()[0]))
    return false;

  inputMemref = inputLoad.getMemRef();
  weightMemref = weightLoad.getMemRef();
  return inputMemref != weightMemref && inputMemref != outputMemref &&
         weightMemref != outputMemref;
}

static bool matchStoreStyleReduction2D(ArrayRef<Operation *> ops, Value row,
                                       Value col, Value red, Value outputMemref,
                                       Value &inputMemref,
                                       Value &weightMemref,
                                       bool &weightTransposed) {
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
  bool matchedTransposedWeight = false;
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
      matchedTransposedWeight = false;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(), {colCheck, redCheck})) {
      weightLoad = load;
      matchedTransposedWeight = true;
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
  weightTransposed = matchedTransposedWeight;
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
                                       Value &weightMemref,
                                       bool &weightTransposed) {
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
  bool matchedTransposedWeight = false;
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
      matchedTransposedWeight = false;
      continue;
    }
    if (sameMemrefAccess(load, load.getMemRef(),
                         {batchCheck, colCheck, redCheck})) {
      weightLoad = load;
      matchedTransposedWeight = true;
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
  weightTransposed = matchedTransposedWeight;
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
    if (sameConvOutputAccess(load, outputMemref, ocCheck, ohCheck, owCheck)) {
      outputLoad = load;
      continue;
    }
    if (sameConvInputAccess(load, load.getMemRef(), icCheck, ohPlusKhCheck,
                            owPlusKwCheck)) {
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
  if (!sameConvOutputAccess(store, outputMemref, ocCheck, ohCheck, owCheck))
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

static LogicalResult matchStripMinedGEMMV(AffineForOp rowTileLoop,
                                          AffineCandidateInfo &info) {
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
  bool weightTransposed = false;
  if (!matchStoreStyleReduction2D(redOps, rowLoop.getInductionVar(),
                                  colLoop.getInductionVar(),
                                  redLoop.getInductionVar(),
                                  outputStore.getMemRef(), inputMemref,
                                  weightMemref, weightTransposed))
    if (!matchStoreStyleReduction2DLinearized(
            redOps, rowLoop.getInductionVar(), colLoop.getInductionVar(),
            redLoop.getInductionVar(), colTileLoop.getConstantUpperBound(),
            redTileLoop.getConstantUpperBound(), outputStore.getMemRef(),
            inputMemref, weightMemref))
      return failure();
  info.hasExistingInput = true;

  info.family = StringRef("GEMMV");
  info.sourceFamily = info.family;
  info.rootLoop = rowTileLoop;
  info.hasExistingInput = true;
  info.weightTransposed = weightTransposed;
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

static LogicalResult matchStripMinedGEMM(AffineForOp batchLoop,
                                         AffineCandidateInfo &info) {
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
  bool weightTransposed = false;
  if (!matchStoreStyleReduction3D(redOps, batchLoop.getInductionVar(),
                                  rowLoop.getInductionVar(),
                                  colLoop.getInductionVar(),
                                  redLoop.getInductionVar(),
                                  outputStore.getMemRef(), inputMemref,
                                  weightMemref, weightTransposed))
    if (!matchStoreStyleReduction3DLinearized(
            redOps, batchLoop.getInductionVar(), rowLoop.getInductionVar(),
            colLoop.getInductionVar(), redLoop.getInductionVar(),
            rowTileLoop.getConstantUpperBound(), colLoop.getConstantUpperBound(),
            redTileLoop.getConstantUpperBound(), outputStore.getMemRef(),
            inputMemref, weightMemref))
      return failure();
  info.hasExistingInput = true;

  info.family = StringRef("GEMM");
  info.sourceFamily = info.family;
  info.rootLoop = batchLoop;
  info.hasExistingInput = true;
  info.weightTransposed = weightTransposed;
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

static LogicalResult matchStripMinedCONV(AffineForOp ocTileLoop,
                                         AffineCandidateInfo &info) {
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
  info.rootLoop = ocTileLoop;
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
  auto context = ocTileLoop.getContext();
  info.strides = DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1, 1});
  info.dilations = DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1, 1});
  return success();
}

static LogicalResult matchCanonicalGEMMV(AffineForOp rowLoop,
                                         AffineCandidateInfo &info) {
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
  info.rowIV = rowLoop.getInductionVar();
  info.colIV = colLoop.getInductionVar();
  if (colOps.size() == 1 && isa<AffineForOp>(colOps.front())) {
    auto redLoop = cast<AffineForOp>(colOps.front());
    if (!isRectangularLoop(redLoop))
      return failure();
    auto redOps = getBodyOps(*redLoop.getBody());
    auto outputStore = dyn_cast<AffineStoreOp>(redOps.back());
    if (!outputStore)
      return failure();
    bool weightTransposed = false;
    if (!matchStoreStyleReduction2D(redOps, rowLoop.getInductionVar(),
                                    colLoop.getInductionVar(),
                                    redLoop.getInductionVar(),
                                    outputStore.getMemRef(), inputMemref,
                                    weightMemref, weightTransposed))
      if (!matchStoreStyleReduction2DLinearized(
              redOps, rowLoop.getInductionVar(), colLoop.getInductionVar(),
              redLoop.getInductionVar(), colLoop.getConstantUpperBound(),
              redLoop.getConstantUpperBound(), outputStore.getMemRef(),
              inputMemref, weightMemref))
        return failure();
    info.hasExistingInput = true;
    info.weightTransposed = weightTransposed;
    reductionExtent = redLoop.getConstantUpperBound();
    outputMemref = outputStore.getMemRef();
    info.redIV = redLoop.getInductionVar();
  } else if (!matchIterArgReduction2DCanonical(colOps, rowLoop.getInductionVar(),
                                               colLoop.getInductionVar(),
                                               inputMemref, weightMemref,
                                               outputMemref,
                                               info.hasExistingInput,
                                               info.weightTransposed,
                                               reductionExtent)) {
    if (!matchSemanticYieldedReduction2D(colOps, rowLoop.getInductionVar(),
                                         colLoop.getInductionVar(), inputMemref,
                                         weightMemref, outputMemref,
                                         info.hasExistingInput,
                                         info.weightTransposed,
                                         info.semanticInput,
                                         info.semanticWeight, info.redIV))
      return failure();
  }

  info.family = StringRef("GEMMV");
  info.sourceFamily = info.family;
  info.rootLoop = rowLoop;
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputMemref;
  int64_t kExtent = reductionExtent;
  if (kExtent < 0) {
    if (info.redIV) {
      if (auto parentLoop = getOwningAffineFor(info.redIV))
        kExtent = parentLoop.getConstantUpperBound();
    }
  }
  if (kExtent < 0)
    return failure();
  info.inputShape = {rowLoop.getConstantUpperBound(), kExtent};
  info.weightShape = {kExtent, colLoop.getConstantUpperBound()};
  info.outputShape = {rowLoop.getConstantUpperBound(), colLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchCanonicalGEMM(AffineForOp batchLoop,
                                        AffineCandidateInfo &info) {
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
    bool weightTransposed = false;
    if (!matchStoreStyleReduction3D(redOps, batchLoop.getInductionVar(),
                                    rowLoop.getInductionVar(),
                                    colLoop.getInductionVar(),
                                    redLoop.getInductionVar(),
                                    outputStore.getMemRef(), inputMemref,
                                    weightMemref, weightTransposed))
      if (!matchStoreStyleReduction3DLinearized(
              redOps, batchLoop.getInductionVar(), rowLoop.getInductionVar(),
              colLoop.getInductionVar(), redLoop.getInductionVar(),
              rowLoop.getConstantUpperBound(), colLoop.getConstantUpperBound(),
              redLoop.getConstantUpperBound(), outputStore.getMemRef(),
              inputMemref, weightMemref))
        return failure();
    info.hasExistingInput = true;
    info.weightTransposed = weightTransposed;
    reductionExtent = redLoop.getConstantUpperBound();
    outputMemref = outputStore.getMemRef();
  } else if (!matchIterArgReduction3DCanonical(
                 colOps, batchLoop.getInductionVar(), rowLoop.getInductionVar(),
                 colLoop.getInductionVar(), inputMemref, weightMemref,
                 outputMemref, info.hasExistingInput,
                 info.weightTransposed, reductionExtent)) {
    return failure();
  }

  info.family = StringRef("GEMM");
  info.sourceFamily = info.family;
  info.rootLoop = batchLoop;
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputMemref;
  int64_t kExtent = reductionExtent;
  if (kExtent < 0)
    return failure();
  info.inputShape = {batchLoop.getConstantUpperBound(), rowLoop.getConstantUpperBound(),
                     kExtent};
  info.weightShape = {batchLoop.getConstantUpperBound(), kExtent,
                      colLoop.getConstantUpperBound()};
  info.outputShape = {batchLoop.getConstantUpperBound(), rowLoop.getConstantUpperBound(),
                      colLoop.getConstantUpperBound()};
  return success();
}

static LogicalResult matchCanonicalCONV(AffineForOp ocLoop,
                                        AffineCandidateInfo &info) {
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
  int64_t icExtent = -1;
  int64_t khExtent = -1;
  int64_t kwExtent = -1;
  info.convOcIV = ocLoop.getInductionVar();
  info.convOhIV = ohLoop.getInductionVar();
  info.convOwIV = owLoop.getInductionVar();
  if (matchIterArgReductionConv1x1Canonical(
          owOps, ocLoop.getInductionVar(), ohLoop.getInductionVar(),
          owLoop.getInductionVar(), inputMemref, weightMemref, outputMemref,
          info.hasExistingInput, icExtent)) {
    khExtent = 1;
    kwExtent = 1;
  } else if (owOps.size() == 1 && isa<AffineForOp>(owOps.front())) {
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
    icExtent = icLoop.getConstantUpperBound();
    khExtent = khLoop.getConstantUpperBound();
    kwExtent = kwLoop.getConstantUpperBound();

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
    info.hasExistingInput = true;
    outputMemref = outputStore.getMemRef();
  } else if (!matchIterArgReductionConvCanonical(
                 owOps, ocLoop.getInductionVar(), ohLoop.getInductionVar(),
                 owLoop.getInductionVar(), inputMemref, weightMemref,
                 outputMemref, info.hasExistingInput, icExtent, khExtent,
                 kwExtent)) {
    inputMemref = Value();
    weightMemref = Value();
    if (!matchSemanticIterArgReductionConvCanonical(
            owOps, ocLoop.getInductionVar(), ohLoop.getInductionVar(),
            owLoop.getInductionVar(), inputMemref, weightMemref, outputMemref,
            info.hasExistingInput, icExtent, khExtent, kwExtent,
            info.semanticInput, info.semanticWeight, info.convIcIV,
            info.convKhIV, info.convKwIV))
      return failure();
  }

  info.family = StringRef("CONV");
  info.sourceFamily = info.family;
  info.rootLoop = ocLoop;
  info.inputMemref = inputMemref;
  info.weightMemref = weightMemref;
  info.outputMemref = outputMemref;
  auto inputType = inputMemref ? dyn_cast<MemRefType>(inputMemref.getType())
                               : MemRefType();
  auto weightType = weightMemref ? dyn_cast<MemRefType>(weightMemref.getType())
                                 : MemRefType();
  if (icExtent >= 0 && khExtent >= 0 && kwExtent >= 0) {
    info.inputShape = {1, icExtent,
                       ohLoop.getConstantUpperBound() + khExtent - 1,
                       owLoop.getConstantUpperBound() + kwExtent - 1};
    info.weightShape = {ocLoop.getConstantUpperBound(), icExtent, khExtent,
                        kwExtent};
  } else {
    if (!inputType || !weightType)
      return failure();
    info.inputShape = llvm::to_vector(inputType.getShape());
    info.weightShape = llvm::to_vector(weightType.getShape());
  }
  info.outputShape = {1, ocLoop.getConstantUpperBound(), ohLoop.getConstantUpperBound(),
                      owLoop.getConstantUpperBound()};
  auto context = ocLoop.getContext();
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

static FailureOr<int64_t>
getConvPackedElementCount(const AffineCandidateInfo &candidate) {
  if (candidate.sourceInputShape.size() != 4 ||
      candidate.sourceWeightShape.size() != 4 ||
      candidate.sourceOutputShape.size() != 4)
    return failure();
  const int64_t oc = candidate.sourceOutputShape[1];
  const int64_t oh = candidate.sourceOutputShape[2];
  const int64_t ow = candidate.sourceOutputShape[3];
  const int64_t ic = candidate.sourceInputShape[1];
  const int64_t kh = candidate.sourceWeightShape[2];
  const int64_t kw = candidate.sourceWeightShape[3];
  const int64_t flatK = ic * kh * kw;
  const int64_t flatN = oh * ow;
  int64_t total = oc * flatK + flatK * flatN;
  if (candidate.hasExistingInput)
    total += oc * flatN;
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
    if (!sourceType.getLayout().isIdentity())
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
                                                     builder.getIndexAttr(0),
                                                     sizes, strides);
  }
  if (!sourceType.hasStaticShape())
    return source;
  if (!sourceType.getLayout().isIdentity())
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

static RankedTensorType getExactTensorTypeForMemref(Value memref) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType || !memrefType.hasStaticShape())
    return RankedTensorType();
  return RankedTensorType::get(memrefType.getShape(),
                               memrefType.getElementType());
}

static bool hasExactStaticMemrefShape(Value memref, ArrayRef<int64_t> shape) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  return memrefType && memrefType.hasStaticShape() &&
         memrefType.getShape().size() == shape.size() &&
         llvm::equal(memrefType.getShape(), shape);
}

static Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static Value materializeTransposedWeightMemref(OpBuilder &builder, Location loc,
                                               Value weightMemref,
                                               ArrayRef<int64_t> canonicalShape) {
  auto weightType = dyn_cast<MemRefType>(weightMemref.getType());
  if (!weightType || !weightType.hasStaticShape())
    return weightMemref;
  if (weightType.getRank() != 2 && weightType.getRank() != 3)
    return weightMemref;

  auto transposedType =
      MemRefType::get(canonicalShape, weightType.getElementType());
  Value transposed = builder.create<memref::AllocOp>(loc, transposedType);

  if (weightType.getRank() == 2) {
    auto kExtent = canonicalShape[0];
    auto nExtent = canonicalShape[1];
    auto kLoop = builder.create<AffineForOp>(loc, 0, kExtent, 1);
    builder.setInsertionPointToStart(kLoop.getBody());
    auto nLoop = builder.create<AffineForOp>(loc, 0, nExtent, 1);
    builder.setInsertionPointToStart(nLoop.getBody());
    Value sourceElt = builder.create<memref::LoadOp>(
        loc, weightMemref, ValueRange{nLoop.getInductionVar(),
                                      kLoop.getInductionVar()});
    builder.create<memref::StoreOp>(
        loc, sourceElt, transposed,
        ValueRange{kLoop.getInductionVar(), nLoop.getInductionVar()});
    builder.setInsertionPointAfter(nLoop);
    builder.setInsertionPointAfter(kLoop);
    return transposed;
  }

  auto bExtent = canonicalShape[0];
  auto kExtent = canonicalShape[1];
  auto nExtent = canonicalShape[2];
  auto bLoop = builder.create<AffineForOp>(loc, 0, bExtent, 1);
  builder.setInsertionPointToStart(bLoop.getBody());
  auto kLoop = builder.create<AffineForOp>(loc, 0, kExtent, 1);
  builder.setInsertionPointToStart(kLoop.getBody());
  auto nLoop = builder.create<AffineForOp>(loc, 0, nExtent, 1);
  builder.setInsertionPointToStart(nLoop.getBody());
  Value sourceElt = builder.create<memref::LoadOp>(
      loc, weightMemref,
      ValueRange{bLoop.getInductionVar(), nLoop.getInductionVar(),
                 kLoop.getInductionVar()});
  builder.create<memref::StoreOp>(
      loc, sourceElt, transposed,
      ValueRange{bLoop.getInductionVar(), kLoop.getInductionVar(),
                 nLoop.getInductionVar()});
  builder.setInsertionPointAfter(nLoop);
  builder.setInsertionPointAfter(kLoop);
  builder.setInsertionPointAfter(bLoop);
  return transposed;
}

static accel::GEMMOp lowerConvAsGemmOnly(OpBuilder &builder, Location loc,
                                         const AffineCandidateInfo &candidate) {
  Value inputMemref =
      candidate.inputMemref
          ? materializeStaticMemrefView(builder, loc, candidate.inputMemref,
                                        candidate.sourceInputShape)
          : Value();
  Value weightMemref =
      candidate.weightMemref
          ? materializeStaticMemrefView(builder, loc, candidate.weightMemref,
                                        candidate.sourceWeightShape)
          : Value();
  Value outputMemref = materializeStaticMemrefView(
      builder, loc, candidate.outputMemref, candidate.sourceOutputShape);

  Type inputElementType =
      inputMemref ? cast<MemRefType>(inputMemref.getType()).getElementType()
                  : candidate.semanticInput->rootValue.getType();
  Type weightElementType =
      weightMemref ? cast<MemRefType>(weightMemref.getType()).getElementType()
                   : candidate.semanticWeight->rootValue.getType();
  auto outputType = cast<MemRefType>(outputMemref.getType());
  const int64_t oc = candidate.sourceOutputShape[1];
  const int64_t oh = candidate.sourceOutputShape[2];
  const int64_t ow = candidate.sourceOutputShape[3];
  const int64_t ic = candidate.sourceInputShape[1];
  const int64_t kh = candidate.sourceWeightShape[2];
  const int64_t kw = candidate.sourceWeightShape[3];
  const int64_t flatK = ic * kh * kw;
  const int64_t flatN = oh * ow;

  auto gemmAType = MemRefType::get({oc, flatK}, weightElementType);
  auto gemmBType = MemRefType::get({flatK, flatN}, inputElementType);
  auto gemmOutType = MemRefType::get({oc, flatN}, outputType.getElementType());

  Value gemmA = builder.create<memref::AllocOp>(loc, gemmAType);
  Value gemmB = builder.create<memref::AllocOp>(loc, gemmBType);
  auto c0 = createIndexConstant(builder, loc, 0);
  auto cKh = createIndexConstant(builder, loc, kh);
  auto cKw = createIndexConstant(builder, loc, kw);
  auto cOw = createIndexConstant(builder, loc, ow);

  auto ocLoop = builder.create<AffineForOp>(loc, 0, oc, 1);
  builder.setInsertionPointToStart(ocLoop.getBody());
  auto icLoopA = builder.create<AffineForOp>(loc, 0, ic, 1);
  builder.setInsertionPointToStart(icLoopA.getBody());
  auto khLoopA = builder.create<AffineForOp>(loc, 0, kh, 1);
  builder.setInsertionPointToStart(khLoopA.getBody());
  auto kwLoopA = builder.create<AffineForOp>(loc, 0, kw, 1);
  builder.setInsertionPointToStart(kwLoopA.getBody());
  Value flatKIdxA = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(
          loc,
          builder.create<arith::AddIOp>(
              loc, builder.create<arith::MulIOp>(
                       loc, icLoopA.getInductionVar(), cKh),
              khLoopA.getInductionVar()),
          cKw),
      kwLoopA.getInductionVar());
  Value aElt;
  if (succeeded(candidate.semanticWeight) &&
      candidate.semanticWeight->needsMaterialization) {
    DenseMap<Value, Value> valueMap;
    valueMap[candidate.convOcIV] = ocLoop.getInductionVar();
    valueMap[candidate.convIcIV] = icLoopA.getInductionVar();
    valueMap[candidate.convKhIV] = khLoopA.getInductionVar();
    valueMap[candidate.convKwIV] = kwLoopA.getInductionVar();
    aElt = cloneSemanticProducerValue(builder, candidate.semanticWeight->rootValue,
                                      valueMap);
  } else {
    aElt = builder.create<memref::LoadOp>(
        loc, weightMemref,
        ValueRange{ocLoop.getInductionVar(), icLoopA.getInductionVar(),
                   khLoopA.getInductionVar(), kwLoopA.getInductionVar()});
  }
  builder.create<memref::StoreOp>(
      loc, aElt, gemmA, ValueRange{ocLoop.getInductionVar(), flatKIdxA});
  builder.setInsertionPointAfter(kwLoopA);
  builder.setInsertionPointAfter(khLoopA);
  builder.setInsertionPointAfter(icLoopA);
  builder.setInsertionPointAfter(ocLoop);

  auto icLoopB = builder.create<AffineForOp>(loc, 0, ic, 1);
  builder.setInsertionPointToStart(icLoopB.getBody());
  auto khLoopB = builder.create<AffineForOp>(loc, 0, kh, 1);
  builder.setInsertionPointToStart(khLoopB.getBody());
  auto kwLoopB = builder.create<AffineForOp>(loc, 0, kw, 1);
  builder.setInsertionPointToStart(kwLoopB.getBody());
  auto ohLoopB = builder.create<AffineForOp>(loc, 0, oh, 1);
  builder.setInsertionPointToStart(ohLoopB.getBody());
  auto owLoopB = builder.create<AffineForOp>(loc, 0, ow, 1);
  builder.setInsertionPointToStart(owLoopB.getBody());
  Value flatKIdxB = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(
          loc,
          builder.create<arith::AddIOp>(
              loc, builder.create<arith::MulIOp>(
                       loc, icLoopB.getInductionVar(), cKh),
              khLoopB.getInductionVar()),
          cKw),
      kwLoopB.getInductionVar());
  Value flatNIdxB = builder.create<arith::AddIOp>(
      loc, builder.create<arith::MulIOp>(loc, ohLoopB.getInductionVar(), cOw),
      owLoopB.getInductionVar());
  Value ihIdx =
      builder.create<arith::AddIOp>(loc, ohLoopB.getInductionVar(),
                                    khLoopB.getInductionVar());
  Value iwIdx =
      builder.create<arith::AddIOp>(loc, owLoopB.getInductionVar(),
                                    kwLoopB.getInductionVar());
  Value bElt;
  if (succeeded(candidate.semanticInput) &&
      candidate.semanticInput->needsMaterialization) {
    DenseMap<Value, Value> valueMap;
    valueMap[candidate.convOhIV] = ohLoopB.getInductionVar();
    valueMap[candidate.convOwIV] = owLoopB.getInductionVar();
    valueMap[candidate.convIcIV] = icLoopB.getInductionVar();
    valueMap[candidate.convKhIV] = khLoopB.getInductionVar();
    valueMap[candidate.convKwIV] = kwLoopB.getInductionVar();
    bElt = cloneSemanticProducerValue(builder, candidate.semanticInput->rootValue,
                                      valueMap);
  } else {
    bElt = builder.create<memref::LoadOp>(
        loc, inputMemref,
        ValueRange{c0, icLoopB.getInductionVar(), ihIdx, iwIdx});
  }
  builder.create<memref::StoreOp>(
      loc, bElt, gemmB, ValueRange{flatKIdxB, flatNIdxB});
  builder.setInsertionPointAfter(owLoopB);
  builder.setInsertionPointAfter(ohLoopB);
  builder.setInsertionPointAfter(kwLoopB);
  builder.setInsertionPointAfter(khLoopB);
  builder.setInsertionPointAfter(icLoopB);

  auto gemmATensor = builder.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get({oc, flatK}, weightElementType), gemmA);
  auto gemmBTensor = builder.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get({flatK, flatN}, inputElementType), gemmB);
  Value existingTensor;
  if (candidate.hasExistingInput) {
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

static LogicalResult convertCONVToGEMMOnly(AffineCandidateInfo &info) {
  auto inputShape = info.inputShape;
  auto weightShape = info.weightShape;
  auto outputShape = info.outputShape;
  if (inputShape.size() != 4 || weightShape.size() != 4 ||
      outputShape.size() != 4) {
    info.skipReason =
        "direct CONV lowering requires 4D NCHW/FCHW tensors";
    return failure();
  }
  if (inputShape[0] != 1 || outputShape[0] != 1) {
    info.skipReason =
        "direct CONV lowering only supports batch size 1";
    return failure();
  }
  info.sourceInputShape = inputShape;
  info.sourceWeightShape = weightShape;
  info.sourceOutputShape = outputShape;
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
  return success();
}

static FailureOr<AffineCandidateInfo> matchSingleAffineCandidate(AffineForOp rootLoop) {
  AffineCandidateInfo info;
  if (succeeded(matchCanonicalGEMM(rootLoop, info)))
    return info;
  if (succeeded(matchStripMinedGEMM(rootLoop, info)))
    return info;
  if (succeeded(matchCanonicalCONV(rootLoop, info))) {
    if (succeeded(convertCONVToGEMMOnly(info)))
      return info;
    if (info.skipReason.empty())
      info.skipReason = "direct CONV candidate failed checked lowering contract";
    return info;
  }
  if (succeeded(matchStripMinedCONV(rootLoop, info))) {
    if (succeeded(convertCONVToGEMMOnly(info)))
      return info;
    if (info.skipReason.empty())
      info.skipReason = "direct CONV candidate failed checked lowering contract";
    return info;
  }
  if (succeeded(matchCanonicalGEMMV(rootLoop, info))) {
    convertGEMMVToGEMMOnly(info);
    return info;
  }
  if (succeeded(matchStripMinedGEMMV(rootLoop, info))) {
    convertGEMMVToGEMMOnly(info);
    return info;
  }
  return failure();
}

static SmallVector<AffineForOp, 8> collectCandidateRoots(func::FuncOp func) {
  SmallVector<AffineForOp, 8> roots;
  func.walk([&](AffineForOp loop) {
    if (loop->getParentOfType<AffineForOp>())
      return;
    roots.push_back(loop);
  });
  return roots;
}

struct LowerAffineToAccel : public LowerAffineToAccelBase<LowerAffineToAccel> {
  LowerAffineToAccel() {
    allowLargeNormalization = true;
    maxNormalizedOperandElements = 4096;
  }
  LowerAffineToAccel(bool allowLargeNormalizationValue,
                     unsigned maxNormalizedOperandElementsValue) {
    allowLargeNormalization = allowLargeNormalizationValue;
    maxNormalizedOperandElements = maxNormalizedOperandElementsValue;
  }
  void runOnOperation() override {
    auto func = getOperation();
    func->removeAttr(kSkipReasonAttr);
    auto roots = collectCandidateRoots(func);
    int64_t candidateIndex = 0;
    std::string firstSkipReason;
    auto recordSkipReason = [&](StringRef reason) {
      if (firstSkipReason.empty() && !reason.empty())
        firstSkipReason = reason.str();
    };
    for (AffineForOp rootLoop : roots) {
      if (!rootLoop || rootLoop->getBlock() == nullptr)
        continue;

      auto candidate = matchSingleAffineCandidate(rootLoop);
      if (failed(candidate))
        continue;
      if (!candidate->skipReason.empty()) {
        recordSkipReason(candidate->skipReason);
        continue;
      }

      OpBuilder builder(rootLoop);
      builder.setInsertionPointAfter(rootLoop);

      Value shapedSemanticInput;
      Value shapedSemanticWeight;
      if (candidate->sourceFamily != StringRef("CONV") &&
          succeeded(candidate->semanticInput) &&
          candidate->semanticInput->needsMaterialization) {
        auto materialized = materializeSemanticOperand2D(
            builder, rootLoop.getLoc(), *candidate->semanticInput,
            candidate->inputShape, candidate->rowIV, candidate->redIV,
            allowLargeNormalization, maxNormalizedOperandElements);
        if (failed(materialized)) {
          recordSkipReason(
              "semantic GEMM normalization for lhs operand exceeds "
              "max-normalized-operand-elements");
          continue;
        }
        shapedSemanticInput = *materialized;
      }
      if (candidate->sourceFamily != StringRef("CONV") &&
          succeeded(candidate->semanticWeight) &&
          candidate->semanticWeight->needsMaterialization) {
        auto materialized = materializeSemanticOperand2D(
            builder, rootLoop.getLoc(), *candidate->semanticWeight,
            candidate->weightShape, candidate->redIV, candidate->colIV,
            allowLargeNormalization, maxNormalizedOperandElements);
        if (failed(materialized)) {
          recordSkipReason(
              "semantic GEMM normalization for rhs operand exceeds "
              "max-normalized-operand-elements");
          continue;
        }
        shapedSemanticWeight = *materialized;
      }

      if (candidate->family == "GEMM" &&
          candidate->sourceFamily == StringRef("CONV")) {
        if (candidate->sourceInputShape.size() != 4 ||
            candidate->sourceWeightShape.size() != 4 ||
            candidate->sourceOutputShape.size() != 4) {
          recordSkipReason(
              "direct CONV lowering requires 4D NCHW/FCHW tensors");
          continue;
        }
        auto packedElementCount = getConvPackedElementCount(*candidate);
        if (failed(packedElementCount)) {
          recordSkipReason(
              "direct CONV candidate failed packed-shape derivation");
          continue;
        }
        if (!allowLargeNormalization && maxNormalizedOperandElements != 0 &&
            static_cast<unsigned>(*packedElementCount) >
                maxNormalizedOperandElements) {
          recordSkipReason(
              "direct CONV packing exceeds max-normalized-operand-elements");
          continue;
        }
        auto gemm = lowerConvAsGemmOnly(builder, rootLoop.getLoc(), *candidate);
        annotateCandidateAttrs(gemm.getOperation(), func, *candidate);
        gemm->setAttr(
            kCandidateIndexAttr,
            IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                             candidateIndex++));
        rootLoop.erase();
        continue;
      }

      Value shapedInput =
          shapedSemanticInput
              ? shapedSemanticInput
              : materializeStaticMemrefView(builder, rootLoop.getLoc(),
                                            candidate->inputMemref,
                                            candidate->inputShape);
      Value shapedWeightSource =
          shapedSemanticWeight
              ? shapedSemanticWeight
              : materializeStaticMemrefView(builder, rootLoop.getLoc(),
                                            candidate->weightMemref, {});
      if (!shapedSemanticWeight && candidate->weightTransposed &&
          !allowLargeNormalization && maxNormalizedOperandElements != 0 &&
          static_cast<unsigned>(getNumElements(candidate->weightShape)) >
              maxNormalizedOperandElements) {
        recordSkipReason(
            "transposed GEMM weight materialization exceeds max-normalized-operand-elements");
        continue;
      }
      Value shapedWeight = shapedSemanticWeight
                               ? shapedSemanticWeight
                               : (candidate->weightTransposed
                                      ? materializeTransposedWeightMemref(
                                            builder, rootLoop.getLoc(),
                                            shapedWeightSource,
                                            candidate->weightShape)
                                      : materializeStaticMemrefView(
                                            builder, rootLoop.getLoc(),
                                            candidate->weightMemref,
                                            candidate->weightShape));
      Value shapedOutput =
          materializeStaticMemrefView(builder, rootLoop.getLoc(),
                                      candidate->outputMemref,
                                      candidate->outputShape);
      if (!hasExactStaticMemrefShape(shapedInput, candidate->inputShape) ||
          !hasExactStaticMemrefShape(shapedWeight, candidate->weightShape) ||
          !hasExactStaticMemrefShape(shapedOutput, candidate->outputShape)) {
        recordSkipReason(
            "matched affine candidate requires static ranked tensor views");
        continue;
      }
      auto inputType = getExactTensorTypeForMemref(shapedInput);
      auto weightType = getExactTensorTypeForMemref(shapedWeight);
      auto outputType = getExactTensorTypeForMemref(shapedOutput);
      if (!inputType || !weightType || !outputType) {
        recordSkipReason(
            "matched affine candidate requires static ranked tensor views");
        continue;
      }
      auto input =
          builder.create<bufferization::ToTensorOp>(rootLoop.getLoc(), inputType,
                                                    shapedInput);
      auto weight = builder.create<bufferization::ToTensorOp>(
          rootLoop.getLoc(), weightType, shapedWeight);
      Value existing;
      if (candidate->hasExistingInput)
        existing = builder.create<bufferization::ToTensorOp>(
            rootLoop.getLoc(), outputType, shapedOutput);

      Value lowered = builder
                          .create<accel::GEMMOp>(rootLoop.getLoc(), outputType,
                                                 input, weight, Value(), existing)
                          .getResult();
      auto *loweredOp = lowered.getDefiningOp();
      annotateCandidateAttrs(loweredOp, func, *candidate);
      loweredOp->setAttr(
          kCandidateIndexAttr,
          IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                           candidateIndex++));
      auto resultMemref = builder.create<bufferization::ToMemrefOp>(
          rootLoop.getLoc(), shapedOutput.getType(), lowered);
      builder.create<memref::CopyOp>(rootLoop.getLoc(), resultMemref,
                                     shapedOutput);
      rootLoop.erase();
    }
    if (!firstSkipReason.empty())
      func->setAttr(kSkipReasonAttr,
                    StringAttr::get(func.getContext(), firstSkipReason));
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createLowerAffineToAccelPass(bool allowLargeNormalization,
                                       unsigned maxNormalizedOperandElements) {
  return std::make_unique<LowerAffineToAccel>(allowLargeNormalization,
                                              maxNormalizedOperandElements);
}
