//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "scalehls/Dialect/Accel/Accel.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

namespace {
constexpr StringLiteral kParentFuncAttr = "scalehls.gemm_parent_func";
constexpr StringLiteral kCandidateIndexAttr = "scalehls.gemm_candidate_index";
constexpr StringLiteral kSkipReasonAttr = "scalehls.gemm_skip_reason";
constexpr StringLiteral kDetectedAttr = "scalehls.gemm";
constexpr StringLiteral kContractAttr = "scalehls.gemm_contract";
constexpr StringLiteral kTensorContract = "tensor_family";
constexpr StringLiteral kFamilyAttr = "scalehls.accelerator_family";
constexpr StringLiteral kShapeAttr = "scalehls.gemm_shape";
constexpr StringLiteral kAShapeAttr = "scalehls.gemm_a_shape";
constexpr StringLiteral kBShapeAttr = "scalehls.gemm_b_shape";
constexpr StringLiteral kCShapeAttr = "scalehls.gemm_c_shape";
constexpr StringLiteral kBiasShapeAttr = "scalehls.gemm_bias_shape";
constexpr StringLiteral kPrecisionAttr = "scalehls.gemm_precision";
constexpr StringLiteral kElementBitsAttr = "scalehls.gemm_element_bits";
constexpr StringLiteral kElementBytesAttr = "scalehls.gemm_element_bytes";
constexpr StringLiteral kHasBiasAttr = "scalehls.accelerator_has_bias";
} // namespace

namespace {
static bool isSimpleAddGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getNumInputs() != 2 || generic.getNumOutputs() != 1)
    return false;
  if (generic.getBody()->getOperations().size() != 2)
    return false;
  auto add = dyn_cast<arith::AddFOp>(generic.getBody()->front());
  auto addi = dyn_cast<arith::AddIOp>(generic.getBody()->front());
  auto yield = dyn_cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  return yield && yield.getValues().size() == 1 &&
         ((add && yield.getValues().front() == add.getResult()) ||
          (addi && yield.getValues().front() == addi.getResult()));
}

static bool hasStaticShape(Value value) {
  auto type = dyn_cast<ShapedType>(value.getType());
  return type && type.hasStaticShape();
}

static bool sameShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  return lhsType && rhsType && lhsType.hasStaticShape() &&
         rhsType.hasStaticShape() && lhsType.getShape() == rhsType.getShape();
}

struct CandidateInfo {
  Operation *baseOp = nullptr;
  Operation *absorbedAdd = nullptr;
  Value input;
  Value weight;
  Value bias;
  Value existingInput;
  Value finalResult;
  StringRef family;
};

static LogicalResult classifyAdditiveOperand(Value additive, Value result,
                                             Value &bias,
                                             Value &existingInput) {
  if (sameShape(additive, result)) {
    if (existingInput)
      return failure();
    existingInput = additive;
    return success();
  }
  if (bias)
    return failure();
  bias = additive;
  return success();
}

static linalg::GenericOp findSimpleAddUser(Value result) {
  linalg::GenericOp addUser;
  for (Operation *user : result.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!isSimpleAddGeneric(generic))
      continue;
    if (addUser)
      return {};
    addUser = generic;
  }
  return addUser;
}

static ArrayAttr buildI64ArrayAttr(MLIRContext *context,
                                   ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(IntegerAttr::get(IntegerType::get(context, 64), value));
  return ArrayAttr::get(context, attrs);
}

static void annotateOperandShape(Operation *op, StringRef attrName, Value value) {
  auto type = dyn_cast<ShapedType>(value.getType());
  if (!type || !type.hasStaticShape())
    return;
  op->setAttr(attrName, buildI64ArrayAttr(op->getContext(), type.getShape()));
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

static void annotateCandidateAttrs(Operation *op, const CandidateInfo &candidate) {
  auto outputType = dyn_cast<ShapedType>(candidate.finalResult.getType());
  if (!outputType)
    return;
  op->setAttr(kDetectedAttr, UnitAttr::get(op->getContext()));
  op->setAttr(kContractAttr,
              StringAttr::get(op->getContext(), kTensorContract));
  op->setAttr(kFamilyAttr,
              StringAttr::get(op->getContext(), candidate.family));
  op->setAttr(kHasBiasAttr,
              BoolAttr::get(op->getContext(), static_cast<bool>(candidate.bias)));
  annotateOperandShape(op, kShapeAttr, candidate.finalResult);
  annotateOperandShape(op, kAShapeAttr, candidate.input);
  annotateOperandShape(op, kBShapeAttr, candidate.weight);
  annotateOperandShape(op, kCShapeAttr, candidate.finalResult);
  if (candidate.bias)
    annotateOperandShape(op, kBiasShapeAttr, candidate.bias);
  annotatePrecision(op, outputType.getElementType());
}

static FailureOr<CandidateInfo> matchMatmulCandidate(Operation *baseOp,
                                                     Value input,
                                                     Value weight,
                                                     Value result) {
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType || !resultType.hasStaticShape())
    return failure();

  CandidateInfo info;
  info.baseOp = baseOp;
  info.input = input;
  info.weight = weight;
  info.finalResult = result;
  info.family = StringRef("GEMM");

  if (auto addUser = findSimpleAddUser(result)) {
    Value additive =
        addUser.getInputs()[0] == result ? addUser.getInputs()[1]
                                         : addUser.getInputs()[0];
    if (failed(classifyAdditiveOperand(additive, addUser.getResult(0),
                                       info.bias, info.existingInput)))
      return failure();
    info.absorbedAdd = addUser;
    info.finalResult = addUser.getResult(0);
  }

  if (!hasStaticShape(info.input) || !hasStaticShape(info.weight) ||
      !hasStaticShape(info.finalResult))
    return failure();
  return info;
}

static FailureOr<CandidateInfo> matchCandidate(Operation *op,
                                               std::string &skipReason) {
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
    return matchMatmulCandidate(matmul, matmul.getInputs()[0],
                                matmul.getInputs()[1], matmul.getResult(0));
  if (auto batchMatmul = dyn_cast<linalg::BatchMatmulOp>(op))
    return matchMatmulCandidate(batchMatmul, batchMatmul.getInputs()[0],
                                batchMatmul.getInputs()[1],
                                batchMatmul.getResult(0));
  if (isa<linalg::Conv2DNchwFchwOp>(op)) {
    skipReason =
        "unsupported linalg accel candidate: conv_2d_nchw_fchw remains tensor-side only";
    return failure();
  }
  if (isa<linalg::LinalgOp>(op)) {
    skipReason =
        "unsupported linalg accel candidate: only matmul and batch_matmul contracts are lowered";
    return failure();
  }
  return failure();
}

static SmallVector<Operation *> collectCandidateBases(func::FuncOp func) {
  SmallVector<Operation *> bases;
  func.walk([&](Operation *op) {
    if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::Conv2DNchwFchwOp>(op))
      bases.push_back(op);
  });
  return bases;
}

struct LowerLinalgToAccel
    : public LowerLinalgToAccelBase<LowerLinalgToAccel> {
  void runOnOperation() override {
    auto func = getOperation();
    auto bases = collectCandidateBases(func);
    if (bases.empty()) {
      func->setAttr(kSkipReasonAttr,
                    StringAttr::get(func.getContext(),
                                    "no_supported_accel_candidate"));
      return;
    }

    int64_t candidateIndex = 0;
    bool mappedAny = false;
    std::string firstSkipReason;
    auto recordSkipReason = [&](StringRef reason) {
      if (firstSkipReason.empty() && !reason.empty())
        firstSkipReason = reason.str();
    };
    for (Operation *baseOp : bases) {
      if (!baseOp || baseOp->getBlock() == nullptr)
        continue;

      std::string candidateSkipReason;
      auto candidate = matchCandidate(baseOp, candidateSkipReason);
      if (failed(candidate)) {
        recordSkipReason(candidateSkipReason);
        continue;
      }

      OpBuilder builder(candidate->absorbedAdd ? candidate->absorbedAdd : baseOp);
      Value lowered =
          builder
              .create<accel::GEMMOp>(baseOp->getLoc(),
                                     candidate->finalResult.getType(),
                                     candidate->input, candidate->weight,
                                     candidate->bias, candidate->existingInput)
              .getResult();

      auto *loweredOp = lowered.getDefiningOp();
      annotateCandidateAttrs(loweredOp, *candidate);
      loweredOp->setAttr(kParentFuncAttr, StringAttr::get(func.getContext(),
                                                          func.getName()));
      loweredOp->setAttr(
          kCandidateIndexAttr,
          IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                           candidateIndex++));
      mappedAny = true;

      candidate->finalResult.replaceAllUsesWith(lowered);
      if (candidate->absorbedAdd)
        candidate->absorbedAdd->erase();
      baseOp->erase();
    }

    if (!firstSkipReason.empty())
      func->setAttr(
          kSkipReasonAttr,
          StringAttr::get(func.getContext(), firstSkipReason));
    else if (mappedAny)
      func->removeAttr(kSkipReasonAttr);
    else
      func->setAttr(kSkipReasonAttr,
                    StringAttr::get(func.getContext(),
                                    "no_supported_accel_candidate"));
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createLowerLinalgToAccelPass() {
  return std::make_unique<LowerLinalgToAccel>();
}
