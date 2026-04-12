//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/FormatVariadic.h"
#include "scalehls/Dialect/Accel/Accel.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

namespace {
constexpr StringLiteral kDetectedAttr = "scalehls.gemm";
constexpr StringLiteral kContractAttr = "scalehls.gemm_contract";
constexpr StringLiteral kTensorContract = "tensor_family";
constexpr StringLiteral kOutlinedAttr = "scalehls.gemm_outlined";
constexpr StringLiteral kFamilyAttr = "scalehls.accelerator_family";
constexpr StringLiteral kShapeAttr = "scalehls.gemm_shape";
constexpr StringLiteral kAShapeAttr = "scalehls.gemm_a_shape";
constexpr StringLiteral kBShapeAttr = "scalehls.gemm_b_shape";
constexpr StringLiteral kCShapeAttr = "scalehls.gemm_c_shape";
constexpr StringLiteral kBiasShapeAttr = "scalehls.gemm_bias_shape";
constexpr StringLiteral kAArgAttr = "scalehls.gemm_a_arg";
constexpr StringLiteral kBArgAttr = "scalehls.gemm_b_arg";
constexpr StringLiteral kCArgAttr = "scalehls.gemm_c_arg";
constexpr StringLiteral kBiasArgAttr = "scalehls.gemm_bias_arg";
constexpr StringLiteral kPrecisionAttr = "scalehls.gemm_precision";
constexpr StringLiteral kElementBitsAttr = "scalehls.gemm_element_bits";
constexpr StringLiteral kElementBytesAttr = "scalehls.gemm_element_bytes";
constexpr StringLiteral kHasBiasAttr = "scalehls.accelerator_has_bias";
constexpr StringLiteral kParentFuncAttr = "scalehls.gemm_parent_func";
constexpr StringLiteral kCandidateIndexAttr = "scalehls.gemm_candidate_index";
constexpr StringLiteral kPointerTransportAttr =
    "scalehls.accelerator_pointer_transport";
constexpr StringLiteral kUntiledPointerTransport = "untiled";
constexpr StringLiteral kTiledPointerTransport = "tiled_persistent";
constexpr StringLiteral kABIModeAttr = "scalehls.accelerator_abi_mode";
} // namespace

namespace {
static std::string stringifyShape(ArrayRef<int64_t> shape) {
  std::string out;
  llvm::raw_string_ostream os(out);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      os << "x";
    os << shape[i];
  }
  return os.str();
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

static SmallVector<int64_t> getStaticShape(Value value) {
  SmallVector<int64_t> shape;
  if (!value)
    return shape;
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasStaticShape())
    return shape;
  shape.append(shapedType.getShape().begin(), shapedType.getShape().end());
  return shape;
}

static Optional<uint64_t> getStaticElementCount(Value value) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasStaticShape())
    return llvm::None;
  uint64_t count = 1;
  for (int64_t dim : shapedType.getShape()) {
    if (dim < 0)
      return llvm::None;
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

static StringRef getFamilyName(Operation *op) {
  if (isa<accel::GEMMVOp>(op))
    return StringRef("GEMMV");
  if (isa<accel::GEMMOp>(op))
    return StringRef("GEMM");
  return StringRef("CONV");
}

static Value getBiasOperand(Operation *op) {
  if (auto gemmv = dyn_cast<accel::GEMMVOp>(op))
    return gemmv.getBias();
  if (auto gemm = dyn_cast<accel::GEMMOp>(op))
    return gemm.getBias();
  if (auto conv = dyn_cast<accel::CONVOp>(op))
    return conv.getBias();
  return Value();
}

static Value getExistingInputOperand(Operation *op) {
  if (auto gemmv = dyn_cast<accel::GEMMVOp>(op))
    return gemmv.getExistingInput();
  if (auto gemm = dyn_cast<accel::GEMMOp>(op))
    return gemm.getExistingInput();
  if (auto conv = dyn_cast<accel::CONVOp>(op))
    return conv.getExistingInput();
  return Value();
}

static std::string buildAcceleratorName(func::FuncOp func, Operation *op) {
  StringRef family = getFamilyName(op);
  auto result = op->getResult(0);
  auto inputShape = getStaticShape(op->getOperand(0));
  auto weightShape = getStaticShape(op->getOperand(1));
  auto outputShape = getStaticShape(result);
  auto biasShape = getStaticShape(getBiasOperand(op));
  auto existingShape = getStaticShape(getExistingInputOperand(op));

  std::string name = func.getName().str() + "_accel_";
  name += family.lower();
  name += "_";
  name += getPrecisionString(cast<ShapedType>(result.getType()).getElementType()).str();
  if (!inputShape.empty())
    name += "_in" + stringifyShape(inputShape);
  if (!weightShape.empty())
    name += "_w" + stringifyShape(weightShape);
  if (!biasShape.empty())
    name += "_b" + stringifyShape(biasShape);
  if (!existingShape.empty())
    name += "_ei" + stringifyShape(existingShape);
  if (!outputShape.empty())
    name += "_out" + stringifyShape(outputShape);
  return name;
}

static std::string uniquifySymbolName(SymbolTable &symbolTable,
                                      StringRef baseName) {
  std::string uniqueName = baseName.str();
  unsigned suffix = 1;
  while (symbolTable.lookup(uniqueName))
    uniqueName = (baseName + "_" + Twine(suffix++)).str();
  return uniqueName;
}

static void copyMappingAttrs(Operation *from, Operation *to, Operation *accelOp) {
  for (StringRef attrName :
       {kDetectedAttr, kContractAttr, kFamilyAttr, kShapeAttr, kAShapeAttr,
        kBShapeAttr, kCShapeAttr, kBiasShapeAttr, kAArgAttr, kBArgAttr,
        kCArgAttr, kBiasArgAttr, kPrecisionAttr, kElementBitsAttr,
        kElementBytesAttr, kHasBiasAttr, kPointerTransportAttr,
        kABIModeAttr}) {
    if (auto attr = from->getAttr(attrName))
      to->setAttr(attrName, attr);
  }
  if (!to->hasAttr(kDetectedAttr))
    to->setAttr(kDetectedAttr, UnitAttr::get(to->getContext()));
  if (!to->hasAttr(kContractAttr))
    to->setAttr(kContractAttr, StringAttr::get(to->getContext(), kTensorContract));
  if (!to->hasAttr(kFamilyAttr))
    to->setAttr(kFamilyAttr,
                StringAttr::get(to->getContext(), getFamilyName(accelOp)));
  if (!to->hasAttr(kHasBiasAttr))
    to->setAttr(
        kHasBiasAttr,
        BoolAttr::get(to->getContext(), static_cast<bool>(getBiasOperand(accelOp))));
  if (auto parent = accelOp->getAttr(kParentFuncAttr))
    to->setAttr(kParentFuncAttr, parent);
  if (auto candidateIndex = accelOp->getAttr(kCandidateIndexAttr))
    to->setAttr(kCandidateIndexAttr, candidateIndex);
}

static void enumerateFlatIndexTuples(OpBuilder &builder, Location loc,
                                     ArrayRef<int64_t> shape,
                                     SmallVectorImpl<SmallVector<Value>> &all) {
  all.clear();
  if (shape.empty()) {
    all.push_back({});
    return;
  }

  SmallVector<int64_t> current(shape.size(), 0);
  while (true) {
    SmallVector<Value> tuple;
    tuple.reserve(shape.size());
    for (int64_t index : current)
      tuple.push_back(builder.create<arith::ConstantIndexOp>(loc, index));
    all.push_back(std::move(tuple));

    int64_t dim = static_cast<int64_t>(shape.size()) - 1;
    while (dim >= 0) {
      ++current[dim];
      if (current[dim] < shape[dim])
        break;
      current[dim] = 0;
      --dim;
    }
    if (dim < 0)
      break;
  }
}

static LogicalResult flattenTensorOperand(OpBuilder &builder, Location loc,
                                          Value tensor,
                                          SmallVectorImpl<Value> &flatValues) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return failure();

  SmallVector<SmallVector<Value>> allIndices;
  enumerateFlatIndexTuples(builder, loc, tensorType.getShape(), allIndices);
  for (ArrayRef<Value> indices : allIndices)
    flatValues.push_back(builder.create<tensor::ExtractOp>(loc, tensor, indices));
  return success();
}

static LogicalResult rebuildTensorResultFromFlatValues(
    OpBuilder &builder, Location loc, RankedTensorType tensorType,
    ValueRange flatValues, Value &rebuiltTensor) {
  if (!tensorType.hasStaticShape())
    return failure();
  auto elementCount = tensorType.getNumElements();
  if (flatValues.size() != static_cast<size_t>(elementCount))
    return failure();
  rebuiltTensor = builder.create<tensor::FromElementsOp>(loc, tensorType,
                                                         flatValues);
  return success();
}

struct LowerAccelToCalls : public LowerAccelToCallsBase<LowerAccelToCalls> {
  LowerAccelToCalls() = default;
  LowerAccelToCalls(std::string abiModeValue, unsigned maxElementsValue) {
    abiMode = std::move(abiModeValue);
    maxElements = maxElementsValue;
  }

  void runOnOperation() override {
    auto module = getOperation();
    SymbolTable symbolTable(module);
    OpBuilder builder(module.getContext());

    for (auto func : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      SmallVector<Operation *> accelOps;
      func.walk([&](Operation *op) {
        if (isa<accel::GEMMVOp, accel::GEMMOp, accel::CONVOp>(op))
          accelOps.push_back(op);
      });
      if (accelOps.empty())
        continue;

      SmallVector<Attribute> outlinedSymbols;
      for (Operation *accelOp : accelOps) {
        StringRef selectedABIMode = abiMode;
        accelOp->setAttr(kABIModeAttr,
                         StringAttr::get(module.getContext(), selectedABIMode));
        if (selectedABIMode != "pointer" && selectedABIMode != "full-data") {
          accelOp->emitError()
              << "ABI mode '" << selectedABIMode
              << "' is not implemented yet; supported modes are pointer and full-data";
          signalPassFailure();
          return;
        }

        if (selectedABIMode == "full-data") {
          SymbolRefAttr outlinedSymbol;
          if (failed(lowerToFullDataABI(builder, symbolTable, func, accelOp,
                                        outlinedSymbol))) {
            signalPassFailure();
            return;
          }
          outlinedSymbols.push_back(outlinedSymbol);
          continue;
        }

        auto outputElements = getStaticElementCount(accelOp->getResult(0));
        const bool useTiledPointer =
            maxElements != 0 && outputElements && *outputElements > maxElements;
        accelOp->setAttr(
            kPointerTransportAttr,
            StringAttr::get(module.getContext(),
                            useTiledPointer ? kTiledPointerTransport
                                            : kUntiledPointerTransport));
        if (useTiledPointer) {
          accelOp->emitError()
              << "persistent tiled pointer ABI not implemented yet for logical output size "
              << *outputElements << " > max-elements="
              << maxElements.getValue();
          signalPassFailure();
          return;
        }

        SmallVector<Type> operandTypes(accelOp->getOperandTypes().begin(),
                                       accelOp->getOperandTypes().end());
        SmallVector<Type> resultTypes(accelOp->getResultTypes().begin(),
                                      accelOp->getResultTypes().end());
        auto baseSymbolName = buildAcceleratorName(func, accelOp);
        auto symbolName = uniquifySymbolName(symbolTable, baseSymbolName);

        builder.setInsertionPoint(func);
        auto symbol = builder.create<func::FuncOp>(
            func.getLoc(), symbolName,
            builder.getFunctionType(operandTypes, resultTypes));
        symbol->setAttr("sym_visibility",
                        StringAttr::get(module.getContext(), "private"));
        copyMappingAttrs(accelOp, symbol, accelOp);
        symbolTable.insert(symbol);

        builder.setInsertionPoint(accelOp);
        auto call = builder.create<func::CallOp>(accelOp->getLoc(), symbol,
                                                 accelOp->getOperands());
        accelOp->getResult(0).replaceAllUsesWith(call.getResult(0));
        accelOp->erase();
        outlinedSymbols.push_back(SymbolRefAttr::get(symbol));
      }

      if (!outlinedSymbols.empty())
        func->setAttr(kOutlinedAttr, outlinedSymbols.front());
    }
  }

private:
  LogicalResult lowerToFullDataABI(OpBuilder &builder, SymbolTable &symbolTable,
                                   func::FuncOp func, Operation *accelOp,
                                   SymbolRefAttr &outlinedSymbol) {
    auto resultType =
        dyn_cast<RankedTensorType>(accelOp->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape()) {
      accelOp->emitError()
          << "full-data ABI requires a static ranked tensor result";
      return failure();
    }

    SmallVector<Value> flatOperands;
    builder.setInsertionPoint(accelOp);
    for (Value operand : accelOp->getOperands()) {
      if (failed(flattenTensorOperand(builder, accelOp->getLoc(), operand,
                                      flatOperands))) {
        accelOp->emitError()
            << "full-data ABI requires static ranked tensor operands";
        return failure();
      }
    }

    SmallVector<Type> flatOperandTypes;
    flatOperandTypes.reserve(flatOperands.size());
    for (Value value : flatOperands)
      flatOperandTypes.push_back(value.getType());

    SmallVector<Type> flatResultTypes(resultType.getNumElements(),
                                      resultType.getElementType());
    auto baseSymbolName = buildAcceleratorName(func, accelOp);
    auto symbolName = uniquifySymbolName(symbolTable, baseSymbolName);

    builder.setInsertionPoint(func);
    auto symbol = builder.create<func::FuncOp>(
        func.getLoc(), symbolName,
        builder.getFunctionType(flatOperandTypes, flatResultTypes));
    symbol->setAttr("sym_visibility",
                    StringAttr::get(func.getContext(), "private"));
    copyMappingAttrs(accelOp, symbol, accelOp);
    symbolTable.insert(symbol);
    outlinedSymbol = SymbolRefAttr::get(symbol);

    builder.setInsertionPoint(accelOp);
    auto call = builder.create<func::CallOp>(accelOp->getLoc(), flatResultTypes,
                                             symbol.getName(), flatOperands);

    Value rebuiltTensor;
    if (failed(rebuildTensorResultFromFlatValues(builder, accelOp->getLoc(),
                                                 resultType, call.getResults(),
                                                 rebuiltTensor))) {
      accelOp->emitError()
          << "failed to rebuild full-data ABI tensor result from scalar outputs";
      return failure();
    }
    accelOp->getResult(0).replaceAllUsesWith(rebuiltTensor);
    accelOp->erase();
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createLowerAccelToCallsPass(std::string abiMode,
                                                            unsigned maxElements) {
  return std::make_unique<LowerAccelToCalls>(std::move(abiMode), maxElements);
}
