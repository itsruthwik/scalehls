//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
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
constexpr StringLiteral kOutlinedHelpersAttr = "scalehls.gemm_outlined_helpers";
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
constexpr StringLiteral kExistingInputArgAttr = "scalehls.gemm_existing_input_arg";
constexpr StringLiteral kPrecisionAttr = "scalehls.gemm_precision";
constexpr StringLiteral kElementBitsAttr = "scalehls.gemm_element_bits";
constexpr StringLiteral kElementBytesAttr = "scalehls.gemm_element_bytes";
constexpr StringLiteral kHasBiasAttr = "scalehls.accelerator_has_bias";
constexpr StringLiteral kABIModeAttr = "scalehls.accelerator_abi_mode";
constexpr StringLiteral kParentFuncAttr = "scalehls.gemm_parent_func";
constexpr StringLiteral kCandidateIndexAttr = "scalehls.gemm_candidate_index";
constexpr StringLiteral kPointerTransportAttr =
    "scalehls.accelerator_pointer_transport";
constexpr StringLiteral kPointerABIMode = "pointer";
constexpr StringLiteral kUntiledPointerTransport = "untiled";
constexpr StringLiteral kIPTileSizeAttr = "scalehls.accelerator_ip_tile_size";
constexpr StringLiteral kIPTilingModeAttr = "scalehls.accelerator_ip_tiling_mode";
constexpr StringLiteral kIPTileRowAttr = "scalehls.accelerator_ip_tile_row";
constexpr StringLiteral kIPTileColAttr = "scalehls.accelerator_ip_tile_col";
constexpr StringLiteral kIPTileKAttr = "scalehls.accelerator_ip_tile_k";
constexpr StringLiteral kSerialIPTilingMode = "serial";
} // namespace

namespace {
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

static SmallVector<int64_t> getHelperShape(Operation *op, Value value) {
  auto shape = getStaticShape(value);
  if (isa<accel::GEMMOp>(op) && shape.size() == 3)
    return SmallVector<int64_t>(llvm::drop_begin(shape));
  return shape;
}

static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getCanonicalGemmDims(Operation *op) {
  auto inputShape = getHelperShape(op, op->getOperand(0));
  auto weightShape = getHelperShape(op, op->getOperand(1));
  auto outputShape = getHelperShape(op, op->getResult(0));
  if (inputShape.size() != 2 || weightShape.size() != 2 || outputShape.size() != 2)
    return failure();
  int64_t m = inputShape[0];
  int64_t k = inputShape[1];
  int64_t n = weightShape[1];
  if (weightShape[0] != k || outputShape[0] != m || outputShape[1] != n)
    return failure();
  return std::make_tuple(m, k, n);
}

static Value getBiasOperand(Operation *op) {
  if (auto gemm = dyn_cast<accel::GEMMOp>(op))
    return gemm.getBias();
  return Value();
}

static Value getExistingInputOperand(Operation *op) {
  if (auto gemm = dyn_cast<accel::GEMMOp>(op))
    return gemm.getExistingInput();
  return Value();
}

static StringRef getContractAttrValue(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kContractAttr))
    return attr.getValue();
  return kTensorContract;
}

static StringRef getFamilyAttrValue(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kFamilyAttr))
    return attr.getValue();
  return StringRef("GEMM");
}

static StringRef getAbiModeAttrValue(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kABIModeAttr))
    return attr.getValue();
  return kPointerABIMode;
}

static StringRef getPointerTransportAttrValue(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kPointerTransportAttr))
    return attr.getValue();
  return kUntiledPointerTransport;
}

static bool hasCompatibleHelperContract(Operation *accelOp, func::FuncOp helper,
                                        FunctionType expectedType) {
  auto getHelperStringAttr = [&](StringRef attrName) -> StringRef {
    if (auto attr = helper->getAttrOfType<StringAttr>(attrName))
      return attr.getValue();
    return StringRef();
  };
  auto getHelperBoolAttr = [&](StringRef attrName, bool defaultValue) -> bool {
    if (auto attr = helper->getAttrOfType<BoolAttr>(attrName))
      return attr.getValue();
    return defaultValue;
  };
  if (helper.getFunctionType() != expectedType)
    return false;
  if (!helper->hasAttr(kDetectedAttr))
    return false;
  if (getHelperStringAttr(kContractAttr) != getContractAttrValue(accelOp))
    return false;
  if (getHelperStringAttr(kFamilyAttr) != getFamilyAttrValue(accelOp))
    return false;
  if (getHelperStringAttr(kABIModeAttr) != getAbiModeAttrValue(accelOp))
    return false;
  if (getHelperStringAttr(kPointerTransportAttr) !=
      getPointerTransportAttrValue(accelOp))
    return false;
  if (getHelperBoolAttr(kHasBiasAttr, false) !=
      static_cast<bool>(getBiasOperand(accelOp)))
    return false;
  for (StringRef attrName :
       {kIPTileSizeAttr, kIPTilingModeAttr, kIPTileRowAttr, kIPTileColAttr,
        kIPTileKAttr}) {
    if (helper->getAttr(attrName) != accelOp->getAttr(attrName))
      return false;
  }
  return true;
}

using SerialHelperMap = llvm::StringMap<SmallVector<func::FuncOp, 2>>;

static func::FuncOp lookupCompatibleSerialHelper(SerialHelperMap &helpers,
                                                 StringRef baseName,
                                                 Operation *accelOp,
                                                 FunctionType expectedType) {
  auto it = helpers.find(baseName);
  if (it == helpers.end())
    return func::FuncOp();
  for (func::FuncOp helper : it->second) {
    if (hasCompatibleHelperContract(accelOp, helper, expectedType))
      return helper;
  }
  return func::FuncOp();
}

static void registerSerialHelper(SerialHelperMap &helpers, StringRef baseName,
                                 func::FuncOp helper) {
  helpers[baseName].push_back(helper);
}

static std::string buildAcceleratorName(Operation *op) {
  std::string name = "gemm";
  auto dims = getCanonicalGemmDims(op);
  if (succeeded(dims)) {
    auto [m, k, n] = *dims;
    name += llvm::formatv("_{0}x{1}x{2}", m, k, n).str();
  }
  return name;
}

static std::string buildCandidateAcceleratorName(Operation *op) {
  int64_t candidateIndex = 0;
  if (auto attr = op->getAttrOfType<IntegerAttr>(kCandidateIndexAttr))
    candidateIndex = attr.getInt();
  std::string name = buildAcceleratorName(op);
  name += llvm::formatv("_call{0}", candidateIndex).str();
  return name;
}

static std::string buildTiledAcceleratorName(Operation *op) {
  std::string name = buildCandidateAcceleratorName(op);
  if (auto tileSize = op->getAttrOfType<StringAttr>(kIPTileSizeAttr))
    name += "_ip" + tileSize.getValue().str();
  auto tileRow = op->getAttrOfType<IntegerAttr>(kIPTileRowAttr);
  auto tileCol = op->getAttrOfType<IntegerAttr>(kIPTileColAttr);
  auto tileK = op->getAttrOfType<IntegerAttr>(kIPTileKAttr);
  if (tileRow && tileCol) {
    name += llvm::formatv("_tile{0}_{1}", tileRow.getInt(), tileCol.getInt()).str();
    if (tileK)
      name += llvm::formatv("_k{0}", tileK.getInt()).str();
  }
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
        kCArgAttr, kBiasArgAttr, kExistingInputArgAttr, kPrecisionAttr, kElementBitsAttr,
        kElementBytesAttr, kHasBiasAttr, kABIModeAttr, kPointerTransportAttr,
        kIPTileSizeAttr, kIPTilingModeAttr, kIPTileRowAttr,
        kIPTileColAttr, kIPTileKAttr}) {
    if (auto attr = from->getAttr(attrName))
      to->setAttr(attrName, attr);
  }
  if (!to->hasAttr(kDetectedAttr))
    to->setAttr(kDetectedAttr, UnitAttr::get(to->getContext()));
  if (!to->hasAttr(kContractAttr))
    to->setAttr(kContractAttr, StringAttr::get(to->getContext(), kTensorContract));
  if (!to->hasAttr(kFamilyAttr))
    to->setAttr(kFamilyAttr,
                StringAttr::get(to->getContext(), "GEMM"));
  if (!to->hasAttr(kHasBiasAttr))
    to->setAttr(
        kHasBiasAttr,
        BoolAttr::get(to->getContext(), static_cast<bool>(getBiasOperand(accelOp))));
  if (!to->hasAttr(kAArgAttr))
    to->setAttr(kAArgAttr,
                IntegerAttr::get(IntegerType::get(to->getContext(), 64), 0));
  if (!to->hasAttr(kBArgAttr))
    to->setAttr(kBArgAttr,
                IntegerAttr::get(IntegerType::get(to->getContext(), 64), 1));
  unsigned nextArg = 2;
  if (getBiasOperand(accelOp)) {
    if (!to->hasAttr(kBiasArgAttr))
      to->setAttr(kBiasArgAttr,
                  IntegerAttr::get(IntegerType::get(to->getContext(), 64), nextArg));
    ++nextArg;
  }
  if (getExistingInputOperand(accelOp)) {
    if (!to->hasAttr(kExistingInputArgAttr))
      to->setAttr(kExistingInputArgAttr,
                  IntegerAttr::get(IntegerType::get(to->getContext(), 64), nextArg));
    ++nextArg;
  }
  if (!to->hasAttr(kCArgAttr))
    to->setAttr(kCArgAttr,
                IntegerAttr::get(IntegerType::get(to->getContext(), 64), nextArg));
  if (!to->hasAttr(kABIModeAttr))
    to->setAttr(kABIModeAttr,
                StringAttr::get(to->getContext(), kPointerABIMode));
  if (auto parent = accelOp->getAttr(kParentFuncAttr))
    to->setAttr(kParentFuncAttr, parent);
  if (auto candidateIndex = accelOp->getAttr(kCandidateIndexAttr))
    to->setAttr(kCandidateIndexAttr, candidateIndex);
}

static RankedTensorType getHelperTensorType(Operation *op, Value value) {
  auto shapedType = dyn_cast<RankedTensorType>(value.getType());
  if (!shapedType || !shapedType.hasStaticShape())
    return RankedTensorType();
  auto shape = getHelperShape(op, value);
  return RankedTensorType::get(shape, shapedType.getElementType());
}

static Value collapseLeadingBatchSlice(OpBuilder &builder, Location loc, Value slice) {
  auto sliceType = dyn_cast<RankedTensorType>(slice.getType());
  if (!sliceType || sliceType.getRank() != 3 || sliceType.getShape().front() != 1)
    return slice;
  auto collapsedType = RankedTensorType::get(
      {sliceType.getShape()[1], sliceType.getShape()[2]}, sliceType.getElementType());
  SmallVector<ReassociationIndices, 2> reassociation = {{0, 1}, {2}};
  return builder.create<tensor::CollapseShapeOp>(loc, collapsedType, slice,
                                                 reassociation);
}

static Value expandLeadingBatchSlice(OpBuilder &builder, Location loc, Value slice,
                                     RankedTensorType targetType) {
  SmallVector<ReassociationIndices, 2> reassociation = {{0, 1}, {2}};
  return builder.create<tensor::ExpandShapeOp>(loc, targetType, slice, reassociation);
}

struct LowerAccelToCalls : public LowerAccelToCallsBase<LowerAccelToCalls> {
  LowerAccelToCalls() = default;

  LogicalResult lowerBatchedGemmTo2DHelper(OpBuilder &builder,
                                           SymbolTable &symbolTable,
                                           func::FuncOp func,
                                           accel::GEMMOp gemmOp,
                                           func::FuncOp &symbol,
                                           SerialHelperMap *serialTileHelpers = nullptr) {
    auto inputType = dyn_cast<RankedTensorType>(gemmOp.getInput().getType());
    auto weightType = dyn_cast<RankedTensorType>(gemmOp.getWeight().getType());
    auto outputType = dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
    if (!inputType || !weightType || !outputType || inputType.getRank() != 3 ||
        weightType.getRank() != 3 || outputType.getRank() != 3 ||
        !inputType.hasStaticShape() || !weightType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return failure();

    auto helperInputType = getHelperTensorType(gemmOp, gemmOp.getInput());
    auto helperWeightType = getHelperTensorType(gemmOp, gemmOp.getWeight());
    auto helperOutputType = getHelperTensorType(gemmOp, gemmOp.getResult());
    RankedTensorType helperBiasType;
    RankedTensorType helperExistingType;
    if (gemmOp.getBias())
      helperBiasType = getHelperTensorType(gemmOp, gemmOp.getBias());
    if (gemmOp.getExistingInput())
      helperExistingType = getHelperTensorType(gemmOp, gemmOp.getExistingInput());
    if (!helperInputType || !helperWeightType || !helperOutputType)
      return failure();

    SmallVector<Type> operandTypes = {helperInputType, helperWeightType};
    if (helperBiasType)
      operandTypes.push_back(helperBiasType);
    if (helperExistingType)
      operandTypes.push_back(helperExistingType);

    const bool isSerialTiled =
        serialTileHelpers &&
        gemmOp->getAttrOfType<StringAttr>(kIPTilingModeAttr) &&
        gemmOp->getAttrOfType<StringAttr>(kIPTilingModeAttr).getValue() ==
            kSerialIPTilingMode;

    auto helperType =
        builder.getFunctionType(operandTypes, TypeRange{helperOutputType});

    if (isSerialTiled) {
      std::string serialBaseName = buildAcceleratorName(gemmOp);
      if (auto tileSize = gemmOp->getAttrOfType<StringAttr>(kIPTileSizeAttr))
        serialBaseName += "_ip" + tileSize.getValue().str();
      symbol = lookupCompatibleSerialHelper(*serialTileHelpers, serialBaseName,
                                            gemmOp, helperType);
      if (!symbol) {
        auto symbolName = uniquifySymbolName(symbolTable, serialBaseName);
        builder.setInsertionPoint(func);
        symbol = builder.create<func::FuncOp>(
            func.getLoc(), symbolName, helperType);
        symbol->setAttr("sym_visibility",
                        StringAttr::get(func.getContext(), "private"));
        copyMappingAttrs(gemmOp, symbol, gemmOp);
        symbolTable.insert(symbol);
        registerSerialHelper(*serialTileHelpers, serialBaseName, symbol);
      }
    } else {
      auto baseSymbolName = buildCandidateAcceleratorName(gemmOp);
      auto symbolName = uniquifySymbolName(symbolTable, baseSymbolName);
      builder.setInsertionPoint(func);
      symbol = builder.create<func::FuncOp>(
          func.getLoc(), symbolName, helperType);
      symbol->setAttr("sym_visibility",
                      StringAttr::get(func.getContext(), "private"));
      copyMappingAttrs(gemmOp, symbol, gemmOp);
      symbolTable.insert(symbol);
    }

    builder.setInsertionPoint(gemmOp);
    auto loc = gemmOp.getLoc();
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto batch = builder.create<arith::ConstantIndexOp>(loc, outputType.getShape()[0]);

    Value init = gemmOp.getExistingInput();
    if (!init)
      init = builder.create<tensor::EmptyOp>(loc, outputType.getShape(),
                                             outputType.getElementType());

    auto loop = builder.create<scf::ForOp>(loc, zero, batch, one, ValueRange{init});
    builder.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value current = loop.getRegionIterArgs().front();

    auto extract2D = [&](Value tensorValue, RankedTensorType rank2Type) -> Value {
      if (!tensorValue)
        return Value();
      auto rank3Type = dyn_cast<RankedTensorType>(tensorValue.getType());
      if (rank3Type && rank3Type.getRank() == 3) {
        SmallVector<OpFoldResult> offsets = {
            OpFoldResult(iv), builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes = {
            builder.getIndexAttr(1), builder.getIndexAttr(rank3Type.getShape()[1]),
            builder.getIndexAttr(rank3Type.getShape()[2])};
        SmallVector<OpFoldResult> strides = {
            builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
        auto sliceType = RankedTensorType::get(
            {1, rank3Type.getShape()[1], rank3Type.getShape()[2]},
            rank3Type.getElementType());
        Value slice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceType, tensorValue, offsets, sizes, strides);
        return collapseLeadingBatchSlice(builder, loc, slice);
      }
      return tensorValue;
    };

    SmallVector<Value> callOperands;
    callOperands.push_back(extract2D(gemmOp.getInput(), helperInputType));
    callOperands.push_back(extract2D(gemmOp.getWeight(), helperWeightType));
    if (gemmOp.getBias())
      callOperands.push_back(extract2D(gemmOp.getBias(), helperBiasType));
    if (gemmOp.getExistingInput())
      callOperands.push_back(extract2D(current, helperExistingType));
    auto call = builder.create<func::CallOp>(loc, TypeRange{helperOutputType},
                                             symbol.getName(), callOperands);
    auto expanded = expandLeadingBatchSlice(
        builder, loc, call.getResult(0),
        RankedTensorType::get({1, helperOutputType.getShape()[0],
                               helperOutputType.getShape()[1]},
                              helperOutputType.getElementType()));
    SmallVector<OpFoldResult> offsets = {
        OpFoldResult(iv), builder.getIndexAttr(0), builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {
        builder.getIndexAttr(1), builder.getIndexAttr(helperOutputType.getShape()[0]),
        builder.getIndexAttr(helperOutputType.getShape()[1])};
    SmallVector<OpFoldResult> strides = {
        builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
    auto inserted = builder.create<tensor::InsertSliceOp>(loc, expanded, current,
                                                          offsets, sizes, strides);
    builder.create<scf::YieldOp>(loc, inserted.getResult());

    gemmOp.getResult().replaceAllUsesWith(loop.getResult(0));
    gemmOp.erase();
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();
    SymbolTable symbolTable(module);
    OpBuilder builder(module.getContext());

    for (auto func : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      SmallVector<Operation *> accelOps;
      func.walk([&](Operation *op) {
        if (isa<accel::GEMMOp>(op))
          accelOps.push_back(op);
      });
      if (accelOps.empty())
        continue;

      SmallVector<Attribute> outlinedSymbols;
      SerialHelperMap serialTileHelpers;
      for (Operation *accelOp : accelOps) {
        accelOp->setAttr(kPointerTransportAttr,
                         StringAttr::get(module.getContext(),
                                         kUntiledPointerTransport));

        SmallVector<Type> operandTypes(accelOp->getOperandTypes().begin(),
                                       accelOp->getOperandTypes().end());
        SmallVector<Type> resultTypes(accelOp->getResultTypes().begin(),
                                      accelOp->getResultTypes().end());
        func::FuncOp symbol;
        const bool isTiled =
            accelOp->hasAttr(kIPTileSizeAttr) && isa<accel::GEMMOp>(accelOp);
        const bool isSerialTiled =
            isTiled &&
            accelOp->getAttrOfType<StringAttr>(kIPTilingModeAttr) &&
            accelOp->getAttrOfType<StringAttr>(kIPTilingModeAttr).getValue() ==
                kSerialIPTilingMode;

        if (auto gemmOp = dyn_cast<accel::GEMMOp>(accelOp)) {
          auto resultType = dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
          if (resultType && resultType.getRank() == 3) {
            if (failed(lowerBatchedGemmTo2DHelper(builder, symbolTable, func, gemmOp,
                                                  symbol, &serialTileHelpers))) {
              accelOp->emitError("failed to lower batched GEMM to 2D helper ABI");
              signalPassFailure();
              return;
            }
            if (llvm::none_of(outlinedSymbols, [&](Attribute attr) {
                  return attr.cast<SymbolRefAttr>().getLeafReference() ==
                         symbol.getName();
                }))
              outlinedSymbols.push_back(SymbolRefAttr::get(symbol));
            continue;
          }
        }

        if (isSerialTiled) {
          std::string serialBaseName = buildAcceleratorName(accelOp);
          if (auto tileSize = accelOp->getAttrOfType<StringAttr>(kIPTileSizeAttr))
            serialBaseName += "_ip" + tileSize.getValue().str();
          auto helperType = builder.getFunctionType(operandTypes, resultTypes);
          symbol = lookupCompatibleSerialHelper(serialTileHelpers,
                                                serialBaseName, accelOp,
                                                helperType);
          if (!symbol) {
            auto symbolName = uniquifySymbolName(symbolTable, serialBaseName);
            builder.setInsertionPoint(func);
            symbol = builder.create<func::FuncOp>(
                func.getLoc(), symbolName, helperType);
            symbol->setAttr("sym_visibility",
                            StringAttr::get(module.getContext(), "private"));
            copyMappingAttrs(accelOp, symbol, accelOp);
            symbolTable.insert(symbol);
            registerSerialHelper(serialTileHelpers, serialBaseName, symbol);
            outlinedSymbols.push_back(SymbolRefAttr::get(symbol));
          }
        } else {
          auto baseSymbolName =
              isTiled ? buildTiledAcceleratorName(accelOp)
                      : buildCandidateAcceleratorName(accelOp);
          auto symbolName = uniquifySymbolName(symbolTable, baseSymbolName);
          builder.setInsertionPoint(func);
          symbol = builder.create<func::FuncOp>(
              func.getLoc(), symbolName,
              builder.getFunctionType(operandTypes, resultTypes));
          symbol->setAttr("sym_visibility",
                          StringAttr::get(module.getContext(), "private"));
          copyMappingAttrs(accelOp, symbol, accelOp);
          symbolTable.insert(symbol);
          outlinedSymbols.push_back(SymbolRefAttr::get(symbol));
        }

        builder.setInsertionPoint(accelOp);
        auto call = builder.create<func::CallOp>(accelOp->getLoc(), symbol,
                                                 accelOp->getOperands());
        accelOp->getResult(0).replaceAllUsesWith(call.getResult(0));
        accelOp->erase();
      }

      func->removeAttr(kOutlinedAttr);
      func->removeAttr(kOutlinedHelpersAttr);
      if (outlinedSymbols.size() == 1) {
        func->setAttr(kOutlinedAttr, outlinedSymbols.front());
      } else if (!outlinedSymbols.empty()) {
        func->setAttr(kOutlinedHelpersAttr,
                      ArrayAttr::get(module.getContext(), outlinedSymbols));
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createLowerAccelToCallsPass() {
  return std::make_unique<LowerAccelToCalls>();
}
