//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/StringRef.h"
#include "scalehls/Dialect/Accel/Accel.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

namespace {
constexpr StringLiteral kIPTileSizeAttr = "scalehls.accelerator_ip_tile_size";
constexpr StringLiteral kIPTilingModeAttr = "scalehls.accelerator_ip_tiling_mode";
constexpr StringLiteral kIPTileRowAttr = "scalehls.accelerator_ip_tile_row";
constexpr StringLiteral kIPTileColAttr = "scalehls.accelerator_ip_tile_col";
constexpr StringLiteral kIPTileKAttr = "scalehls.accelerator_ip_tile_k";
constexpr StringLiteral kParallelIPTilingMode = "parallel";
constexpr StringLiteral kSerialIPTilingMode = "serial";

struct GemmIPTileSize {
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t depth = 0;
};

static FailureOr<GemmIPTileSize> parseGemmIPTileSize(StringRef spec) {
  if (spec.empty())
    return failure();
  SmallVector<StringRef> parts;
  spec.split(parts, 'x');
  if (parts.size() != 3)
    return failure();
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t depth = 0;
  if (parts[0].getAsInteger(10, rows) || parts[1].getAsInteger(10, cols) ||
      parts[2].getAsInteger(10, depth))
    return failure();
  if (rows <= 0 || cols <= 0 || depth <= 0)
    return failure();
  return GemmIPTileSize{rows, cols, depth};
}

static Value extractStaticSlice(OpBuilder &builder, Location loc, Value tensor,
                                ArrayRef<int64_t> offsets,
                                ArrayRef<int64_t> sizes) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> strides(sizes.size(), 1);
  SmallVector<OpFoldResult> mixedOffsets;
  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
  for (int64_t offset : offsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  for (int64_t size : sizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : strides)
    mixedStrides.push_back(builder.getIndexAttr(stride));
  auto sliceType = RankedTensorType::get(sizes, tensorType.getElementType());
  return builder.create<tensor::ExtractSliceOp>(loc, sliceType, tensor,
                                                mixedOffsets, mixedSizes,
                                                mixedStrides);
}

static Value insertStaticSlice(OpBuilder &builder, Location loc, Value source,
                               Value dest, ArrayRef<int64_t> offsets,
                               ArrayRef<int64_t> sizes);

static int64_t ceilDiv(int64_t value, int64_t factor) {
  return (value + factor - 1) / factor;
}

static TypedAttr getZeroAttr(Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return FloatAttr::get(floatType, 0.0);
  if (auto integerType = dyn_cast<IntegerType>(elementType))
    return IntegerAttr::get(integerType, 0);
  return {};
}

static Value createZeroInitializedTensor(OpBuilder &builder, Location loc,
                                         ArrayRef<int64_t> shape,
                                         Type elementType) {
  auto zeroAttr = getZeroAttr(elementType);
  assert(zeroAttr && "expected integer or float tensor element type");
  Value empty =
      builder.create<tensor::EmptyOp>(loc, shape, elementType).getResult();
  Value zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
  return builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty})
      .getResult(0);
}

static Value padTensorWithZeros(OpBuilder &builder, Location loc, Value tensor,
                                ArrayRef<int64_t> paddedShape) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  if (llvm::equal(tensorType.getShape(), paddedShape))
    return tensor;
  Value padded =
      createZeroInitializedTensor(builder, loc, paddedShape,
                                  tensorType.getElementType());
  return insertStaticSlice(builder, loc, tensor, padded,
                           SmallVector<int64_t>(tensorType.getRank(), 0),
                           tensorType.getShape());
}

static Value broadcastRank1BiasToResultShape(OpBuilder &builder, Location loc,
                                             Value bias,
                                             ArrayRef<int64_t> resultShape) {
  auto biasType = cast<RankedTensorType>(bias.getType());
  auto resultType =
      RankedTensorType::get(resultShape, biasType.getElementType());
  Value empty =
      builder.create<tensor::EmptyOp>(loc, resultShape, biasType.getElementType());
  auto biasMap = AffineMap::get(3, 0, {builder.getAffineDimExpr(2)},
                                builder.getContext());
  auto resultMap = AffineMap::getMultiDimIdentityMap(3, builder.getContext());
  return builder
      .create<linalg::GenericOp>(
          loc, resultType, ValueRange{bias}, ValueRange{empty},
          ArrayRef<AffineMap>{biasMap, resultMap},
          ArrayRef<StringRef>{"parallel", "parallel", "parallel"},
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
          })
      .getResult(0);
}

static FailureOr<Value> materializeTiledBias(OpBuilder &builder, Location loc,
                                             Value bias,
                                             ArrayRef<int64_t> paddedResultShape) {
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || !biasType.hasStaticShape())
    return failure();
  if (biasType.getRank() == 3)
    return padTensorWithZeros(builder, loc, bias, paddedResultShape);
  if (biasType.getRank() != 1)
    return failure();

  Value paddedBias = padTensorWithZeros(builder, loc, bias,
                                        ArrayRef<int64_t>{paddedResultShape[2]});
  return broadcastRank1BiasToResultShape(builder, loc, paddedBias,
                                         paddedResultShape);
}

static Value insertStaticSlice(OpBuilder &builder, Location loc, Value source,
                               Value dest, ArrayRef<int64_t> offsets,
                               ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> strides(sizes.size(), 1);
  SmallVector<OpFoldResult> mixedOffsets;
  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
  for (int64_t offset : offsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  for (int64_t size : sizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : strides)
    mixedStrides.push_back(builder.getIndexAttr(stride));
  return builder.create<tensor::InsertSliceOp>(loc, source, dest, mixedOffsets,
                                               mixedSizes, mixedStrides);
}

static void copyAttrs(Operation *from, Operation *to) {
  for (auto namedAttr : from->getAttrs()) {
    if (namedAttr.getName() == "operand_segment_sizes")
      continue;
    to->setAttr(namedAttr.getName(), namedAttr.getValue());
  }
}

static Value createTensorAdd(OpBuilder &builder, Location loc, Value lhs,
                             Value rhs) {
  auto tensorType = cast<RankedTensorType>(lhs.getType());
  auto empty = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                               tensorType.getElementType());
  auto identityMap =
      AffineMap::getMultiDimIdentityMap(tensorType.getRank(), builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, identityMap, identityMap};
  SmallVector<StringRef> iteratorTypes(tensorType.getRank(), "parallel");
  return builder
      .create<linalg::GenericOp>(
          loc, tensorType, ValueRange{lhs, rhs}, ValueRange{empty},
          indexingMaps, iteratorTypes,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value sum;
            auto elementType = tensorType.getElementType();
            if (elementType.isa<FloatType>()) {
              sum = nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0],
                                                        args[1]);
            } else {
              sum = nestedBuilder.create<arith::AddIOp>(nestedLoc, args[0],
                                                        args[1]);
            }
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
          })
      .getResult(0);
}

struct TileAccelGemm : public TileAccelGemmBase<TileAccelGemm> {
  TileAccelGemm() = default;
  TileAccelGemm(std::string ipSizeValue, bool serialValue) {
    ipSize = std::move(ipSizeValue);
    serial = serialValue;
  }

  void runOnOperation() override {
    if (ipSize.empty())
      return;

    auto parsed = parseGemmIPTileSize(ipSize);
    if (failed(parsed)) {
      getOperation()->emitError()
          << "invalid GEMM IP tile size '" << ipSize
          << "'; expected MxNxK with positive integers";
      signalPassFailure();
      return;
    }

    auto func = getOperation();
    SmallVector<accel::GEMMOp> gemmOps;
    func.walk([&](accel::GEMMOp op) { gemmOps.push_back(op); });

    for (accel::GEMMOp gemmOp : gemmOps) {
      if (failed(tileGemmOp(gemmOp, *parsed, serial.getValue()))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult tileGemmOp(accel::GEMMOp gemmOp, GemmIPTileSize tileSize,
                           bool serialMode) {
    auto inputType = dyn_cast<RankedTensorType>(gemmOp.getInput().getType());
    auto weightType = dyn_cast<RankedTensorType>(gemmOp.getWeight().getType());
    auto resultType = dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
    if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
        !weightType.hasStaticShape() || !resultType.hasStaticShape()) {
      return gemmOp.emitError()
             << "GEMM IP tiling requires static ranked tensor operands and result";
    }
    if (inputType.getRank() != 3 || weightType.getRank() != 3 ||
        resultType.getRank() != 3) {
      return gemmOp.emitError()
             << "GEMM IP tiling currently supports rank-3 GEMM tensors only";
    }

    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();
    auto resultShape = resultType.getShape();
    if (inputShape[0] != weightShape[0] || inputShape[0] != resultShape[0] ||
        inputShape[1] != resultShape[1] || inputShape[2] != weightShape[1] ||
        weightShape[2] != resultShape[2]) {
      return gemmOp.emitError()
             << "GEMM IP tiling requires canonical batch GEMM shapes";
    }
    Value bias = gemmOp.getBias();
    if (bias) {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || !biasType.hasStaticShape()) {
        return gemmOp.emitError()
               << "GEMM IP tiling requires bias to be a static ranked tensor";
      }
      if (biasType.getRank() == 3) {
        if (biasType.getShape() != resultShape) {
          return gemmOp.emitError()
                 << "GEMM IP tiling requires rank-matched bias to match the full result shape";
        }
      } else if (biasType.getRank() == 1) {
        if (biasType.getShape()[0] != resultShape[2]) {
          return gemmOp.emitError()
                 << "GEMM IP tiling requires rank-1 bias length to match the GEMM N dimension";
        }
      } else {
        return gemmOp.emitError()
               << "GEMM IP tiling requires bias to be rank-1 or match the full result shape";
      }
    }

    Value existingInput = gemmOp.getExistingInput();
    if (existingInput) {
      auto existingType = dyn_cast<RankedTensorType>(existingInput.getType());
      if (!existingType || !existingType.hasStaticShape() ||
          existingType.getRank() != 3 || existingType.getShape() != resultShape) {
        return gemmOp.emitError()
               << "GEMM IP tiling currently requires existing_input to match the full result shape";
      }
    }

    const int64_t batch = resultShape[0];
    const int64_t m = resultShape[1];
    const int64_t n = resultShape[2];
    const int64_t k = inputShape[2];
    const int64_t paddedM = ceilDiv(m, tileSize.rows) * tileSize.rows;
    const int64_t paddedN = ceilDiv(n, tileSize.cols) * tileSize.cols;
    const int64_t paddedK = ceilDiv(k, tileSize.depth) * tileSize.depth;
    const unsigned rowTiles = static_cast<unsigned>(paddedM / tileSize.rows);
    const unsigned colTiles = static_cast<unsigned>(paddedN / tileSize.cols);
    const unsigned kTiles = static_cast<unsigned>(paddedK / tileSize.depth);

    OpBuilder builder(gemmOp);
    Location loc = gemmOp.getLoc();
    SmallVector<int64_t> paddedInputShape{batch, paddedM, paddedK};
    SmallVector<int64_t> paddedWeightShape{batch, paddedK, paddedN};
    SmallVector<int64_t> paddedResultShape{batch, paddedM, paddedN};
    Value paddedInput =
        padTensorWithZeros(builder, loc, gemmOp.getInput(), paddedInputShape);
    Value paddedWeight =
        padTensorWithZeros(builder, loc, gemmOp.getWeight(), paddedWeightShape);
    Value paddedBias;
    if (bias) {
      auto materializedBias =
          materializeTiledBias(builder, loc, bias, paddedResultShape);
      if (failed(materializedBias)) {
        return gemmOp.emitError()
               << "GEMM IP tiling failed to materialize tiled bias";
      }
      paddedBias = *materializedBias;
    }
    Value paddedExistingInput =
        existingInput
            ? padTensorWithZeros(builder, loc, existingInput, paddedResultShape)
            : Value();

    Value tiledResult =
        createZeroInitializedTensor(builder, loc, paddedResultShape,
                                    resultType.getElementType());

    for (unsigned tileRow = 0; tileRow < rowTiles; ++tileRow) {
      for (unsigned tileCol = 0; tileCol < colTiles; ++tileCol) {
        SmallVector<int64_t> resultOffsets{
            0, static_cast<int64_t>(tileRow) * tileSize.rows,
            static_cast<int64_t>(tileCol) * tileSize.cols};
        SmallVector<int64_t> resultSizes{batch, tileSize.rows, tileSize.cols};
        Value accumulatedTile;
        if (paddedExistingInput)
          accumulatedTile = extractStaticSlice(builder, loc, paddedExistingInput,
                                               resultOffsets, resultSizes);
        if (paddedBias) {
          Value biasTile =
              extractStaticSlice(builder, loc, paddedBias, resultOffsets,
                                 resultSizes);
          accumulatedTile =
              accumulatedTile ? createTensorAdd(builder, loc, accumulatedTile, biasTile)
                              : biasTile;
        }

        for (unsigned tileK = 0; tileK < kTiles; ++tileK) {
          SmallVector<int64_t> inputOffsets{
              0, static_cast<int64_t>(tileRow) * tileSize.rows,
              static_cast<int64_t>(tileK) * tileSize.depth};
          SmallVector<int64_t> inputSizes{batch, tileSize.rows, tileSize.depth};
          SmallVector<int64_t> weightOffsets{
              0, static_cast<int64_t>(tileK) * tileSize.depth,
              static_cast<int64_t>(tileCol) * tileSize.cols};
          SmallVector<int64_t> weightSizes{batch, tileSize.depth, tileSize.cols};

          Value inputTile = extractStaticSlice(builder, loc, paddedInput,
                                               inputOffsets, inputSizes);
          Value weightTile = extractStaticSlice(builder, loc, paddedWeight,
                                                weightOffsets, weightSizes);

          auto tileResultType =
              RankedTensorType::get(resultSizes, resultType.getElementType());
          auto tiledOp = builder.create<accel::GEMMOp>(loc, tileResultType,
                                                       inputTile, weightTile,
                                                       Value(), Value());
          copyAttrs(gemmOp, tiledOp);
          tiledOp->setAttr(
              kIPTileSizeAttr,
              StringAttr::get(
                  gemmOp.getContext(),
                  (Twine(tileSize.rows) + "x" + Twine(tileSize.cols) + "x" +
                   Twine(tileSize.depth))
                      .str()));
          tiledOp->setAttr(
              kIPTilingModeAttr,
              StringAttr::get(gemmOp.getContext(),
                              serialMode ? kSerialIPTilingMode
                                         : kParallelIPTilingMode));
          tiledOp->setAttr(
              kIPTileRowAttr,
              builder.getI64IntegerAttr(static_cast<int64_t>(tileRow)));
          tiledOp->setAttr(
              kIPTileColAttr,
              builder.getI64IntegerAttr(static_cast<int64_t>(tileCol)));
          tiledOp->setAttr(kIPTileKAttr,
                           builder.getI64IntegerAttr(static_cast<int64_t>(tileK)));

          accumulatedTile =
              accumulatedTile
                  ? createTensorAdd(builder, loc, accumulatedTile,
                                    tiledOp.getResult())
                  : tiledOp.getResult();
        }

        tiledResult = insertStaticSlice(builder, loc, accumulatedTile, tiledResult,
                                        resultOffsets, resultSizes);
      }
    }

    Value finalResult = tiledResult;
    if (paddedM != m || paddedN != n) {
      finalResult = extractStaticSlice(builder, loc, tiledResult, {0, 0, 0},
                                       resultShape);
    }

    gemmOp.getResult().replaceAllUsesWith(finalResult);
    gemmOp.erase();
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createTileAccelGemmPass(std::string ipSize,
                                                        bool serial) {
  return std::make_unique<TileAccelGemm>(std::move(ipSize), serial);
}
