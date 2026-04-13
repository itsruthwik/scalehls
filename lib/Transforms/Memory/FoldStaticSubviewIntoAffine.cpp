//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

namespace {
static LogicalResult getStaticSubviewLayout(memref::SubViewOp subview,
                                            SmallVectorImpl<int64_t> &offsets,
                                            SmallVectorImpl<int64_t> &strides) {
  offsets.clear();
  strides.clear();
  for (OpFoldResult offset : subview.getMixedOffsets()) {
    auto attr = offset.dyn_cast<Attribute>();
    if (!attr)
      return failure();
    offsets.push_back(attr.cast<IntegerAttr>().getInt());
  }
  for (OpFoldResult stride : subview.getMixedStrides()) {
    auto attr = stride.dyn_cast<Attribute>();
    if (!attr)
      return failure();
    strides.push_back(attr.cast<IntegerAttr>().getInt());
  }
  return success();
}

template <typename AffineOpTy>
static AffineMap buildSubviewAffineMap(MLIRContext *context,
                                       ArrayRef<int64_t> offsets,
                                       ArrayRef<int64_t> strides) {
  SmallVector<AffineExpr> exprs;
  exprs.reserve(offsets.size());
  for (auto [index, offset] : llvm::enumerate(offsets)) {
    auto dim = getAffineDimExpr(index, context);
    exprs.push_back(dim * strides[index] + offset);
  }
  return AffineMap::get(static_cast<unsigned>(offsets.size()), 0, exprs, context);
}

static void emitCopyLoopNest(PatternRewriter &rewriter, Location loc,
                             Value source, AffineMap sourceMap,
                             Value target, AffineMap targetMap,
                             ArrayRef<int64_t> shape,
                             SmallVectorImpl<Value> &ivs,
                             unsigned dim) {
  if (dim == shape.size()) {
    auto load = rewriter.create<AffineLoadOp>(loc, source, sourceMap, ivs);
    rewriter.create<AffineStoreOp>(loc, load, target, targetMap, ivs);
    return;
  }

  auto loop = rewriter.create<AffineForOp>(loc, 0, shape[dim]);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(loop.getBody());
  ivs.push_back(loop.getInductionVar());
  emitCopyLoopNest(rewriter, loc, source, sourceMap, target, targetMap, shape,
                   ivs, dim + 1);
  ivs.pop_back();
}

static LogicalResult expandStaticSubviewCopy(memref::CopyOp copy,
                                             PatternRewriter &rewriter) {
  auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!sourceType || !targetType || !sourceType.hasStaticShape())
    return failure();

  auto sourceSubview = copy.getSource().getDefiningOp<memref::SubViewOp>();
  auto targetSubview = copy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!sourceSubview && !targetSubview)
    return failure();

  SmallVector<int64_t> sourceOffsets, sourceStrides, targetOffsets, targetStrides;
  Value source = copy.getSource();
  Value target = copy.getTarget();
  AffineMap sourceMap;
  AffineMap targetMap;

  if (sourceSubview) {
    if (failed(getStaticSubviewLayout(sourceSubview, sourceOffsets, sourceStrides)))
      return failure();
    source = sourceSubview.getSource();
    sourceMap = buildSubviewAffineMap<AffineLoadOp>(copy.getContext(),
                                                    sourceOffsets, sourceStrides);
  } else {
    sourceMap = AffineMap::getMultiDimIdentityMap(sourceType.getRank(),
                                                  copy.getContext());
  }

  if (targetSubview) {
    if (failed(getStaticSubviewLayout(targetSubview, targetOffsets, targetStrides)))
      return failure();
    target = targetSubview.getSource();
    targetMap = buildSubviewAffineMap<AffineStoreOp>(copy.getContext(),
                                                     targetOffsets, targetStrides);
  } else {
    targetMap = AffineMap::getMultiDimIdentityMap(targetType.getRank(),
                                                  copy.getContext());
  }

  SmallVector<Value> ivs;
  ivs.reserve(sourceType.getRank());
  emitCopyLoopNest(rewriter, copy.getLoc(), source, sourceMap, target, targetMap,
                   sourceType.getShape(), ivs, 0);
  rewriter.eraseOp(copy);
  if (sourceSubview && sourceSubview->use_empty())
    rewriter.eraseOp(sourceSubview);
  if (targetSubview && targetSubview->use_empty())
    rewriter.eraseOp(targetSubview);
  return success();
}

struct FoldSubviewLoad : public OpRewritePattern<AffineLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp load,
                                PatternRewriter &rewriter) const override {
    auto subview = load.getMemRef().getDefiningOp<memref::SubViewOp>();
    if (!subview)
      return failure();

    SmallVector<int64_t> offsets, strides;
    if (failed(getStaticSubviewLayout(subview, offsets, strides)))
      return failure();

    auto map = buildSubviewAffineMap<AffineLoadOp>(load.getContext(), offsets, strides);
    rewriter.replaceOpWithNewOp<AffineLoadOp>(load, subview.getSource(), map,
                                              load.getMapOperands());
    if (subview->use_empty())
      rewriter.eraseOp(subview);
    return success();
  }
};

struct FoldSubviewStore : public OpRewritePattern<AffineStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp store,
                                PatternRewriter &rewriter) const override {
    auto subview = store.getMemRef().getDefiningOp<memref::SubViewOp>();
    if (!subview)
      return failure();

    SmallVector<int64_t> offsets, strides;
    if (failed(getStaticSubviewLayout(subview, offsets, strides)))
      return failure();

    auto map =
        buildSubviewAffineMap<AffineStoreOp>(store.getContext(), offsets, strides);
    rewriter.replaceOpWithNewOp<AffineStoreOp>(store, store.getValueToStore(),
                                               subview.getSource(), map,
                                               store.getMapOperands());
    if (subview->use_empty())
      rewriter.eraseOp(subview);
    return success();
  }
};

struct FoldSubviewCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    return expandStaticSubviewCopy(copy, rewriter);
  }
};

struct FoldStaticSubviewIntoAffine
    : public scalehls::FoldStaticSubviewIntoAffineBase<
          FoldStaticSubviewIntoAffine> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldSubviewLoad, FoldSubviewStore, FoldSubviewCopy>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createFoldStaticSubviewIntoAffinePass() {
  return std::make_unique<FoldStaticSubviewIntoAffine>();
}
