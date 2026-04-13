//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

namespace {
struct FoldEmptyTensorToMemrefAllocPattern
    : public OpRewritePattern<bufferization::ToMemrefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToMemrefOp toMemref,
                                PatternRewriter &rewriter) const override {
    Value tensor = toMemref.getTensor();
    if (!isa_and_nonnull<tensor::EmptyOp>(tensor.getDefiningOp()))
      return failure();

    auto memrefType = dyn_cast<MemRefType>(toMemref.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      return failure();

    auto alloc = rewriter.create<memref::AllocOp>(toMemref.getLoc(), memrefType);
    rewriter.replaceOp(toMemref, alloc.getResult());
    if (tensor.use_empty())
      rewriter.eraseOp(tensor.getDefiningOp());
    return success();
  }
};

struct FoldEmptyTensorToMemrefAlloc
    : public scalehls::FoldEmptyTensorToMemrefAllocBase<
          FoldEmptyTensorToMemrefAlloc> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldEmptyTensorToMemrefAllocPattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createFoldEmptyTensorToMemrefAllocPass() {
  return std::make_unique<FoldEmptyTensorToMemrefAlloc>();
}
