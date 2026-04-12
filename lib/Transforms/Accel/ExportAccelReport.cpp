//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "scalehls/Transforms/Passes.h"
#include <optional>

using namespace mlir;
using namespace scalehls;

namespace {
constexpr StringLiteral kSkipReasonAttr = "scalehls.gemm_skip_reason";
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
constexpr StringLiteral kContractAttr = "scalehls.gemm_contract";
constexpr StringLiteral kHasBiasAttr = "scalehls.accelerator_has_bias";
constexpr StringLiteral kParentFuncAttr = "scalehls.gemm_parent_func";
constexpr StringLiteral kCandidateIndexAttr = "scalehls.gemm_candidate_index";
} // namespace

namespace {
static FailureOr<SmallVector<int64_t>> readI64Array(Operation *op,
                                                    StringRef attrName) {
  auto attr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!attr)
    return failure();
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute element : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    if (!intAttr)
      return failure();
    values.push_back(intAttr.getInt());
  }
  return values;
}

static std::optional<int64_t> readI64(Operation *op, StringRef attrName) {
  auto attr = op->getAttrOfType<IntegerAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static llvm::json::Array toJsonArray(ArrayRef<int64_t> values) {
  llvm::json::Array array;
  for (int64_t value : values)
    array.push_back(value);
  return array;
}

static LogicalResult ensureParentDir(StringRef filePath) {
  if (filePath.empty())
    return success();
  auto parent = llvm::sys::path::parent_path(filePath);
  if (parent.empty())
    return success();
  if (std::error_code err = llvm::sys::fs::create_directories(parent))
    return failure();
  return success();
}

static LogicalResult writeJsonFile(StringRef filePath,
                                   const llvm::json::Value &value) {
  if (failed(ensureParentDir(filePath)))
    return failure();
  std::error_code err;
  llvm::raw_fd_ostream os(filePath, err, llvm::sys::fs::OF_Text);
  if (err)
    return failure();
  os << llvm::formatv("{0:2}\n", value);
  return success();
}

static LogicalResult ensureDir(StringRef dir) {
  if (dir.empty())
    return success();
  if (std::error_code err = llvm::sys::fs::create_directories(dir))
    return failure();
  return success();
}

static llvm::json::Object buildMatrixDescriptor(Operation *op,
                                                StringRef shapeAttr,
                                                StringRef argAttr) {
  llvm::json::Object obj;
  auto shape = readI64Array(op, shapeAttr);
  if (succeeded(shape))
    obj["shape"] = toJsonArray(*shape);
  if (auto arg = readI64(op, argAttr))
    obj["arg_index"] = *arg;
  return obj;
}

static llvm::json::Object buildMappedEntry(func::FuncOp helper) {
  llvm::json::Object obj;
  if (auto parent = helper->getAttrOfType<StringAttr>(kParentFuncAttr))
    obj["function"] = parent.str();
  else
    obj["function"] = helper.getName().str();
  obj["status"] = "mapped";
  obj["symbol"] = helper.getName().str();
  if (auto candidateIndex = readI64(helper, kCandidateIndexAttr))
    obj["candidate_index"] = *candidateIndex;
  if (auto contract = helper->getAttrOfType<StringAttr>(kContractAttr))
    obj["contract"] = contract.str();
  if (auto family = helper->getAttrOfType<StringAttr>(kFamilyAttr))
    obj["family"] = family.str();
  auto shape = readI64Array(helper, kShapeAttr);
  if (succeeded(shape))
    obj["shape"] = toJsonArray(*shape);
  auto inputs = buildMatrixDescriptor(helper, kAShapeAttr, kAArgAttr);
  auto weights = buildMatrixDescriptor(helper, kBShapeAttr, kBArgAttr);
  auto outputs = buildMatrixDescriptor(helper, kCShapeAttr, kCArgAttr);
  obj["A"] = llvm::json::Object(inputs);
  obj["B"] = llvm::json::Object(weights);
  obj["C"] = llvm::json::Object(outputs);
  obj["inputs"] = std::move(inputs);
  obj["weights"] = std::move(weights);
  obj["outputs"] = std::move(outputs);
  if (auto hasBias = helper->getAttrOfType<BoolAttr>(kHasBiasAttr))
    obj["has_bias"] = hasBias.getValue();
  if (helper->hasAttr(kBiasShapeAttr))
    obj["bias"] = buildMatrixDescriptor(helper, kBiasShapeAttr, kBiasArgAttr);
  if (auto precision = helper->getAttrOfType<StringAttr>(kPrecisionAttr))
    obj["data_precision"] = precision.str();
  if (auto bits = readI64(helper, kElementBitsAttr))
    obj["data_bits"] = *bits;
  if (auto bytes = readI64(helper, kElementBytesAttr))
    obj["data_size_bytes"] = *bytes;
  return obj;
}

static llvm::json::Object buildSkippedEntry(func::FuncOp func) {
  llvm::json::Object obj;
  obj["function"] = func.getName().str();
  obj["status"] = "unmapped";
  if (auto reason = func->getAttrOfType<StringAttr>(kSkipReasonAttr))
    obj["skip_reason"] = reason.str();
  return obj;
}

struct ExportAccelReport : public ExportAccelReportBase<ExportAccelReport> {
  ExportAccelReport() = default;
  ExportAccelReport(StringRef manifestDirValue, StringRef candidateLogValue) {
    manifestDir = manifestDirValue.str();
    candidateLog = candidateLogValue.str();
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::json::Array candidateEntries;
    llvm::DenseMap<StringRef, SmallVector<func::FuncOp>> mappedByParent;

    if (!manifestDir.empty() && failed(ensureDir(manifestDir))) {
      module.emitError() << "failed to create manifest directory: "
                         << manifestDir;
      signalPassFailure();
      return;
    }

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal()) {
        if (auto parent = func->getAttrOfType<StringAttr>(kParentFuncAttr))
          mappedByParent[parent.getValue()].push_back(func);
      }
    }

    for (auto &it : mappedByParent) {
      auto parentName = it.first;
      auto &helpers = it.second;
      llvm::sort(helpers, [](func::FuncOp lhs, func::FuncOp rhs) {
        return readI64(lhs, kCandidateIndexAttr).value_or(0) <
               readI64(rhs, kCandidateIndexAttr).value_or(0);
      });

      for (func::FuncOp helper : helpers) {
        auto entry = buildMappedEntry(helper);
        candidateEntries.push_back(llvm::json::Object(entry));

        if (!manifestDir.empty()) {
          SmallString<256> path(manifestDir);
          llvm::sys::path::append(path, parentName);
          if (helpers.size() > 1) {
            path += ".";
            path += llvm::utostr(
                readI64(helper, kCandidateIndexAttr).value_or(0));
          }
          path += ".json";
          if (failed(writeJsonFile(path, llvm::json::Value(std::move(entry))))) {
            module.emitError() << "failed to write accelerator manifest: "
                               << path;
            signalPassFailure();
            return;
          }
        }
      }
    }

    for (auto func : module.getOps<func::FuncOp>()) {
      if (mappedByParent.count(func.getName()))
        continue;
      if (func->hasAttr(kSkipReasonAttr))
        candidateEntries.push_back(buildSkippedEntry(func));
    }

    if (!candidateLog.empty()) {
      llvm::json::Object log;
      log["candidates"] = std::move(candidateEntries);
      if (failed(writeJsonFile(candidateLog, llvm::json::Value(std::move(log))))) {
        module.emitError() << "failed to write accelerator candidate log: "
                           << candidateLog;
        signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createExportAccelReportPass(std::string manifestDir,
                                      std::string candidateLog) {
  return std::make_unique<ExportAccelReport>(manifestDir, candidateLog);
}
