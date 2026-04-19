//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
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

static FailureOr<SmallVector<int64_t>> getShapedTypeShape(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasStaticShape())
    return failure();
  SmallVector<int64_t> shape(shapedType.getShape().begin(),
                             shapedType.getShape().end());
  return shape;
}

static std::optional<std::string> getDTypeString(Type type) {
  if (type.isF16())
    return std::string("f16");
  if (type.isF32())
    return std::string("f32");
  if (type.isF64())
    return std::string("f64");
  if (auto integerType = dyn_cast<IntegerType>(type))
    return std::string(integerType.isUnsigned() ? "u" : "i") +
           llvm::utostr(integerType.getWidth());
  return std::nullopt;
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

static LogicalResult writeTextFile(StringRef filePath, StringRef text) {
  if (failed(ensureParentDir(filePath)))
    return failure();
  std::error_code err;
  llvm::raw_fd_ostream os(filePath, err, llvm::sys::fs::OF_Text);
  if (err)
    return failure();
  os << text;
  return success();
}

static std::string formatShapeArray(const llvm::json::Array &shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0)
      text += ", ";
    if (auto value = shape[i].getAsInteger())
      text += llvm::utostr(*value);
    else
      text += "?";
  }
  text += "]";
  return text;
}

static std::string formatCompactObject(
    std::initializer_list<std::pair<StringRef, std::string>> fields) {
  std::string text = "{ ";
  bool first = true;
  for (const auto &field : fields) {
    if (!first)
      text += ", ";
    first = false;
    text += "\"";
    text += field.first.str();
    text += "\": ";
    text += field.second;
  }
  text += " }";
  return text;
}

static std::string quoteJsonString(StringRef value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::json::Value(value.str());
  os.flush();
  return out;
}

static std::string formatManifestJson(const llvm::json::Array &helpers) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "{\n";
  os << "  \"helpers\": [\n";
  for (size_t i = 0; i < helpers.size(); ++i) {
    const auto *helper = helpers[i].getAsObject();
    if (!helper)
      continue;
    os << "    {\n";
    os << "      \"symbol\": "
       << quoteJsonString(helper->getString("symbol").value_or("")) << ",\n";
    os << "      \"function\": "
       << quoteJsonString(helper->getString("function").value_or("")) << ",\n";
    os << "      \"call_index\": "
       << helper->getInteger("call_index").value_or(0) << ",\n";
    if (auto gemm = helper->getObject("gemm")) {
      os << "      \"gemm\": "
         << formatCompactObject({
                {"M", llvm::utostr(gemm->getInteger("M").value_or(0))},
                {"K", llvm::utostr(gemm->getInteger("K").value_or(0))},
                {"N", llvm::utostr(gemm->getInteger("N").value_or(0))},
            })
         << ",\n";
    }
    if (auto sizes = helper->getObject("sizes")) {
      os << "      \"sizes\": {\n";
      SmallVector<StringRef> roleOrder = {"A", "B", "Bias", "ExistingInput", "C"};
      bool firstSize = true;
      for (StringRef role : roleOrder) {
        auto shape = sizes->getArray(role);
        if (!shape)
          continue;
        if (!firstSize)
          os << ",\n";
        firstSize = false;
        os << "        \"" << role << "\": " << formatShapeArray(*shape);
      }
      if (!firstSize)
        os << "\n";
      os << "      },\n";
    }
    if (auto precision = helper->getObject("precision")) {
      os << "      \"precision\": "
         << formatCompactObject({
                {"element", quoteJsonString(precision->getString("element").value_or(""))},
                {"bits", llvm::utostr(precision->getInteger("bits").value_or(0))},
                {"bytes", llvm::utostr(precision->getInteger("bytes").value_or(0))},
            })
         << "\n";
    } else {
      os << "      \"precision\": { }\n";
    }
    os << "    }";
    if (i + 1 != helpers.size())
      os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  os.flush();
  return buffer;
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

static llvm::json::Object buildAbiOperandDescriptor(func::FuncOp helper,
                                                    unsigned argIndex) {
  llvm::json::Object obj;
  auto funcType = helper.getFunctionType();
  if (argIndex >= funcType.getNumInputs())
    return obj;
  Type inputType = funcType.getInput(argIndex);
  auto shape = getShapedTypeShape(inputType);
  if (succeeded(shape))
    obj["shape"] = toJsonArray(*shape);
  if (auto shapedType = dyn_cast<ShapedType>(inputType)) {
    if (auto dtype = getDTypeString(shapedType.getElementType()))
      obj["dtype"] = *dtype;
  }
  obj["arg_index"] = static_cast<int64_t>(argIndex);
  return obj;
}

static llvm::json::Object buildAbiResultDescriptor(func::FuncOp helper) {
  llvm::json::Object obj;
  auto funcType = helper.getFunctionType();
  if (funcType.getNumResults() == 0)
    return obj;
  Type resultType = funcType.getResult(0);
  auto shape = getShapedTypeShape(resultType);
  if (succeeded(shape))
    obj["shape"] = toJsonArray(*shape);
  if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
    if (auto dtype = getDTypeString(shapedType.getElementType()))
      obj["dtype"] = *dtype;
  }
  return obj;
}

static void maybeSetGemmDims(llvm::json::Object &obj,
                             const llvm::json::Object &aAbi,
                             const llvm::json::Object &bAbi,
                             const llvm::json::Object &cAbi) {
  auto aShape = aAbi.getArray("shape");
  auto bShape = bAbi.getArray("shape");
  auto cShape = cAbi.getArray("shape");
  if (!aShape || !bShape || !cShape || aShape->size() != 2 || bShape->size() != 2 ||
      cShape->size() != 2)
    return;

  auto aRows = (*aShape)[0].getAsInteger();
  auto aCols = (*aShape)[1].getAsInteger();
  auto bRows = (*bShape)[0].getAsInteger();
  auto bCols = (*bShape)[1].getAsInteger();
  auto cRows = (*cShape)[0].getAsInteger();
  auto cCols = (*cShape)[1].getAsInteger();
  if (!aRows || !aCols || !bRows || !bCols || !cRows || !cCols)
    return;
  if (*aCols != *bRows || *aRows != *cRows || *bCols != *cCols)
    return;

  llvm::json::Object dims;
  dims["M"] = *aRows;
  dims["K"] = *aCols;
  dims["N"] = *bCols;
  obj["gemm_dimensions"] = std::move(dims);
}

static std::optional<unsigned> getArgIndexAttr(Operation *op, StringRef attrName) {
  auto value = readI64(op, attrName);
  if (!value || *value < 0)
    return std::nullopt;
  return static_cast<unsigned>(*value);
}

static void maybeSetArrayField(llvm::json::Object &obj, StringRef fieldName,
                               const llvm::json::Object &descriptor) {
  if (auto shape = descriptor.getArray("shape"))
    obj[fieldName] = llvm::json::Array(*shape);
}

static llvm::json::Object buildPrecisionDescriptor(func::FuncOp helper) {
  llvm::json::Object obj;
  if (auto precision = helper->getAttrOfType<StringAttr>(kPrecisionAttr))
    obj["element"] = precision.str();
  if (auto bits = readI64(helper, kElementBitsAttr))
    obj["bits"] = *bits;
  if (auto bytes = readI64(helper, kElementBytesAttr))
    obj["bytes"] = *bytes;
  return obj;
}

static llvm::json::Object buildGeneratorManifest(func::FuncOp helper) {
  llvm::json::Object obj;
  if (auto parent = helper->getAttrOfType<StringAttr>(kParentFuncAttr))
    obj["function"] = parent.str();
  else
    obj["function"] = helper.getName().str();
  if (auto candidateIndex = readI64(helper, kCandidateIndexAttr))
    obj["call_index"] = *candidateIndex;
  obj["symbol"] = helper.getName().str();

  auto aArgIndex = getArgIndexAttr(helper, kAArgAttr).value_or(0);
  auto bArgIndex = getArgIndexAttr(helper, kBArgAttr).value_or(1);
  auto aAbi = buildAbiOperandDescriptor(helper, aArgIndex);
  auto bAbi = buildAbiOperandDescriptor(helper, bArgIndex);
  llvm::json::Object cAbi;
  if (auto cArgIndex = getArgIndexAttr(helper, kCArgAttr))
    cAbi = buildAbiOperandDescriptor(helper, *cArgIndex);
  if (cAbi.empty())
    cAbi = buildAbiResultDescriptor(helper);

  llvm::json::Object sizes;
  auto aSource = buildMatrixDescriptor(helper, kAShapeAttr, kAArgAttr);
  auto bSource = buildMatrixDescriptor(helper, kBShapeAttr, kBArgAttr);
  auto cSource = buildMatrixDescriptor(helper, kCShapeAttr, kCArgAttr);
  if (!aSource.empty())
    maybeSetArrayField(sizes, "A", aSource);
  else
    maybeSetArrayField(sizes, "A", aAbi);
  if (!bSource.empty())
    maybeSetArrayField(sizes, "B", bSource);
  else
    maybeSetArrayField(sizes, "B", bAbi);
  if (!cSource.empty())
    maybeSetArrayField(sizes, "C", cSource);
  else
    maybeSetArrayField(sizes, "C", cAbi);
  if (helper->hasAttr(kBiasShapeAttr)) {
    auto biasSource = buildMatrixDescriptor(helper, kBiasShapeAttr, kBiasArgAttr);
    maybeSetArrayField(sizes, "Bias", biasSource);
  } else if (auto biasArgIndex = getArgIndexAttr(helper, kBiasArgAttr)) {
    maybeSetArrayField(sizes, "Bias",
                       buildAbiOperandDescriptor(helper, *biasArgIndex));
  }
  if (auto existingInputArgIndex = getArgIndexAttr(helper, kExistingInputArgAttr)) {
    if (!cSource.empty())
      maybeSetArrayField(sizes, "ExistingInput", cSource);
    else
      maybeSetArrayField(sizes, "ExistingInput",
                         buildAbiOperandDescriptor(helper, *existingInputArgIndex));
  }
  obj["sizes"] = std::move(sizes);
  obj["precision"] = buildPrecisionDescriptor(helper);

  llvm::json::Object dimsContainer;
  maybeSetGemmDims(dimsContainer, aAbi, bAbi, cAbi);
  if (auto dims = dimsContainer.getObject("gemm_dimensions"))
    obj["gemm"] = llvm::json::Object(*dims);
  return obj;
}

static llvm::json::Object buildMappedEntry(func::FuncOp helper) {
  llvm::json::Object obj;
  if (auto parent = helper->getAttrOfType<StringAttr>(kParentFuncAttr))
    obj["function"] = parent.str();
  else
    obj["function"] = helper.getName().str();
  obj["symbol"] = helper.getName().str();
  if (auto candidateIndex = readI64(helper, kCandidateIndexAttr))
    obj["candidate_index"] = *candidateIndex;

  auto aArgIndex = getArgIndexAttr(helper, kAArgAttr).value_or(0);
  auto bArgIndex = getArgIndexAttr(helper, kBArgAttr).value_or(1);
  auto abiInputs = buildAbiOperandDescriptor(helper, aArgIndex);
  auto abiWeights = buildAbiOperandDescriptor(helper, bArgIndex);
  llvm::json::Object abiOutputs;
  if (helper.getFunctionType().getNumResults() != 0)
    abiOutputs = buildAbiResultDescriptor(helper);
  else if (auto cArgIndex = getArgIndexAttr(helper, kCArgAttr))
    abiOutputs = buildAbiOperandDescriptor(helper, *cArgIndex);

  obj["A"] = llvm::json::Object(abiInputs);
  obj["B"] = llvm::json::Object(abiWeights);
  obj["C"] = llvm::json::Object(abiOutputs);
  maybeSetGemmDims(obj, abiInputs, abiWeights, abiOutputs);
  obj["precision"] = buildPrecisionDescriptor(helper);

  auto sourceInputs = buildMatrixDescriptor(helper, kAShapeAttr, kAArgAttr);
  auto sourceWeights = buildMatrixDescriptor(helper, kBShapeAttr, kBArgAttr);
  auto sourceOutputs = buildMatrixDescriptor(helper, kCShapeAttr, kCArgAttr);
  if (!sourceInputs.empty())
    obj["source_A"] = llvm::json::Object(sourceInputs);
  if (!sourceWeights.empty())
    obj["source_B"] = llvm::json::Object(sourceWeights);
  if (!sourceOutputs.empty())
    obj["source_C"] = llvm::json::Object(sourceOutputs);

  if (auto hasBias = helper->getAttrOfType<BoolAttr>(kHasBiasAttr))
    obj["has_bias"] = hasBias.getValue();
  if (auto biasArgIndex = getArgIndexAttr(helper, kBiasArgAttr)) {
    auto bias = buildAbiOperandDescriptor(helper, *biasArgIndex);
    if (!bias.empty())
      obj["bias"] = std::move(bias);
  } else if (helper->hasAttr(kBiasShapeAttr)) {
    obj["bias"] = buildMatrixDescriptor(helper, kBiasShapeAttr, kBiasArgAttr);
  }
  if (auto existingInputArgIndex = getArgIndexAttr(helper, kExistingInputArgAttr)) {
    auto existingInput = buildAbiOperandDescriptor(helper, *existingInputArgIndex);
    if (!existingInput.empty())
      obj["existing_input"] = std::move(existingInput);
  } else if (helper->hasAttr(kExistingInputArgAttr)) {
    obj["existing_input"] =
        buildMatrixDescriptor(helper, kCShapeAttr, kExistingInputArgAttr);
  }
  return obj;
}

static llvm::json::Object buildSkippedEntry(func::FuncOp func, bool hasMappedHelpers) {
  llvm::json::Object obj;
  obj["function"] = func.getName().str();
  obj["status"] = hasMappedHelpers ? "partially_mapped" : "unmapped";
  if (auto reason = func->getAttrOfType<StringAttr>(kSkipReasonAttr))
    obj["skip_reason"] = reason.str();
  return obj;
}

static std::string formatShape(const llvm::json::Object &obj) {
  auto shape = obj.getArray("shape");
  if (!shape)
    return "[]";
  std::string text = "[";
  for (size_t i = 0; i < shape->size(); ++i) {
    if (i != 0)
      text += ", ";
    if (auto value = (*shape)[i].getAsInteger())
      text += llvm::utostr(*value);
    else
      text += "?";
  }
  text += "]";
  return text;
}

static std::string formatDescriptor(const llvm::json::Object &obj) {
  std::string text = formatShape(obj);
  if (auto dtype = obj.getString("dtype"))
    text += " dtype=" + dtype->str();
  if (auto argIndex = obj.getInteger("arg_index"))
    text += " arg=" + llvm::utostr(*argIndex);
  return text;
}

static void appendPrecisionReport(llvm::raw_string_ostream &os,
                                  const llvm::json::Object &entry) {
  auto precision = entry.getObject("precision");
  if (!precision)
    return;
  os << "  Precision: "
     << precision->getString("element").value_or("<unknown>");
  if (auto bits = precision->getInteger("bits"))
    os << " (" << *bits << " bits";
  if (auto bytes = precision->getInteger("bytes"))
    os << ", " << *bytes << " bytes";
  if (precision->getInteger("bits") || precision->getInteger("bytes"))
    os << ")";
  os << "\n";
}

static void appendMappedReport(llvm::raw_string_ostream &os,
                               const llvm::json::Object &entry) {
  os << "- Function: " << entry.getString("function").value_or("<unknown>") << "\n";
  os << "  Status: mapped\n";
  os << "  Symbol: " << entry.getString("symbol").value_or("<unknown>") << "\n";
  if (auto candidateIndex = entry.getInteger("candidate_index"))
    os << "  Candidate index: " << *candidateIndex << "\n";
  if (auto dims = entry.getObject("gemm_dimensions")) {
    os << "  GEMM dimensions: M=" << dims->getInteger("M").value_or(-1)
       << " K=" << dims->getInteger("K").value_or(-1)
       << " N=" << dims->getInteger("N").value_or(-1) << "\n";
  }
  appendPrecisionReport(os, entry);
  if (auto a = entry.getObject("A"))
    os << "  Helper A: " << formatDescriptor(*a) << "\n";
  if (auto b = entry.getObject("B"))
    os << "  Helper B: " << formatDescriptor(*b) << "\n";
  if (auto c = entry.getObject("C"))
    os << "  Helper C: " << formatDescriptor(*c) << "\n";
  if (auto sourceA = entry.getObject("source_A"))
    os << "  Logical A: " << formatDescriptor(*sourceA) << "\n";
  if (auto sourceB = entry.getObject("source_B"))
    os << "  Logical B: " << formatDescriptor(*sourceB) << "\n";
  if (auto sourceC = entry.getObject("source_C"))
    os << "  Logical C: " << formatDescriptor(*sourceC) << "\n";
  if (auto hasBias = entry.getBoolean("has_bias"))
    os << "  Has bias epilogue: " << (*hasBias ? "yes" : "no") << "\n";
  if (auto bias = entry.getObject("bias"))
    os << "  Bias: " << formatDescriptor(*bias) << "\n";
  if (auto existingInput = entry.getObject("existing_input"))
    os << "  Existing input: " << formatDescriptor(*existingInput) << "\n";
  os << "\n";
}

static void appendSkippedReport(llvm::raw_string_ostream &os,
                                const llvm::json::Object &entry) {
  os << "- Function: " << entry.getString("function").value_or("<unknown>") << "\n";
  os << "  Status: "
     << entry.getString("status").value_or("unmapped") << "\n";
  if (auto reason = entry.getString("skip_reason"))
    os << "  Skip reason: " << *reason << "\n";
  os << "\n";
}

struct ExportAccelReport : public ExportAccelReportBase<ExportAccelReport> {
  ExportAccelReport() = default;
  ExportAccelReport(StringRef manifestFileValue, StringRef candidateReportValue) {
    manifestFile = manifestFileValue.str();
    candidateReport = candidateReportValue.str();
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::json::Array mappedEntries;
    llvm::json::Array skippedEntries;
    llvm::json::Array manifestHelpers;
    llvm::DenseMap<StringRef, SmallVector<func::FuncOp>> mappedByParent;

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal()) {
        if (auto parent = func->getAttrOfType<StringAttr>(kParentFuncAttr))
          mappedByParent[parent.getValue()].push_back(func);
      }
    }

    for (auto &it : mappedByParent) {
      auto &helpers = it.second;
      llvm::sort(helpers, [](func::FuncOp lhs, func::FuncOp rhs) {
        return readI64(lhs, kCandidateIndexAttr).value_or(0) <
               readI64(rhs, kCandidateIndexAttr).value_or(0);
      });

      for (func::FuncOp helper : helpers) {
        auto entry = buildMappedEntry(helper);
        mappedEntries.push_back(llvm::json::Object(entry));
        manifestHelpers.push_back(buildGeneratorManifest(helper));
      }
    }

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr(kSkipReasonAttr))
        skippedEntries.push_back(
            buildSkippedEntry(func, mappedByParent.count(func.getName())));
    }

    if (!manifestFile.empty()) {
      llvm::json::Object manifest;
      manifest["helpers"] = std::move(manifestHelpers);
      auto *helpers = manifest.getArray("helpers");
      std::string formattedManifest =
          helpers ? formatManifestJson(*helpers) : std::string("{\"helpers\":[]}\n");
      if (failed(writeTextFile(manifestFile, formattedManifest))) {
        module.emitError() << "failed to write accelerator manifest: "
                           << manifestFile;
        signalPassFailure();
        return;
      }
    }

    if (!candidateReport.empty()) {
      std::string buffer;
      llvm::raw_string_ostream os(buffer);
      os << "ScaleHLS Accelerator Mapping Report\n";
      os << "=================================\n\n";
      os << "Mapped candidates: " << mappedEntries.size() << "\n";
      os << "Unmapped functions: " << skippedEntries.size() << "\n\n";
      os << "[Mapped]\n";
      if (mappedEntries.empty()) {
        os << "(none)\n\n";
      } else {
        for (const auto &value : mappedEntries)
          appendMappedReport(os, *value.getAsObject());
      }
      os << "[Unmapped]\n";
      if (skippedEntries.empty()) {
        os << "(none)\n";
      } else {
        for (const auto &value : skippedEntries)
          appendSkippedReport(os, *value.getAsObject());
      }
      os.flush();
      if (failed(writeTextFile(candidateReport, buffer))) {
        module.emitError() << "failed to write accelerator mapping report: "
                           << candidateReport;
        signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createExportAccelReportPass(std::string manifestFile,
                                      std::string candidateReport) {
  return std::make_unique<ExportAccelReport>(manifestFile, candidateReport);
}
