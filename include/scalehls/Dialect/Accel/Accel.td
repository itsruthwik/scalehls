//===----------------------------------------------------------------------===//
//
// Author: Ruthwik Reddy Sunketa
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_DIALECT_ACCEL_ACCEL_TD
#define SCALEHLS_DIALECT_ACCEL_ACCEL_TD

include "mlir/IR/OpBase.td"

def AccelDialect : Dialect {
  let name = "accel";
  let summary = "Shared accelerator-family IR for mapped kernels";
  let description = [{
    The accel dialect captures logical accelerator-family semantics shared by
    frontend-specific recognizers. Ops model full-result production for GEMMV,
    GEMM, and CONV without committing to a physical interface style.
  }];
  let cppNamespace = "::mlir::scalehls::accel";
}

class AccelOp<string mnemonic, list<Trait> traits = []> :
    Op<AccelDialect, mnemonic, traits>;

def Accel_GEMMVOp : AccelOp<"gemmv", [AttrSizedOperandSegments]> {
  let summary = "Logical GEMMV accelerator invocation";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       Optional<AnyRankedTensor>:$existing_input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Accel_GEMMOp : AccelOp<"gemm", [AttrSizedOperandSegments]> {
  let summary = "Logical GEMM accelerator invocation";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       Optional<AnyRankedTensor>:$existing_input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Accel_CONVOp : AccelOp<"conv", [AttrSizedOperandSegments]> {
  let summary = "Logical CONV accelerator invocation";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       Optional<AnyRankedTensor>:$existing_input,
                       DenseI64ArrayAttr:$strides,
                       DenseI64ArrayAttr:$dilations);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

#endif // SCALEHLS_DIALECT_ACCEL_ACCEL_TD
