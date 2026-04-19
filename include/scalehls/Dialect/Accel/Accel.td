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
  let summary = "Logical GEMM accelerator IR for mapped kernels";
  let description = [{
    The accel dialect captures logical GEMM semantics shared by
    frontend-specific recognizers. Ops model full-result GEMM production
    without committing to a physical interface style. Surrounding transpose,
    packing, or epilogue logic is represented in compiler IR outside the GEMM
    boundary.
  }];
  let cppNamespace = "::mlir::scalehls::accel";
}

class AccelOp<string mnemonic, list<Trait> traits = []> :
    Op<AccelDialect, mnemonic, traits>;

def Accel_GEMMOp : AccelOp<"gemm", [AttrSizedOperandSegments]> {
  let summary = "Logical GEMM accelerator invocation";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       Optional<AnyRankedTensor>:$existing_input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

#endif // SCALEHLS_DIALECT_ACCEL_ACCEL_TD
