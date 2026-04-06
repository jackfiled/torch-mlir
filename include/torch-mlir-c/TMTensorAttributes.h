#ifndef TORCH_MLIR_CAPI_TMTENSORATTRIBUTES_H
#define TORCH_MLIR_CAPI_TMTENSORATTRIBUTES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool
mlirAttributeIsADenseExternalElements(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef mlirDenseExternalElementsAttrGetName(void);

MLIR_CAPI_EXPORTED MlirTypeID mlirDenseExternalElementsAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute mlirDenseExternalElementsAttrGetFromBuffer(
    MlirContext context, MlirType shapedType, MlirStringRef filename,
    intptr_t size, const void *buffer);

#ifdef __cplusplus
}
#endif

#endif // TORCH_MLIR_CAPI_TORCHATTRIBUTES_H
