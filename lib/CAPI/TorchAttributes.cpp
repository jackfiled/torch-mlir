
#include "torch-mlir-c/TorchAttributes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;

bool mlirAttributeIsADenseExternalElements(MlirAttribute attr) {
  return llvm::isa<torch::Torch::DenseExternalElementsAttr>(unwrap(attr));
}

MlirStringRef mlirDenseExternalElementsAttrGetName() {
  return wrap(torch::Torch::DenseExternalElementsAttr::name);
}

MlirTypeID mlirDenseExternalElementsAttrGetTypeID() {
  return wrap(torch::Torch::DenseExternalElementsAttr::getTypeID());
}

MlirAttribute mlirDenseExternalElementsAttrGetFromBuffer(
    const MlirContext context, const MlirType shapedType,
    const MlirStringRef filename, const intptr_t size, const void *buffer) {
  const auto *ptr = static_cast<const char *>(buffer);
  return wrap(torch::Torch::DenseExternalElementsAttr::get(
      unwrap(context), cast<ShapedType>(unwrap(shapedType)), unwrap(filename),
      ArrayRef(ptr, size)));
}
