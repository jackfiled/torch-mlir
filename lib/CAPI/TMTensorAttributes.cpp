
#include "torch-mlir-c/TMTensorAttributes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorAttributes.h"

using namespace mlir;

bool mlirAttributeIsADenseExternalElements(MlirAttribute attr) {
  return llvm::isa<torch::TMTensor::DenseExternalElementsAttr>(unwrap(attr));
}

MlirStringRef mlirDenseExternalElementsAttrGetName() {
  return wrap(torch::TMTensor::DenseExternalElementsAttr::name);
}

MlirTypeID mlirDenseExternalElementsAttrGetTypeID() {
  return wrap(torch::TMTensor::DenseExternalElementsAttr::getTypeID());
}

MlirAttribute mlirDenseExternalElementsAttrGetFromBuffer(
    const MlirContext context, const MlirType shapedType,
    const MlirStringRef filename, const intptr_t size, const void *buffer) {
  const auto *ptr = static_cast<const char *>(buffer);
  return wrap(torch::TMTensor::DenseExternalElementsAttr::get(
      unwrap(context), cast<ShapedType>(unwrap(shapedType)), unwrap(filename),
      ArrayRef(ptr, size)));
}
