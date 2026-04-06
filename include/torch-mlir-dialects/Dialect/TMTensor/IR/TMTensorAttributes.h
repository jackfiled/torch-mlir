#ifndef TORCH_MLIR_TMTENSOR_ATTRIBUTES_H
#define TORCH_MLIR_TMTENSOR_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

namespace mlir::torch::TMTensor::detail {
struct DenseExternalElementsAttrStorage;
} // namespace mlir::torch::Torch::detail

#define GET_ATTRDEF_CLASSES
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorAttributes.h.inc"

namespace mlir::torch::TMTensor {
template <typename T>
DenseExternalElementsAttr
DenseExternalElementsAttr::get(MLIRContext *context, ShapedType shapedType,
                               StringRef filename, ArrayRef<T> data) {
  const auto length = data.size() * sizeof(T);
  auto buffer = appendBufferIntoFile(
      filename, reinterpret_cast<const char *>(data.data()), length);
  assert(succeeded(buffer) && "Failed to write into data file.");

  return get(context, shapedType, filename, buffer.value(), length);
}

template <typename T>
FailureOr<const T *>
DenseExternalElementsAttr::try_value_begin_impl(OverloadToken<T>) const {
  const auto buffer = loadBuffer();
  if (failed(buffer)) {
    return failure();
  }

  return reinterpret_cast<const T *>(buffer->data());
}

Attribute parseTMTensorDialectAttributes(AsmParser &parser, Type type);

void printTMTensorDialectAttributes(Attribute attr, AsmPrinter &printer);

} // namespace mlir::torch::Torch

#endif // TORCH_MLIR_TMTENSOR_ATTRIBUTES_H
