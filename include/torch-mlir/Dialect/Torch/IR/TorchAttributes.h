#ifndef TORCH_MLIR_TORCHATTRIBUTES_H
#define TORCH_MLIR_TORCHATTRIBUTES_H

#define DEBUG_TYPE "torch-mlir-torch-attributes"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

namespace mlir::torch::Torch::detail {
struct DenseExternalElementsAttrStorage;
} // namespace mlir::torch::Torch::detail

#define GET_ATTRDEF_CLASSES
#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.h.inc"

namespace mlir::torch::Torch {
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

Attribute parseTorchDialectAttributes(AsmParser &parser, Type type);

void printTorchDialectAttributes(Attribute attr, AsmPrinter &printer);

} // namespace mlir::torch::Torch

#endif // TORCH_MLIR_TORCHATTRIBUTES_H
