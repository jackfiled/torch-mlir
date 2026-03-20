#ifndef TORCH_MLIR_TORCHATTRIBUTES_H
#define TORCH_MLIR_TORCHATTRIBUTES_H

#define DEBUG_TYPE "torch-mlir-torch-attributes"

#include <fstream>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/Support/Debug.h"

namespace mlir::torch::Torch {

struct BufferFileInfo {
  int64_t length;
  int64_t offset;
};

namespace detail {
struct DenseExternalElementsAttrStorage;
}
} // namespace mlir::torch::Torch

#define GET_ATTRDEF_CLASSES
#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.h.inc"

namespace mlir::torch::Torch {
template <typename T>
FailureOr<BufferFileInfo>
DenseExternalElementsAttr::appendBufferToFile(StringRef filename,
                                              ArrayRef<T> array) {
  std::fstream file{filename.data(), std::ios::binary | std::ios::in |
                                         std::ios::out | std::ios::ate};
  if (!file.is_open()) {
    // If file not existed
    file.open(filename.data(),
              std::ios::binary | std::ios::out | std::ios::trunc);

    if (!file.is_open()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to create file: " << filename);
      return failure();
    }
  }

  const int64_t offset = file.tellp();
  const int64_t length = array.size() * sizeof(T);

  file.write(reinterpret_cast<const char *>(array.data()), length);
  if (!file) {
    LLVM_DEBUG(llvm::dbgs() << "failed to write into file: " << filename);
    return failure();
  }

  file.flush();
  return BufferFileInfo{length, offset};
}
} // namespace mlir::torch::Torch

#endif // TORCH_MLIR_TORCHATTRIBUTES_H
