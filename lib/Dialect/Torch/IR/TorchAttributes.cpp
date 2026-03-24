#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <fstream>

using namespace llvm;
using namespace mlir;

namespace {

FailureOr<std::vector<char>> readArray(const StringRef filename,
                                       const int64_t offset,
                                       const int64_t length) {
  auto result = std::vector<char>{};
  result.reserve(length);

  std::ifstream file(filename.data(), std::ios::binary);

  if (!file.is_open()) {
    LLVM_DEBUG(dbgs() << "Failed to open data file: " << filename);
    return failure();
  }

  file.seekg(offset, std::ios::beg);
  if (!file) {
    LLVM_DEBUG(dbgs() << "Failed to seek data file: " << filename);
    return failure();
  }

  file.read(result.data(), length);
  if (!file) {
    LLVM_DEBUG(dbgs() << "Failed to read data file: " << filename);
    return failure();
  }

  return result;
}

template <typename T> struct DenseExternalAttributeUtil;

template <> struct DenseExternalAttributeUtil<float> {
  static bool checkElementType(const Type &elementType) {
    return elementType.isF32();
  }
};

template <> struct DenseExternalAttributeUtil<double> {
  static bool checkElementType(const Type &elementType) {
    return elementType.isF64();
  }
};
} // namespace

#define GET_ATTRDEF_CLASSES
#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.cpp.inc"

namespace mlir::torch::Torch {
namespace detail {
struct DenseExternalElementsAttrStorage : AttributeStorage {
  using KeyTy = std::tuple<ShapedType, StringRef, int64_t, int64_t>;
  DenseExternalElementsAttrStorage(ShapedType type, StringRef filename,
                                   int64_t offset, int64_t length)
      : type(type), filename(filename), offset(offset), length(length) {}

  KeyTy getAsKey() const { return KeyTy{type, filename, offset, length}; }

  bool operator==(const KeyTy &tblgenKey) const {
    return type == llvm::cast<Type>(std::get<0>(tblgenKey)) &&
           filename == std::get<1>(tblgenKey) &&
           offset == std::get<2>(tblgenKey) && length == std::get<3>(tblgenKey);
  }

  static hash_code hashKey(const KeyTy &tblgenKey) {
    return hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey),
                        std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static DenseExternalElementsAttrStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::get<0>(tblgenKey);
    auto filename = std::get<1>(tblgenKey);
    auto offset = std::get<2>(tblgenKey);
    auto length = std::get<3>(tblgenKey);
    filename = allocator.copyInto(filename);
    return new (allocator.allocate<DenseExternalElementsAttrStorage>())
        DenseExternalElementsAttrStorage(type, filename, offset, length);
  }

  ShapedType type;
  StringRef filename;
  int64_t offset;
  int64_t length;
  std::optional<std::vector<char>> buffer;
};
} // namespace detail

bool DenseExternalElementsAttr::isSplat() { return false; }

ShapedType DenseExternalElementsAttr::getType() const {
  return getImpl()->type;
}

StringRef DenseExternalElementsAttr::getFilename() const {
  return getImpl()->filename;
}

int64_t DenseExternalElementsAttr::getOffset() const {
  return getImpl()->offset;
}

int64_t DenseExternalElementsAttr::getLength() const {
  return getImpl()->length;
}

FailureOr<ArrayRef<char>> DenseExternalElementsAttr::loadBuffer() const {
  if (getImpl()->buffer) {
    return ArrayRef<char>{getImpl()->buffer.value()};
  }

  auto buffer = readArray(getFilename(), getOffset(), getLength());

  if (failed(buffer)) {
    return failure();
  }

  getImpl()->buffer = std::move(buffer);
  return ArrayRef<char>{getImpl()->buffer.value()};
}

FailureOr<int64_t> DenseExternalElementsAttr::appendBufferIntoFile(
    StringRef filename, const char *data, int64_t length) {
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

  file.write(data, length);
  if (!file) {
    LLVM_DEBUG(llvm::dbgs() << "failed to write into file: " << filename);
    return failure();
  }

  file.flush();
  return offset;
}

void TorchDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.cpp.inc"

      >();
}

Attribute parseTorchDialectAttributes(AsmParser &parser, Type type) {
  const auto currentLocation = parser.getCurrentLocation();

  StringRef mnemonic;
  Attribute result;
  auto parsedResult = generatedAttributeParser(parser, &mnemonic, type, result);
  if (parsedResult.has_value()) {
    return result;
  }

  parser.emitError(currentLocation)
      << "Failed to parse attribute '" << mnemonic << "' in dialect '"
      << TorchDialect::getDialectNamespace() << "'";
  return {};
}

void printTorchDialectAttributes(Attribute attr, AsmPrinter &printer) {
  if (failed(generatedAttributePrinter(attr, printer))) {
    LLVM_DEBUG(dbgs() << "Failed to print attributes of torch dialect.");
  }
}

} // namespace mlir::torch::Torch
