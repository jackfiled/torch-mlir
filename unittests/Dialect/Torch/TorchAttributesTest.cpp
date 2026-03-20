#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "gtest/gtest.h"
#include <filesystem>
#include <random>

using namespace llvm;
using namespace mlir;
using namespace mlir::torch::Torch;
namespace fs = std::filesystem;

namespace {
fs::path generateUniqueTempFile() {
  const fs::path tempDir = fs::temp_directory_path();
  std::random_device dev;
  std::mt19937 generator(dev());
  std::uniform_int_distribution<int> distribution{100000, 999999};

  const std::string filename = "torch_test_data_" + std::to_string(distribution(generator)) + ".data";
  return tempDir / filename;
}
}

TEST(TorchAttributesTests, CreateTest) {
  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<TorchDialect>();

  std::string filename = generateUniqueTempFile();

  auto floatType = Float32Type::get(&context);
  auto shapedType = MemRefType::get(ArrayRef<int64_t>({10}), floatType);
  std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const auto result =
      DenseExternalElementsAttr::appendBufferToFile(filename, ArrayRef(data));
  ASSERT_TRUE(succeeded(result));
  auto denseAttr = DenseExternalElementsAttr::get(
      &context, shapedType, filename, result->offset, result->length);

  auto elementsAttr = dyn_cast<ElementsAttr>(denseAttr);

  ASSERT_TRUE(elementsAttr);
  ASSERT_EQ(10, elementsAttr.getNumElements());
  ASSERT_FALSE(elementsAttr.isSplat());
  ASSERT_EQ(0, denseAttr.getOffset());
  ASSERT_EQ(40, denseAttr.getLength());

  auto it = elementsAttr.getValuesImpl(Float32Type::getTypeID());
  ASSERT_TRUE(succeeded(it));

  for (int64_t i = 0; i < 10; ++i) {
    ASSERT_FLOAT_EQ(i + 1, it->at<float>(i));
  }
}