#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "gtest/gtest.h"
#include <filesystem>
#include <mlir/Parser/Parser.h>
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

  const std::string filename =
      "torch_test_data_" + std::to_string(distribution(generator)) + ".data";
  return tempDir / filename;
}
} // namespace

TEST(TorchAttributesTests, CreateTest) {
  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<TorchDialect>();

  std::string filename = generateUniqueTempFile();

  auto floatType = Float32Type::get(&context);
  auto shapedType = MemRefType::get(ArrayRef<int64_t>({10}), floatType);
  std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto denseAttr = DenseExternalElementsAttr::get(&context, shapedType,
                                                  filename, ArrayRef<float>{data});

  auto elementsAttr = dyn_cast<ElementsAttr>(denseAttr);

  ASSERT_TRUE(elementsAttr);
  ASSERT_EQ(10, elementsAttr.getNumElements());
  ASSERT_FALSE(elementsAttr.isSplat());
  ASSERT_EQ(0, denseAttr.getOffset());
  ASSERT_EQ(40, denseAttr.getLength());

  for (const auto [i, value] : enumerate(elementsAttr.getValues<float>())) {
    ASSERT_FLOAT_EQ(i + 1, value);
  }
}

TEST(TorchAttributesTests, DumpTest) {
  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<TorchDialect>();
  OpBuilder builder{&context};

  auto module = ModuleOp::create(builder, UnknownLoc::get(&context));

  builder.setInsertionPointToStart(module.getBody());
  auto returnType = ValueTensorType::get(&context, ArrayRef<int64_t>{10},
                                         cast<Type>(builder.getF32Type()));
  auto funcType = FunctionType::get(&context, {}, {returnType});

  auto funcOp =
      func::FuncOp::create(UnknownLoc::get(&context), "test", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  std::string filename = "/home/ricardo/Documents/Code/Cpp/chameleon/"
                         "third-party/torch-mlir/test.data";
  auto tensorType = RankedTensorType::get(ArrayRef<int64_t>{10},
                                          cast<Type>(builder.getF32Type()));
  std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto denseAttr = DenseExternalElementsAttr::get(&context, tensorType,
                                                  filename, ArrayRef<float>{data});

  auto literalOpType = ValueTensorType::get(&context, ArrayRef<int64_t>{10},
                                            cast<Type>(builder.getF32Type()));
  auto op = ValueTensorLiteralOp::create(builder, UnknownLoc::get(&context),
                                         literalOpType,
                                         cast<ElementsAttr>(denseAttr));

  func::ReturnOp::create(builder, UnknownLoc::get(&context), op.getResult());

  module.push_back(funcOp);

  module.dump();
}

TEST(TorchAttributesTests, ParseTest) {
  const std::string input = R"(
module {
  func.func @test() -> !torch.vtensor<[10],f32> {
    %0 = torch.vtensor.literal(#torch.external<"/home/ricardo/Documents/Code/Cpp/chameleon/third-party/torch-mlir/test.data" : 0 + 40> : tensor<10xf32>) : !torch.vtensor<[10],f32>
    return %0 : !torch.vtensor<[10],f32>
  }
}
)";

  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<TorchDialect>();
  ParserConfig config{&context};

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(input, config);

  ASSERT_TRUE(module);
  module->print(outs());

  module->walk([](ValueTensorLiteralOp op) {
    auto attr = op.getValueAttr();
    ASSERT_EQ(10, attr.getNumElements());

    for (const auto [i, value] : enumerate(attr.getValues<float>())) {
      ASSERT_FLOAT_EQ(i + 1, value);
    }
  });
}
