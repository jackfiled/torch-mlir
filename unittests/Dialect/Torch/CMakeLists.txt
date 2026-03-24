add_torch_mlir_unittest(TorchAttributesTests
  TorchAttributesTest.cpp
)

mlir_target_link_libraries(TorchAttributesTests
  PRIVATE
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  MLIRParser
  MLIRSparseTensorDialect
)