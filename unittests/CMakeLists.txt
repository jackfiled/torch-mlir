add_custom_target(TorchMLIRUnitTest)

function(add_torch_mlir_unittest test_dirname)
    add_unittest(TorchMLIRUnitTest ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)