add_mlir_library(TorchMLIRTorchDialect
        TorchDialect.cpp
        TorchOps.cpp
        TorchOpsODSGenerated.cpp
        TorchTypes.cpp
        UtilsForODSGenerated.cpp
        TorchAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/Torch

        DEPENDS
        MLIRTorchOpsIncGen
        MLIRTorchTypesIncGen
        MLIRTorchAttributesIncGen

        LINK_LIBS PUBLIC
        MLIRBytecodeOpInterface
        MLIRBytecodeReader
        MLIRBytecodeWriter
        MLIRFuncDialect
        MLIRIR
        MLIRSupport
        MLIRControlFlowInterfaces
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
)

torch_mlir_target_includes(TorchMLIRTorchDialect)
