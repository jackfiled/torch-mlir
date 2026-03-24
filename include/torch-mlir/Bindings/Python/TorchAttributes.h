#ifndef TORCH_MLIR_TORCHATTRIBUTES_H
#define TORCH_MLIR_TORCHATTRIBUTES_H

#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "torch-mlir-c/TorchAttributes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchAttributes.h"

using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace mlir::python::torch::Torch {
class MLIR_PYTHON_API_EXPORTED PyDenseExternalElementsAttribute
    : public PyConcreteAttribute<PyDenseExternalElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsADenseExternalElements;
  static constexpr auto pyClassName = "DenseExternalElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypedIdFunction =
      mlirDenseArrayAttrGetTypeID;

  static PyDenseExternalElementsAttribute
  getFromBuffer(const nb_buffer &, const std::string &, const PyType &,
                DefaultingPyMlirContext);

  static void bindDerived(ClassTy &c);
};

MLIR_PYTHON_API_EXPORTED void populateTorchAttributes(nanobind::module_ &m);
} // namespace mlir::python::torch::Torch

#endif // TORCH_MLIR_TORCHATTRIBUTES_H
