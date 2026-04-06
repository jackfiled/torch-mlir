#ifndef TORCH_MLIR_TMTENSORATTRIBUTES_H
#define TORCH_MLIR_TMTENSORATTRIBUTES_H

#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "torch-mlir-c/TMTensorAttributes.h"

using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace mlir::python::torch::TMTensor {
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

MLIR_PYTHON_API_EXPORTED void populateTMTensorAttributes(nanobind::module_ &m);
} // namespace mlir::python::torch::TMTensor

#endif // TORCH_MLIR_TMTENSORATTRIBUTES_H
