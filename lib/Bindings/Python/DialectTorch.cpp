#include "torch-mlir/Bindings/Python/TorchAttributes.h"

#include "mlir-c/BuiltinTypes.h"
#include "llvm/ADT/ScopeExit.h"

#include <llvm/IR/Comdat.h>

namespace nb = nanobind;

namespace mlir::python::torch::Torch {

PyDenseExternalElementsAttribute
PyDenseExternalElementsAttribute::getFromBuffer(
    const nb_buffer &buffer, const std::string &filename,
    const PyType &shapedType, DefaultingPyMlirContext context) {
  if (!mlirTypeIsAShaped(shapedType)) {
    throw std::invalid_argument(
        "Constructing a DenseExternalElementsAttr requires a ShapedType.");
  }

  const int flags = PyBUF_STRIDES;
  Py_buffer bufferView;
  if (PyObject_GetBuffer(buffer.ptr(), &bufferView, flags) != 0) {
    throw nb::python_error{};
  }

  llvm::scope_exit freeBuffer([&] { PyBuffer_Release(&bufferView); });

  if (PyBuffer_IsContiguous(&bufferView, 'A') == 0) {
    throw std::invalid_argument("Contigous buffer is required.");
  }

  MlirAttribute attr = mlirDenseExternalElementsAttrGetFromBuffer(
      context->get(), shapedType, toMlirStringRef(filename), bufferView.len,
      bufferView.buf);
  if (mlirAttributeIsNull(attr)) {
    throw std::invalid_argument("DenseExternalElementsAttr cloud not be "
                                "constructed from the given buffer.");
  }

  return {context->getRef(), attr};
}

void PyDenseExternalElementsAttribute::bindDerived(ClassTy &c) {
  c.def_static("get", getFromBuffer, nb::arg("array"), nb::arg("filename"),
               nb::arg("shapedType"), nb::arg("context") = nb::none());
}

void populateTorchAttributes(nanobind::module_ &m) {
  PyDenseExternalElementsAttribute::bind(m);
}
} // namespace mlir::python::torch::Torch
