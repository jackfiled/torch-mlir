//===-- TorchBind.td - Torch dialect bind ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"

namespace nb = nanobind;

namespace mlir::python::torch::TMTensor {
void populateTMTensorAttributes(nb::module_ &m);
}

NB_MODULE(_torchMlir, m) {
  torchMlirRegisterAllPasses();

  m.doc() = "torch-mlir main python extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle torchHandler = mlirGetDialectHandle__torch__();
        mlirDialectHandleRegisterDialect(torchHandler, context);
        if (load) {
          mlirDialectHandleLoadDialect(torchHandler, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def(
      "register_tmtensor_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle tmTensorHandler = mlirGetDialectHandle__tmtensor__();
        mlirDialectHandleRegisterDialect(tmTensorHandler, context);
        if (load) {
          mlirDialectHandleLoadDialect(tmTensorHandler, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("get_int64_max", []() { return INT64_MAX; });

  m.def("get_int64_min", []() { return INT64_MIN; });

  mlir::python::torch::TMTensor::populateTMTensorAttributes(m);
}
