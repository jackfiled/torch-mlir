//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_ATTRIBUTES_TD
#define TORCH_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchBase.td"

class TorchDialect_Attr<string name, string attrMnemonic,
  list<Trait> traits = []> : AttrDef<Torch_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Torch_DenseExternalElementsAttr : TorchDialect_Attr<"DenseExternalElements", "external",
  [TypedAttrInterface, ElementsAttrInterface]> {
  let summary = "An attribute containing a dense multi-dimensional array storaging in external file.";

  let description = [{

  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "ShapedType">:$type,
    StringRefParameter<"">:$filename, "int64_t":$offset, "int64_t":$length);

  let assemblyFormat = "`<` $filename `:` $offset `+` $length `>`";

  let genStorageClass = 0;

  let extraClassDeclaration = [{
    static bool isSplat();

    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> getValuesImpl(::mlir::TypeID);

    template <typename T>
    static ::mlir::FailureOr<::mlir::torch::Torch::BufferFileInfo> appendBufferToFile(::llvm::StringRef, ::llvm::ArrayRef<T>);
  }];

}

#endif // TORCH_ATTRIBUTES_TD
