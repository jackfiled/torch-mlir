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

    using ContiguousIterableTypesT = std::tuple<float, double>;

    template <typename T>
    static DenseExternalElementsAttr get(::mlir::MLIRContext *, ::mlir::ShapedType, ::llvm::StringRef, ::llvm::ArrayRef<T>);

    template <typename T>
    ::mlir::FailureOr<const T *> try_value_begin_impl(OverloadToken<T>) const;

  private:
    ::mlir::FailureOr<::llvm::ArrayRef<char>> loadBuffer() const;

    static ::mlir::FailureOr<int64_t> appendBufferIntoFile(::llvm::StringRef, const char *, int64_t);

  public:
  }];

}

#endif // TORCH_ATTRIBUTES_TD
