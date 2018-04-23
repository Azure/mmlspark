// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize.params

import com.microsoft.ml.spark.core.serialize.ComplexParam
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{NamespaceInjections, Params}

/** Param for Transformer.  Needed as spark has explicit params for many different
  * types but not Transformer.
  */
class TransformerParam(parent: Params, name: String, doc: String, isValid: Transformer => Boolean)
  extends ComplexParam[Transformer](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, NamespaceInjections.alwaysTrue)

}
