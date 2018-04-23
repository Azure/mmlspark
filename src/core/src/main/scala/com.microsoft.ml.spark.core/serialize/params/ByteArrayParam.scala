// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize.params

import com.microsoft.ml.spark.core.serialize.ComplexParam
import org.apache.spark.ml.param.{NamespaceInjections, Params}

/** Param for ByteArray.  Needed as spark has explicit params for many different
  * types but not ByteArray.
  */
class ByteArrayParam(parent: Params, name: String, doc: String, isValid: Array[Byte] => Boolean)
    extends ComplexParam[Array[Byte]](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, NamespaceInjections.alwaysTrue)

}
