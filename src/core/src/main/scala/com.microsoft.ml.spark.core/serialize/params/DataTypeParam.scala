// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize.params

import com.microsoft.ml.spark.core.serialize.ComplexParam
import org.apache.spark.ml.param.{NamespaceInjections, Params}
import org.apache.spark.sql.types.DataType

/** Param for DataType */
class DataTypeParam(parent: Params, name: String, doc: String, isValid: DataType => Boolean)
    extends ComplexParam[DataType](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, NamespaceInjections.alwaysTrue)

}
