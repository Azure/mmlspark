// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize.params

import com.microsoft.ml.spark.core.serialize.ComplexParam
import org.apache.spark.ml.param.{NamespaceInjections, Params}
import org.apache.spark.sql.execution.python.UserDefinedPythonFunction

/** Param for UserDefinedPythonFunction.  Needed as spark has explicit params for many different
  * types but not UserDefinedPythonFunction.
  */
class UDPyFParam(parent: Params, name: String, doc: String, isValid: UserDefinedPythonFunction => Boolean)
  extends ComplexParam[UserDefinedPythonFunction](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, NamespaceInjections.alwaysTrue)

}
