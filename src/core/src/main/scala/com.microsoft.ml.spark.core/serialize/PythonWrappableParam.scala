// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize

import org.apache.spark.ml.param.Param
trait PythonWrappableParam[T] extends Param[T]{

  def pythonValueEncoding(scalaValue: T): String

}
