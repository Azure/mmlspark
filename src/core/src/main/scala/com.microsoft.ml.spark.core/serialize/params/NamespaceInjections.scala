// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.param

object NamespaceInjections {
  val alwaysTrue: Any => Boolean = ParamValidators.alwaysTrue
}
