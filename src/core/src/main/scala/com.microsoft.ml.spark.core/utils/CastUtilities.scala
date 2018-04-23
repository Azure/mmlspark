// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.utils

import org.apache.spark.sql.catalyst.expressions.{Cast, Literal}
import org.apache.spark.sql.types.DataType

/** Utilities for casting values. */
object CastUtilities {
  /** Implicit method that casts a value to the given datatype.
    * @param any The value to cast, can be any type.
    */
  implicit class CastValue(val any: Any) extends AnyVal {
    def toDataType(dataType: DataType): Any = {
      val literal = Literal(any)
      new Cast(literal, dataType).eval()
    }
  }
}
