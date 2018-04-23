// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.udfs

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable

object udfs {

  def get_value_at(colname: String, i: Int): Column = {
    udf({
      vec: org.apache.spark.ml.linalg.Vector => vec(i)
    }, DoubleType)(col(colname))
  }

  val to_vector: UserDefinedFunction = udf({
    arr: mutable.WrappedArray[Double] => Vectors.dense(arr.toArray)
  }, VectorType)

  def to_vector(colName: String): Column = to_vector(col(colName))

}
