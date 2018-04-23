// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize

import scala.reflect.runtime.universe.{TypeTag, typeTag}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.SparkSession

abstract class ComplexParam[T: TypeTag](parent: Params, name: String, doc: String, isValid: T => Boolean)
  extends Param[T](parent, name, doc, isValid) {

  def ttag: TypeTag[T] = typeTag[T]

  def save(obj: T, sparkSession: SparkSession, path: Path, overwrite: Boolean): Unit = {
    Serializer.typeToSerializer[T](ttag.tpe, sparkSession).write(obj, path, overwrite)
  }

  def load(sparkSession: SparkSession, path: Path): T = {
    Serializer.typeToSerializer[T](ttag.tpe, sparkSession).read(path)
  }

  override def jsonEncode(value: T): String = {
    throw new NotImplementedError("The parameter is a ComplexParam and cannot be JSON encoded.")
  }

  override def jsonDecode(json: String): T = {
    throw new NotImplementedError("The parameter is a ComplexParam and cannot be JSON decoded.")
  }

}
