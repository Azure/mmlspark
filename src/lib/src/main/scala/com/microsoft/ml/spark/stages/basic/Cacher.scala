// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class Cacher(val uid: String) extends Transformer with DefaultParamsWritable {

  val disable = new BooleanParam(this,
    "disable", "Whether or disable caching (so that you can turn it off during evaluation)")

  def getDisable: Boolean = $(disable)

  def setDisable(value: Boolean): this.type = set(disable, value)

  setDefault(disable->false)

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (!getDisable) {
      dataset.toDF.cache()
    } else {
      dataset.toDF
    }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  def this() = this(Identifiable.randomUID("Cacher"))
}

object Cacher extends DefaultParamsReadable[Cacher]
