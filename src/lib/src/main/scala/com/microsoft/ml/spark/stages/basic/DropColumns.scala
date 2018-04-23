// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.contracts.MMLParams
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

object DropColumns extends DefaultParamsReadable[DropColumns]

/** <code>DropColumns</code> takes a dataframe and a list of columns to drop as input and returns
  * a dataframe comprised of only those columns not listed in the input list.
  *
  */

class DropColumns(val uid: String) extends Transformer with MMLParams {
  def this() = this(Identifiable.randomUID("DropColumns"))

  val cols: StringArrayParam = new StringArrayParam(this, "cols", "Comma separated list of column names")

  /** @group getParam */
  final def getCols: Array[String] = $(cols)

  /** @group setParam */
  def setCols(value: Array[String]): this.type = set(cols, value)

  def setCol(value: String): this.type = set(cols, Array(value))

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from column selection
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    verifySchema(dataset.schema)
    dataset.toDF().drop(getCols: _*)
  }

  def transformSchema(schema: StructType): StructType = {
    verifySchema(schema)
    val droppedCols = getCols.toSet
    StructType(schema.fields.filter(f => !droppedCols(f.name)))
  }

  def copy(extra: ParamMap): DropColumns = defaultCopy(extra)

  private def verifySchema(schema: StructType): Unit = {
    val providedCols = schema.fields.map(_.name).toSet
    val invalidCols = getCols.filter(!providedCols(_))

    if (invalidCols.length > 0) {
      throw new NoSuchElementException(
        s"DataFrame does not contain specified columns: ${invalidCols.reduce(_ + "," + _)}")
    }

  }

}
