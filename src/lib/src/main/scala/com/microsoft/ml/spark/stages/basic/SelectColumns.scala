// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.contracts.MMLParams
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

object SelectColumns extends DefaultParamsReadable[SelectColumns]

/** <code>SelectColumns</code> takes a dataframe and a list of columns to select as input and returns
  * a dataframe comprised of only those columns listed in the input list.
  *
  * The columns to be selected is a list of column names
  */

class SelectColumns(val uid: String) extends Transformer with MMLParams {
  def this() = this(Identifiable.randomUID("SelectColumns"))

  val cols: StringArrayParam = new StringArrayParam(this, "cols", "Comma separated list of selected column names")

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
    dataset.toDF().select(getCols.map(col): _*)
  }

  def transformSchema(schema: StructType): StructType = {
    verifySchema(schema)
    val selectedCols = getCols.toSet
    StructType(schema.fields.filter(f => selectedCols(f.name)))
  }

  def copy(extra: ParamMap): SelectColumns = defaultCopy(extra)

  private def verifySchema(schema: StructType): Unit = {
    val providedCols = schema.fields.map(_.name).toSet
    val invalidCols = getCols.filter(!providedCols(_))

    if (invalidCols.length > 0) {
      throw new NoSuchElementException(
        s"DataFrame does not contain specified columns: ${invalidCols.reduce(_ + "," + _)}")
    }

  }

}
