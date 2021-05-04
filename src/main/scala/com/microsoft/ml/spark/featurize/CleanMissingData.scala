// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.featurize

import com.microsoft.ml.spark.codegen.Wrappable
import com.microsoft.ml.spark.core.contracts.{HasInputCols, HasOutputCols}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

object CleanMissingData extends DefaultParamsReadable[CleanMissingData] {
  val MeanOpt = "Mean"
  val MedianOpt = "Median"
  val CustomOpt = "Custom"
  val Modes: Array[String] = Array(MeanOpt, MedianOpt, CustomOpt)

  def validateAndTransformSchema(schema: StructType,
                                 inputCols: Array[String],
                                 outputCols: Array[String]): StructType = {
    inputCols.zip(outputCols).foldLeft(schema)((oldSchema, io) => {
      if (oldSchema.fieldNames.contains(io._2)) {
        val index = oldSchema.fieldIndex(io._2)
        val fields = oldSchema.fields
        fields(index) = StructField(io._1, oldSchema.fields(oldSchema.fieldIndex(io._1)).dataType)
        StructType(fields)
      } else {
        oldSchema.add(io._2, oldSchema.fields(oldSchema.fieldIndex(io._1)).dataType)
      }
    })
  }
}

/** Removes missing values from input dataset.
  * The following modes are supported:
  * Mean   - replaces missings with mean of fit column
  * Median - replaces missings with approximate median of fit column
  * Custom - replaces missings with custom value specified by user
  * For mean and median modes, only numeric column types are supported, specifically:
  * `Int`, `Long`, `Float`, `Double`
  * For custom mode, the types above are supported and additionally:
  * `String`, `Boolean`
  */
class CleanMissingData(override val uid: String) extends Estimator[CleanMissingDataModel]
  with HasInputCols with HasOutputCols with Wrappable with DefaultParamsWritable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("CleanMissingData"))

  val cleaningMode: Param[String] = new Param[String](this, "cleaningMode", "Cleaning mode")
  setDefault(cleaningMode -> CleanMissingData.MeanOpt)

  def setCleaningMode(value: String): this.type = set(cleaningMode, value)

  def getCleaningMode: String = $(cleaningMode)

  /** Custom value for imputation, supports numeric, string and boolean types.
    * Date and Timestamp currently not supported.
    */
  val customValue: Param[String] = new Param[String](this, "customValue", "Custom value for replacement")

  def setCustomValue(value: String): this.type = set(customValue, value)

  def getCustomValue: String = $(customValue)

  /** Fits the dataset, prepares the transformation function.
    *
    * @param dataset The input dataset.
    * @return The model for removing missings.
    */
  override def fit(dataset: Dataset[_]): CleanMissingDataModel = {
    logFit({
      val (colsToFill, fillValues) = getReplacementValues(
        dataset, getInputCols, getOutputCols, getCleaningMode).toSeq.unzip
      new CleanMissingDataModel(uid)
        .setColsToFill(colsToFill.toArray)
        .setFillValues(fillValues.toArray)
        .setInputCols(getInputCols)
        .setOutputCols(getOutputCols)
    })
  }

  override def copy(extra: ParamMap): Estimator[CleanMissingDataModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    CleanMissingData.validateAndTransformSchema(schema, getInputCols, getOutputCols)

  // Only support numeric types (no string, boolean or Date/Timestamp) for numeric imputation
  private def isSupportedTypeForImpute(dataType: DataType): Boolean = dataType.isInstanceOf[NumericType]

  private def verifyColumnsSupported(dataset: Dataset[_], colsToClean: Array[String]): Unit = {
    if (colsToClean.exists(name => !isSupportedTypeForImpute(dataset.schema(name).dataType)))
      throw new UnsupportedOperationException("Only numeric types supported for numeric imputation")
  }

  private def rowToValues(row: Row): Array[Double] = {
    (0 until row.size).map { i =>
      row.get(i) match {
        case n: Double => n
        case n: Int => n.toDouble
        case _ => throw new UnsupportedOperationException("Unknown type in row")
      }
    }.toArray
  }

  private def getReplacementValues(dataset: Dataset[_],
                                   colsToClean: Array[String],
                                   outputCols: Array[String],
                                   mode: String): Map[String, Any] = {
    import org.apache.spark.sql.functions._
    val columns = colsToClean.map(col => dataset(col))
    val metrics = getCleaningMode match {
      case CleanMissingData.MeanOpt =>
        // Verify columns are supported for imputation
        verifyColumnsSupported(dataset, colsToClean)
        val row = dataset.select(columns.map(column => avg(column)): _*).collect()(0)
        rowToValues(row)
      case CleanMissingData.MedianOpt =>
        // Verify columns are supported for imputation
        verifyColumnsSupported(dataset, colsToClean)
        val row =
          dataset.select(columns.map(column => callUDF("percentile_approx",
            column, lit(0.5))): _*)
            .collect()(0)
        rowToValues(row)
      case CleanMissingData.CustomOpt =>
        // Note: All column types supported for custom value
        colsToClean.map(_ => getCustomValue)
    }
    outputCols.zip(metrics).toMap
  }

}

/** Model produced by [[CleanMissingData]]. */
class CleanMissingDataModel(val uid: String)
  extends Model[CleanMissingDataModel] with ComplexParamsWritable with Wrappable
    with HasInputCols with HasOutputCols with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("CleanMissingDataModel"))

  val colsToFill = new StringArrayParam(
    this, "colsToFill", "The columns to fill with")
  val fillValues = new UntypedArrayParam(
    this, "fillValues", "what to replace in the columns")

  def getColsToFill: Array[String] = $(colsToFill)

  def getFillValues: Array[Any] = $(fillValues)

  def setColsToFill(v: Array[String]): this.type = set(colsToFill, v)

  def setFillValues(v: Array[Any]): this.type = set(fillValues, v)

  override def copy(extra: ParamMap): CleanMissingDataModel = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val datasetCols = dataset.columns.map(name => dataset(name)).toList
      val datasetInputCols = getInputCols.zip(getOutputCols)
        .flatMap(io =>
          if (io._1 == io._2) {
            None
          } else {
            Some(dataset(io._1).as(io._2))
          }).toList
      val addedCols = dataset.select(datasetCols ::: datasetInputCols: _*)
      addedCols.na.fill(getColsToFill.zip(getFillValues).toMap)
    })
  }

  override def transformSchema(schema: StructType): StructType =
    CleanMissingData.validateAndTransformSchema(schema, getInputCols, getOutputCols)
}

object CleanMissingDataModel extends ComplexParamsReadable[CleanMissingDataModel]
