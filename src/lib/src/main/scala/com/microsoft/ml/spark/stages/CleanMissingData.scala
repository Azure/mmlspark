// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.{HasInputCols, HasOutputCols, MMLParams}
import com.microsoft.ml.spark.core.serialize.{ConstructorReadable, ConstructorWritable}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer
import scala.reflect.runtime.universe.{TypeTag, typeTag}

object CleanMissingData extends DefaultParamsReadable[CleanMissingData] {
  val meanOpt = "Mean"
  val medianOpt = "Median"
  val customOpt = "Custom"
  val modes = Array(meanOpt, medianOpt, customOpt)

  def validateAndTransformSchema(schema: StructType,
                                 inputCols: Array[String],
                                 outputCols: Array[String]): StructType = {
    inputCols.zip(outputCols).foldLeft(schema)((oldSchema, io) => {
      if (oldSchema.fieldNames.contains(io._2)) {
        val index = oldSchema.fieldIndex(io._2)
        val fields = oldSchema.fields
        fields(index) = oldSchema.fields(oldSchema.fieldIndex(io._1))
        StructType(fields)
      } else {
        oldSchema.add(oldSchema.fields(oldSchema.fieldIndex(io._1)))
      }
    })
  }
}

/** Removes missing values from input dataset.
  * The following modes are supported:
  *   Mean   - replaces missings with mean of fit column
  *   Median - replaces missings with approximate median of fit column
  *   Custom - replaces missings with custom value specified by user
  * For mean and median modes, only numeric column types are supported, specifically:
  *   `Int`, `Long`, `Float`, `Double`
  * For custom mode, the types above are supported and additionally:
  *   `String`, `Boolean`
  */
class CleanMissingData(override val uid: String) extends Estimator[CleanMissingDataModel]
  with HasInputCols with HasOutputCols with MMLParams {

  def this() = this(Identifiable.randomUID("CleanMissingData"))

  val cleaningMode: Param[String] = StringParam(this, "cleaningMode", "Cleaning mode", CleanMissingData.meanOpt)
  def setCleaningMode(value: String): this.type = set(cleaningMode, value)
  def getCleaningMode: String = $(cleaningMode)

  /** Custom value for imputation, supports numeric, string and boolean types.
    * Date and Timestamp currently not supported.
    */
  val customValue: Param[String] = StringParam(this, "customValue", "Custom value for replacement")
  def setCustomValue(value: String): this.type = set(customValue, value)
  def getCustomValue: String = $(customValue)

  /** Fits the dataset, prepares the transformation function.
    *
    * @param dataset The input dataset.
    * @return The model for removing missings.
    */
  override def fit(dataset: Dataset[_]): CleanMissingDataModel = {
    val replacementValues = getReplacementValues(dataset, getInputCols, getOutputCols, getCleaningMode)
    new CleanMissingDataModel(uid, replacementValues, getInputCols, getOutputCols)
  }

  override def copy(extra: ParamMap): Estimator[CleanMissingDataModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    CleanMissingData.validateAndTransformSchema(schema, getInputCols, getOutputCols)

  // Only support numeric types (no string, boolean or Date/Timestamp) for numeric imputation
  private def isSupportedTypeForImpute(dataType: DataType): Boolean = dataType.isInstanceOf[NumericType]

  private def verifyColumnsSupported(dataset: Dataset[_], colsToClean: Array[String]): Unit = {
    colsToClean.foreach(columnName =>
      if (!isSupportedTypeForImpute(dataset.schema(columnName).dataType))
        throw new UnsupportedOperationException("Only numeric types supported for numeric imputation"))
  }

  private def getReplacementValues(dataset: Dataset[_],
                                   colsToClean: Array[String],
                                   outputCols: Array[String],
                                   mode: String): Map[String, Any] = {
    import org.apache.spark.sql.functions._
    val columns = colsToClean.map(col => dataset(col))
    val metrics = getCleaningMode match {
      case CleanMissingData.meanOpt =>
        // Verify columns are supported for imputation
        verifyColumnsSupported(dataset, colsToClean)
        val row = dataset.select(columns.map(column => avg(column)): _*).collect()(0)
        rowToValues(row)
      case CleanMissingData.medianOpt =>
        // Verify columns are supported for imputation
        verifyColumnsSupported(dataset, colsToClean)
        val row = dataset.select(columns.map(column => callUDF("percentile_approx", column, lit(0.5))): _*).collect()(0)
        rowToValues(row)
      case CleanMissingData.customOpt =>
        // Note: All column types supported for custom value
        colsToClean.map(col => getCustomValue)
    }
    outputCols.zip(metrics).toMap
  }

  private def rowToValues(row: Row): Array[Double] = {
    val avgs = ListBuffer[Double]()
    for (i <- 0 until row.size) {
      avgs += row.getDouble(i)
    }
    avgs.toArray
  }
}

/** Model produced by [[CleanMissingData]]. */
class CleanMissingDataModel(val uid: String,
                            val replacementValues: Map[String, Any],
                            val inputCols: Array[String],
                            val outputCols: Array[String])
    extends Model[CleanMissingDataModel] with ConstructorWritable[CleanMissingDataModel] {

  val ttag: TypeTag[CleanMissingDataModel] = typeTag[CleanMissingDataModel]
  def objectsToSave: List[Any] = List(uid, replacementValues, inputCols, outputCols)

  override def copy(extra: ParamMap): CleanMissingDataModel =
    new CleanMissingDataModel(uid, replacementValues, inputCols, outputCols)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val datasetCols = dataset.columns.map(name => dataset(name)).toList
    val datasetInputCols = inputCols.zip(outputCols)
      .flatMap(io =>
        if (io._1 == io._2) {
          None
        } else {
          Some(dataset(io._1).as(io._2))
        }).toList
    val addedCols = dataset.select(datasetCols ::: datasetInputCols:_*)
    addedCols.na.fill(replacementValues)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    CleanMissingData.validateAndTransformSchema(schema, inputCols, outputCols)
}

object CleanMissingDataModel extends ConstructorReadable[CleanMissingDataModel]
