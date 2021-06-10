// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.explainers

import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector => SV, Vectors => SVS}
import org.apache.spark.ml.param._
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

trait HasMetricsCol extends Params {
  val metricsCol = new Param[String](
    this,
    "metricsCol",
    "Column name for fitting metrics"
  )

  def getMetricsCol: String = $(metricsCol)

  def setMetricsCol(v: String): this.type = this.set(metricsCol, v)
}

trait HasModel extends Params {
  val model = new TransformerParam(this, "model", "The model to be interpreted.")

  def getModel: Transformer = $(model)

  def setModel(v: Transformer): this.type = set(model, v)
}

trait HasNumSamples extends Params {
  final val numSamples: IntParam = new IntParam(
    this,
    "numSamples",
    "Number of samples to generate.",
    ParamValidators.gt(0)
  )

  final def getNumSamples: Int = $(numSamples)

  final def getNumSamplesOpt: Option[Int] = this.get(numSamples)

  final def setNumSamples(value: Int): this.type = this.set(numSamples, value)
}

trait HasSamplingFraction extends Params {
  val samplingFraction = new DoubleParam(
    this,
    "samplingFraction",
    "The fraction of superpixels (for image) or tokens (for text) to keep on",
    ParamValidators.inRange(0, 1)
  )

  def getSamplingFraction: Double = $(samplingFraction)

  def setSamplingFraction(d: Double): this.type = set(samplingFraction, d)
}

trait HasBackgroundData extends Params {
  val backgroundData: DataFrameParam = new DataFrameParam(
    this,
    "backgroundData",
    "A dataframe containing background data"
  )

  def getBackgroundData: DataFrame = $(backgroundData)

  def setBackgroundData(value: DataFrame): this.type = set(backgroundData, value)
}

trait HasExplainTarget extends Params {
  final val targetCol: Param[String] = new Param[String](
    this,
    "targetCol",
    "The column name of the prediction target to explain (i.e. the response variable). " +
      "This is usually set to \"prediction\" for regression models and " +
      "\"probability\" for probabilistic classification models. Default value: probability")
  setDefault(targetCol, "probability")

  final def getTargetCol: String = $(targetCol)
  final def setTargetCol(value: String): this.type = this.set(targetCol, value)

  final val targetClasses: IntArrayParam = new IntArrayParam(
    this,
    "targetClasses",
    "The indices of the classes for multinomial classification models. Default: 0." +
      "For regression models this parameter is ignored."
  )

  final def getTargetClasses: Array[Int] = $(targetClasses)
  final def setTargetClasses(values: Array[Int]): this.type = this.set(targetClasses, values)

  final val targetClassesCol: Param[String] = new Param[String](
    this,
    "targetClassesCol",
    "The name of the column that specifies the indices of the classes for multinomial classification models."
  )

  final def getTargetClassesCol: String = $(targetClassesCol)
  final def setTargetClassesCol(value: String): this.type = this.set(targetClassesCol, value)

  private def slice[T](values: Int => T, indices: Seq[Int])(num: Numeric[_]): SV = {
    val n = num.asInstanceOf[Numeric[T]]
    SVS.dense(indices.map(values.apply).map(n.toDouble).toArray)
  }

  private val dataTypeToNumericMap: Map[NumericType, Numeric[_]] = Map(
    FloatType -> implicitly[Numeric[Float]],
    DoubleType -> implicitly[Numeric[Double]],
    ByteType -> implicitly[Numeric[Byte]],
    ShortType -> implicitly[Numeric[Short]],
    IntegerType -> implicitly[Numeric[Int]],
    LongType -> implicitly[Numeric[Long]]
  )

  def getExplainTarget(schema: StructType): Column = {
    val toVector = UDFUtils.oldUdf(
      (values: Seq[Double]) => SVS.dense(values.toArray),
      VectorType
    )

    val explainTarget = schema(getTargetCol).dataType match {
      case _: NumericType =>
        toVector(array(col(getTargetCol)))
      case VectorType =>
        val vectorSlicer = UDFUtils.oldUdf(
          (v: SV, indices: Seq[Int]) => slice(v.apply, indices)(implicitly[Numeric[Double]]),
          VectorType
        )

        val classesCol = this.get(targetClassesCol).map(col).getOrElse(lit(getTargetClasses))
        vectorSlicer(col(getTargetCol), classesCol)
      case ArrayType(et: NumericType, _) =>
        val arraySlicer = UDFUtils.oldUdf(
          (v: Seq[Any], indices: Seq[Int]) => slice(v.apply, indices)(dataTypeToNumericMap(et)),
          VectorType
        )

        val classesCol = this.get(targetClassesCol).map(col).getOrElse(lit(getTargetClasses))
        arraySlicer(col(getTargetCol), classesCol)
      case MapType(_: IntegerType, et: NumericType, _) =>
        val mapSlicer = UDFUtils.oldUdf(
          (m: Map[Int, Any], indices: Seq[Int]) => slice(m.apply, indices)(dataTypeToNumericMap(et)),
          VectorType
        )

        val classesCol = this.get(targetClassesCol).map(col).getOrElse(lit(getTargetClasses))
        mapSlicer(col(getTargetCol), classesCol)
      case other =>
        throw new IllegalArgumentException(
          s"Only numeric types, vector type, array of numeric types and map types with numeric value type " +
            s"are supported as target column. The current type is $other."
        )
    }

    explainTarget
  }
}
