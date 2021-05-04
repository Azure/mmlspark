// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.downloader.FaultToleranceUtils
import com.microsoft.ml.spark.io.binary.BinaryFileFormat
import org.apache.spark.ml.param.{ByteArrayParam, DataFrameParam, Param}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{col, struct, udf}
import org.apache.spark.sql.types.StructType
import org.vowpalwabbit.spark.{VowpalWabbitArguments, VowpalWabbitExample, VowpalWabbitNative}
import org.vowpalwabbit.spark.prediction.ScalarPrediction

import scala.io.Source

case class PathAndData(path: String, bytes: Array[Byte])

/**
  * Base trait to wrap the model for prediction.
  */
trait VowpalWabbitBaseModel extends org.apache.spark.ml.param.shared.HasFeaturesCol
  with org.apache.spark.ml.param.shared.HasRawPredictionCol
  with HasAdditionalFeatures {
  @transient
  lazy val vw: VowpalWabbitNative = FaultToleranceUtils.retryWithTimeout() {
    new VowpalWabbitNative(s"--testonly ${getTestArgs}", getModel)
  }

  @transient
  lazy val example: VowpalWabbitExample = vw.createExample()

  @transient
  lazy val vwArgs: VowpalWabbitArguments = vw.getArguments

  val testArgs = new Param[String](this, "testArgs", "Additional arguments passed to VW at test time")
  setDefault(testArgs -> "")

  def getTestArgs: String = $(testArgs)

  def setTestArgs(value: String): this.type = set(testArgs, value)

  protected def transformImplInternal(dataset: Dataset[_]): DataFrame = {
    // select the columns we care for
    val featureColumnNames = Seq(getFeaturesCol) ++ getAdditionalFeatures
    val featureColumns = dataset.schema.filter({ f => featureColumnNames.contains(f.name) })

    // pre-compute namespace hashes
    val featureColIndices = VowpalWabbitUtil.generateNamespaceInfos(
      StructType(featureColumns),
      vwArgs.getHashSeed,
      Seq(getFeaturesCol) ++ getAdditionalFeatures)

    // define UDF
    val predictUDF = udf { (namespaces: Row) => predictInternal(featureColIndices, namespaces) }

    // add prediction column
    dataset.withColumn(
      $(rawPredictionCol),
      predictUDF(struct(featureColumns.map(f => col(f.name)): _*)))
  }

  // Model parameter
  val model = new ByteArrayParam(this, "model", "The VW model....")

  def setModel(v: Array[Byte]): this.type = set(model, v)

  def getModel: Array[Byte] = $(model)

  def getReadableModel: String = {
    val readableModelFile = java.io.File.createTempFile("vowpalwabbit", ".txt")

    try {
      // start VW, write the model, stop
      new VowpalWabbitNative(s"--readable_model ${readableModelFile.getAbsolutePath}", getModel).close()

      // need to read the model after the VW object is disposed as this triggers the file write
      StreamUtilities.using(Source.fromFile(readableModelFile)) {
        _.mkString
      }.get
    } finally {
      readableModelFile.delete
    }
  }

  // Perf stats parameter
  val performanceStatistics = new DataFrameParam(this,
    "performanceStatistics",
    "Performance statistics collected during training")

  def setPerformanceStatistics(v: DataFrame): this.type = set(performanceStatistics, v)

  def getPerformanceStatistics: DataFrame = $(performanceStatistics)

  protected def predictInternal(featureColIndices: Seq[NamespaceInfo], row: Row): Double = {
    example.clear()

    // transfer features
    VowpalWabbitUtil.addFeaturesToExample(featureColIndices, row, example)

    // TODO: surface prediction confidence
    example.predict.asInstanceOf[ScalarPrediction].getValue.toDouble
  }

  def saveNativeModel(path: String): Unit = {
    val session = SparkSession.builder().getOrCreate()

    val f = new java.io.File(path)

    session.createDataFrame(Seq(PathAndData(f.getCanonicalPath, getModel)))
      .write.mode("overwrite")
      .format(classOf[BinaryFileFormat].getName)
      .save(f.getName)
  }
}
