// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.contracts.Wrappable
import com.microsoft.ml.spark.core.serialize.params.PipelineStageParam
import org.apache.spark.ml.param.{ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

object MultiColumnAdapter extends ComplexParamsReadable[MultiColumnAdapter]

/** The <code>MultiColumnAdapter</code> takes a unary pipeline stage and a list of input output column pairs
  * and applies the pipeline stage to each input column after being fit
  */
class MultiColumnAdapter(override val uid: String) extends Estimator[PipelineModel]
  with Wrappable with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("MultiColumnAdapter"))

  /** List of input column names, encoded as a string. These are the columns for the pipeline stage.
    * @group param
    */
  val inputCols: StringArrayParam =
    new StringArrayParam(this,
      "inputCols",
      "list of column names encoded as a string")

  /** @group getParam */
  final def getInputCols: Array[String] = $(inputCols)

  /** @group setParam */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** List of column names for the pipeline staged columns, encoded as a string.
    * @group param
    */
  val outputCols: StringArrayParam =
    new StringArrayParam(this,
      "outputCols",
      "list of column names encoded as a string")

  /** @group getParam */
  final def getOutputCols: Array[String] = $(outputCols)

  /** @group setParam */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  /** @return List of input/output column name pairs. */
  def getInputOutputPairs: List[(String, String)] = getInputCols.zip(getOutputCols).toList

  def getStages: Array[PipelineStage] = getInputOutputPairs.toArray.map(getInOutPairStage)

  private def getInOutPairStage(inputOutputPair: (String, String)): PipelineStage = {
    val model = getBaseStage.copy(new ParamMap())
    setParamInternal(setParamInternal(model,
      "inputCol", inputOutputPair._1),
      "outputCol", inputOutputPair._2)
    model
  }

  /** Base pipeline stage to apply to every column in the input column list.
    * @group param
    */
  val baseStage: PipelineStageParam =
    new PipelineStageParam(this,
                           "baseStage",
                           "base pipeline stage to apply to every column")

  /** @group getParam */
  final def getBaseStage: PipelineStage = $(baseStage)

  /** @group setParam */
  def setBaseStage(value: PipelineStage): this.type = {
    try {
      //Test to see whether the class has the appropriate getters and setters
      value.getParam("inputCol")
      value.getParam("outputCol")
      setParamInternal(value, "inputCol", this.uid + "_in")
      setParamInternal(value, "outputCol", this.uid + "_out")
    } catch {
      case e: Exception =>
        throw new IllegalArgumentException(
          "Need to pass a pipeline stage with inputCol and outputCol params")
    }
    set(baseStage, value)
  }

  private def setParamInternal[M <: PipelineStage, V](model: M,
                                                      name: String,
                                                      value: V) = {
    model.set(model.getParam(name), value)
  }

  private def getParamInternal[M <: PipelineStage](model: M, name: String) = {
    model.getOrDefault(model.getParam(name))
  }

  /** Fit the pipeline stage to all the columns in the input column list
    * @param dataset
    * @return PipelineModel fit on the columns bearing the input column names
    */
  override def fit(dataset: Dataset[_]): PipelineModel = {
    transformSchema(dataset.schema)
    new Pipeline(uid).setStages(getStages).fit(dataset)
  }

  def copy(extra: ParamMap): this.type = defaultCopy(extra)

  private def verifyCols(df: DataFrame,
                         inputOutputPairs: List[(String, String)]): Unit = {
    inputOutputPairs.foreach {
      case (s1, s2) if !df.columns.contains(s1) =>
        throw new IllegalArgumentException(
          s"DataFrame does not contain specified column: $s1")
      case (s1, s2) if df.columns.contains(s2) =>
        throw new IllegalArgumentException(
          s"DataFrame already contains specified column: $s2")
      case _ =>
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    getInputOutputPairs.foldLeft(schema) { (schema, pair) =>
      getInOutPairStage(pair).transformSchema(schema)
    }
  }

}
