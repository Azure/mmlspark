// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.contracts

import scala.collection.mutable.Map
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}

trait MMLParams extends Wrappable with DefaultParamsWritable

trait Wrappable extends Params {

  // Use this function when instantiating sparkML Identifiable for your
  // own use - it allows us to locate the origin of any stacks
  def chainedUid(origin: String): String = Identifiable.randomUID(this.uid)

  private var orderCounter = 0
  // TODO: Support non-string "enums"?
  val paramDomains = Map[String, Seq[String]]()

  def BooleanParam(i: Identifiable, name: String, description: String): BooleanParam =
    BooleanParam(i, name, description, false)

  def BooleanParam(i: Identifiable, name: String, description: String,
                   default: Boolean): BooleanParam = {
    val baseParam = new BooleanParam(i, name, description)
    MMLParam(baseParam, Some(default), None)
    baseParam
  }

  def IntParam(i: Identifiable, name: String, description: String): IntParam = {
    val baseParam = new IntParam(i, name, description)
    MMLParam(baseParam, None, None)
    baseParam
  }

  def IntParam(i: Identifiable, name: String, description: String,
               default: Int): IntParam = {
    val baseParam = new IntParam(i, name, description)
    MMLParam(baseParam, Some(default), None)
    baseParam
  }

  def IntParam(i: Identifiable, name: String, description: String, validation: Int => Boolean): IntParam = {
    val baseParam = new IntParam(i, name, description, validation)
    MMLParam(baseParam, None, None)
    baseParam
  }

  def LongParam(i: Identifiable, name: String, description: String): LongParam = {
    val baseParam = new LongParam(i, name, description)
    MMLParam(baseParam, None, None)
    baseParam
  }

  def LongParam(i: Identifiable, name: String, description: String,
                default: Long): LongParam = {
    val baseParam = new LongParam(i, name, description)
    MMLParam(baseParam, Some(default), None)
    baseParam
  }

  def DoubleParam(i: Identifiable, name: String, description: String): DoubleParam = {
    val baseParam = new DoubleParam(i, name, description)
    MMLParam(baseParam, None, None)
    baseParam
  }

  def DoubleParam(i: Identifiable, name: String, description: String,
                  default: Double): DoubleParam = {
    val baseParam = new DoubleParam(i, name, description)
    MMLParam(baseParam, Some(default), None)
    baseParam
  }

  def StringParam(i: Identifiable, name: String, description: String): Param[String] = {
    val baseParam = new Param[String](i, name, description)
    MMLParam(baseParam, None, None)
    baseParam
  }

  def StringParam(i: Identifiable, name: String, description: String, validation: String => Boolean): Param[String] = {
    val baseParam = new Param[String](i, name, description, validation)
    MMLParam(baseParam, None, None)
    baseParam
  }

  def StringParam(i: Identifiable, name: String, description: String,
                  default: String): Param[String] = {
    val baseParam = new Param[String](i, name, description)
    MMLParam(baseParam, Some(default), None)
    baseParam
  }

  def StringParam(i: Identifiable, name: String, description: String,
                  default: String, domain: Seq[String]): Param[String] = {
    val baseParam = new Param[String](i, name, description)
    MMLParam(baseParam, Some(default), Some(domain))
    baseParam
  }

  private def MMLParam[T](param: Param[T],
                          default: Option[T], domain: Option[Seq[String]]): Unit = {
    if (default.isDefined) setDefault(param, default.get)
    if (domain.isDefined) paramDomains.put(param.name, domain.get)
    orderCounter += 1
  }

}

trait HasInputCol extends Wrappable {
  /** The name of the input column
    * @group param
    */
  val inputCol = StringParam(this, "inputCol", "The name of the input column")
  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)
  /** @group getParam */
  def getInputCol: String = $(inputCol)
}

trait HasOutputCol extends Wrappable {
  /** The name of the output column
    * @group param
    */
  val outputCol = StringParam(this, "outputCol", "The name of the output column")
  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)
  /** @group getParam */
  def getOutputCol: String = $(outputCol)
}

trait HasInputCols extends Wrappable {
  /** The names of the inputColumns
    * @group param
    */
  val inputCols = new StringArrayParam(this, "inputCols", "The names of the input columns")
  /** @group setParam */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  /** @group getParam */
  def getInputCols: Array[String] = $(inputCols)
}

trait HasOutputCols extends Wrappable {
  /** The names of the output columns
    * @group param
    */
  val outputCols = new StringArrayParam(this, "outputCols", "The names of the output columns")
  /** @group setParam */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)
  /** @group getParam */
  def getOutputCols: Array[String] = $(outputCols)
}

trait HasLabelCol extends Wrappable {
  /** The name of the label column
    * @group param
    */
  val labelCol = StringParam(this, "labelCol", "The name of the label column")
  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)
  /** @group getParam */
  def getLabelCol: String = $(labelCol)
}

trait HasFeaturesCol extends Wrappable {
  /** The name of the features column
    * @group param
    */
  val featuresCol = StringParam(this, "featuresCol", "The name of the features column")
  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  /** @group getParam */
  def getFeaturesCol: String = $(featuresCol)
}

trait HasScoredLabelsCol extends Wrappable {
  /** The name of the scored labels column
    * @group param
    */
  val scoredLabelsCol =
    StringParam(this, "scoredLabelsCol",
                "Scored labels column name, only required if using SparkML estimators")
  /** @group setParam */
  def setScoredLabelsCol(value: String): this.type = set(scoredLabelsCol, value)
  /** @group getParam */
  def getScoredLabelsCol: String = $(scoredLabelsCol)
}

trait HasScoresCol extends Wrappable {
  /** The name of the scores column
    * @group param
    */
  val scoresCol =
    StringParam(this, "scoresCol",
                "Scores or raw prediction column name, only required if using SparkML estimators")
  /** @group setParam */
  def setScoresCol(value: String): this.type = set(scoresCol, value)
  /** @group getParam */
  def getScoresCol: String = $(scoresCol)
}

trait HasScoredProbabilitiesCol extends Wrappable {
  /** The name of the scored probabilities column
    * @group param
    */
  val scoredProbabilitiesCol =
    StringParam(this, "scoredProbabilitiesCol",
                "Scored probabilities, usually calibrated from raw scores, only required if using SparkML estimators")
  /** @group setParam */
  def setScoredProbabilitiesCol(value: String): this.type = set(scoredProbabilitiesCol, value)
  /** @group getParam */
  def getScoredProbabilitiesCol: String = $(scoredProbabilitiesCol)
}

trait HasEvaluationMetric extends Wrappable {
  val evaluationMetric: Param[String] =
    StringParam(this, "evaluationMetric", "Metric to evaluate models with")

  /** @group getParam */
  def getEvaluationMetric: String = $(evaluationMetric)

  /** @group setParam */
  def setEvaluationMetric(value: String): this.type = set(evaluationMetric, value)
}
