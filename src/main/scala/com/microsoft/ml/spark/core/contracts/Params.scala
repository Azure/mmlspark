// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.contracts

import org.apache.spark.ml.param._

trait HasInputCol extends Params {
  /** The name of the input column
    *
    * @group param
    */
  val inputCol = new Param[String](this, "inputCol", "The name of the input column")

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group getParam */
  def getInputCol: String = $(inputCol)
}

trait HasOutputCol extends Params {
  /** The name of the output column
    *
    * @group param
    */
  val outputCol = new Param[String](this, "outputCol", "The name of the output column")

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group getParam */
  def getOutputCol: String = $(outputCol)
}

trait HasInputCols extends Params {
  /** The names of the inputColumns
    *
    * @group param
    */
  val inputCols = new StringArrayParam(this, "inputCols", "The names of the input columns")

  /** @group setParam */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group getParam */
  def getInputCols: Array[String] = $(inputCols)
}

trait HasOutputCols extends Params {
  /** The names of the output columns
    *
    * @group param
    */
  val outputCols = new StringArrayParam(this, "outputCols", "The names of the output columns")

  /** @group setParam */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  /** @group getParam */
  def getOutputCols: Array[String] = $(outputCols)
}

trait HasLabelCol extends Params {
  /** The name of the label column
    *
    * @group param
    */
  val labelCol = new Param[String](this, "labelCol", "The name of the label column")

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group getParam */
  def getLabelCol: String = $(labelCol)
}

trait HasFeaturesCol extends Params {
  /** The name of the features column
    *
    * @group param
    */
  val featuresCol = new Param[String](this, "featuresCol", "The name of the features column")

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group getParam */
  def getFeaturesCol: String = $(featuresCol)
}

trait HasWeightCol extends Params {
  /** The name of the weight column
    *
    * @group param
    */
  val weightCol = new Param[String](this, "weightCol", "The name of the weight column")

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** @group getParam */
  def getWeightCol: String = $(weightCol)
}

trait HasScoredLabelsCol extends Params {
  /** The name of the scored labels column
    *
    * @group param
    */
  val scoredLabelsCol =
    new Param[String](this, "scoredLabelsCol",
      "Scored labels column name, only required if using SparkML estimators")

  /** @group setParam */
  def setScoredLabelsCol(value: String): this.type = set(scoredLabelsCol, value)

  /** @group getParam */
  def getScoredLabelsCol: String = $(scoredLabelsCol)
}

trait HasScoresCol extends Params {
  /** The name of the scores column
    *
    * @group param
    */
  val scoresCol =
    new Param[String](this, "scoresCol",
      "Scores or raw prediction column name, only required if using SparkML estimators")

  /** @group setParam */
  def setScoresCol(value: String): this.type = set(scoresCol, value)

  /** @group getParam */
  def getScoresCol: String = $(scoresCol)
}

trait HasScoredProbabilitiesCol extends Params {
  /** The name of the scored probabilities column
    *
    * @group param
    */
  val scoredProbabilitiesCol =
    new Param[String](this, "scoredProbabilitiesCol",
      "Scored probabilities, usually calibrated from raw scores, only required if using SparkML estimators")

  /** @group setParam */
  def setScoredProbabilitiesCol(value: String): this.type = set(scoredProbabilitiesCol, value)

  /** @group getParam */
  def getScoredProbabilitiesCol: String = $(scoredProbabilitiesCol)
}

trait HasEvaluationMetric extends Params {
  val evaluationMetric: Param[String] =
    new Param[String](this, "evaluationMetric", "Metric to evaluate models with")

  /** @group getParam */
  def getEvaluationMetric: String = $(evaluationMetric)

  /** @group setParam */
  def setEvaluationMetric(value: String): this.type = set(evaluationMetric, value)
}

trait HasValidationIndicatorCol extends Params {
  /** The name of the validation indicator column
    *
    * @group param
    */
  val validationIndicatorCol: Param[String] =
    new Param[String](this, "validationIndicatorCol", "Indicates whether the row is for training or validation")

  /** @group getParam */
  def getValidationIndicatorCol: String = $(validationIndicatorCol)

  /** @group setParam */
  def setValidationIndicatorCol(value: String): this.type = set(validationIndicatorCol, value)
}

trait HasInitScoreCol extends Params {
  /** The name of the initial score column
    *
    * @group param
    */
  val initScoreCol = new Param[String](this, "initScoreCol",
    "The name of the initial score column, used for continued training")

  /** @group setParam */
  def setInitScoreCol(value: String): this.type = set(initScoreCol, value)

  /** @group getParam */
  def getInitScoreCol: String = $(initScoreCol)
}

trait HasGroupCol extends Params {
  /** The name of the group column
    *
    * @group param
    */
  val groupCol = new Param[String](this, "groupCol", "The name of the group column")

  /** @group setParam */
  def setGroupCol(value: String): this.type = set(groupCol, value)

  /** @group getParam */
  def getGroupCol: String = $(groupCol)
}

trait HasConfidenceScoreCol extends Params {
  /** The name of the confidence score column
   *
   * @group param
   */
  val confidenceScoreCol =
    new Param[String](this, "confidenceScoreCol",
      "Confidence score, usually a value between 0-1. Higher value implies higher model confidence.")

  /** @group setParam */
  def setConfidenceScoreCol(value: String): this.type = set(confidenceScoreCol, value)

  /** @group getParam */
  def getConfidenceScoreCol: String = $(confidenceScoreCol)
  setDefault(confidenceScoreCol -> "ConfidenceScore")
}

trait HasLangCol extends Params {
  /** The name of the language of the document column
    *
    * @group param
    */
  val langCol = new Param[String](this, "langCol",
    "Name of the language column of the document.")

  /** @group setParam */
  def setLangCol(value: String): this.type = set(langCol, value)

  /** @group getParam */
  def getLangCol: String = $(langCol)
  setDefault(langCol -> null)
}
