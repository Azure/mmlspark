// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import com.microsoft.ml.spark.core.env.InternalWrapper
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{ComplexParamsReadable, PredictionModel, Predictor}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, DoubleType, IntegerType, StructType}
import org.vowpalwabbit.spark.{VowpalWabbitExample, VowpalWabbitNative}
import vowpalWabbit.responses.ActionProbs
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

import scala.collection.mutable
import scala.math.max

object VowpalWabbitContextualBandit extends DefaultParamsReadable[VowpalWabbitContextualBandit]

class ExampleStack(val vw: VowpalWabbitNative) {
  val stack = new mutable.Stack[VowpalWabbitExample]

  def getOrCreateExample(): VowpalWabbitExample = {
    val ex = if (stack.isEmpty)
      vw.createExample
    else
      stack.pop

    ex.clear
    ex.setDefaultLabel

    ex
  }

  def returnExample(ex: VowpalWabbitExample): Unit = {
    stack.push(ex)
  }

  def close(): Unit =
    while (stack.nonEmpty)
      stack.pop.close()
}

// https://github.com/VowpalWabbit/estimators/blob/master/ips_snips.py
class ContextualBanditMetrics extends Serializable {
  var snips_numerator: Double = 0
  var total_events: Double = 0
  var snips_denominator: Double = 0
  var policy_events: Double = 0 // the number of times the policy had a nonzero probability on the action played
  var max_ips_numerator: Double = 0

  // r is a reward
  def add_example(p_log: Double, reward: Double, p_pred: Double, count: Int = 1): Unit = {
    total_events += count
    if (p_pred > 0) {
      val p_over_p = p_pred / p_log
      snips_denominator += p_over_p * count
      policy_events += count
      if (reward != 0) {
        snips_numerator += reward * p_over_p * count
        max_ips_numerator = max(max_ips_numerator, reward * p_over_p)
      }
    }
  }

  def get_snips_estimate(): Double = {
    snips_numerator / snips_denominator
  }

  def get_ips_estimate(): Double = {
    snips_numerator / total_events
  }
}

@InternalWrapper
trait VowpalWabbitContextualBanditBase extends VowpalWabbitBase {
  val sharedCol = new Param[String](this, "sharedCol", "Column name of shared features")
  def getSharedCol: String = $(sharedCol)
  def setSharedCol(value: String): this.type = set(sharedCol, value)

  setDefault(sharedCol -> "shared")
}

@InternalWrapper
class VowpalWabbitContextualBandit(override val uid: String)
  extends Predictor[Row, VowpalWabbitContextualBandit, VowpalWabbitContextualBanditModel]
    with VowpalWabbitContextualBanditBase {
  def this() = this(Identifiable.randomUID("VowpalWabbitContextualBandit"))

  val probabilityCol = new Param[String](this, "probabilityCol", "Column name of probability of chosen action")
  def getProbabilityCol: String = $(probabilityCol)
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
  setDefault(probabilityCol -> "probability")

  val chosenActionCol = new Param[String](this, "chosenActionCol", "Column name of chosen action")
  def getChosenActionCol: String = $(chosenActionCol)
  def setChosenActionCol(value: String): this.type = set(chosenActionCol, value)
  setDefault(chosenActionCol -> "chosenAction")

  val additionalSharedFeatures = new StringArrayParam(this, "additionalSharedFeatures", "Additional namespaces for the shared example")
  def getAdditionalSharedFeatures: Array[String] = $(additionalSharedFeatures)
  def setAdditionalSharedFeatures(value: Array[String]): this.type = set(additionalSharedFeatures, value)
  setDefault(additionalSharedFeatures -> Array.empty)

  // Used in the base class to remove unneeded columns from the dataframe.
  protected override def getAdditionalColumns(): Seq[String] = Seq(getChosenActionCol, getProbabilityCol, getSharedCol)

  override def transformSchema(schema: StructType): StructType = {
    val allActionFeatureColumns = Seq(getFeaturesCol) ++ getAdditionalFeatures
    val allSharedFeatureColumns = Seq(getSharedCol) ++ getAdditionalSharedFeatures
    val actionCol = getChosenActionCol
    val labelCol = getLabelCol
    val probCol = getProbabilityCol

    // Validate action columns
    for (colName <- allActionFeatureColumns)
    {
      val dt = schema(colName).dataType
      assert(dt match {
        case ArrayType(VectorType, _ ) => true
        case _ => false
      },  s"$colName must be a list of sparse vectors of features. Found: $dt. Each item in the list corresponds to a specific action and the overall list is the namespace.")
    }

    // Validate shared columns
    for (colName <- allSharedFeatureColumns)
    {
      val dt = schema(colName).dataType
      assert(dt match {
        case VectorType => true
        case _ => false
      }, s"$colName must be a sparse vector of features. Found $dt")
    }

    val actionDt = schema(actionCol).dataType
    assert(actionDt match {
      case IntegerType => true
      case _ => false
    },  s" $actionCol must be an integer. Found: $actionDt")

    val labelDt = schema(labelCol).dataType
    assert(labelDt match {
      case IntegerType => true
      case DoubleType => true
      case _ => false
    }, s"$labelCol must be an double. Found: $labelDt")

    val probDt = schema(probCol).dataType
    assert(probDt match {
      case IntegerType => true
      case DoubleType => true
      case _ => false
    }, s"$probCol must be an double. Found: $probDt")

    schema
  }

  protected override def trainRow(schema: StructType,
                                  inputRows: Iterator[Row],
                                  ctx: TrainContext
                                 ): Unit = {
    val allActionFeatureColumns = Seq(getFeaturesCol) ++ getAdditionalFeatures
    val allSharedFeatureColumns = Seq(getSharedCol) ++ getAdditionalSharedFeatures

    val actionNamespaceInfos = VowpalWabbitUtil.generateNamespaceInfos(
      schema,
      getHashSeed,
      allActionFeatureColumns)

    val sharedNamespaceInfos = VowpalWabbitUtil.generateNamespaceInfos(schema, getHashSeed, allSharedFeatureColumns);
    val chosenActionColIdx = schema.fieldIndex(getChosenActionCol)
    val labelIdx = schema.fieldIndex(getLabelCol)
    val probabilityIdx = schema.fieldIndex(getProbabilityCol)

    val exampleStack = new ExampleStack(ctx.vw)

    for (row <- inputRows) {
      VowpalWabbitUtil.prepareMultilineExample(row, actionNamespaceInfos, sharedNamespaceInfos, ctx.vw, exampleStack,
        examples => {
          // It's one-based but we need to skip the shared example anyway
          val selectedActionIdx = row.getInt(chosenActionColIdx)
          if (selectedActionIdx == 0) {
            throw new IllegalArgumentException("Chosen action index is 1 based - cannot be 0")
          }

          val cost = row.getDouble(labelIdx)
          val loggedProbability = row.getDouble(probabilityIdx)

          // Set the label for learning
          examples(selectedActionIdx).setContextualBanditLabel(
            selectedActionIdx,
            cost,
            loggedProbability)

          // Learn from the examples
          ctx.vw.learn(examples)

          // Update the IPS/SNIPS estimator
          val prediction: ActionProbs = examples(0).getPrediction.asInstanceOf[ActionProbs]
          val probs = prediction.getActionProbs
          val selectedActionIdxZeroBased = selectedActionIdx - 1
          ctx.contextualBanditMetrics.add_example(loggedProbability, cost, probs.apply(selectedActionIdxZeroBased).getProbability)
        })
    }
  }

  override protected def train(dataset: Dataset[_]): VowpalWabbitContextualBanditModel = {
    val model = new VowpalWabbitContextualBanditModel(uid)
      .setFeaturesCol(getFeaturesCol)
      .setAdditionalFeatures(getAdditionalFeatures)
      .setSharedCol(getSharedCol)
      .setPredictionCol(getPredictionCol)

    trainInternal(dataset, model)
  }

  override def copy(extra: ParamMap): VowpalWabbitContextualBandit = defaultCopy(extra)
}

@InternalWrapper
class VowpalWabbitContextualBanditModel(override val uid: String)
  extends PredictionModel[Row, VowpalWabbitContextualBanditModel]
    with VowpalWabbitBaseModel with VowpalWabbitContextualBanditBase {

  override def transformSchema(schema: StructType): StructType = schema

  lazy val exampleStack = new ExampleStack(vw)

  override def transform(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError("transform is not implemented")
  }

  protected override def transformImpl(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError("transformImpl is not implemented")
  }

  override def predict(features: Row): Double = {
    throw new NotImplementedError("predict is not implemented")
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
}

object VowpalWabbitContextualBanditModel extends ComplexParamsReadable[VowpalWabbitContextualBanditModel]