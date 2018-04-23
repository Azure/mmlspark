// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.metrics.MetricConstants
import com.microsoft.ml.spark.core.schema.SchemaConstants
import com.microsoft.ml.spark.stages.featurize._
import org.apache.spark.ml.classification.{ClassificationModel, Classifier}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression._
import org.apache.spark.ml.{PipelineStage, RegressionUtils, Transformer}

object EvaluationUtils {
  val modelTypeUnsupportedErr = "Model type not supported for evaluation"
  // Find type of trained models
  def getModelType(model: PipelineStage):String = {
    model match {
      case _: TrainRegressor => SchemaConstants.RegressionKind
      case _: TrainClassifier => SchemaConstants.ClassificationKind
      case _: Classifier[_, _, _] => SchemaConstants.ClassificationKind
      case regressor: PipelineStage if RegressionUtils.isRegressor(regressor) => SchemaConstants.RegressionKind
      case _: DecisionTreeRegressor => SchemaConstants.RegressionKind
      case _: GBTRegressor => SchemaConstants.RegressionKind
      case _: RandomForestRegressor => SchemaConstants.RegressionKind
      case _: TrainedRegressorModel => SchemaConstants.RegressionKind
      case _: TrainedClassifierModel => SchemaConstants.ClassificationKind
      case evm: BestModel => getModelType(evm.getBestModel)
      case _: ClassificationModel[_, _] => SchemaConstants.ClassificationKind
      case _: RegressionModel[_, _] => SchemaConstants.RegressionKind
      case _ => throw new Exception(modelTypeUnsupportedErr)
    }
  }

  def getMetricWithOperator(model: PipelineStage, evaluationMetric: String): (String, Ordering[Double]) = {
    val modelType = getModelType(model)
    getMetricWithOperator(modelType, evaluationMetric)
  }

  def getMetricWithOperator(modelType: String, evaluationMetric: String): (String, Ordering[Double]) = {
    val chooseHighest = Ordering.Double
    val chooseLowest = Ordering.Double.reverse
    val (evaluationMetricColumnName, operator): (String, Ordering[Double]) = modelType match {
      case SchemaConstants.RegressionKind => evaluationMetric match {
        case MetricConstants.MseSparkMetric  => (MetricConstants.MseColumnName,  chooseLowest)
        case MetricConstants.RmseSparkMetric => (MetricConstants.RmseColumnName, chooseLowest)
        case MetricConstants.R2SparkMetric   => (MetricConstants.R2ColumnName,   chooseHighest)
        case MetricConstants.MaeSparkMetric  => (MetricConstants.MaeColumnName,  chooseLowest)
        case _ => throw new Exception("Metric is not supported for regressors")
      }
      case SchemaConstants.ClassificationKind => evaluationMetric match {
        case MetricConstants.AucSparkMetric       => (MetricConstants.AucColumnName, chooseHighest)
        case MetricConstants.PrecisionSparkMetric => (MetricConstants.PrecisionColumnName, chooseHighest)
        case MetricConstants.RecallSparkMetric    => (MetricConstants.RecallColumnName, chooseHighest)
        case MetricConstants.AccuracySparkMetric  => (MetricConstants.AccuracyColumnName, chooseHighest)
        case _ => throw new Exception("Metric is not supported for classifiers")
      }
      case _ => throw new Exception("Model type not supported for evaluation")
    }
    (evaluationMetricColumnName, operator)
  }

  def getModelParams(model: Transformer): ParamMap = {
    model match {
      case reg: TrainedRegressorModel => reg.getParamMap
      case cls: TrainedClassifierModel => cls.getParamMap
      case evm: BestModel => getModelParams(evm.getBestModel)
      case _ => throw new Exception("Model type not supported for evaluation")
    }
  }

  /** Returns a string representation of the model.
    * @param model The model output by TrainClassifier or TrainRegressor
    * @return A comma delimited representation of the model parameter names and values
    */
  def modelParamsToString(model: Transformer): String =
    getModelParams(model).toSeq.map(pv => s"${pv.param.name}: ${pv.value}").sorted.mkString(", ")

}
