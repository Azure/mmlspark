// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.{HasEvaluationMetric, Wrappable}
import com.microsoft.ml.spark.core.metrics.MetricConstants
import com.microsoft.ml.spark.core.serialize.params.TransformerArrayParam
import com.microsoft.ml.spark.core.serialize.{ConstructorReadable, ConstructorWritable}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable.ListBuffer
import scala.reflect.runtime.universe.{TypeTag, typeTag}

object FindBestModel extends ComplexParamsReadable[FindBestModel] {
  val modelNameCol = "model_name"
  val metricsCol = "metric"
  val paramsCol = "parameters"
}

trait FindBestModelParams extends Wrappable with ComplexParamsWritable with HasEvaluationMetric {
  /** Metric to evaluate the models with. Default is "accuracy"
    *
    * The metrics that can be chosen are:
    *
    *   For Binary Classifiers:
    *     - AreaUnderROC
    *     - AUC
    *     - accuracy
    *     - precision
    *     - recall
    *
    *   For Regression Classifiers:
    *     - mse
    *     - rmse
    *     - r2
    *     - mae
    *
    *   Or, for either type of classifier:
    *     - all - This will report all the relevant metrics
    *
    * @group param
    */
  setDefault(evaluationMetric -> MetricConstants.AccuracySparkMetric)
}

/** Evaluates and chooses the best model from a list of models. */
class FindBestModel(override val uid: String) extends Estimator[BestModel] with FindBestModelParams {

  def this() = this(Identifiable.randomUID("FindBestModel"))
  /** List of models to be evaluated. The list is an Array of models
    * @group param
    */
  val models: TransformerArrayParam = new TransformerArrayParam(this, "models", "List of models to be evaluated")

  /** @group getParam */
  def getModels: Array[Transformer] = $(models)

  /** @group setParam */
  def setModels(value: Array[Transformer]): this.type = set(models, value)

  var selectedModel: Transformer = null

  var selectedScoredDataset: Dataset[_] = null

  var selectedROCCurve: DataFrame = null

  var selectedBestModelMetrics: Dataset[_] = null

  /** @param dataset - The input dataset, to be fitted
    * @return The Model that results from the fitting
    */
  override def fit(dataset: Dataset[_]): BestModel = {
    // Staging
    val trainedModels = getModels
    if (trainedModels.isEmpty) {
      throw new Exception("No trained models to evaluate.")
    }
    if (!(MetricConstants.findBestModelMetrics contains getEvaluationMetric)) {
      throw new Exception("Invalid evaluation metric")
    }
    val evaluator = new ComputeModelStatistics()
    evaluator.set(evaluator.evaluationMetric, getEvaluationMetric)

    var bestMetric: Double = Double.NaN
    // Setup to store metrics and model name data for model metrics table
    val modelMetrics = ListBuffer[Double]()
    val models = ListBuffer[String]()
    val parameters = ListBuffer[String]()
    val firstModel = trainedModels(0)

    val (evaluationMetricColumnName, operator): (String, Ordering[Double]) =
      EvaluationUtils.getMetricWithOperator(firstModel, getEvaluationMetric)
    val modelType = EvaluationUtils.getModelType(firstModel)

    val compareModels = (model: Transformer, metrics: DataFrame, scoredDataset: Dataset[_]) => {
      val currentMetric = metrics.select(evaluationMetricColumnName).first()(0).toString.toDouble
      modelMetrics += currentMetric
      models += model.uid
      parameters += EvaluationUtils.modelParamsToString(model)
      if (bestMetric.isNaN || operator.gt(currentMetric, bestMetric)) {
        bestMetric = currentMetric
        selectedModel = model
        selectedScoredDataset = scoredDataset
      }
    }

    for (trainedModel <- trainedModels) {
      // Check that models are consistent
      if (EvaluationUtils.getModelType(trainedModel) != modelType) {
        throw new Exception("Models are inconsistent. Please evaluate only regressors or classifiers.")
      }
      val df = trainedModel.transform(dataset)
      val metrics = evaluator.transform(df)
      compareModels(trainedModel, metrics, df)
    }

    // compute ROC curve
    evaluator.set(evaluator.evaluationMetric, MetricConstants.AllSparkMetrics)
    selectedBestModelMetrics = evaluator.transform(selectedScoredDataset)
    selectedROCCurve = evaluator.rocCurve

    val spark = dataset.sparkSession
    val allModelMetricsSchema = StructType(Seq(StructField(FindBestModel.modelNameCol, StringType, true),
      StructField(FindBestModel.metricsCol, DoubleType, true),
      StructField(FindBestModel.paramsCol, StringType, true)))
    var allModelMetrics = spark.createDataFrame(spark.sparkContext.parallelize(models.zip(modelMetrics).zip(parameters)
        .map(mmp => Row(mmp._1._1, mmp._1._2, mmp._2))), allModelMetricsSchema)
    new BestModel(uid,
      selectedModel,
      selectedScoredDataset,
      selectedROCCurve,
      selectedBestModelMetrics,
      allModelMetrics)
  }

  // Choose a random model as we don't know which one will be chosen yet - all will transform schema in same way
  def transformSchema(schema: StructType): StructType = getModels(0).transformSchema(schema)

  def copy(extra: ParamMap): FindBestModel = defaultCopy(extra)

}

/** Model produced by [[FindBestModel]]. */
class BestModel(val uid: String,
                val model: Transformer,
                val scoredDataset: Dataset[_],
                val rocCurve: DataFrame,
                val bestModelMetrics: Dataset[_],
                val allModelMetrics: Dataset[_])
    extends Model[BestModel] with ConstructorWritable[BestModel] {

  val ttag: TypeTag[BestModel] = typeTag[BestModel]
  def objectsToSave: List[Any] = List(uid, model, scoredDataset, rocCurve, bestModelMetrics, allModelMetrics)

  override def copy(extra: ParamMap): BestModel =
    new BestModel(uid, model.copy(extra), scoredDataset, rocCurve, bestModelMetrics, allModelMetrics)

  override def transform(dataset: Dataset[_]): DataFrame = model.transform(dataset)

  /** The best model found during evaluation.
    * @return The best model.
    */
  def getBestModel: Transformer = model

  /** Gets the scored dataset.
    * @return The scored dataset for the best model.
    */
  def getScoredDataset: Dataset[_] = scoredDataset

  /** Gets the ROC curve with TPR, FPR.
    * @return The evaluation results.
    */
  def getEvaluationResults: Dataset[_] = rocCurve

  /** Gets all of the best model metrics results from the evaluator.
    * @return All of the best model metrics results.
    */
  def getBestModelMetrics: Dataset[_] = bestModelMetrics

  /** Gets a table of metrics from all models compared from the evaluation comparison.
    * @return The model metrics results from all models.
    */
  def getAllModelMetrics: Dataset[_] = allModelMetrics

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = model.transformSchema(schema)

}

object BestModel extends ConstructorReadable[BestModel]
