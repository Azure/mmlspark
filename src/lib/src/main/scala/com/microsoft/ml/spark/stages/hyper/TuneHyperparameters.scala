// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.hyper

import java.util.concurrent._

import com.google.common.util.concurrent.{MoreExecutors, ThreadFactoryBuilder}
import com.microsoft.ml.spark.core.contracts.{HasEvaluationMetric, Wrappable}
import com.microsoft.ml.spark.core.metrics.MetricConstants
import com.microsoft.ml.spark.core.serialize.params.{EstimatorArrayParam, ParamSpace, ParamSpaceParam}
import com.microsoft.ml.spark.core.serialize.{ConstructorReadable, ConstructorWritable}
import com.microsoft.ml.spark.stages.featurize.{TrainedClassifierModel, TrainedRegressorModel}
import com.microsoft.ml.spark.stages.{ComputeModelStatistics, EvaluationUtils}
import org.apache.spark.SparkException
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable.ListBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Awaitable, ExecutionContext, Future}
import scala.reflect.runtime.universe.{TypeTag, typeTag}
import scala.util.control.NonFatal

/** Tunes model hyperparameters
  *
  * Allows user to specify multiple untrained models to tune using various search strategies.
  * Currently supports cross validation with random grid search.
  */
class TuneHyperparameters(override val uid: String) extends Estimator[TuneHyperparametersModel]
  with Wrappable with ComplexParamsWritable with HasEvaluationMetric {

  def this() = this(Identifiable.randomUID("TuneHyperparameters"))

  /** Estimators to run
    * @group param
    */
  val models = new EstimatorArrayParam(this, "models", "Estimators to run")
  /** @group getParam */
  def getModels: Array[Estimator[_]] = $(models)
  /** @group setParam */
  def setModels(value: Array[Estimator[_]]): this.type = set(models, value)

  val numFolds = IntParam(this, "numFolds", "Number of folds")
  /** @group getParam */
  def getNumFolds: Int = $(numFolds)
  /** @group setParam */
  def setNumFolds(value: Int): this.type = set(numFolds, value)

  val seed = LongParam(this, "seed", "Random number generator seed")
  /** @group getParam */
  def getSeed: Long = $(seed)
  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)
  setDefault(seed -> 0)

  val numRuns = IntParam(this, "numRuns", "Termination criteria for randomized search")
  /** @group getParam */
  def getNumRuns: Int = $(numRuns)
  /** @group setParam */
  def setNumRuns(value: Int): this.type = set(numRuns, value)

  val parallelism = IntParam(this, "parallelism", "The number of models to run in parallel")
  /** @group getParam */
  def getParallelism: Int = $(parallelism)
  /** @group setParam */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  val paramSpace: ParamSpaceParam =
    new ParamSpaceParam(this, "paramSpace", "Parameter space for generating hyperparameters")
  /** @group getParam */
  def getParamSpace: ParamSpace = $(paramSpace)
  /** @group setParam */
  def setParamSpace(value: ParamSpace): this.type  = set(paramSpace, value)

  private def getExecutionContext: ExecutionContext = {
    getParallelism match {
      case 1 =>
        ExecutionContext.fromExecutorService(MoreExecutors.sameThreadExecutor())
      case n =>
        val keepAliveSeconds = 60L
        val prefix = s"${this.getClass.getSimpleName}-thread-pool"
        val threadFactory = new ThreadFactoryBuilder().setDaemon(true).setNameFormat(prefix + "-%d").build()
        val threadPool = new ThreadPoolExecutor(getParallelism, getParallelism, keepAliveSeconds,
          TimeUnit.SECONDS, new LinkedBlockingQueue[Runnable], threadFactory)
        threadPool.allowCoreThreadTimeOut(true)
        ExecutionContext.fromExecutorService(threadPool)
    }
  }

  /** Private function taken from spark - waits for the task to complete
    */
  private def awaitResult[T](awaitable: Awaitable[T], atMost: Duration): T = {
    try {
      val awaitPermission = null.asInstanceOf[scala.concurrent.CanAwait]
      awaitable.result(atMost)(awaitPermission)
    } catch {
      case NonFatal(t) if !t.isInstanceOf[TimeoutException] =>
        throw new SparkException("Exception thrown in awaitResult: ", t)
    }
  }

  /** Tunes model hyperparameters for given number of runs and returns the best model
    * found based on evaluation metric.
    *
    * @param dataset The input dataset to train.
    * @return The trained classification model.
    */
  override def fit(dataset: Dataset[_]): TuneHyperparametersModel = {
    val sparkSession = dataset.sparkSession
    val splits = MLUtils.kFold(dataset.toDF.rdd, getNumFolds, getSeed)
    val hyperParams = getParamSpace.paramMaps
    val schema = dataset.schema
    val executionContext = getExecutionContext
    val (evaluationMetricColumnName, operator): (String, Ordering[Double]) =
      EvaluationUtils.getMetricWithOperator(getModels.head, getEvaluationMetric)
    val paramsPerRun = ListBuffer[ParamMap]()
    for (_ <- 0 until getNumRuns) {
      // Generate the new parameters, stepping through estimators sequentially
      paramsPerRun += hyperParams.next()
    }
    val numModels = getModels.length

    val metrics = splits.zipWithIndex.map { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()

      val modelParams = ListBuffer[ParamMap]()
      for (n <- 0 until getNumRuns) {
        val params = paramsPerRun(n)
        modelParams += params
      }
      val foldMetricFutures = modelParams.zipWithIndex.map { case (paramMap, paramIndex) =>
          Future[Double] {
            val model = getModels(paramIndex % numModels).fit(trainingDataset, paramMap).asInstanceOf[Model[_]]
            val scoredDataset = model.transform(validationDataset, paramMap)
            val evaluator = new ComputeModelStatistics()
            evaluator.set(evaluator.evaluationMetric, getEvaluationMetric)
            model match {
              case _: TrainedRegressorModel => {
                logDebug("Evaluating trained regressor model.")
              }
              case _: TrainedClassifierModel => {
                logDebug("Evaluating trained classifier model.")
              }
              case classificationModel: ClassificationModel[_, _] => {
                logDebug(s"Evaluating SparkML ${model.uid} classification model.")
                evaluator
                  .setLabelCol(classificationModel.getLabelCol)
                  .setScoredLabelsCol(classificationModel.getPredictionCol)
                  .setScoresCol(classificationModel.getRawPredictionCol)
                if (getEvaluationMetric == MetricConstants.AllSparkMetrics)
                  evaluator.setEvaluationMetric(MetricConstants.ClassificationMetrics)
              }
              case regressionModel: RegressionModel[_, _] => {
                logDebug(s"Evaluating SparkML ${model.uid} regression model.")
                evaluator
                  .setLabelCol(regressionModel.getLabelCol)
                  .setScoredLabelsCol(regressionModel.getPredictionCol)
                if (getEvaluationMetric == MetricConstants.AllSparkMetrics)
                  evaluator.setEvaluationMetric(MetricConstants.RegressionMetrics)
              }
            }
            val metrics = evaluator.transform(scoredDataset)
            val metric = metrics.select(evaluationMetricColumnName).first()(0).toString.toDouble
            logDebug(s"Got metric $metric for model trained with $paramMap.")
            metric
          } (executionContext)
      }
      val foldMetrics = foldMetricFutures.toArray.map(awaitResult(_, Duration.Inf))

      trainingDataset.unpersist()
      validationDataset.unpersist()
      foldMetrics
    }.transpose.map(_.sum / $(numFolds)) // Calculate average metric over all splits

    val (bestMetric, bestIndex) = metrics.zipWithIndex.maxBy(_._1)(operator)
    // Compute best model fit on dataset
    val bestModel = getModels(bestIndex % numModels).fit(dataset, paramsPerRun(bestIndex)).asInstanceOf[Model[_]]
    new TuneHyperparametersModel(uid, bestModel, bestMetric)
  }

  override def copy(extra: ParamMap): Estimator[TuneHyperparametersModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = getModels(0).transformSchema(schema)
}

object TuneHyperparameters extends ComplexParamsReadable[TuneHyperparameters]

/** Model produced by [[TuneHyperparameters]]. */
class TuneHyperparametersModel(val uid: String,
                               val model: Transformer,
                               val bestMetric: Double)
    extends Model[TuneHyperparametersModel] with ConstructorWritable[TuneHyperparametersModel] {

  override val ttag: TypeTag[TuneHyperparametersModel] = typeTag[TuneHyperparametersModel]
  override def objectsToSave: List[Any] = List(uid, model, bestMetric)

  override def copy(extra: ParamMap): TuneHyperparametersModel =
    new TuneHyperparametersModel(uid, model.copy(extra), bestMetric)

  override def transform(dataset: Dataset[_]): DataFrame = model.transform(dataset)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = model.transformSchema(schema)

  def getBestMetric: Double = bestMetric

  def getBestModel: Transformer = model

  def getBestModelInfo: String = EvaluationUtils.modelParamsToString(model)

}

object TuneHyperparametersModel extends ConstructorReadable[TuneHyperparametersModel]
