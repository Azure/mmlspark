// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml._
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.recommendation._
import org.apache.spark.ml.util._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Future}
import scala.util.Random
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasCollectSubModels
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{RankingHelper, TrainValidRecommendSplitParams}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.util.ThreadUtils

import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.language.existentials

/**
  * Validation for hyper-parameter tuning.
  * Randomly splits the input dataset into train and validation sets,
  * and uses evaluation metric on the validation set to select the best model.
  */
@InternalWrapper
class RankingTrainValidationSplit(override val uid: String)
  extends Estimator[RankingTrainValidationSplitModel]
    with ComplexParamsWritable with RankingFunctions
    with TrainValidRecommendSplitParams with HasCollectSubModels {

  /** @group setParam */
  def setEstimator(value: Estimator[_ <: Model[_]]): this.type = set(estimator, value)

  /** @group setParam */
  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /** @group setParam */
  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  /** @group setParam */
  def setTrainRatio(value: Double): this.type = set(trainRatio, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /**
    * Set the mamixum level of parallelism to evaluate models in parallel.
    * Default is 1 for serial evaluation
    *
    * @group expertSetParam
    */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  /**
    * Whether to collect submodels when fitting. If set, we can get submodels from
    * the returned model.
    *
    * Note: If set this param, when you save the returned model, you can set an option
    * "persistSubModels" to be "true" before saving, in order to save these submodels.
    *
    * @group expertSetParam
    */
  def setCollectSubModels(value: Boolean): this.type = set(collectSubModels, value)

  val collectSubMetrics: BooleanParam = new BooleanParam(this, "collectSubMetrics", "")

  def setCollectSubMetrics(value: Boolean): this.type = set(collectSubMetrics, value)

  def getCollectSubMetrics: Boolean = $(collectSubMetrics)

  def this() = this(Identifiable.randomUID("RankingTrainValidationSplit"))

  def transformSchema(schema: StructType): StructType = {
    require($(estimatorParamMaps).nonEmpty, s"Validator requires non-empty estimatorParamMaps")
    val firstEstimatorParamMap = $(estimatorParamMaps).head
    val est = $(estimator)
    for (paramMap <- $(estimatorParamMaps).tail) {
      est.copy(paramMap).transformSchema(schema)
    }
    est.copy(firstEstimatorParamMap).transformSchema(schema)
  }

  setDefault(minRatingsPerUser -> 1, minRatingsPerItem -> 1)

  override def getExecutionContext: ExecutionContext = {

    getParallelism match {
      case 1 =>
        RankingHelper.getThreadUtils().sameThread
      case n =>
        ExecutionContext.fromExecutorService(RankingHelper.getThreadUtils()
          .newDaemonCachedThreadPool(s"${this.getClass.getSimpleName}-thread-pool", n))
    }
  }

  def fit(dataset: Dataset[_]): RankingTrainValidationSplitModel = {
    val schema = dataset.schema
    transformSchema(schema)

    val eval = $(evaluator).asInstanceOf[RankingEvaluator]

    val est = new RankingAdapter()
      .setMode("allUsers") //allItems does not work, not sure if it would be used
      .setK(eval.getK)
      .setRecommender($(estimator).asInstanceOf[Estimator[_ <: Model[_]]])
      .setUserCol($(estimator).asInstanceOf[ALS].getUserCol)
      .setRatingCol($(estimator).asInstanceOf[ALS].getRatingCol)
      .setItemCol($(estimator).asInstanceOf[ALS].getItemCol)
    //todo cast to something more generic than ALS

    val epm = $(estimatorParamMaps)

    // Create execution context based on $(parallelism)
    val executionContext = getExecutionContext

    val (trainingDataset, validationDataset) =
      split(dataset, getTrainRatio, est.getItemCol, est.getUserCol, est.getRatingCol)
    trainingDataset.cache()
    validationDataset.cache()

    val collectSubModelsParam = $(collectSubModels)
    val collectSubMetricsParam = $(collectSubMetrics)

    val subModels: Option[Array[Model[_]]] = if (collectSubModelsParam) {
      Some(Array.fill[Model[_]](epm.length)(null))
    } else None

    val subMetrics: Option[Array[Map[String, Double]]] = if (collectSubMetricsParam) {
      Some(Array.fill[Map[String, Double]](epm.length)(null))
    } else None

    // Fit models in a Future for training in parallel
    logDebug(s"Train split with multiple sets of parameters.")
    val metricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
      Future[Double] {
        val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]

        if (collectSubModelsParam) {
          subModels.get(paramIndex) = model
        }
        // TODO: duplicate evaluator to take extra params from input
        val df = model.transform(validationDataset, paramMap)
        if (collectSubMetricsParam) {
          subMetrics.get(paramIndex) = eval.asInstanceOf[RankingEvaluator].getMetricsMap(df)
        }
        val metric = eval.evaluate(df)
        logDebug(s"Got metric $metric for model trained with $paramMap.")
        metric
      }(executionContext)
    }

    // Wait for all metrics to be calculated
    val metrics = metricFutures.map(RankingHelper.getThreadUtils().awaitResult(_, Duration.Inf))

    // Unpersist training & validation set once all metrics have been produced
    trainingDataset.unpersist()
    validationDataset.unpersist()

    logInfo(s"Train validation split metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best train validation split metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    new RankingTrainValidationSplitModel(uid)
      .setBestModel(bestModel)
      .setMetrics(metrics)
      .setSubModels(subModels)
      .setSubMetrics(subMetrics)
  }

  override def copy(extra: ParamMap): RankingTrainValidationSplit = {
    defaultCopy(extra)
  }

}

object RankingTrainValidationSplit extends ComplexParamsReadable[RankingTrainValidationSplit]

/**
  * Model from train validation split.
  *
  * @param uid Id.
  */
@InternalWrapper
class RankingTrainValidationSplitModel private[spark](val uid: String)
  extends Model[RankingTrainValidationSplitModel]
    with ComplexParamsWritable with Wrappable {

  private var _subModels: Option[Array[Model[_]]] = None

  private[spark] def setSubModels(subModels: Option[Array[Model[_]]])
  : RankingTrainValidationSplitModel = {
    _subModels = subModels
    this
  }

  /**
    * @return submodels represented in array. The index of array corresponds to the ordering of
    *         estimatorParamMaps
    */
  def subModels: Array[Model[_]] = {
    require(_subModels.isDefined, "subModels not available, To retrieve subModels, make sure " +
      "to set collectSubModels to true before fitting.")
    _subModels.get
  }

  def hasSubModels: Boolean = _subModels.isDefined

  private var _subMetrics: Option[Array[Map[String, Double]]] = None

  private[spark] def setSubMetrics(subMetrics: Option[Array[Map[String, Double]]])
  : RankingTrainValidationSplitModel = {
    _subMetrics = subMetrics
    this
  }

  /**
    * @return submodels represented in array. The index of array corresponds to the ordering of
    *         estimatorParamMaps
    */
  def subMetrics: Array[Map[String, Double]] = {
    require(_subMetrics.isDefined, "subModels not available, To retrieve subModels, make sure " +
      "to set collectSubModels to true before fitting.")
    _subMetrics.get
  }

  def hasSubMetrics: Boolean = _subMetrics.isDefined

  def recommendForAllUsers(k: Int): DataFrame = getBestModel.asInstanceOf[RankingAdapterModel]
    .recommendForAllUsers(k)

  def recommendForAllItems(k: Int): DataFrame = getBestModel.asInstanceOf[RankingAdapterModel]
    .recommendForAllItems(k)

  val metrics = new DoubleArrayParam(this, "metrics", "metrics")

  def setMetrics(m: Array[Double]): RankingTrainValidationSplitModel.this.type = set(metrics, m)

  def getMetrics: Array[Double] = $(metrics)

  val bestModel = new TransformerParam(this, "bestModel", "bestModel", { x: Transformer => true })

  def setBestModel(m: Model[_]): RankingTrainValidationSplitModel.this.type = set(bestModel, m)

  def getBestModel: Model[_] = $(bestModel).asInstanceOf[Model[_]]

  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    getBestModel.transform(dataset)
  }

  def transformSchema(schema: StructType): StructType = {
    getBestModel.transformSchema(schema)
  }

  override def copy(extra: ParamMap): RankingTrainValidationSplitModel = {
    defaultCopy(extra)
  }

}

object RankingTrainValidationSplitModel extends ComplexParamsReadable[RankingTrainValidationSplitModel]