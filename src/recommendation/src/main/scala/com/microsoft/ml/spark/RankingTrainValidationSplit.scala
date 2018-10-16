/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tuning

import java.util.{List => JList}

import com.microsoft.ml.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.shared.{HasCollectSubModels, HasParallelism}
import org.apache.spark.ml.param.{ParamMap, TransformerParam}
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
  * Similar to [[CrossValidator]], but only splits the set once.
  */
class RankingTrainValidationSplit(override val uid: String)
  extends Estimator[RankingTrainValidationSplitModel]
    with ComplexParamsWritable with RecommendationSplitFunctions
    with TrainValidationSplitParams with HasParallelism with HasCollectSubModels
    with Logging {

  /** @group setParam */
  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

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
    * You can check documents of
    * {@link org.apache.spark.ml.tuning.TrainValidationSplitModel.TrainValidationSplitModelWriter}
    * for more information.
    *
    * @group expertSetParam
    */
  def setCollectSubModels(value: Boolean): this.type = set(collectSubModels, value)

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

  def fit(dataset: Dataset[_]): RankingTrainValidationSplitModel = {
    val schema = dataset.schema
    transformSchema(schema)
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)

    // Create execution context based on $(parallelism)
    val executionContext = getExecutionContext

    val (trainingDataset, validationDataset) =
      splitDF(filterRatings(dataset.dropDuplicates()), getTrainRatio)
    trainingDataset.cache()
    validationDataset.cache()

    val collectSubModelsParam = $(collectSubModels)

    var subModels: Option[Array[Model[_]]] = if (collectSubModelsParam) {
      Some(Array.fill[Model[_]](epm.length)(null))
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
        val metric = eval.evaluate(model.transform(validationDataset, paramMap))
        logDebug(s"Got metric $metric for model trained with $paramMap.")
        metric
      }(executionContext)
    }

    // Wait for all metrics to be calculated
    val metrics = metricFutures.map(ThreadUtils.awaitResult(_, Duration.Inf))

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
    new RankingTrainValidationSplitModel(uid).setBestModel(bestModel)
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
class RankingTrainValidationSplitModel private[ml](val uid: String)
  extends Model[RankingTrainValidationSplitModel]
    with ComplexParamsWritable with Wrappable {

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