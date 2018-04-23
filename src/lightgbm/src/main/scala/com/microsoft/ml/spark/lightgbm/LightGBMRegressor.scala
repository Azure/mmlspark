// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.lightgbm.lightgbmlibConstants
import com.microsoft.ml.spark.core.serialize.{ConstructorReadable, ConstructorWritable}
import org.apache.spark.ml.BaseRegressor
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util._
import org.apache.spark.sql._

import scala.reflect.runtime.universe.{TypeTag, typeTag}

object LightGBMRegressor extends DefaultParamsReadable[LightGBMRegressor]

/** Trains a LightGBM Regression model, a fast, distributed, high performance gradient boosting
  * framework based on decision tree algorithms.
  * For more information please see here: https://github.com/Microsoft/LightGBM.
  * For parameter information see here: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
  * Note: The application parameter supports the following values:
  *  - regression_l2, L2 loss, alias=regression, mean_squared_error, mse, l2_root, root_mean_squared_error, rmse
  *  - regression_l1, L1 loss, alias=mean_absolute_error, mae
  *  - huber, Huber loss
  *  - fair, Fair loss
  *  - poisson, Poisson regression
  *  - quantile, Quantile regression
  *  - mape, MAPE loss, alias=mean_absolute_percentage_error
  *  - gamma, Gamma regression with log-link. It might be useful, e.g., for modeling insurance claims severity,
  *     or for any target that might be gamma-distributed
  *  - tweedie, Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in
  *    insurance, or for any target that might be tweedie-distributed
  * @param uid The unique ID.
  */
class LightGBMRegressor(override val uid: String)
  extends BaseRegressor[Vector, LightGBMRegressor, LightGBMRegressionModel]
    with LightGBMParams {
  def this() = this(Identifiable.randomUID("LightGBMRegressor"))

  val alpha = DoubleParam(this, "alpha", "parameter for Huber loss and Quantile regression", 0.9)

  def getAlpha: Double = $(alpha)
  def setAlpha(value: Double): this.type = set(alpha, value)

  val application = StringParam(this, "application",
    "Regression application, regression_l2, regression_l1, huber, fair, poisson, quantile, mape, gamma or tweedie",
    "regression")

  def getApplication: String = $(application)
  def setApplication(value: String): this.type = set(application, value)

  /** Trains the LightGBM Regression model.
    *
    * @param dataset The input dataset to train.
    * @return The trained model.
    */
  override protected def train(dataset: Dataset[_]): LightGBMRegressionModel = {
    val numExecutors = LightGBMUtils.getNumExecutors(dataset)
    // Reduce number of partitions to number of executors
    val df = dataset.toDF().coalesce(numExecutors).cache()

    val (nodes, numNodes) = LightGBMUtils.getNodes(df, getDefaultListenPort)
    /* Run a parallel job via map partitions to initialize the native library and network,
     * translate the data to the LightGBM in-memory representation and train the models
     */
    val encoder = Encoders.kryo[LightGBMBooster]
    log.info(s"Nodes used for LightGBM: $nodes")
    val trainParams = new RegressorTrainParams(getParallelism, getNumIterations, getLearningRate, getNumLeaves,
      getApplication, getAlpha, getMaxBin, getBaggingFraction, getBaggingFreq, getBaggingSeed, getFeatureFraction,
      getMaxDepth, getMinSumHessianInLeaf)
    val lightGBMBooster = df
      .mapPartitions(TrainUtils.trainLightGBM(nodes, numNodes, getLabelCol, getFeaturesCol,
        getDefaultListenPort, log, trainParams))(encoder)
      .reduce((booster1, booster2) => booster1)
    new LightGBMRegressionModel(uid, lightGBMBooster, getLabelCol, getFeaturesCol, getPredictionCol)
  }

  override def copy(extra: ParamMap): LightGBMRegressor = defaultCopy(extra)
}

/** Model produced by [[LightGBMRegressor]]. */
class LightGBMRegressionModel(override val uid: String, model: LightGBMBooster, labelColName: String,
                                  featuresColName: String, predictionColName: String)
  extends RegressionModel[Vector, LightGBMRegressionModel]
    with ConstructorWritable[LightGBMRegressionModel] {

  // Update the underlying Spark ML params
  // (for proper serialization to work we put them on constructor instead of using copy as in Spark ML)
  set(labelCol, labelColName)
  set(featuresCol, featuresColName)
  set(predictionCol, predictionColName)

  override protected def predict(features: Vector): Double = {
    model.score(features, false)
  }

  override def copy(extra: ParamMap): LightGBMRegressionModel = defaultCopy(extra)

  override val ttag: TypeTag[LightGBMRegressionModel] = typeTag[LightGBMRegressionModel]

  override def objectsToSave: List[Any] = List(uid, model, getLabelCol, getFeaturesCol, getPredictionCol)
}

object LightGBMRegressionModel extends ConstructorReadable[LightGBMRegressionModel]
