// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{BaseRegressor, RegressionModel}
import org.apache.spark.sql._

import scala.concurrent.Await
import scala.concurrent.duration.{Duration, SECONDS}
import scala.math.min
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
@InternalWrapper
class LightGBMRegressor(override val uid: String)
  extends BaseRegressor[Vector, LightGBMRegressor, LightGBMRegressionModel]
    with LightGBMParams {
  def this() = this(Identifiable.randomUID("LightGBMRegressor"))

  // Set default objective to be regression
  setDefault(objective -> "regression")

  val alpha = new DoubleParam(this, "alpha", "parameter for Huber loss and Quantile regression")
  setDefault(alpha -> 0.9)

  def getAlpha: Double = $(alpha)
  def setAlpha(value: Double): this.type = set(alpha, value)

  val tweedieVariancePower = new DoubleParam(this, "tweedieVariancePower",
    "control the variance of tweedie distribution, must be between 1 and 2")
  setDefault(tweedieVariancePower -> 1.5)

  def getTweedieVariancePower: Double = $(tweedieVariancePower)
  def setTweedieVariancePower(value: Double): this.type = set(tweedieVariancePower, value)

  // Set default objective to be regression
  setDefault(objective -> "regression")

  /** Trains the LightGBM Regression model.
    *
    * @param dataset The input dataset to train.
    * @return The trained model.
    */
  override protected def train(dataset: Dataset[_]): LightGBMRegressionModel = {
    val numCoresPerExec = LightGBMUtils.getNumCoresPerExecutor(dataset)
    val numExecutorCores = LightGBMUtils.getNumExecutorCores(dataset, numCoresPerExec)
    val numWorkers = min(numExecutorCores, dataset.rdd.getNumPartitions)
    // Reduce number of partitions to number of executor cores if needed
    val df = dataset.toDF().coalesce(numWorkers).cache()
    val (inetAddress, port, future, iters) =
      LightGBMUtils.createDriverNodesThread(numWorkers, df, log, getTimeout, getAutoscale, getNumIterations)

    /* Run a parallel job via map partitions to initialize the native library and network,
     * translate the data to the LightGBM in-memory representation and train the models
     */
    val encoder = Encoders.kryo[LightGBMBooster]

    val categoricalSlotIndexesArr = get(categoricalSlotIndexes).getOrElse(Array.empty[Int])
    val categoricalSlotNamesArr = get(categoricalSlotNames).getOrElse(Array.empty[String])
    val categoricalIndexes = LightGBMUtils.getCategoricalIndexes(df, getFeaturesCol,
      categoricalSlotIndexesArr, categoricalSlotNamesArr)
    val trainParams = RegressorTrainParams(getParallelism, getNumIterations, getLearningRate, getNumLeaves,
      getObjective, getAlpha, getTweedieVariancePower, getMaxBin, getBaggingFraction, getBaggingFreq, getBaggingSeed,
      getEarlyStoppingRound, getFeatureFraction, getMaxDepth, getMinSumHessianInLeaf, numWorkers, getModelString,
      getVerbosity, categoricalIndexes, getBoostFromAverage, getBoostingType, getAutoscale)
    log.info(s"LightGBMRegressor parameters: ${trainParams.toString}")
    val networkParams = NetworkParams(getDefaultListenPort, inetAddress, port)

    val lightGBMBooster = df
      .mapPartitions(TrainUtils.trainLightGBM(networkParams, getLabelCol, getFeaturesCol, get(weightCol),
        log, trainParams, numCoresPerExec))(encoder)
      .reduce((booster1, _) => booster1)
    // Wait for future to complete (should be done by now)
    Await.result(future, Duration(getTimeout, SECONDS))
    if (iters(0) != getNumIterations) {
      setModelString(lightGBMBooster.model)
      setNumIterations(getNumIterations - iters(0))
      train(dataset)
    } else {
      new LightGBMRegressionModel(uid, lightGBMBooster, getLabelCol, getFeaturesCol, getPredictionCol)
    }
  }

  override def copy(extra: ParamMap): LightGBMRegressor = defaultCopy(extra)
}

/** Model produced by [[LightGBMRegressor]]. */
@InternalWrapper
class LightGBMRegressionModel(override val uid: String, model: LightGBMBooster, labelColName: String,
                                  featuresColName: String, predictionColName: String)
  extends RegressionModel[Vector, LightGBMRegressionModel]
    with ConstructorWritable[LightGBMRegressionModel] {

  // Update the underlying Spark ML params
  // (for proper serialization to work we put them on constructor instead of using copy as in Spark ML)
  set(labelCol, labelColName)
  set(featuresCol, featuresColName)
  set(predictionCol, predictionColName)

  override def predict(features: Vector): Double = {
    model.score(features, false, false)(0)
  }

  override def copy(extra: ParamMap): LightGBMRegressionModel =
    new LightGBMRegressionModel(uid, model, labelColName, featuresColName, predictionColName)

  override val ttag: TypeTag[LightGBMRegressionModel] = typeTag[LightGBMRegressionModel]

  override def objectsToSave: List[Any] = List(uid, model, getLabelCol, getFeaturesCol, getPredictionCol)

  def saveNativeModel(filename: String, overwrite: Boolean): Unit = {
    val session = SparkSession.builder().getOrCreate()
    model.saveNativeModel(session, filename, overwrite)
  }

  def getFeatureImportances(importanceType: String): Array[Double] = {
    model.getFeatureImportances(importanceType)
  }

  def getModel: LightGBMBooster = this.model
}

object LightGBMRegressionModel extends ConstructorReadable[LightGBMRegressionModel] {
  def loadNativeModelFromFile(filename: String, labelColName: String = "label",
                              featuresColName: String = "features",
                              predictionColName: String = "prediction"): LightGBMRegressionModel = {
    val uid = Identifiable.randomUID("LightGBMRegressor")
    val session = SparkSession.builder().getOrCreate()
    val textRdd = session.read.text(filename)
    val text = textRdd.collect().map { row => row.getString(0) }.mkString("\n")
    val lightGBMBooster = new LightGBMBooster(text)
    new LightGBMRegressionModel(uid, lightGBMBooster, labelColName, featuresColName, predictionColName)
  }

  def loadNativeModelFromString(model: String, labelColName: String = "label",
                                featuresColName: String = "features",
                                predictionColName: String = "prediction"): LightGBMRegressionModel = {
    val uid = Identifiable.randomUID("LightGBMRegressor")
    val lightGBMBooster = new LightGBMBooster(model)
    new LightGBMRegressionModel(uid, lightGBMBooster, labelColName, featuresColName, predictionColName)
  }
}
