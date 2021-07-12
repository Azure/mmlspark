// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.spark.lightgbm.booster.LightGBMBooster
import com.microsoft.ml.spark.lightgbm.params.{LightGBMModelParams, LightGBMPredictionParams,
  RegressorTrainParams, TrainParams}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.ml.{BaseRegressor, ComplexParamsReadable, ComplexParamsWritable}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}

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
    with LightGBMBase[LightGBMRegressionModel] with BasicLogging {
  logClass()

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

  def getTrainParams(numTasks: Int, categoricalIndexes: Array[Int], dataset: Dataset[_]): TrainParams = {
    val modelStr = if (getModelString == null || getModelString.isEmpty) None else get(modelString)
    RegressorTrainParams(getParallelism, getTopK, getNumIterations, getLearningRate, getNumLeaves,
      getAlpha, getTweedieVariancePower, getMaxBin, getBinSampleCount, getBaggingFraction, getPosBaggingFraction,
      getNegBaggingFraction, getBaggingFreq, getBaggingSeed, getEarlyStoppingRound, getImprovementTolerance,
      getFeatureFraction, getMaxDepth, getMinSumHessianInLeaf, numTasks, modelStr, getVerbosity, categoricalIndexes,
      getBoostFromAverage, getBoostingType, getLambdaL1, getLambdaL2, getIsProvideTrainingMetric, getMetric,
      getMinGainToSplit, getMaxDeltaStep, getMaxBinByFeature, getMinDataInLeaf, getSlotNames, getDelegate,
      getDartParams(), getExecutionParams(), getObjectiveParams(), getDeviceType)
  }

  def getModel(trainParams: TrainParams, lightGBMBooster: LightGBMBooster): LightGBMRegressionModel = {
    new LightGBMRegressionModel(uid)
      .setLightGBMBooster(lightGBMBooster)
      .setFeaturesCol(getFeaturesCol)
      .setPredictionCol(getPredictionCol)
      .setLeafPredictionCol(getLeafPredictionCol)
      .setFeaturesShapCol(getFeaturesShapCol)
      .setNumIterations(lightGBMBooster.bestIteration)
  }

  def stringFromTrainedModel(model: LightGBMRegressionModel): String = {
    model.getModel.modelStr.get
  }

  override def copy(extra: ParamMap): LightGBMRegressor = defaultCopy(extra)
}

/** Model produced by [[LightGBMRegressor]]. */
class LightGBMRegressionModel(override val uid: String)
  extends RegressionModel[Vector, LightGBMRegressionModel]
    with LightGBMModelParams
    with LightGBMModelMethods
    with LightGBMPredictionParams
    with ComplexParamsWritable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LightGBMRegressionModel"))

  override protected lazy val pyInternalWrapper = true

  /**
    * Adds additional Leaf Index and SHAP columns if specified.
    *
    * @param dataset input dataset
    * @return transformed dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      updateBoosterParamsBeforePredict()
      var outputData = super.transform(dataset)
      if (getLeafPredictionCol.nonEmpty) {
        val predLeafUDF = udf(predictLeaf _)
        outputData = outputData.withColumn(getLeafPredictionCol, predLeafUDF(col(getFeaturesCol)))
      }
      if (getFeaturesShapCol.nonEmpty) {
        val featureShapUDF = udf(featuresShap _)
        outputData = outputData.withColumn(getFeaturesShapCol, featureShapUDF(col(getFeaturesCol)))
      }
      outputData.toDF
    })
  }

  override def predict(features: Vector): Double = {
    logPredict(
      getModel.score(features, false, false)(0)
    )
  }

  override def copy(extra: ParamMap): LightGBMRegressionModel = defaultCopy(extra)

  def saveNativeModel(filename: String, overwrite: Boolean): Unit = {
    val session = SparkSession.builder().getOrCreate()
    getModel.saveNativeModel(session, filename, overwrite)
  }
}

object LightGBMRegressionModel extends ComplexParamsReadable[LightGBMRegressionModel] {
  def loadNativeModelFromFile(filename: String): LightGBMRegressionModel = {
    val uid = Identifiable.randomUID("LightGBMRegressionModel")
    val session = SparkSession.builder().getOrCreate()
    val textRdd = session.read.text(filename)
    val text = textRdd.collect().map { row => row.getString(0) }.mkString("\n")
    val lightGBMBooster = new LightGBMBooster(text)
    new LightGBMRegressionModel(uid).setLightGBMBooster(lightGBMBooster)
  }

  def loadNativeModelFromString(model: String): LightGBMRegressionModel = {
    val uid = Identifiable.randomUID("LightGBMRegressionModel")
    val lightGBMBooster = new LightGBMBooster(model)
    new LightGBMRegressionModel(uid).setLightGBMBooster(lightGBMBooster)
  }
}
