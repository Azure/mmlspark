// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import com.microsoft.ml.spark.schema.SparkSchema
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql._

import scala.concurrent.Await
import scala.concurrent.duration.{Duration, SECONDS}
import scala.math.min
import scala.reflect.runtime.universe.{TypeTag, typeTag}

object LightGBMClassifier extends DefaultParamsReadable[LightGBMClassifier]

/** Trains a LightGBM Binary Classification model, a fast, distributed, high performance gradient boosting
  * framework based on decision tree algorithms.
  * For more information please see here: https://github.com/Microsoft/LightGBM.
  * For parameter information see here: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
  * @param uid The unique ID.
  */
@InternalWrapper
class LightGBMClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, LightGBMClassifier, LightGBMClassificationModel]
  with LightGBMParams {
  def this() = this(Identifiable.randomUID("LightGBMClassifier"))

  // Set default objective to be binary classification
  setDefault(objective -> LightGBMConstants.binaryObjective)

  val isUnbalance = new BooleanParam(this, "isUnbalance",
    "Set to true if training data is unbalanced in binary classification scenario")
  setDefault(isUnbalance -> false)

  def getIsUnbalance: Boolean = $(isUnbalance)
  def setIsUnbalance(value: Boolean): this.type = set(isUnbalance, value)

  /** Trains the LightGBM Classification model.
    *
    * @param dataset The input dataset to train.
    * @return The trained model.
    */
  override protected def train(dataset: Dataset[_]): LightGBMClassificationModel = {
    val numCoresPerExec = LightGBMUtils.getNumCoresPerExecutor(dataset)
    val numExecutorCores = LightGBMUtils.getNumExecutorCores(dataset, numCoresPerExec)
    val numWorkers = min(numExecutorCores, dataset.rdd.getNumPartitions)
    // Reduce number of partitions to number of executor cores
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
    /* The native code for getting numClasses is always 1 unless it is multiclass-classification problem
     * so we infer the actual numClasses from the dataset here
     */
    val actualNumClasses = getNumClasses(dataset)
    val classes =
      if (getObjective == LightGBMConstants.binaryObjective) None
      else Some(actualNumClasses)
    val metric =
      if (classes.isDefined) "multiclass"
      else "binary_logloss,auc"
    val trainParams = ClassifierTrainParams(getParallelism, getNumIterations, getLearningRate, getNumLeaves,
      getMaxBin, getBaggingFraction, getBaggingFreq, getBaggingSeed, getEarlyStoppingRound,
      getFeatureFraction, getMaxDepth, getMinSumHessianInLeaf, numWorkers, getObjective, getModelString,
      getIsUnbalance, getVerbosity, categoricalIndexes, classes, metric, getBoostFromAverage, getBoostingType,
      getAutoscale)
    log.info(s"LightGBMClassifier parameters: ${trainParams.toString}")
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
      new LightGBMClassificationModel(uid, lightGBMBooster, getLabelCol, getFeaturesCol,
        getPredictionCol, getProbabilityCol, getRawPredictionCol,
        if (isDefined(thresholds)) Some(getThresholds) else None, actualNumClasses)
    }
  }

  override def copy(extra: ParamMap): LightGBMClassifier = defaultCopy(extra)
}

/** Model produced by [[LightGBMClassifier]]. */
@InternalWrapper
class LightGBMClassificationModel(
  override val uid: String, model: LightGBMBooster, labelColName: String,
  featuresColName: String, predictionColName: String, probColName: String,
  rawPredictionColName: String, thresholdValues: Option[Array[Double]],
  actualNumClasses: Int)
    extends ProbabilisticClassificationModel[Vector, LightGBMClassificationModel]
    with ConstructorWritable[LightGBMClassificationModel] {

  // Update the underlying Spark ML params
  // (for proper serialization to work we put them on constructor instead of using copy as in Spark ML)
  set(labelCol, labelColName)
  set(featuresCol, featuresColName)
  set(predictionCol, predictionColName)
  set(probabilityCol, probColName)
  set(rawPredictionCol, rawPredictionColName)
  if (thresholdValues.isDefined) set(thresholds, thresholdValues.get)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        dv.values(0) = 1.0 / (1.0 + math.exp(-dv.values(0)))
        dv.values(1) = 1.0 - dv.values(0)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in LightGBMClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def numClasses: Int = this.actualNumClasses

  override protected def predictRaw(features: Vector): Vector = {
    Vectors.dense(model.score(features, true, true))
  }

  override protected def predictProbability(features: Vector): Vector = {
    Vectors.dense(model.score(features, false, true))
  }

  override def copy(extra: ParamMap): LightGBMClassificationModel =
    new LightGBMClassificationModel(uid, model, labelColName, featuresColName, predictionColName, probColName,
      rawPredictionColName, thresholdValues, actualNumClasses)

  override val ttag: TypeTag[LightGBMClassificationModel] =
    typeTag[LightGBMClassificationModel]

  override def objectsToSave: List[Any] =
    List(uid, model, getLabelCol, getFeaturesCol, getPredictionCol,
         getProbabilityCol, getRawPredictionCol, thresholdValues, actualNumClasses)

  def saveNativeModel(filename: String, overwrite: Boolean): Unit = {
    val session = SparkSession.builder().getOrCreate()
    model.saveNativeModel(session, filename, overwrite)
  }

  def getFeatureImportances(importanceType: String): Array[Double] = {
    model.getFeatureImportances(importanceType)
  }

  def getModel: LightGBMBooster = this.model
}

object LightGBMClassificationModel extends ConstructorReadable[LightGBMClassificationModel] {
  def loadNativeModelFromFile(filename: String, labelColName: String = "label",
                              featuresColName: String = "features", predictionColName: String = "prediction",
                              probColName: String = "probability",
                              rawPredictionColName: String = "rawPrediction"): LightGBMClassificationModel = {
    val uid = Identifiable.randomUID("LightGBMClassifier")
    val session = SparkSession.builder().getOrCreate()
    val textRdd = session.read.text(filename)
    val text = textRdd.collect().map { row => row.getString(0) }.mkString("\n")
    val lightGBMBooster = new LightGBMBooster(text)
    val actualNumClasses = lightGBMBooster.getNumClasses()
    new LightGBMClassificationModel(uid, lightGBMBooster, labelColName, featuresColName,
      predictionColName, probColName, rawPredictionColName, None, actualNumClasses)
  }

  def loadNativeModelFromString(model: String, labelColName: String = "label",
                                featuresColName: String = "features", predictionColName: String = "prediction",
                                probColName: String = "probability",
                                rawPredictionColName: String = "rawPrediction"): LightGBMClassificationModel = {
    val uid = Identifiable.randomUID("LightGBMClassifier")
    val lightGBMBooster = new LightGBMBooster(model)
    val actualNumClasses = lightGBMBooster.getNumClasses()
    new LightGBMClassificationModel(uid, lightGBMBooster, labelColName, featuresColName,
      predictionColName, probColName, rawPredictionColName, None, actualNumClasses)
  }
}
