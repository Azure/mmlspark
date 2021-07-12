// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.spark.lightgbm.booster.LightGBMBooster
import com.microsoft.ml.spark.lightgbm.params.{ClassifierTrainParams, LightGBMModelParams,
  LightGBMPredictionParams, TrainParams}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}

object LightGBMClassifier extends DefaultParamsReadable[LightGBMClassifier]

/** Trains a LightGBM Classification model, a fast, distributed, high performance gradient boosting
  * framework based on decision tree algorithms.
  * For more information please see here: https://github.com/Microsoft/LightGBM.
  * For parameter information see here: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
  * @param uid The unique ID.
  */
class LightGBMClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, LightGBMClassifier, LightGBMClassificationModel]
  with LightGBMBase[LightGBMClassificationModel] with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LightGBMClassifier"))

  // Set default objective to be binary classification
  setDefault(objective -> LightGBMConstants.BinaryObjective)

  val isUnbalance = new BooleanParam(this, "isUnbalance",
    "Set to true if training data is unbalanced in binary classification scenario")
  setDefault(isUnbalance -> false)

  def getIsUnbalance: Boolean = $(isUnbalance)
  def setIsUnbalance(value: Boolean): this.type = set(isUnbalance, value)

  def getTrainParams(numTasks: Int, categoricalIndexes: Array[Int], dataset: Dataset[_]): TrainParams = {
    /* The native code for getting numClasses is always 1 unless it is multiclass-classification problem
     * so we infer the actual numClasses from the dataset here
     */
    val actualNumClasses = getNumClasses(dataset)
    val modelStr = if (getModelString == null || getModelString.isEmpty) None else get(modelString)
    ClassifierTrainParams(getParallelism, getTopK, getNumIterations, getLearningRate, getNumLeaves, getMaxBin,
      getBinSampleCount, getBaggingFraction, getPosBaggingFraction, getNegBaggingFraction,
      getBaggingFreq, getBaggingSeed, getEarlyStoppingRound, getImprovementTolerance,
      getFeatureFraction, getMaxDepth, getMinSumHessianInLeaf, numTasks, modelStr,
      getIsUnbalance, getVerbosity, categoricalIndexes, actualNumClasses, getBoostFromAverage,
      getBoostingType, getLambdaL1, getLambdaL2, getIsProvideTrainingMetric,
      getMetric, getMinGainToSplit, getMaxDeltaStep, getMaxBinByFeature, getMinDataInLeaf, getSlotNames,
      getDelegate, getDartParams(), getExecutionParams(), getObjectiveParams(), getDeviceType)
  }

  def getModel(trainParams: TrainParams, lightGBMBooster: LightGBMBooster): LightGBMClassificationModel = {
    val classifierTrainParams = trainParams.asInstanceOf[ClassifierTrainParams]
    val model = new LightGBMClassificationModel(uid)
      .setLightGBMBooster(lightGBMBooster)
      .setFeaturesCol(getFeaturesCol)
      .setPredictionCol(getPredictionCol)
      .setProbabilityCol(getProbabilityCol)
      .setRawPredictionCol(getRawPredictionCol)
      .setLeafPredictionCol(getLeafPredictionCol)
      .setFeaturesShapCol(getFeaturesShapCol)
      .setActualNumClasses(classifierTrainParams.numClass)
      .setNumIterations(lightGBMBooster.bestIteration)
    if (isDefined(thresholds)) model.setThresholds(getThresholds) else model
  }

  def stringFromTrainedModel(model: LightGBMClassificationModel): String = {
    model.getModel.modelStr.get
  }

  override def copy(extra: ParamMap): LightGBMClassifier = defaultCopy(extra)
}

/** Special parameter for classification model for actual number of classes in dataset
  */
trait HasActualNumClasses extends Params {
  val actualNumClasses = new IntParam(this, "actualNumClasses",
    "Inferred number of classes based on dataset metadata or, if there is no metadata, unique count")

  def getActualNumClasses: Int = $(actualNumClasses)
  def setActualNumClasses(value: Int): this.type = set(actualNumClasses, value)
}

/** Model produced by [[LightGBMClassifier]]. */
class LightGBMClassificationModel(override val uid: String)
    extends ProbabilisticClassificationModel[Vector, LightGBMClassificationModel]
      with LightGBMModelParams with LightGBMModelMethods with LightGBMPredictionParams
      with HasActualNumClasses with ComplexParamsWritable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LightGBMClassificationModel"))

  override protected lazy val pyInternalWrapper = true

  /**
    * Implementation based on ProbabilisticClassifier with slight modifications to
    * avoid calling raw2probabilityInPlace to defer the probability calculation
    * to lightgbm native code.
    *
    * @param dataset input dataset
    * @return transformed dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      updateBoosterParamsBeforePredict()
      transformSchema(dataset.schema, logging = true)
      if (isDefined(thresholds)) {
        require(getThresholds.length == numClasses, this.getClass.getSimpleName +
          ".transform() called with non-matching numClasses and thresholds.length." +
          s" numClasses=$numClasses, but thresholds has length ${getThresholds.length}")
      }

      // Output selected columns only.
      var outputData = dataset
      var numColsOutput = 0
      if (getRawPredictionCol.nonEmpty) {
        val predictRawUDF = udf(predictRaw _)
        outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)))
        numColsOutput += 1
      }
      if (getProbabilityCol.nonEmpty) {
        val probabilityUDF = udf(predictProbability _)
        outputData = outputData.withColumn(getProbabilityCol, probabilityUDF(col(getFeaturesCol)))
        numColsOutput += 1
      }
      if (getPredictionCol.nonEmpty) {
        val predUDF = predictColumn
        outputData = outputData.withColumn(getPredictionCol, predUDF)
        numColsOutput += 1
      }
      if (getLeafPredictionCol.nonEmpty) {
        val predLeafUDF = udf(predictLeaf _)
        outputData = outputData.withColumn(getLeafPredictionCol, predLeafUDF(col(getFeaturesCol)))
        numColsOutput += 1
      }
      if (getFeaturesShapCol.nonEmpty) {
        val featureShapUDF = udf(featuresShap _)
        outputData = outputData.withColumn(getFeaturesShapCol, featureShapUDF(col(getFeaturesCol)))
        numColsOutput += 1
      }

      if (numColsOutput == 0) {
        this.logWarning(s"$uid: LightGBMClassificationModel.transform() was called as NOOP" +
          " since no output columns were set.")
      }
      outputData.toDF
    })
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    throw new NotImplementedError("Unexpected error in LightGBMClassificationModel:" +
      " raw2probabilityInPlace should not be called!")
  }

  override def numClasses: Int = getActualNumClasses

  override def predictRaw(features: Vector): Vector = {
    Vectors.dense(getModel.score(features, true, true))
  }

  override def predictProbability(features: Vector): Vector = {
    Vectors.dense(getModel.score(features, false, true))
  }

  override def copy(extra: ParamMap): LightGBMClassificationModel = defaultCopy(extra)

  protected def predictColumn: Column = {
    if (getRawPredictionCol.nonEmpty && !isDefined(thresholds)) {
      // Note: Only call raw2prediction if thresholds not defined
      udf(raw2prediction _).apply(col(getRawPredictionCol))
    } else if (getProbabilityCol.nonEmpty) {
      udf(probability2prediction _).apply(col(getProbabilityCol))
    } else {
      udf(predict _).apply(col(getFeaturesCol))
    }
  }

  def saveNativeModel(filename: String, overwrite: Boolean): Unit = {
    val session = SparkSession.builder().getOrCreate()
    getModel.saveNativeModel(session, filename, overwrite)
  }
}

object LightGBMClassificationModel extends ComplexParamsReadable[LightGBMClassificationModel] {
  def loadNativeModelFromFile(filename: String): LightGBMClassificationModel = {
    val uid = Identifiable.randomUID("LightGBMClassificationModel")
    val session = SparkSession.builder().getOrCreate()
    val textRdd = session.read.text(filename)
    val text = textRdd.collect().map { row => row.getString(0) }.mkString("\n")
    val lightGBMBooster = new LightGBMBooster(text)
    val actualNumClasses = lightGBMBooster.numClasses
    new LightGBMClassificationModel(uid).setLightGBMBooster(lightGBMBooster).setActualNumClasses(actualNumClasses)
  }

  def loadNativeModelFromString(model: String): LightGBMClassificationModel = {
    val uid = Identifiable.randomUID("LightGBMClassificationModel")
    val lightGBMBooster = new LightGBMBooster(model)
    val actualNumClasses = lightGBMBooster.numClasses
    new LightGBMClassificationModel(uid).setLightGBMBooster(lightGBMBooster).setActualNumClasses(actualNumClasses)
  }
}
