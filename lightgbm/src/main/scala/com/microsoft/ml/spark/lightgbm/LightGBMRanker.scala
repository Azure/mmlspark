// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.spark.lightgbm.booster.LightGBMBooster
import com.microsoft.ml.spark.lightgbm.params.{LightGBMModelParams, LightGBMPredictionParams,
  RankerTrainParams, TrainParams}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Ranker, RankerModel}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.DataType

object LightGBMRanker extends DefaultParamsReadable[LightGBMRanker]

/** Trains a LightGBMRanker model, a fast, distributed, high performance gradient boosting
  * framework based on decision tree algorithms.
  * For more information please see here: https://github.com/Microsoft/LightGBM.
  * For parameter information see here: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
  * @param uid The unique ID.
  */
class LightGBMRanker(override val uid: String)
  extends Ranker[Vector, LightGBMRanker, LightGBMRankerModel]
    with LightGBMBase[LightGBMRankerModel] with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LightGBMRanker"))

  // Set default objective to be ranking classification
  setDefault(objective -> LightGBMConstants.RankObjective)

  val maxPosition = new IntParam(this, "maxPosition", "optimized NDCG at this position")
  setDefault(maxPosition -> 20)

  def getMaxPosition: Int = $(maxPosition)
  def setMaxPosition(value: Int): this.type = set(maxPosition, value)

  val labelGain = new DoubleArrayParam(this, "labelGain", "graded relevance for each label in NDCG")
  setDefault(labelGain -> Array.empty[Double])

  def getLabelGain: Array[Double] = $(labelGain)
  def setLabelGain(value: Array[Double]): this.type = set(labelGain, value)

  val evalAt = new IntArrayParam(this, "evalAt", "NDCG and MAP evaluation positions, separated by comma")
  setDefault(evalAt -> (1 to 5).toArray)

  def getEvalAt: Array[Int] = $(evalAt)
  def setEvalAt(value: Array[Int]): this.type = set(evalAt, value)

  def getTrainParams(numTasks: Int, categoricalIndexes: Array[Int], dataset: Dataset[_]): TrainParams = {
    val modelStr = if (getModelString == null || getModelString.isEmpty) None else get(modelString)
    RankerTrainParams(getParallelism, getTopK, getNumIterations, getLearningRate, getNumLeaves,
      getMaxBin, getBinSampleCount, getBaggingFraction, getPosBaggingFraction, getNegBaggingFraction,
      getBaggingFreq, getBaggingSeed, getEarlyStoppingRound, getImprovementTolerance,
      getFeatureFraction, getMaxDepth, getMinSumHessianInLeaf, numTasks, modelStr,
      getVerbosity, categoricalIndexes, getBoostingType, getLambdaL1, getLambdaL2, getMaxPosition, getLabelGain,
      getIsProvideTrainingMetric, getMetric, getEvalAt, getMinGainToSplit, getMaxDeltaStep,
      getMaxBinByFeature, getMinDataInLeaf, getSlotNames, getDelegate, getDartParams(), getExecutionParams(),
      getObjectiveParams(), getDeviceType)
  }

  def getModel(trainParams: TrainParams, lightGBMBooster: LightGBMBooster): LightGBMRankerModel = {
    new LightGBMRankerModel(uid)
      .setLightGBMBooster(lightGBMBooster)
      .setFeaturesCol(getFeaturesCol)
      .setPredictionCol(getPredictionCol)
      .setLeafPredictionCol(getLeafPredictionCol)
      .setFeaturesShapCol(getFeaturesShapCol)
      .setNumIterations(lightGBMBooster.bestIteration)
  }

  def stringFromTrainedModel(model: LightGBMRankerModel): String = {
    model.getModel.modelStr.get
  }

  override def getOptGroupCol: Option[String] = Some(getGroupCol)

  /** For Ranking, we need to sort the data within partitions by group prior to training to ensure training succeeds.
    * @param dataset The dataset to preprocess prior to training.
    * @return The preprocessed data, sorted within partiton by group.
    */
  override def preprocessData(dataset: DataFrame): DataFrame = {
    dataset.sortWithinPartitions(getOptGroupCol.get)
  }

  override def copy(extra: ParamMap): LightGBMRanker = defaultCopy(extra)

  override def prepareDataframe(dataset: Dataset[_], trainingCols: Array[(String, Seq[DataType])],
                                numTasks: Int): DataFrame = {
    if (getRepartitionByGroupingColumn) {
      val repartitionedDataset = getOptGroupCol match {
        case None => dataset
        case Some(groupingCol) => {
          val df = dataset.repartition(new Column(groupingCol)).cache()
          //force materialization
          df.count
          df
        }
      }
      super.prepareDataframe(repartitionedDataset, trainingCols, numTasks)
    } else {
      super.prepareDataframe(dataset, trainingCols, numTasks)
    }
  }
}

/** Model produced by [[LightGBMRanker]]. */
class LightGBMRankerModel(override val uid: String)
  extends RankerModel[Vector, LightGBMRankerModel]
    with LightGBMModelParams
    with LightGBMModelMethods
    with LightGBMPredictionParams
    with ComplexParamsWritable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LightGBMRankerModel"))

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

  override def copy(extra: ParamMap): LightGBMRankerModel = defaultCopy(extra)

  override def numFeatures: Int = getModel.numFeatures

  def saveNativeModel(filename: String, overwrite: Boolean): Unit = {
    val session = SparkSession.builder().getOrCreate()
    getModel.saveNativeModel(session, filename, overwrite)
  }
}

object LightGBMRankerModel extends ComplexParamsReadable[LightGBMRankerModel] {
  def loadNativeModelFromFile(filename: String): LightGBMRankerModel = {
    val uid = Identifiable.randomUID("LightGBMRankerModel")
    val session = SparkSession.builder().getOrCreate()
    val textRdd = session.read.text(filename)
    val text = textRdd.collect().map { row => row.getString(0) }.mkString("\n")
    val lightGBMBooster = new LightGBMBooster(text)
    new LightGBMRankerModel(uid).setLightGBMBooster(lightGBMBooster)
  }

  def loadNativeModelFromString(model: String): LightGBMRankerModel = {
    val uid = Identifiable.randomUID("LightGBMRankerModel")
    val lightGBMBooster = new LightGBMBooster(model)
    new LightGBMRankerModel(uid).setLightGBMBooster(lightGBMBooster)
  }
}
