// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm.params

import com.microsoft.ml.spark.codegen.Wrappable
import com.microsoft.ml.spark.core.contracts.{HasInitScoreCol, HasValidationIndicatorCol, HasWeightCol}
import com.microsoft.ml.spark.lightgbm.booster.LightGBMBooster
import com.microsoft.ml.spark.lightgbm.{LightGBMConstants, LightGBMDelegate}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.DefaultParamsWritable

/** Defines common LightGBM execution parameters.
  */
trait LightGBMExecutionParams extends Wrappable {
  val parallelism = new Param[String](this, "parallelism",
    "Tree learner parallelism, can be set to data_parallel or voting_parallel")
  setDefault(parallelism->"data_parallel")

  def getParallelism: String = $(parallelism)
  def setParallelism(value: String): this.type = set(parallelism, value)

  val topK = new IntParam(this, "topK",
    "The top_k value used in Voting parallel, " +
      "set this to larger value for more accurate result, but it will slow down the training speed. " +
      "It should be greater than 0")
  setDefault(topK -> LightGBMConstants.DefaultTopK)

  def getTopK: Int = $(topK)
  def setTopK(value: Int): this.type = set(topK, value)

  val defaultListenPort = new IntParam(this, "defaultListenPort",
    "The default listen port on executors, used for testing")

  def getDefaultListenPort: Int = $(defaultListenPort)
  def setDefaultListenPort(value: Int): this.type = set(defaultListenPort, value)

  setDefault(defaultListenPort -> LightGBMConstants.DefaultLocalListenPort)

  val driverListenPort = new IntParam(this, "driverListenPort",
    "The listen port on a driver. Default value is 0 (random)")

  def getDriverListenPort: Int = $(driverListenPort)
  def setDriverListenPort(value: Int): this.type = set(driverListenPort, value)

  setDefault(driverListenPort -> LightGBMConstants.DefaultDriverListenPort)

  val timeout = new DoubleParam(this, "timeout", "Timeout in seconds")
  setDefault(timeout -> 1200)

  def getTimeout: Double = $(timeout)
  def setTimeout(value: Double): this.type = set(timeout, value)

  val useBarrierExecutionMode = new BooleanParam(this, "useBarrierExecutionMode",
    "Use new barrier execution mode in Beta testing, off by default.")
  setDefault(useBarrierExecutionMode -> false)

  def getUseBarrierExecutionMode: Boolean = $(useBarrierExecutionMode)
  def setUseBarrierExecutionMode(value: Boolean): this.type = set(useBarrierExecutionMode, value)

  val numBatches = new IntParam(this, "numBatches",
    "If greater than 0, splits data into separate batches during training")
  setDefault(numBatches -> 0)

  def getNumBatches: Int = $(numBatches)
  def setNumBatches(value: Int): this.type = set(numBatches, value)

  val repartitionByGroupingColumn = new BooleanParam(this, "repartitionByGroupingColumn",
    "Repartition training data according to grouping column, on by default.")
  setDefault(repartitionByGroupingColumn -> true)

  def getRepartitionByGroupingColumn: Boolean = $(repartitionByGroupingColumn)
  def setRepartitionByGroupingColumn(value: Boolean): this.type = set(repartitionByGroupingColumn, value)

  val numTasks = new IntParam(this, "numTasks",
    "Advanced parameter to specify the number of tasks.  " +
      "MMLSpark tries to guess this based on cluster configuration, but this parameter can be used to override.")
  setDefault(numTasks -> 0)

  def getNumTasks: Int = $(numTasks)
  def setNumTasks(value: Int): this.type = set(numTasks, value)

  val chunkSize = new IntParam(this, "chunkSize",
    "Advanced parameter to specify the chunk size for copying Java data to native.  " +
      "If set too high, memory may be wasted, but if set too low, performance may be reduced during data copy." +
      "If dataset size is known beforehand, set to the number of rows in the dataset.")
  setDefault(chunkSize -> 10000)

  def getChunkSize: Int = $(chunkSize)
  def setChunkSize(value: Int): this.type = set(chunkSize, value)

  val matrixType = new Param[String](this, "matrixType",
    "Advanced parameter to specify whether the native lightgbm matrix constructed should be sparse or dense.  " +
      "Values can be auto, sparse or dense. Default value is auto, which samples first ten rows to determine type.")
  setDefault(matrixType -> "auto")

  def getMatrixType: String = $(matrixType)
  def setMatrixType(value: String): this.type = set(matrixType, value)

  val numThreads = new IntParam(this, "numThreads",
    "Number of threads for LightGBM. For the best speed, set this to the number of real CPU cores.")
  setDefault(numThreads -> 0)

  def getNumThreads: Int = $(numThreads)
  def setNumThreads(value: Int): this.type = set(numThreads, value)
}

/** Defines common parameters across all LightGBM learners related to learning score evolution.
  */
trait LightGBMLearnerParams extends Wrappable {
  val earlyStoppingRound = new IntParam(this, "earlyStoppingRound", "Early stopping round")
  setDefault(earlyStoppingRound -> 0)

  def getEarlyStoppingRound: Int = $(earlyStoppingRound)
  def setEarlyStoppingRound(value: Int): this.type = set(earlyStoppingRound, value)

  val improvementTolerance = new DoubleParam(this, "improvementTolerance",
    "Tolerance to consider improvement in metric")
  setDefault(improvementTolerance -> 0.0)

  def getImprovementTolerance: Double = $(improvementTolerance)
  def setImprovementTolerance(value: Double): this.type = set(improvementTolerance, value)
}

/** Defines common parameters across all LightGBM learners related to histogram bin construction.
  */
trait LightGBMBinParams extends Wrappable {
  val maxBin = new IntParam(this, "maxBin", "Max bin")
  setDefault(maxBin -> 255)

  def getMaxBin: Int = $(maxBin)
  def setMaxBin(value: Int): this.type = set(maxBin, value)

  val binSampleCount = new IntParam(this, "binSampleCount", "Number of samples considered at computing histogram bins")
  setDefault(binSampleCount -> 200000)

  def getBinSampleCount: Int = $(binSampleCount)
  def setBinSampleCount(value: Int): this.type = set(binSampleCount, value)
}

/** Defines parameters for dart mode across all LightGBM learners.
 */
trait LightGBMDartParams extends Wrappable {
  val dropRate = new DoubleParam(this, "dropRate",
    "Dropout rate: a fraction of previous trees to drop during the dropout")
  setDefault(dropRate -> 0.1)

  def getDropRate: Double = $(dropRate)
  def setDropRate(value: Double): this.type = set(dropRate, value)

  val maxDrop = new IntParam(this, "maxDrop",
    "Max number of dropped trees during one boosting iteration")
  setDefault(maxDrop -> 50)

  def getMaxDrop: Int = $(maxDrop)
  def setMaxDrop(value: Int): this.type = set(maxDrop, value)

  val skipDrop = new DoubleParam(this, "skipDrop",
    "Probability of skipping the dropout procedure during a boosting iteration")
  setDefault(skipDrop -> 0.5)

  def getSkipDrop: Double = $(skipDrop)
  def setSkipDrop(value: Double): this.type = set(skipDrop, value)

  val xgboostDartMode = new BooleanParam(this, "xgboostDartMode",
    "Set this to true to use xgboost dart mode")
  setDefault(xgboostDartMode -> false)

  def getXGBoostDartMode: Boolean = $(xgboostDartMode)
  def setXGBoostDartMode(value: Boolean): this.type = set(xgboostDartMode, value)

  val uniformDrop = new BooleanParam(this, "uniformDrop",
    "Set this to true to use uniform drop in dart mode")
  setDefault(uniformDrop -> false)

  def getUniformDrop: Boolean = $(uniformDrop)
  def setUniformDrop(value: Boolean): this.type = set(uniformDrop, value)
}

/** Defines parameters for slots across all LightGBM learners.
 */
trait LightGBMSlotParams extends Wrappable {
  val slotNames = new StringArrayParam(this, "slotNames",
    "List of slot names in the features column")

  def getSlotNames: Array[String] = $(slotNames)
  def setSlotNames(value: Array[String]): this.type = set(slotNames, value)

  setDefault(slotNames -> Array.empty)

  val categoricalSlotIndexes = new IntArrayParam(this, "categoricalSlotIndexes",
    "List of categorical column indexes, the slot index in the features column")

  def getCategoricalSlotIndexes: Array[Int] = $(categoricalSlotIndexes)
  def setCategoricalSlotIndexes(value: Array[Int]): this.type = set(categoricalSlotIndexes, value)

  setDefault(categoricalSlotIndexes -> Array.empty)

  val categoricalSlotNames = new StringArrayParam(this, "categoricalSlotNames",
    "List of categorical column slot names, the slot name in the features column")

  def getCategoricalSlotNames: Array[String] = $(categoricalSlotNames)
  def setCategoricalSlotNames(value: Array[String]): this.type = set(categoricalSlotNames, value)

  setDefault(categoricalSlotNames -> Array.empty)
}

/** Defines parameters for fraction across all LightGBM learners.
  */
trait LightGBMFractionParams extends Wrappable {
  val baggingFraction = new DoubleParam(this, "baggingFraction", "Bagging fraction")
  setDefault(baggingFraction->1)

  def getBaggingFraction: Double = $(baggingFraction)
  def setBaggingFraction(value: Double): this.type = set(baggingFraction, value)

  val posBaggingFraction = new DoubleParam(this, "posBaggingFraction", "Positive Bagging fraction")
  setDefault(posBaggingFraction->1)

  def getPosBaggingFraction: Double = $(posBaggingFraction)
  def setPosBaggingFraction(value: Double): this.type = set(posBaggingFraction, value)

  val negBaggingFraction = new DoubleParam(this, "negBaggingFraction", "Negative Bagging fraction")
  setDefault(negBaggingFraction->1)

  def getNegBaggingFraction: Double = $(negBaggingFraction)
  def setNegBaggingFraction(value: Double): this.type = set(negBaggingFraction, value)

  val featureFraction = new DoubleParam(this, "featureFraction", "Feature fraction")
  setDefault(featureFraction->1)

  def getFeatureFraction: Double = $(featureFraction)
  def setFeatureFraction(value: Double): this.type = set(featureFraction, value)
}

/** Defines common prediction parameters across LightGBM Ranker, Classifier and Regressor
  */
trait LightGBMPredictionParams extends Wrappable {
  val leafPredictionCol = new Param[String](this, "leafPredictionCol",
    "Predicted leaf indices's column name")
  setDefault(leafPredictionCol -> "")

  def getLeafPredictionCol: String = $(leafPredictionCol)
  def setLeafPredictionCol(value: String): this.type = set(leafPredictionCol, value)

  val featuresShapCol = new Param[String](this, "featuresShapCol",
    "Output SHAP vector column name after prediction containing the feature contribution values")
  setDefault(featuresShapCol -> "")

  def getFeaturesShapCol: String = $(featuresShapCol)
  def setFeaturesShapCol(value: String): this.type = set(featuresShapCol, value)
}

/** Defines parameters for LightGBM models
  */
trait LightGBMModelParams extends Wrappable {
  val lightGBMBooster = new LightGBMBoosterParam(this, "lightGBMBooster",
    "The trained LightGBM booster")

  def getLightGBMBooster: LightGBMBooster = $(lightGBMBooster)
  def setLightGBMBooster(value: LightGBMBooster): this.type = set(lightGBMBooster, value)

  /**
    * Alias for same method
    * @return The LightGBM Booster.
    */
  def getModel: LightGBMBooster = this.getLightGBMBooster

  val startIteration = new IntParam(this, "startIteration",
    "Sets the start index of the iteration to predict. If <= 0, starts from the first iteration.")
  setDefault(startIteration -> LightGBMConstants.DefaultStartIteration)

  def getStartIteration: Int = $(startIteration)
  def setStartIteration(value: Int): this.type = set(startIteration, value)

  val numIterations = new IntParam(this, "numIterations",
    "Sets the total number of iterations used in the prediction." +
      "If <= 0, all iterations from ``start_iteration`` are used (no limits).")
  setDefault(numIterations -> LightGBMConstants.DefaultNumIterations)

  def getNumIterations: Int = $(numIterations)
  def setNumIterations(value: Int): this.type = set(numIterations, value)
}

/** Defines common objective parameters
  */
trait LightGBMObjectiveParams extends Wrappable {
  val objective = new Param[String](this, "objective",
    "The Objective. For regression applications, this can be: " +
      "regression_l2, regression_l1, huber, fair, poisson, quantile, mape, gamma or tweedie. " +
      "For classification applications, this can be: binary, multiclass, or multiclassova. ")
  setDefault(objective -> "regression")

  def getObjective: String = $(objective)
  def setObjective(value: String): this.type = set(objective, value)

  val fobj = new FObjParam(this, "fobj", "Customized objective function. " +
    "Should accept two parameters: preds, train_data, and return (grad, hess).")

  def getFObj: FObjTrait = $(fobj)
  def setFObj(value: FObjTrait): this.type = set(fobj, value)
}

/** Defines common parameters across all LightGBM learners.
  */
trait LightGBMParams extends Wrappable with DefaultParamsWritable with HasWeightCol
  with HasValidationIndicatorCol with HasInitScoreCol with LightGBMExecutionParams
  with LightGBMSlotParams with LightGBMFractionParams with LightGBMBinParams with LightGBMLearnerParams
  with LightGBMDartParams with LightGBMPredictionParams with LightGBMObjectiveParams {
  val numIterations = new IntParam(this, "numIterations",
    "Number of iterations, LightGBM constructs num_class * num_iterations trees")
  setDefault(numIterations->100)

  def getNumIterations: Int = $(numIterations)
  def setNumIterations(value: Int): this.type = set(numIterations, value)

  val learningRate = new DoubleParam(this, "learningRate", "Learning rate or shrinkage rate")
  setDefault(learningRate -> 0.1)

  def getLearningRate: Double = $(learningRate)
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  val numLeaves = new IntParam(this, "numLeaves", "Number of leaves")
  setDefault(numLeaves -> 31)

  def getNumLeaves: Int = $(numLeaves)
  def setNumLeaves(value: Int): this.type = set(numLeaves, value)

  val baggingFreq = new IntParam(this, "baggingFreq", "Bagging frequency")
  setDefault(baggingFreq->0)

  def getBaggingFreq: Int = $(baggingFreq)
  def setBaggingFreq(value: Int): this.type = set(baggingFreq, value)

  val baggingSeed = new IntParam(this, "baggingSeed", "Bagging seed")
  setDefault(baggingSeed->3)

  def getBaggingSeed: Int = $(baggingSeed)
  def setBaggingSeed(value: Int): this.type = set(baggingSeed, value)

  val maxDepth = new IntParam(this, "maxDepth", "Max depth")
  setDefault(maxDepth-> -1)

  def getMaxDepth: Int = $(maxDepth)
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  val minSumHessianInLeaf = new DoubleParam(this, "minSumHessianInLeaf", "Minimal sum hessian in one leaf")
  setDefault(minSumHessianInLeaf -> 1e-3)

  def getMinSumHessianInLeaf: Double = $(minSumHessianInLeaf)
  def setMinSumHessianInLeaf(value: Double): this.type = set(minSumHessianInLeaf, value)

  val modelString = new Param[String](this, "modelString", "LightGBM model to retrain")
  setDefault(modelString -> "")

  def getModelString: String = $(modelString)
  def setModelString(value: String): this.type = set(modelString, value)

  val verbosity = new IntParam(this, "verbosity",
    "Verbosity where lt 0 is Fatal, eq 0 is Error, eq 1 is Info, gt 1 is Debug")
  setDefault(verbosity -> -1)

  def getVerbosity: Int = $(verbosity)
  def setVerbosity(value: Int): this.type = set(verbosity, value)

  val boostFromAverage = new BooleanParam(this, "boostFromAverage",
    "Adjusts initial score to the mean of labels for faster convergence")
  setDefault(boostFromAverage -> true)

  def getBoostFromAverage: Boolean = $(boostFromAverage)
  def setBoostFromAverage(value: Boolean): this.type = set(boostFromAverage, value)

  val boostingType = new Param[String](this, "boostingType",
    "Default gbdt = traditional Gradient Boosting Decision Tree. Options are: " +
      "gbdt, gbrt, rf (Random Forest), random_forest, dart (Dropouts meet Multiple " +
      "Additive Regression Trees), goss (Gradient-based One-Side Sampling). ")
  setDefault(boostingType -> "gbdt")

  def getBoostingType: String = $(boostingType)
  def setBoostingType(value: String): this.type = set(boostingType, value)

  val lambdaL1 = new DoubleParam(this, "lambdaL1", "L1 regularization")
  setDefault(lambdaL1 -> 0.0)

  def getLambdaL1: Double = $(lambdaL1)
  def setLambdaL1(value: Double): this.type = set(lambdaL1, value)

  val lambdaL2 = new DoubleParam(this, "lambdaL2", "L2 regularization")
  setDefault(lambdaL2 -> 0.0)

  def getLambdaL2: Double = $(lambdaL2)
  def setLambdaL2(value: Double): this.type = set(lambdaL2, value)

  val isProvideTrainingMetric = new BooleanParam(this, "isProvideTrainingMetric",
    "Whether output metric result over training dataset.")
  setDefault(isProvideTrainingMetric -> false)

  def getIsProvideTrainingMetric: Boolean = $(isProvideTrainingMetric)
  def setIsProvideTrainingMetric(value: Boolean): this.type = set(isProvideTrainingMetric, value)

  val metric = new Param[String](this, "metric",
    "Metrics to be evaluated on the evaluation data.  Options are: " +
      "empty string or not specified means that metric corresponding to specified " +
      "objective will be used (this is possible only for pre-defined objective functions, " +
      "otherwise no evaluation metric will be added). " +
      "None (string, not a None value) means that no metric will be registered, a" +
      "liases: na, null, custom. " +
      "l1, absolute loss, aliases: mean_absolute_error, mae, regression_l1. " +
      "l2, square loss, aliases: mean_squared_error, mse, regression_l2, regression. " +
      "rmse, root square loss, aliases: root_mean_squared_error, l2_root. " +
      "quantile, Quantile regression. " +
      "mape, MAPE loss, aliases: mean_absolute_percentage_error. " +
      "huber, Huber loss. " +
      "fair, Fair loss. " +
      "poisson, negative log-likelihood for Poisson regression. " +
      "gamma, negative log-likelihood for Gamma regression. " +
      "gamma_deviance, residual deviance for Gamma regression. " +
      "tweedie, negative log-likelihood for Tweedie regression. " +
      "ndcg, NDCG, aliases: lambdarank. " +
      "map, MAP, aliases: mean_average_precision. " +
      "auc, AUC. " +
      "binary_logloss, log loss, aliases: binary. " +
      "binary_error, for one sample: 0 for correct classification, 1 for error classification. " +
      "multi_logloss, log loss for multi-class classification, aliases: multiclass, softmax, " +
      "multiclassova, multiclass_ova, ova, ovr. " +
      "multi_error, error rate for multi-class classification. " +
      "cross_entropy, cross-entropy (with optional linear weights), aliases: xentropy. " +
      "cross_entropy_lambda, intensity-weighted cross-entropy, aliases: xentlambda. " +
      "kullback_leibler, Kullback-Leibler divergence, aliases: kldiv. ")
  setDefault(metric -> "")

  def getMetric: String = $(metric)
  def setMetric(value: String): this.type = set(metric, value)

  val minGainToSplit = new DoubleParam(this, "minGainToSplit",
    "The minimal gain to perform split")
  setDefault(minGainToSplit -> 0.0)

  def getMinGainToSplit: Double = $(minGainToSplit)
  def setMinGainToSplit(value: Double): this.type = set(minGainToSplit, value)

  val maxDeltaStep = new DoubleParam(this, "maxDeltaStep",
    "Used to limit the max output of tree leaves")
  setDefault(maxDeltaStep -> 0.0)

  def getMaxDeltaStep: Double = $(maxDeltaStep)
  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  val maxBinByFeature = new IntArrayParam(this, "maxBinByFeature",
    "Max number of bins for each feature")
  setDefault(maxBinByFeature -> Array.empty)

  def getMaxBinByFeature: Array[Int] = $(maxBinByFeature)
  def setMaxBinByFeature(value: Array[Int]): this.type = set(maxBinByFeature, value)

  val minDataInLeaf = new IntParam(this, "minDataInLeaf",
    "Minimal number of data in one leaf. Can be used to deal with over-fitting.")
  setDefault(minDataInLeaf -> 20)

  def getMinDataInLeaf: Int = $(minDataInLeaf)
  def setMinDataInLeaf(value: Int): this.type = set(minDataInLeaf, value)

  var delegate: Option[LightGBMDelegate] = None
  def getDelegate: Option[LightGBMDelegate] = delegate
  def setDelegate(delegate: LightGBMDelegate): this.type = {
    this.delegate = Option(delegate)
    this
  }

  val deviceType = new Param[String](this, "deviceType",
    "Device for the tree learning, you can use GPU to achieve the faster learning")
  setDefault(deviceType -> "cpu")

  def getDeviceType: String = $(deviceType)
  def setDeviceType(value: String): this.type = set(deviceType, value)
}
