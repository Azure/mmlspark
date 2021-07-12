// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm.params

import com.microsoft.ml.spark.lightgbm.{LightGBMConstants, LightGBMDelegate}

/** Defines the common Booster parameters passed to the LightGBM learners.
  */
abstract class TrainParams extends Serializable {
  def parallelism: String
  def topK: Int
  def numIterations: Int
  def learningRate: Double
  def numLeaves: Int
  def maxBin: Int
  def binSampleCount: Int
  def baggingFraction: Double
  def posBaggingFraction: Double
  def negBaggingFraction: Double
  def baggingFreq: Int
  def baggingSeed: Int
  def earlyStoppingRound: Int
  def improvementTolerance: Double
  def featureFraction: Double
  def maxDepth: Int
  def minSumHessianInLeaf: Double
  def numMachines: Int
  def modelString: Option[String]
  def verbosity: Int
  def categoricalFeatures: Array[Int]
  def boostingType: String
  def lambdaL1: Double
  def lambdaL2: Double
  def isProvideTrainingMetric: Boolean
  def metric: String
  def minGainToSplit: Double
  def maxDeltaStep: Double
  def maxBinByFeature: Array[Int]
  def minDataInLeaf: Int
  def featureNames: Array[String]
  def delegate: Option[LightGBMDelegate]
  def dartModeParams: DartModeParams
  def executionParams: ExecutionParams
  def objectiveParams: ObjectiveParams
  def deviceType: String

  override def toString: String = {
    // Since passing `isProvideTrainingMetric` to LightGBM as a config parameter won't work,
    // let's fetch and print training metrics in `TrainUtils.scala` through JNI.
    s"is_pre_partition=True boosting_type=$boostingType tree_learner=$parallelism top_k=$topK " +
      s"num_iterations=$numIterations learning_rate=$learningRate num_leaves=$numLeaves " +
      s"max_bin=$maxBin bagging_fraction=$baggingFraction pos_bagging_fraction=$posBaggingFraction " +
      s"neg_bagging_fraction=$negBaggingFraction bagging_freq=$baggingFreq " +
      s"bagging_seed=$baggingSeed early_stopping_round=$earlyStoppingRound " +
      s"feature_fraction=$featureFraction max_depth=$maxDepth min_sum_hessian_in_leaf=$minSumHessianInLeaf " +
      s"num_machines=$numMachines verbosity=$verbosity " +
      s"lambda_l1=$lambdaL1 lambda_l2=$lambdaL2 metric=$metric min_gain_to_split=$minGainToSplit " +
      s"max_delta_step=$maxDeltaStep min_data_in_leaf=$minDataInLeaf ${objectiveParams.toString()} " +
      (if (categoricalFeatures.isEmpty) "" else s"categorical_feature=${categoricalFeatures.mkString(",")} ") +
      (if (maxBinByFeature.isEmpty) "" else s"max_bin_by_feature=${maxBinByFeature.mkString(",")} ") +
      (if (boostingType == "dart") s"${dartModeParams.toString()} " else "") +
      executionParams.toString() +
      s"device_type=$deviceType"
  }
}

/** Defines the Booster parameters passed to the LightGBM classifier.
  */
case class ClassifierTrainParams(parallelism: String, topK: Int, numIterations: Int, learningRate: Double,
                                 numLeaves: Int, maxBin: Int, binSampleCount: Int,
                                 baggingFraction: Double, posBaggingFraction: Double, negBaggingFraction: Double,
                                 baggingFreq: Int, baggingSeed: Int, earlyStoppingRound: Int,
                                 improvementTolerance: Double, featureFraction: Double,
                                 maxDepth: Int, minSumHessianInLeaf: Double,
                                 numMachines: Int, modelString: Option[String], isUnbalance: Boolean,
                                 verbosity: Int, categoricalFeatures: Array[Int], numClass: Int,
                                 boostFromAverage: Boolean, boostingType: String, lambdaL1: Double, lambdaL2: Double,
                                 isProvideTrainingMetric: Boolean, metric: String, minGainToSplit: Double,
                                 maxDeltaStep: Double, maxBinByFeature: Array[Int], minDataInLeaf: Int,
                                 featureNames: Array[String], delegate: Option[LightGBMDelegate],
                                 dartModeParams: DartModeParams, executionParams: ExecutionParams,
                                 objectiveParams: ObjectiveParams, deviceType: String)
  extends TrainParams {
  override def toString(): String = {
    val extraStr =
      if (objectiveParams.objective != LightGBMConstants.BinaryObjective) s"num_class=$numClass"
      else s"is_unbalance=${isUnbalance.toString}"
    s"metric=$metric boost_from_average=${boostFromAverage.toString} ${super.toString()} $extraStr"
  }
}

/** Defines the Booster parameters passed to the LightGBM regressor.
  */
case class RegressorTrainParams(parallelism: String, topK: Int, numIterations: Int, learningRate: Double,
                                numLeaves: Int, alpha: Double, tweedieVariancePower: Double, maxBin: Int,
                                binSampleCount: Int, baggingFraction: Double, posBaggingFraction: Double,
                                negBaggingFraction: Double, baggingFreq: Int, baggingSeed: Int,
                                earlyStoppingRound: Int, improvementTolerance: Double, featureFraction: Double,
                                maxDepth: Int, minSumHessianInLeaf: Double, numMachines: Int,
                                modelString: Option[String], verbosity: Int,
                                categoricalFeatures: Array[Int], boostFromAverage: Boolean,
                                boostingType: String, lambdaL1: Double, lambdaL2: Double,
                                isProvideTrainingMetric: Boolean, metric: String, minGainToSplit: Double,
                                maxDeltaStep: Double, maxBinByFeature: Array[Int], minDataInLeaf: Int,
                                featureNames: Array[String], delegate: Option[LightGBMDelegate],
                                dartModeParams: DartModeParams, executionParams: ExecutionParams,
                                objectiveParams: ObjectiveParams, deviceType: String)
  extends TrainParams {
  override def toString(): String = {
    s"alpha=$alpha tweedie_variance_power=$tweedieVariancePower boost_from_average=${boostFromAverage.toString} " +
      s"${super.toString()}"
  }
}

/** Defines the Booster parameters passed to the LightGBM ranker.
  */
case class RankerTrainParams(parallelism: String, topK: Int, numIterations: Int, learningRate: Double,
                             numLeaves: Int, maxBin: Int, binSampleCount: Int, baggingFraction: Double,
                             posBaggingFraction: Double, negBaggingFraction: Double, baggingFreq: Int,
                             baggingSeed: Int, earlyStoppingRound: Int, improvementTolerance: Double,
                             featureFraction: Double, maxDepth: Int, minSumHessianInLeaf: Double, numMachines: Int,
                             modelString: Option[String], verbosity: Int,
                             categoricalFeatures: Array[Int], boostingType: String,
                             lambdaL1: Double, lambdaL2: Double, maxPosition: Int,
                             labelGain: Array[Double], isProvideTrainingMetric: Boolean,
                             metric: String, evalAt: Array[Int], minGainToSplit: Double,
                             maxDeltaStep: Double, maxBinByFeature: Array[Int], minDataInLeaf: Int,
                             featureNames: Array[String], delegate: Option[LightGBMDelegate],
                             dartModeParams: DartModeParams, executionParams: ExecutionParams,
                             objectiveParams: ObjectiveParams, deviceType: String)
  extends TrainParams {
  override def toString(): String = {
    val labelGainStr =
      if (labelGain.isEmpty) "" else s"label_gain=${labelGain.mkString(",")}"
    val evalAtStr =
      if (evalAt.isEmpty) "" else s"eval_at=${evalAt.mkString(",")}"
    s"max_position=$maxPosition $labelGainStr $evalAtStr ${super.toString()}"
  }
}

/** Defines the dart mode parameters passed to the LightGBM learners.
  */
case class DartModeParams(dropRate: Double, maxDrop: Int, skipDrop: Double,
                          xgboostDartMode: Boolean, uniformDrop: Boolean) extends Serializable {
  override def toString(): String = {
    s"drop_rate=$dropRate max_drop=$maxDrop skip_drop=$skipDrop xgboost_dart_mode=$xgboostDartMode " +
    s"uniform_drop=$uniformDrop "
  }
}

/** Defines parameters related to lightgbm execution in spark.
  *
  * @param chunkSize Advanced parameter to specify the chunk size for copying Java data to native.
  * @param matrixType Advanced parameter to specify whether the native lightgbm matrix
  *                   constructed should be sparse or dense.
  * @param numThreads The number of threads to run the native lightgbm training with on each worker.
  */
case class ExecutionParams(chunkSize: Int, matrixType: String, numThreads: Int) extends Serializable {
  override def toString(): String = {
    s"num_threads=$numThreads "
  }
}

/** Defines parameters related to the lightgbm objective function.
  *
  * @param objective The Objective. For regression applications, this can be:
  *                  regression_l2, regression_l1, huber, fair, poisson, quantile, mape, gamma or tweedie.
  *                  For classification applications, this can be: binary, multiclass, or multiclassova.
  * @param fobj      Customized objective function.
  *                  Should accept two parameters: preds, train_data, and return (grad, hess).
  */
case class ObjectiveParams(objective: String, fobj: Option[FObjTrait]) extends Serializable {
  override def toString(): String = {
    if (fobj.isEmpty) {
      s"objective=$objective "
    } else {
      ""
    }
  }
}
