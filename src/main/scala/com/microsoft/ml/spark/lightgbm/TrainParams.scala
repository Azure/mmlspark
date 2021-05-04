// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

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
  def objective: String
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

  override def toString: String = {
    // Since passing `isProvideTrainingMetric` to LightGBM as a config parameter won't work,
    // let's fetch and print training metrics in `TrainUtils.scala` through JNI.
    s"is_pre_partition=True boosting_type=$boostingType tree_learner=$parallelism top_k=$topK " +
      s"num_iterations=$numIterations learning_rate=$learningRate num_leaves=$numLeaves " +
      s"max_bin=$maxBin bagging_fraction=$baggingFraction pos_bagging_fraction=$posBaggingFraction " +
      s"neg_bagging_fraction=$negBaggingFraction bagging_freq=$baggingFreq " +
      s"bagging_seed=$baggingSeed early_stopping_round=$earlyStoppingRound " +
      s"feature_fraction=$featureFraction max_depth=$maxDepth min_sum_hessian_in_leaf=$minSumHessianInLeaf " +
      s"num_machines=$numMachines objective=$objective verbosity=$verbosity " +
      s"lambda_l1=$lambdaL1 lambda_l2=$lambdaL2  metric=$metric min_gain_to_split=$minGainToSplit " +
      s"max_delta_step=$maxDeltaStep min_data_in_leaf=$minDataInLeaf " +
      (if (categoricalFeatures.isEmpty) "" else s"categorical_feature=${categoricalFeatures.mkString(",")} ") +
      (if (maxBinByFeature.isEmpty) "" else s"max_bin_by_feature=${maxBinByFeature.mkString(",")}")
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
                                 numMachines: Int, objective: String, modelString: Option[String],
                                 isUnbalance: Boolean, verbosity: Int, categoricalFeatures: Array[Int],
                                 numClass: Int, boostFromAverage: Boolean,
                                 boostingType: String, lambdaL1: Double, lambdaL2: Double,
                                 isProvideTrainingMetric: Boolean, metric: String, minGainToSplit: Double,
                                 maxDeltaStep: Double, maxBinByFeature: Array[Int], minDataInLeaf: Int,
                                 featureNames: Array[String], delegate: Option[LightGBMDelegate])
  extends TrainParams {
  override def toString: String = {
    val extraStr =
      if (objective != LightGBMConstants.BinaryObjective) s"num_class=$numClass"
      else s"is_unbalance=${isUnbalance.toString}"
    s"metric=$metric boost_from_average=${boostFromAverage.toString} ${super.toString()} $extraStr"
  }
}

/** Defines the Booster parameters passed to the LightGBM regressor.
  */
case class RegressorTrainParams(parallelism: String, topK: Int, numIterations: Int, learningRate: Double,
                                numLeaves: Int, objective: String, alpha: Double,
                                tweedieVariancePower: Double, maxBin: Int, binSampleCount: Int,
                                baggingFraction: Double, posBaggingFraction: Double, negBaggingFraction: Double,
                                baggingFreq: Int, baggingSeed: Int, earlyStoppingRound: Int,
                                improvementTolerance: Double, featureFraction: Double,
                                maxDepth: Int, minSumHessianInLeaf: Double, numMachines: Int,
                                modelString: Option[String], verbosity: Int,
                                categoricalFeatures: Array[Int], boostFromAverage: Boolean,
                                boostingType: String, lambdaL1: Double, lambdaL2: Double,
                                isProvideTrainingMetric: Boolean, metric: String, minGainToSplit: Double,
                                maxDeltaStep: Double, maxBinByFeature: Array[Int], minDataInLeaf: Int,
                                featureNames: Array[String], delegate: Option[LightGBMDelegate])
  extends TrainParams {
  override def toString: String = {
    s"alpha=$alpha tweedie_variance_power=$tweedieVariancePower boost_from_average=${boostFromAverage.toString} " +
      s"${super.toString()}"
  }
}

/** Defines the Booster parameters passed to the LightGBM ranker.
  */
case class RankerTrainParams(parallelism: String, topK: Int, numIterations: Int, learningRate: Double,
                             numLeaves: Int, objective: String, maxBin: Int, binSampleCount: Int,
                             baggingFraction: Double, posBaggingFraction: Double, negBaggingFraction: Double,
                             baggingFreq: Int, baggingSeed: Int, earlyStoppingRound: Int, improvementTolerance: Double,
                             featureFraction: Double, maxDepth: Int, minSumHessianInLeaf: Double, numMachines: Int,
                             modelString: Option[String], verbosity: Int,
                             categoricalFeatures: Array[Int], boostingType: String,
                             lambdaL1: Double, lambdaL2: Double, maxPosition: Int,
                             labelGain: Array[Double], isProvideTrainingMetric: Boolean,
                             metric: String, evalAt: Array[Int], minGainToSplit: Double,
                             maxDeltaStep: Double, maxBinByFeature: Array[Int], minDataInLeaf: Int,
                             featureNames: Array[String], delegate: Option[LightGBMDelegate])
  extends TrainParams {
  override def toString: String = {
    val labelGainStr =
      if (labelGain.isEmpty) "" else s"label_gain=${labelGain.mkString(",")}"
    val evalAtStr =
      if (evalAt.isEmpty) "" else s"eval_at=${evalAt.mkString(",")}"
    s"max_position=$maxPosition $labelGainStr $evalAtStr ${super.toString()}"
  }
}
