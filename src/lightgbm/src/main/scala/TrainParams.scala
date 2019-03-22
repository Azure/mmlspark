// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

/** Defines the common Booster parameters passed to the LightGBM learners.
  */
abstract class TrainParams extends Serializable {
  def parallelism: String
  def numIterations: Int
  def learningRate: Double
  def numLeaves: Int
  def maxBin: Int
  def baggingFraction: Double
  def baggingFreq: Int
  def baggingSeed: Int
  def earlyStoppingRound: Int
  def featureFraction: Double
  def maxDepth: Int
  def minSumHessianInLeaf: Double
  def numMachines: Int
  def objective: String
  def modelString: String
  def verbosity: Int
  def categoricalFeatures: Array[Int]
  def boostFromAverage: Boolean
  def boostingType: String
  def autoscale: Boolean

  override def toString(): String = {
    s"is_pre_partition=True boosting_type=$boostingType tree_learner=$parallelism num_iterations=$numIterations " +
      s"learning_rate=$learningRate num_leaves=$numLeaves " +
      s"max_bin=$maxBin bagging_fraction=$baggingFraction bagging_freq=$baggingFreq " +
      s"bagging_seed=$baggingSeed early_stopping_round=$earlyStoppingRound " +
      s"feature_fraction=$featureFraction max_depth=$maxDepth min_sum_hessian_in_leaf=$minSumHessianInLeaf " +
      s"num_machines=$numMachines objective=$objective verbosity=$verbosity " +
      s"boost_from_average=${boostFromAverage.toString} " +
      (if (categoricalFeatures.isEmpty) "" else s"categorical_feature=${categoricalFeatures.mkString(",")}")
  }
}

/** Defines the Booster parameters passed to the LightGBM classifier.
  */
case class ClassifierTrainParams(val parallelism: String, val numIterations: Int, val learningRate: Double,
                                 val numLeaves: Int, val maxBin: Int, val baggingFraction: Double, val baggingFreq: Int,
                                 val baggingSeed: Int, val earlyStoppingRound: Int, val featureFraction: Double,
                                 val maxDepth: Int, val minSumHessianInLeaf: Double,
                                 val numMachines: Int, val objective: String, val modelString: String,
                                 val isUnbalance: Boolean, val verbosity: Int, val categoricalFeatures: Array[Int],
                                 val numClass: Option[Int], val metric: String, val boostFromAverage: Boolean,
                                 val boostingType: String, val autoscale: Boolean)
  extends TrainParams {
  override def toString(): String = {
    val extraStr =
      if (numClass.isDefined) s"num_class=${numClass.get}"
      else s"is_unbalance=${isUnbalance.toString}"
    s"metric=$metric ${super.toString} $extraStr"
  }
}

/** Defines the Booster parameters passed to the LightGBM regressor.
  */
case class RegressorTrainParams(val parallelism: String, val numIterations: Int, val learningRate: Double,
                                val numLeaves: Int, val objective: String, val alpha: Double,
                                val tweedieVariancePower: Double, val maxBin: Int,
                                val baggingFraction: Double, val baggingFreq: Int,
                                val baggingSeed: Int, val earlyStoppingRound: Int, val featureFraction: Double,
                                val maxDepth: Int, val minSumHessianInLeaf: Double, val numMachines: Int,
                                val modelString: String, val verbosity: Int, val categoricalFeatures: Array[Int],
                                val boostFromAverage: Boolean, val boostingType: String, val autoscale: Boolean)
  extends TrainParams {
  override def toString(): String = {
    s"alpha=$alpha tweedie_variance_power=$tweedieVariancePower ${super.toString}"
  }
}
