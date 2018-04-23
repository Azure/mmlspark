// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.hyper

import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.Param

/** Provides good default hyperparameter ranges and values for sweeping.
  * Publicly visible to users so they can easily select the parameters for sweeping.
  */
object DefaultHyperparams {

  /** Defines the default hyperparameter range for logistic regression.
    * @return The hyperparameter search space for logistic regression.
    */
  def defaultRange(learner: LogisticRegression): Array[(Param[_], Dist[_])] = {
    new HyperparamBuilder()
      .addHyperparam(learner.regParam, new DoubleRangeHyperParam(0.001, 1.0))
      .addHyperparam(learner.elasticNetParam, new DoubleRangeHyperParam(0.001, 1.0))
      .addHyperparam(learner.maxIter, new IntRangeHyperParam(5, 10))
      .build()
  }

  /** Defines the default hyperparameter range for decision tree classifier
    * @return The hyperparameter search space for decision tree classifier
    */
  def defaultRange(learner: DecisionTreeClassifier): Array[(Param[_], Dist[_])] = {
    new HyperparamBuilder()
      .addHyperparam(learner.maxBins, new IntRangeHyperParam(16, 32))
      .addHyperparam(learner.maxDepth, new IntRangeHyperParam(2, 5))
      .addHyperparam(learner.minInfoGain, new DoubleRangeHyperParam(0.0, 0.5))
      .addHyperparam(learner.minInstancesPerNode, new IntRangeHyperParam(1, 8))
      .build()
  }

  /** Defines the default hyperparameter range for gradient boosted trees classifier
    * @return The hyperparameter search space for gradient boosted trees classifier
    */
  def defaultRange(learner: GBTClassifier): Array[(Param[_], Dist[_])] = {
    new HyperparamBuilder()
      .addHyperparam(learner.maxBins, new IntRangeHyperParam(16, 32))
      .addHyperparam(learner.maxDepth, new IntRangeHyperParam(2, 5))
      .addHyperparam(learner.minInfoGain, new DoubleRangeHyperParam(0.0, 0.5))
      .addHyperparam(learner.minInstancesPerNode, new IntRangeHyperParam(1, 8))
      .addHyperparam(learner.maxIter, new IntRangeHyperParam(10, 20))
      .addHyperparam(learner.stepSize, new DoubleRangeHyperParam(0.01, 1))
      .addHyperparam(learner.subsamplingRate, new DoubleRangeHyperParam(0.01, 1))
      .build()
  }

  /** Defines the default hyperparameter range for random forest classifier
    * @return The hyperparameter search space for random forest classifier
    */
  def defaultRange(learner: RandomForestClassifier): Array[(Param[_], Dist[_])] = {
    new HyperparamBuilder()
      .addHyperparam(learner.maxBins, new IntRangeHyperParam(16, 32))
      .addHyperparam(learner.maxDepth, new IntRangeHyperParam(2, 5))
      .addHyperparam(learner.minInfoGain, new DoubleRangeHyperParam(0.0, 0.5))
      .addHyperparam(learner.minInstancesPerNode, new IntRangeHyperParam(1, 8))
      .addHyperparam(learner.numTrees, new IntRangeHyperParam(10, 30))
      .addHyperparam(learner.subsamplingRate, new DoubleRangeHyperParam(0.01, 1))
      .build()
  }

  /** Defines the default hyperparameter range for multilayer perceptron classifier
    * @return The hyperparameter search space for multilayer perceptron classifier
    */
  def defaultRange(learner: MultilayerPerceptronClassifier): Array[(Param[_], Dist[_])] = {
    val layers = Array[Int](2, 5, 2)
    new HyperparamBuilder()
      .addHyperparam(learner.blockSize, new IntRangeHyperParam(10, 200))
      .addHyperparam(learner.maxIter, new IntRangeHyperParam(1, 5))
      .addHyperparam(learner.tol, new DoubleRangeHyperParam(1e-6, 1e-3))
      .addHyperparam(learner.layers, new DiscreteHyperParam[Array[Int]](List(layers)))
      .build()
  }

  /** Defines the default hyperparameter range for naive bayes classifier
    * @return The hyperparameter search space for naive bayes classifier
    */
  def defaultRange(learner: NaiveBayes): Array[(Param[_], Dist[_])] = {
    new HyperparamBuilder().addHyperparam(learner.smoothing, new DoubleRangeHyperParam(0, 1))
      .build()
  }

}
