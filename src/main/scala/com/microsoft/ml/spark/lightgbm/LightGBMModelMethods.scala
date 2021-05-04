// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Vector, Vectors}

/** Contains common LightGBM model methods across all LightGBM learner types.
  */
trait LightGBMModelMethods extends LightGBMModelParams with Logging {
  /**
    * Public method to get the global feature importance values.
    * @param importanceType split or gini
    * @return The global feature importance values.
    */
  def getFeatureImportances(importanceType: String): Array[Double] = {
    getLightGBMBooster.getFeatureImportances(importanceType)
  }

  /**
    * Public method to get the vector local SHAP feature importance values for an instance.
    * @param features The local instance or row to compute the SHAP values for.
    * @return The local feature importance values.
    */
  def getFeatureShaps(features: Vector): Array[Double] = {
    getLightGBMBooster.featuresShap(features)
  }

  /**
    * Public method for pyspark API to get the dense local SHAP feature importance values for an instance.
    * @param features The local instance or row to compute the SHAP values for.
    * @return The local feature importance values.
    */
  def getDenseFeatureShaps(features: Array[Double]): Array[Double] = {
    getLightGBMBooster.featuresShap(Vectors.dense(features))
  }

  /**
    * Public method for pyspark API to get the sparse local SHAP feature importance values for an instance.
    * @param size: The size of the sparse vector.
    * @param indices: The local instance or row indices to compute the SHAP values for.
    * @param values: The local instance or row values to compute the SHAP values for.
    * @return The local feature importance values.
    */
  def getSparseFeatureShaps(size: Int, indices: Array[Int], values: Array[Double]): Array[Double] = {
    getLightGBMBooster.featuresShap(Vectors.sparse(size, indices, values))
  }

  /**
    * Public method to get the best iteration from the booster.
    * @return The best iteration, if early stopping was triggered.
    */
  def getBoosterBestIteration: Int = {
    getLightGBMBooster.bestIteration
  }

  /**
    * Public method to get the total number of iterations trained.
    * @return The total number of iterations trained.
    */
  def getBoosterNumTotalIterations: Int = {
    getLightGBMBooster.numTotalIterations
  }

  /**
    * Public method to get the total number of models trained.
    * Note this may be larger than the number of iterations,
    * since in multiclass a model is trained per class for
    * each iteration.
    * @return The total number of models.
    */
  def getBoosterNumTotalModel: Int = {
    getLightGBMBooster.numTotalModel
  }

  /**
    * Public method to get the number of features from the booster.
    * @return The number of features.
    */
  def getBoosterNumFeatures: Int = {
    getLightGBMBooster.numFeatures
  }

  /**
    * Public method to get the number of classes from the booster.
    * @return The number of classes.
    */
  def getBoosterNumClasses: Int = {
    getLightGBMBooster.numClasses
  }

  /**
    * Protected method to predict leaf index.
    * @param features The local instance or row to compute the leaf index for.
    * @return The predicted leaf index.
    */
  protected def predictLeaf(features: Vector): Vector = {
    Vectors.dense(getLightGBMBooster.predictLeaf(features))
  }

  /**
    * Protected method to predict local SHAP feature importance values for an instance.
    * @param features The local instance or row to compute the local SHAP values for.
    * @return The SHAP local feature importance values.
    */
  protected def featuresShap(features: Vector): Vector = {
    Vectors.dense(getLightGBMBooster.featuresShap(features))
  }

  protected def updateBoosterParamsBeforePredict(): Unit = {
    getModel.setNumIterations(getNumIterations)
    getModel.setStartIteration(getStartIteration)
  }
}
