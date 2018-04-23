// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.featurize

import com.microsoft.ml.spark.core.contracts.MMLParams
import com.microsoft.ml.spark.core.serialize.params.MapArrayParam
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel}
import org.apache.spark.sql._
import org.apache.spark.sql.types._

private[ml] object FeaturizeUtilities
{
  // 2^18 features by default
  val numFeaturesDefault = 262144
  // 2^12 features for tree-based or NN-based learners
  val numFeaturesTreeOrNNBased = 4096
}

object Featurize extends DefaultParamsReadable[Featurize]

/** Featurizes a dataset. Converts the specified columns to feature columns. */
class Featurize(override val uid: String) extends Estimator[PipelineModel] with MMLParams {

  def this() = this(Identifiable.randomUID("Featurize"))

  /** Feature columns - the columns to be featurized
    * @group param
    */
  val featureColumns: MapArrayParam = new MapArrayParam(this, "featureColumns", "Feature columns")

  /** @group getParam */
  final def getFeatureColumns: Map[String, Seq[String]] = $(featureColumns)

  /** @group setParam */
  def setFeatureColumns(value: Map[String, Seq[String]]): this.type = set(featureColumns, value)

  /** One hot encode categorical columns when true; default is true
    * @group param
    */
  val oneHotEncodeCategoricals: Param[Boolean] = BooleanParam(this,
    "oneHotEncodeCategoricals",
    "One-hot encode categoricals",
    true)

  /** @group getParam */
  final def getOneHotEncodeCategoricals: Boolean = $(oneHotEncodeCategoricals)

  /** @group setParam */
  def setOneHotEncodeCategoricals(value: Boolean): this.type = set(oneHotEncodeCategoricals, value)

  /** Number of features to hash string columns to
    * @group param
    */
  val numberOfFeatures: IntParam =
    IntParam(this,
      "numberOfFeatures",
      "Number of features to hash string columns to",
      FeaturizeUtilities.numFeaturesDefault)

  /** @group getParam */
  final def getNumberOfFeatures: Int = $(numberOfFeatures)

  /** @group setParam */
  def setNumberOfFeatures(value: Int): this.type = set(numberOfFeatures, value)

  /** Specifies whether to allow featurization of images */
  val allowImages: Param[Boolean] = BooleanParam(this, "allowImages", "Allow featurization of images", false)

  /** @group getParam */
  final def getAllowImages: Boolean = $(allowImages)

  /** @group setParam */
  def setAllowImages(value: Boolean): this.type = set(allowImages, value)

  /** Featurizes the dataset.
    *
    * @param dataset The input dataset to train.
    * @return The featurized model.
    */
  override def fit(dataset: Dataset[_]): PipelineModel = {
    val pipeline = assembleFeaturesEstimators(getFeatureColumns)
    pipeline.fit(dataset)
  }

  private def assembleFeaturesEstimators(featureColumns: Map[String, Seq[String]]): Pipeline = {
    val assembleFeaturesEstimators = featureColumns.map(newColToFeatures => {
      new AssembleFeatures()
        .setColumnsToFeaturize(newColToFeatures._2.toArray)
        .setFeaturesCol(newColToFeatures._1)
        .setNumberOfFeatures(getNumberOfFeatures)
        .setOneHotEncodeCategoricals(getOneHotEncodeCategoricals)
        .setAllowImages(getAllowImages)
    }).toArray

    new Pipeline().setStages(assembleFeaturesEstimators)
  }

  override def copy(extra: ParamMap): Estimator[PipelineModel] = {
    new Featurize()
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    assembleFeaturesEstimators(getFeatureColumns).transformSchema(schema)

}
