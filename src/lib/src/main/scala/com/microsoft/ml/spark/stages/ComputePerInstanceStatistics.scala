// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts._
import com.microsoft.ml.spark.core.metrics.{MetricConstants, MetricUtils}
import com.microsoft.ml.spark.core.schema.SchemaConstants._
import com.microsoft.ml.spark.core.schema.{SchemaConstants, SparkSchema}
import com.microsoft.ml.spark.schema.CategoricalUtilities
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object ComputePerInstanceStatistics extends DefaultParamsReadable[ComputePerInstanceStatistics] {
  val epsilon = 1e-15
}

trait ComputePerInstanceStatisticsParams extends MMLParams
  with HasLabelCol with HasScoresCol with HasScoredLabelsCol with HasScoredProbabilitiesCol with HasEvaluationMetric {
  /** Param "evaluationMetric" is the metric to evaluate the models with. Default is "all"
    *
    *   If using a native Spark ML model, you will need to specify either "classifier" or "regressor"
    *     - classifier
    *     - regressor
    *
    * @group param
    */
  setDefault(evaluationMetric -> MetricConstants.AllSparkMetrics)
}

/** Evaluates the given scored dataset with per instance metrics.
  *
  * The Regression metrics are:
  * - L1_loss
  * - L2_loss
  *
  * The Classification metrics are:
  * - log_loss
  */
class ComputePerInstanceStatistics(override val uid: String) extends Transformer
  with ComputePerInstanceStatisticsParams {

  def this() = this(Identifiable.randomUID("ComputePerInstanceStatistics"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val (modelName, labelColumnName, scoreValueKind) =
      MetricUtils.getSchemaInfo(
        dataset.schema,
        if (isDefined(labelCol)) Some(getLabelCol) else None,
        getEvaluationMetric)

    val dataframe = dataset.toDF()

    if (scoreValueKind == SchemaConstants.ClassificationKind) {
      // Compute the LogLoss for classification case
      val scoredLabelsColumnName =
        if (isDefined(scoredLabelsCol)) getScoredLabelsCol
        else SparkSchema.getScoredLabelsColumnName(dataframe, modelName)

      // Get levels if categorical
      val levels = CategoricalUtilities.getLevels(dataframe.schema, labelColumnName)
      val numLevels =
        if (!levels.isEmpty && levels.get != null) {
          if (levels.get.length > 2) levels.get.length else 2
        } else {
          // Otherwise compute unique levels
          dataset.select(col(labelColumnName).cast(DoubleType)).rdd.distinct().count().toInt
        }

      val logLossFunc = udf((scoredLabel: Double, scores: org.apache.spark.ml.linalg.Vector) =>
        if (scoredLabel < numLevels) {
          -Math.log(Math.min(1, Math.max(ComputePerInstanceStatistics.epsilon, scores(scoredLabel.toInt))))
        } else {
          // penalize if no label seen in training
          -Math.log(ComputePerInstanceStatistics.epsilon)
        })
      val probabilitiesColumnName =
        if (isDefined(scoredProbabilitiesCol)) getScoredProbabilitiesCol
        else SparkSchema.getScoredProbabilitiesColumnName(dataframe, modelName)
      dataframe.withColumn(MetricConstants.LogLossMetric,
        logLossFunc(dataset(scoredLabelsColumnName), dataset(probabilitiesColumnName)))
    } else {
      val scoresColumnName =
        if (isDefined(scoresCol)) getScoresCol
        else SparkSchema.getScoresColumnName(dataframe, modelName)
      // Compute the L1 and L2 loss for regression case
      val scoresAndLabels =
        dataset.select(col(scoresColumnName), col(labelColumnName).cast(DoubleType)).rdd.map {
          case Row(prediction: Double, label: Double) => (prediction, label)
        }
      val l1LossFunc = udf((trueLabel:Double, scoredLabel: Double) => math.abs(trueLabel - scoredLabel))
      val l2LossFunc = udf((trueLabel:Double, scoredLabel: Double) =>
        {
          val loss = math.abs(trueLabel - scoredLabel)
          loss * loss
        })
      dataframe.withColumn(MetricConstants.L1LossMetric,
        l1LossFunc(dataset(labelColumnName), dataset(scoresColumnName)))
        .withColumn(MetricConstants.L2LossMetric,
          l2LossFunc(dataset(labelColumnName), dataset(scoresColumnName)))
    }
  }

  private def getFirstModelName(colMetadata: Metadata): Option[String] = {
    if (!colMetadata.contains(MMLTag)) null
    else {
      val mlTagMetadata = colMetadata.getMetadata(MMLTag)
      val metadataKeys = MetadataUtilities.getMetadataKeys(mlTagMetadata)
      metadataKeys.find(key => key.startsWith(SchemaConstants.ScoreModelPrefix))
    }
  }

  override def copy(extra: ParamMap): Transformer = new ComputePerInstanceStatistics()

  // TODO: This should be based on the retrieved score value kind
  override def transformSchema(schema: StructType): StructType =
    schema.add(new StructField(MetricConstants.L1LossMetric, DoubleType))
          .add(new StructField(MetricConstants.L2LossMetric, DoubleType))

}
