// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.metrics.MetricConstants
import com.microsoft.ml.spark.core.schema.{SchemaConstants, SparkSchema}
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.stages.featurize.{TrainClassifierTestUtilities, TrainRegressorTestUtilities}
import com.microsoft.ml.spark.stages.featurize.TrainClassifierTestUtilities._
import com.microsoft.ml.spark.stages.featurize.TrainRegressorTestUtilities._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.FastVectorAssembler
import org.apache.spark.sql._

/** Tests to validate the functionality of Compute Per Instance Statistics module. */
class VerifyComputePerInstanceStatistics extends TestBase {

  lazy val labelColumn = "label"
  lazy val predictionColumn = SchemaConstants.SparkPredictionColumn
  lazy val dataset = session.createDataFrame(Seq(
    (0.0, 2, 0.50, 0.60, 0.0),
    (1.0, 3, 0.40, 0.50, 1.0),
    (2.0, 4, 0.78, 0.99, 2.0),
    (3.0, 5, 0.12, 0.34, 3.0),
    (0.0, 1, 0.50, 0.60, 0.0),
    (1.0, 3, 0.40, 0.50, 1.0),
    (2.0, 3, 0.78, 0.99, 2.0),
    (3.0, 4, 0.12, 0.34, 3.0),
    (0.0, 0, 0.50, 0.60, 0.0),
    (1.0, 2, 0.40, 0.50, 1.0),
    (2.0, 3, 0.78, 0.99, 2.0),
    (3.0, 4, 0.12, 0.34, 3.0)))
    .toDF(labelColumn, "col1", "col2", "col3", predictionColumn)

  test("Smoke test for evaluating a dataset") {
    val scoreModelName = SchemaConstants.ScoreModelPrefix + "_test model"

    val datasetWithLabel =
      SparkSchema.setLabelColumnName(dataset, scoreModelName, labelColumn, SchemaConstants.RegressionKind)
    val datasetWithScores =
      SparkSchema.setScoresColumnName(datasetWithLabel, scoreModelName, predictionColumn,
                                      SchemaConstants.RegressionKind)

    val evaluatedData = new ComputePerInstanceStatistics().transform(datasetWithScores)
    validatePerInstanceRegressionStatistics(evaluatedData)
  }

  test("Verify computing per instance statistics on generic spark ML estimators is supported") {
    val scoredLabelsCol = "LogRegScoredLabelsCol"
    val scoresCol = "LogRegScoresCol"
    val probCol = "LogRegProbCol"
    val featuresCol = "features"
    val logisticRegression = new LogisticRegression()
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setMaxIter(10)
      .setLabelCol(labelColumn)
      .setPredictionCol(scoredLabelsCol)
      .setRawPredictionCol(scoresCol)
      .setProbabilityCol(probCol)
      .setFeaturesCol(featuresCol)
    val assembler = new FastVectorAssembler()
      .setInputCols(Array("col1", "col2", "col3"))
      .setOutputCol(featuresCol)
    val assembledDataset = assembler.transform(dataset)
    val model = logisticRegression.fit(assembledDataset)
    val scoredData = model.transform(assembledDataset)
    val cms = new ComputePerInstanceStatistics()
      .setLabelCol(labelColumn)
      .setScoredLabelsCol(scoredLabelsCol)
      .setScoresCol(scoresCol)
      .setScoredProbabilitiesCol(probCol)
      .setEvaluationMetric(MetricConstants.ClassificationMetrics)
    val evaluatedData = cms.transform(scoredData)
    validatePerInstanceClassificationStatistics(evaluatedData)
  }

  test("Smoke test to train regressor, score and evaluate on a dataset using all three modules") {
    val dataset = session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, 0),
      (1, 3, 0.40, 0.50, 1),
      (2, 4, 0.78, 0.99, 2),
      (3, 5, 0.12, 0.34, 3),
      (0, 1, 0.50, 0.60, 0),
      (1, 3, 0.40, 0.50, 1),
      (2, 3, 0.78, 0.99, 2),
      (3, 4, 0.12, 0.34, 3),
      (0, 0, 0.50, 0.60, 0),
      (1, 2, 0.40, 0.50, 1),
      (2, 3, 0.78, 0.99, 2),
      (3, 4, 0.12, 0.34, 3)
    )).toDF(labelColumn, "col1", "col2", "col3", "col4")

    val linearRegressor = createLinearRegressor(labelColumn)
    val scoredDataset = TrainRegressorTestUtilities.trainScoreDataset(labelColumn, dataset, linearRegressor)

    val evaluatedData = new ComputePerInstanceStatistics().transform(scoredDataset)
    validatePerInstanceRegressionStatistics(evaluatedData)
  }

  test("Smoke test to train classifier, score and evaluate on a dataset using all three modules") {
    val dataset = session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, 0),
      (1, 3, 0.40, 0.50, 1),
      (0, 4, 0.78, 0.99, 2),
      (1, 5, 0.12, 0.34, 3),
      (0, 1, 0.50, 0.60, 0),
      (1, 3, 0.40, 0.50, 1),
      (0, 3, 0.78, 0.99, 2),
      (1, 4, 0.12, 0.34, 3),
      (0, 0, 0.50, 0.60, 0),
      (1, 2, 0.40, 0.50, 1),
      (0, 3, 0.78, 0.99, 2),
      (1, 4, 0.12, 0.34, 3)
    )).toDF(labelColumn, "col1", "col2", "col3", "col4")

    val logisticRegressor = createLR.setLabelCol(labelColumn)
    val scoredDataset = TrainClassifierTestUtilities.trainScoreDataset(labelColumn, dataset, logisticRegressor)
    val evaluatedData = new ComputePerInstanceStatistics().transform(scoredDataset)
    validatePerInstanceClassificationStatistics(evaluatedData)
  }

  private def validatePerInstanceRegressionStatistics(evaluatedData: DataFrame): Unit = {
    // Validate the per instance statistics
    evaluatedData.collect().foreach(row => {
      val labelUncast = row(0)
      val label =
        labelUncast match {
          case i: Int => i.toDouble
          case _ => labelUncast.asInstanceOf[Double]
        }
      val score = row.getDouble(row.length - 3)
      val l1Loss = row.getDouble(row.length - 2)
      val l2Loss = row.getDouble(row.length - 1)
      val loss = math.abs(label - score)
      assert(l1Loss === loss)
      assert(l2Loss === loss * loss)
    })
  }

  private def validatePerInstanceClassificationStatistics(evaluatedData: DataFrame): Unit = {
    // Validate the per instance statistics
    evaluatedData.collect().foreach(row => {
      val labelUncast = row(0)
      val label =
        labelUncast match {
          case i: Int => i.toDouble
          case _ => labelUncast.asInstanceOf[Double]
        }
      val probabilities = row.get(row.length - 3).asInstanceOf[org.apache.spark.ml.linalg.Vector]
      val scoredLabel = row.getDouble(row.length - 2).toInt
      val logLoss = row.getDouble(row.length - 1)
      val computedLogLoss = -Math.log(Math.min(1, Math.max(ComputePerInstanceStatistics.epsilon,
        probabilities(scoredLabel.toInt))))
      assert(computedLogLoss === logLoss)
    })
  }

}
