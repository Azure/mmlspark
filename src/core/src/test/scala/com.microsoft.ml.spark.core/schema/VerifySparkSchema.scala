// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import com.microsoft.ml.spark.core.test.base.TestBase

/** Verifies the spark schema functions. */
class VerifySparkSchema extends TestBase {

  val labelColumn = "label"
  val scoreColumn = "score"
  val probabilityColumn = "probability"
  val scoredLabelsColumn = "scored label"
  test("Spark schema should be able to set and get label, score, probability and scored labels column name") {
    val dataset = session.createDataFrame(Seq(
      (0, Array("Hi", "I", "can", "not", "foo"), 0.50, 0.60, 0),
      (1, Array("I"),                            0.40, 0.50, 1),
      (2, Array("Logistic", "regression"),       0.78, 0.99, 2),
      (3, Array("Log","f", "reg"),               0.12, 0.34, 3)
    )).toDF(labelColumn, "words", scoreColumn, probabilityColumn, scoredLabelsColumn)

    val modelName = "Test model name"
    val datasetWithLabel =
      SparkSchema.setLabelColumnName(dataset, modelName, labelColumn, SchemaConstants.RegressionKind)
    val labelColumnNameRetrieved =
      SparkSchema.getLabelColumnName(datasetWithLabel, modelName)

    assert(labelColumnNameRetrieved == labelColumn)

    val datasetWithScore =
      SparkSchema.setScoresColumnName(dataset, modelName, scoreColumn, SchemaConstants.RegressionKind)
    val scoreColumnNameRetrieved =
      SparkSchema.getScoresColumnName(datasetWithScore, modelName)

    assert(scoreColumnNameRetrieved == scoreColumn)

    val datasetWithProbability =
      SparkSchema.setScoredProbabilitiesColumnName(dataset, modelName, probabilityColumn,
                                                   SchemaConstants.RegressionKind)
    val probabilityColumnNameRetrieved =
      SparkSchema.getScoredProbabilitiesColumnName(datasetWithProbability, modelName)

    assert(probabilityColumnNameRetrieved == probabilityColumn)

    val datasetWithScoredLabels =
      SparkSchema.setScoredLabelsColumnName(dataset, modelName, scoredLabelsColumn, SchemaConstants.RegressionKind)
    val scoredLabelsColumnNameRetrieved =
      SparkSchema.getScoredLabelsColumnName(datasetWithScoredLabels, modelName)

    assert(scoredLabelsColumnNameRetrieved == scoredLabelsColumn)
  }

}
