// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.env.FileUtilities.File
import com.microsoft.ml.spark.core.metrics.MetricConstants
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import com.microsoft.ml.spark.stages.featurize.TrainClassifierTestUtilities
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

class VerifyFindBestModel extends EstimatorFuzzing[FindBestModel]{
  import TrainClassifierTestUtilities._

  val mockLabelColumn = "Label"

  def createMockDataset: DataFrame = {
    session.createDataFrame(Seq(
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
      (1, 4, 0.12, 0.34, 3)))
      .toDF(mockLabelColumn, "col1", "col2", "col3", "col4")
  }

  test("Smoke test to verify that evaluate can be run") {
    val dataset = createMockDataset
    val randomForestClassifier = createRF.setLabelCol(mockLabelColumn)
    val model = randomForestClassifier.fit(dataset)
    val findBestModel = new FindBestModel()
      .setModels(Array(model.asInstanceOf[Transformer], model.asInstanceOf[Transformer]))
      .setEvaluationMetric(MetricConstants.AccuracySparkMetric)
    val bestModel = findBestModel.fit(dataset)
    bestModel.transform(dataset)
  }

  test("Verify the best model can be saved") {
    val dataset: DataFrame = createMockDataset
    val logisticRegressor = createLR.setLabelCol(mockLabelColumn)
    val model = logisticRegressor.fit(dataset)

    val findBestModel = new FindBestModel()
      .setModels(Array(model.asInstanceOf[Transformer], model.asInstanceOf[Transformer]))
      .setEvaluationMetric(MetricConstants.AucSparkMetric)
    val bestModel = findBestModel.fit(dataset)

    val myModelFile = new File(tmpDir.toFile, "testEvalModel")
    bestModel.save(myModelFile.toString)
    assert(myModelFile.exists())
  }

  test("Verify the best model metrics can be retrieved and are valid") {
    val dataset: DataFrame = createMockDataset
    val logisticRegressor = createLR.setLabelCol(mockLabelColumn)
    val decisionTreeClassifier = createDT.setLabelCol(mockLabelColumn)
    val GBTClassifier = createGBT.setLabelCol(mockLabelColumn)
    val naiveBayesClassifier = createNB.setLabelCol(mockLabelColumn)
    val randomForestClassifier = createRF.setLabelCol(mockLabelColumn)
    val model1 = logisticRegressor.fit(dataset)
    val model2 = decisionTreeClassifier.fit(dataset)
    val model3 = GBTClassifier.fit(dataset)
    val model4 = naiveBayesClassifier.fit(dataset)
    val model5 = randomForestClassifier.fit(dataset)

    val findBestModel = new FindBestModel()
      .setModels(Array(model1.asInstanceOf[Transformer], model2, model3, model4, model5))
      .setEvaluationMetric(MetricConstants.AucSparkMetric)
    val bestModel = findBestModel.fit(dataset)
    // validate schema is as expected
    assert(bestModel.getAllModelMetrics.schema ==
      StructType(Seq(StructField(FindBestModel.modelNameCol, StringType, true),
        StructField(FindBestModel.metricsCol, DoubleType, true),
        StructField(FindBestModel.paramsCol, StringType, true))))
    // validate we got metrics for every model
    assert(bestModel.getAllModelMetrics.count() == 5)
    // validate AUC looks valid
    bestModel.getAllModelMetrics
      .select(FindBestModel.metricsCol)
      .collect()
      .foreach(value => assert(value.getDouble(0) >= 0.5))
  }

  val reader: MLReadable[_] = FindBestModel
  val modelReader: MLReadable[_] = BestModel

  override def testObjects(): Seq[TestObject[FindBestModel]] = Seq(new TestObject({
    val randomForestClassifier = createRF.setLabelCol(mockLabelColumn)
    val model = randomForestClassifier.fit(createMockDataset)
    new FindBestModel()
      .setModels(Array(model, model))
      .setEvaluationMetric(MetricConstants.AccuracySparkMetric)}, createMockDataset))
}
