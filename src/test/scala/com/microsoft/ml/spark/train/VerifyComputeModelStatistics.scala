// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.train

import java.io.File

import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.metrics.MetricConstants
import com.microsoft.ml.spark.core.schema.{CategoricalUtilities, SchemaConstants, SparkSchema}
import com.microsoft.ml.spark.core.test.benchmarks.DatasetUtils
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.FastVectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import com.microsoft.ml.spark.train.TrainRegressorTestUtilities._
import com.microsoft.ml.spark.train.TrainClassifierTestUtilities._

import scala.util.Random

/** Tests to validate the functionality of Evaluate Model module. */
class VerifyComputeModelStatistics extends TransformerFuzzing[ComputeModelStatistics] {
  val labelColumn = "Label"
  lazy val dataset: DataFrame = spark.createDataFrame(Seq(
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

  test("Verify multiclass evaluation is not slow for large number of labels") {
    val numRows = 4096
    import spark.implicits._
    val rand = new Random(1337)
    val labelCol = "label"
    val evaluationMetric = MetricConstants.ClassificationMetricsName
    val predCol = SchemaConstants.SparkPredictionColumn
    val df = Seq.fill(numRows)(rand.nextDouble())
      .zip(Seq.fill(numRows)(rand.nextDouble()))
      .toDF(labelCol, predCol)
    val evaluatedData = new ComputeModelStatistics()
      .setLabelCol(labelCol)
      .setScoredLabelsCol(predCol)
      .setEvaluationMetric(evaluationMetric)
      .transform(df)
    val firstRow = evaluatedData.first()
    (2 to 4).foreach { index =>
      assert(firstRow.getDouble(index) === 0.0)
    }
  }

  test("Smoke test for evaluating a dataset") {
    val predictionColumn = SchemaConstants.SparkPredictionColumn
    val scoreModelName = SchemaConstants.ScoreModelPrefix + "_test model"
    val dataset = spark.createDataFrame(Seq(
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

    val datasetWithLabel =
      SparkSchema.setLabelColumnName(dataset, scoreModelName, labelColumn, SchemaConstants.RegressionKind)
    val datasetWithScores =
      SparkSchema.setScoresColumnName(datasetWithLabel, scoreModelName, predictionColumn,
                                      SchemaConstants.RegressionKind)

    val evaluatedSchema = new ComputeModelStatistics().transformSchema(datasetWithScores.schema)

    val evaluatedData = new ComputeModelStatistics().transform(datasetWithScores)
    val firstRow = evaluatedData.first()
    assert(firstRow.getDouble(0) === 0.0)
    assert(firstRow.getDouble(1) === 0.0)
    assert(firstRow.getDouble(2) === 1.0)
    assert(firstRow.getDouble(3) === 0.0)

    assert(evaluatedSchema == StructType(MetricConstants.RegressionColumns.map(StructField(_, DoubleType))))
  }

  test("Evaluate a dataset with missing values") {
    val predictionColumn = SchemaConstants.SparkPredictionColumn
    val dataset = spark.createDataFrame(sc.parallelize(Seq(
      (0.0, 0.0),
      (0.0, null),
      (1.0, 1.0),
      (2.0, 2.0),
      (null, null),
      (0.0, 0.0),
      (null, 3.0))).map(values => Row(values._1, values._2)),
      StructType(Array(StructField(labelColumn, DoubleType, true),
        StructField(predictionColumn, DoubleType, true))))
      .toDF(labelColumn, predictionColumn)

    val scoreModelName = SchemaConstants.ScoreModelPrefix + "_test model"

    val datasetWithLabel =
      SparkSchema.setLabelColumnName(dataset, scoreModelName, labelColumn, SchemaConstants.RegressionKind)
    val datasetWithScores =
      SparkSchema.setScoresColumnName(datasetWithLabel, scoreModelName, predictionColumn,
        SchemaConstants.RegressionKind)

    val evaluatedData = new ComputeModelStatistics().transform(datasetWithScores)
    val firstRow = evaluatedData.first()
    assert(firstRow.getDouble(0) === 0.0)
    assert(firstRow.getDouble(1) === 0.0)
    assert(firstRow.getDouble(2) === 1.0)
    assert(firstRow.getDouble(3) === 0.0)
  }

  test("Verify compute model statistics does not get stuck in a loop in catalyst") {
    val name = "AutomobilePriceRaw.csv"
    val filePath = FileUtilities.join(
      BuildInfo.datasetDir, "MissingValuesRegression", "Train", name)
    val dataset =
      spark.read.option("header", "true").option("inferSchema", "true")
        .option("nullValue", "?")
        .option("treatEmptyValuesAsNulls", "true")
        .option("delimiter", ",")
        .csv(filePath.toString)
    val glr = new GeneralizedLinearRegression().setFamily("poisson").setLink("log")
    val tr = new TrainRegressor().setModel(glr).setLabelCol("price").setNumFeatures(256)
    val model = tr.fit(dataset)
    val prediction = model.transform(dataset)
    val evaluatedData = new ComputeModelStatistics().transform(prediction)
    assert(math.abs(evaluatedData.collect()(0).getDouble(2) - 0.9772518203539127) < .01)
  }

  test("Smoke test to train regressor, score and evaluate on a dataset using all three modules") {
    val dataset = spark.createDataFrame(Seq(
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

    val otherLabelColumn = "someOtherColumn"

    val datasetWithAddedColumn = dataset.withColumn(otherLabelColumn, org.apache.spark.sql.functions.lit(0.0))

    val linearRegressor = createLinearRegressor(otherLabelColumn)
    val scoredDataset =
      TrainRegressorTestUtilities.trainScoreDataset(otherLabelColumn, datasetWithAddedColumn, linearRegressor)

    val evaluatedData = new ComputeModelStatistics().transform(scoredDataset)
    val firstRow = evaluatedData.first()
    assert(firstRow.getDouble(0) === 0.0)
    assert(firstRow.getDouble(1) === 0.0)
    assert(firstRow.getDouble(2).isNaN)
    assert(firstRow.getDouble(3) === 0.0)
  }

  lazy val logisticRegressor: TrainClassifier = createLR.setLabelCol(labelColumn)
  lazy val scoredDataset: DataFrame = TrainClassifierTestUtilities.trainScoreDataset(
    labelColumn, dataset, logisticRegressor)
  test("Smoke test to train classifier, score and evaluate on a dataset using all three modules") {
    val evaluatedData = new ComputeModelStatistics().transform(scoredDataset)

    val evaluatedSchema = new ComputeModelStatistics().transformSchema(scoredDataset.schema)
    assert(evaluatedSchema == StructType(MetricConstants.ClassificationColumns.map(StructField(_, DoubleType))))
  }

  test("Verify computing statistics on generic spark ML estimators is supported") {
    val scoredLabelsCol = "LogRegScoredLabelsCol"
    val scoresCol = "LogRegScoresCol"
    val featuresCol = "features"
    val logisticRegression = new LogisticRegression()
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setMaxIter(10)
      .setLabelCol(labelColumn)
      .setPredictionCol(scoredLabelsCol)
      .setRawPredictionCol(scoresCol)
      .setFeaturesCol(featuresCol)
    val assembler = new FastVectorAssembler()
      .setInputCols(Array("col1", "col2", "col3", "col4"))
      .setOutputCol(featuresCol)
    val assembledDataset = assembler.transform(dataset)
    val model = logisticRegression.fit(assembledDataset)
    val scoredData = model.transform(assembledDataset)
    val cms = new ComputeModelStatistics()
      .setLabelCol(labelColumn)
      .setScoredLabelsCol(scoredLabelsCol)
      .setScoresCol(scoresCol)
      .setEvaluationMetric(MetricConstants.ClassificationMetricsName)
    val evaluatedData = cms.transform(scoredData)
    val firstRow = evaluatedData.select(col("accuracy"), col("precision"), col("recall"), col("AUC")).first()
    assert(firstRow.getDouble(0) === 1.0)
    assert(firstRow.getDouble(1) === 1.0)
    assert(firstRow.getDouble(2) === 1.0)
    assert(firstRow.getDouble(3) === 1.0)
  }

  test("Verify results of multiclass metrics") {
    val labelColumn = "label"
    val predictionColumn = SchemaConstants.SparkPredictionColumn
    val labelsAndPrediction = spark.createDataFrame(
      Seq(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 2.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (2.0, 2.0))).toDF(labelColumn, predictionColumn)

    val scoreModelName = SchemaConstants.ScoreModelPrefix + "_test model"

    val datasetWithLabel =
      SparkSchema.setLabelColumnName(labelsAndPrediction, scoreModelName, labelColumn,
        SchemaConstants.ClassificationKind)
    val datasetWithScoredLabels =
      SparkSchema.setScoredLabelsColumnName(datasetWithLabel, scoreModelName, predictionColumn,
        SchemaConstants.ClassificationKind)

    val evaluatedData = new ComputeModelStatistics().transform(datasetWithScoredLabels)

    val tp0 = 2.0
    val tp1 = 3.0
    val tp2 = 1.0
    val tn0 = 4.0
    val tn1 = 4.0
    val tn2 = 7.0
    val numLabels = 3.0
    val total = labelsAndPrediction.count()

    val precision0 = 2.0 / (2 + 1)
    val precision1 = 3.0 / (3 + 1)
    val precision2 = 1.0 / (1 + 1)
    val recall0 = 2.0 / (2 + 2)
    val recall1 = 3.0 / (3 + 1)
    val recall2 = 1.0 / (1 + 0)

    val overallAccuracy = (tp0 + tp1 + tp2) / total
    val evalRow = evaluatedData.first()
    assert(evalRow.getAs[Double](MetricConstants.AccuracyColumnName) === overallAccuracy)
    assert(evalRow.getAs[Double](MetricConstants.PrecisionColumnName) === overallAccuracy)
    assert(evalRow.getAs[Double](MetricConstants.RecallColumnName) === overallAccuracy)
    val avgAccuracy = ((tp0 + tn0) / total + (tp1 + tn1) / total + (tp2 + tn2) / total) / numLabels
    val macroPrecision = (precision0 + precision1 + precision2) / numLabels
    val macroRecall = (recall0 + recall1 + recall2) / numLabels
    assert(evalRow.getAs[Double](MetricConstants.AverageAccuracy) === avgAccuracy)
    assert(evalRow.getAs[Double](MetricConstants.MacroAveragedPrecision) === macroPrecision)
    assert(evalRow.getAs[Double](MetricConstants.MacroAveragedRecall) === macroRecall)
  }

  test("validate AUC from compute model statistic and binary classification evaluator gives the same result") {
    val fileLocation = DatasetUtils.binaryTrainFile("transfusion.csv").toString
    val label = "Donated"
    val dataset: DataFrame =
      spark.read.format("com.databricks.spark.csv")
        .option("header", "true").option("inferSchema", "true")
        .option("treatEmptyValuesAsNulls", "false")
        .option("delimiter", ",")
        .load(fileLocation)

    val split = dataset.randomSplit(Array(0.75,0.25))
    val train = split(0)
    val test = split(1)

    val trainClassifier = new TrainClassifier()
    val model = trainClassifier.setModel(new LogisticRegression())
      .set(trainClassifier.labelCol, label)
      .set(trainClassifier.numFeatures, 1 << 18)
      .fit(train)
    val scored = model.transform(test)
    val eval = new ComputeModelStatistics().transform(scored)
    val cmsAUC = eval.first().getAs[Double]("AUC")

    val binaryEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol(label)
      .setRawPredictionCol(SchemaConstants.ScoresColumn)

    val levels = CategoricalUtilities.getLevels(scored.schema, label)
    val levelsToIndexMap: Map[Any, Double] = levels.get.zipWithIndex.map(t => t._1 -> t._2.toDouble).toMap

    // Calculate confusion matrix and output it as DataFrame
    val predictionAndLabels = spark
      .createDataFrame(scored.select(col(SchemaConstants.ScoresColumn), col(label)).rdd.map {
      case Row(prediction: Vector, label) => (prediction(1), levelsToIndexMap(label))
    }).toDF(SchemaConstants.ScoresColumn, label)

    val auc = binaryEvaluator.evaluate(predictionAndLabels)
    assert(auc === cmsAUC)
  }

  override def testObjects(): Seq[TestObject[ComputeModelStatistics]] = Seq(new TestObject(
    new ComputeModelStatistics(), scoredDataset))

  override def reader: MLReadable[_] = ComputeModelStatistics
}
