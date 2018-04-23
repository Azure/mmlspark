// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.hyper

import com.microsoft.ml.spark.core.metrics.MetricConstants
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.benchmarks.{Benchmarks, DatasetUtils}
import com.microsoft.ml.spark.core.test.fuzzing.TestObject
import com.microsoft.ml.spark.stages.featurize.TrainClassifierTestUtilities
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ListBuffer

/** Tests to validate the functionality of Tune Hyperparameters module. */
class VerifyTuneHyperparameters extends Benchmarks {
  import TrainClassifierTestUtilities._

  lazy val moduleName = "tune-hyperparameters"

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

  // Getting negative feature values in NaiveBayes, need to look into
  // verifyLearnerOnMulticlassCsvFile("abalone.csv",                  "Rings", 2, false)
  // Has multiple columns with the same name.  Spark doesn't seem to be able to handle that yet.
  // verifyLearnerOnMulticlassCsvFile("arrhythmia.csv",               "Arrhythmia")
  // Getting negative feature values in NaiveBayes, need to look into
  verifyMulticlassCsv("BreastTissue.csv",             "Class", 2, false)
  verifyMulticlassCsv("CarEvaluation.csv",            "Col7", 2, true)
  // Getting "code generation" exceeded max size limit error
  // verifyLearnerOnMulticlassCsvFile("mnist.train.csv",              "Label")
  // This works with 2.0.0, but on 2.1.0 it looks like it loops infinitely while leaking memory
  // verifyLearnerOnMulticlassCsvFile("au3_25000.csv",                "class", 2, true)
  // This takes way too long for a gated build.  Need to make it something like a p3 test case.
  // verifyLearnerOnMulticlassCsvFile("Seattle911.train.csv",         "Event Clearance Group")

  verifyBinaryCsv("PimaIndian.csv",                   "Diabetes mellitus", 2, true)
  verifyBinaryCsv("data_banknote_authentication.csv", "class", 2, false)
  verifyBinaryCsv("task.train.csv",                   "TaskFailed10", 2, true)
  verifyBinaryCsv("breast-cancer.train.csv",          "Label", 2, true)
  verifyBinaryCsv("random.forest.train.csv",          "#Malignant", 2, true)
  verifyBinaryCsv("transfusion.csv",                  "Donated", 2, true)
  // verifyLearnerOnBinaryCsvFile("au2_10000.csv",                    "class", 1)
  verifyBinaryCsv("breast-cancer-wisconsin.csv",      "Class", 2, true)
  verifyBinaryCsv("fertility_Diagnosis.train.csv",    "Diagnosis", 2, false)
  verifyBinaryCsv("bank.train.csv",                   "y", 2, false)
  verifyBinaryCsv("TelescopeData.csv",                " Class", 2, false)

  test("Compare benchmark results file to generated file", TestBase.Extended) {
    verifyBenchmarks()
  }

  def verifyBinaryCsv(fileName: String,
                      labelCol: String,
                      decimals: Int,
                      includeNaiveBayes: Boolean): Unit = {
    test("Verify classifier can be trained and scored on " + fileName, TestBase.Extended) {
      val fileLocation = DatasetUtils.binaryTrainFile(fileName).toString
      val bestModel = tuneDataset(fileName, labelCol, fileLocation, true, includeNaiveBayes)
      val bestMetric = bestModel.bestMetric
      addBenchmark(s"binary_$fileName", bestMetric, decimals)
    }
  }

  def verifyMulticlassCsv(fileName: String,
                          labelCol: String,
                          decimals: Int,
                          includeNaiveBayes: Boolean): Unit = {
    test("Verify classifier can be trained and scored on multiclass " + fileName, TestBase.Extended) {
      val fileLocation = DatasetUtils.multiclassTrainFile(fileName).toString
      val bestModel = tuneDataset(fileName, labelCol, fileLocation, false, includeNaiveBayes)
      val bestMetric = bestModel.bestMetric
      addBenchmark(s"multiclass_$fileName", bestMetric, 10)
    }
  }

  def tuneDataset(fileName: String,
                  labelCol: String,
                  fileLocation: String,
                  includeNonProb: Boolean,
                  includeNaiveBayes: Boolean): TuneHyperparametersModel = {
    // TODO: Add other file types for testing
    val dataset: DataFrame =
    session.read.format("com.databricks.spark.csv")
      .option("header", "true").option("inferSchema", "true")
      .option("treatEmptyValuesAsNulls", "false")
      .option("delimiter", if (fileName.endsWith(".csv")) "," else "\t")
      .load(fileLocation)

    val models = ListBuffer[Estimator[_]]()
    val hyperParams = ListBuffer[(Param[_], Dist[_])]()
    val logisticRegressor = createLR.setLabelCol(labelCol)
    val decisionTreeClassifier = createDT.setLabelCol(labelCol)
    val randomForestClassifier = createRF.setLabelCol(labelCol)
    models += logisticRegressor
    hyperParams ++= DefaultHyperparams.defaultRange(
      logisticRegressor.getModel.asInstanceOf[LogisticRegression])
    models += decisionTreeClassifier
    hyperParams ++= DefaultHyperparams.defaultRange(
      decisionTreeClassifier.getModel.asInstanceOf[DecisionTreeClassifier])
    models += randomForestClassifier
    hyperParams ++= DefaultHyperparams.defaultRange(
      randomForestClassifier.getModel.asInstanceOf[RandomForestClassifier])

    if (includeNonProb) {
      val gbtClassifier = createGBT.setLabelCol(labelCol)
      val mlpClassifier = createMLP.setLabelCol(labelCol)
      models += gbtClassifier
      hyperParams ++= DefaultHyperparams.defaultRange(
        gbtClassifier.getModel.asInstanceOf[GBTClassifier])
      models += mlpClassifier
      hyperParams ++= DefaultHyperparams.defaultRange(
        mlpClassifier.getModel.asInstanceOf[MultilayerPerceptronClassifier])
    }

    if (includeNaiveBayes) {
      val naiveBayesClassifier = createNB.setLabelCol(labelCol)
      models += naiveBayesClassifier
      hyperParams ++= DefaultHyperparams.defaultRange(
        naiveBayesClassifier.getModel.asInstanceOf[NaiveBayes])
    }

    val randomSpace = new RandomSpace(hyperParams.toArray)

    new TuneHyperparameters()
      .setModels(models.toArray)
      .setEvaluationMetric(MetricConstants.AccuracySparkMetric)
      .setNumFolds(2)
      .setNumRuns(2)
      .setParallelism(1)
      .setParamSpace(randomSpace)
      .setSeed(1234L)
      .fit(dataset)
  }

  val reader: MLReadable[_] = TuneHyperparameters
  val modelReader: MLReadable[_] = TuneHyperparametersModel

  def testObjects(): Seq[TestObject[TuneHyperparameters]] =
    Seq(new TestObject({
      val logisticRegressor = createLR.setLabelCol(mockLabelColumn)
      val decisionTreeClassifier = createDT.setLabelCol(mockLabelColumn)
      val models = ListBuffer[Estimator[_]]()
      val hyperParams = ListBuffer[(Param[_], Dist[_])]()
      hyperParams ++= DefaultHyperparams.defaultRange(logisticRegressor.getModel.asInstanceOf[LogisticRegression])
      hyperParams ++=
        DefaultHyperparams.defaultRange(decisionTreeClassifier.getModel.asInstanceOf[DecisionTreeClassifier])
      models += logisticRegressor
      models += decisionTreeClassifier
      val randomSpace = new RandomSpace(hyperParams.toArray)
      new TuneHyperparameters()
        .setModels(models.toArray)
        .setEvaluationMetric(MetricConstants.AccuracySparkMetric)
        .setNumFolds(2)
        .setNumRuns(2)
        .setParallelism(1)
        .setParamSpace(randomSpace)
        .setSeed(345L)
    }, createMockDataset))

}
