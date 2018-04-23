// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.featurize

import java.io.File

import com.microsoft.ml.spark.core.schema.SchemaConstants
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.benchmarks.Benchmarks
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import com.microsoft.ml.spark.schema.CategoricalUtilities
import com.microsoft.ml.spark.stages.ValueIndexer
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}
import com.microsoft.ml.spark.core.test.benchmarks.DatasetUtils._

/** Tests to validate the functionality of Train Classifier module. */
class VerifyTrainClassifier extends Benchmarks with EstimatorFuzzing[TrainClassifier] {

  import CategoricalUtilities._
  import TrainClassifierTestUtilities._

  val moduleName = "train-classifier"
  val LRName = "LogisticRegression"
  val DTName = "DecisionTreeClassification"
  val RFName = "RandomForestClassification"
  val GBTName = "GradientBoostedTreesClassification"
  val NBName = "NaiveBayesClassifier"
  val MLPName = "MultilayerPerceptronClassifier"
  val mockLabelCol = "Label"

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
      .toDF(mockLabelCol, "col1", "col2", "col3", "col4")
  }

  test("Smoke test for training on a classifier") {
    val dataset: DataFrame = createMockDataset
    val logisticRegressor = createLR.setLabelCol(mockLabelCol)
    trainScoreDataset(mockLabelCol, dataset, logisticRegressor)
  }

  test("Verify you can score on a dataset without a label column") {
    val dataset: DataFrame = createMockDataset

    val logisticRegressor = createLR.setLabelCol(mockLabelCol)

    val data = dataset.randomSplit(Seq(0.6, 0.4).toArray, 42)
    val trainData = data(0)
    val testData = data(1)

    val model = logisticRegressor.fit(trainData)

    model.transform(testData.drop(mockLabelCol))
  }

  test("Verify train classifier works on a dataset with categorical columns") {
    val cat = "Cat"
    val dog = "Dog"
    val bird = "Bird"
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, dog, cat),
      (1, 3, 0.40, 0.50, cat, dog),
      (0, 4, 0.78, 0.99, dog, bird),
      (1, 5, 0.12, 0.34, cat, dog),
      (0, 1, 0.50, 0.60, dog, bird),
      (1, 3, 0.40, 0.50, bird, dog),
      (0, 3, 0.78, 0.99, dog, cat),
      (1, 4, 0.12, 0.34, cat, dog),
      (0, 0, 0.50, 0.60, dog, cat),
      (1, 2, 0.40, 0.50, bird, dog),
      (0, 3, 0.78, 0.99, dog, bird),
      (1, 4, 0.12, 0.34, cat, dog)))
      .toDF(mockLabelCol, "col1", "col2", "col3", "col4", "col5")

    val model1 = new ValueIndexer().setInputCol("col4").setOutputCol("col4").fit(dataset)
    val model2 = new ValueIndexer().setInputCol("col5").setOutputCol("col5").fit(dataset)
    val catDataset = model1.transform(model2.transform(dataset))

    val logisticRegressor = createLR.setLabelCol(mockLabelCol)
    trainScoreDataset(mockLabelCol, catDataset, logisticRegressor)
    val randomForestClassifier = createRF.setLabelCol(mockLabelCol)
    trainScoreDataset(mockLabelCol, catDataset, randomForestClassifier)
  }

  test("Verify you can train on a dataset that contains a vector column") {
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, 0, Vectors.dense(1.0, 0.1, -1.5)),
      (1, 3, 0.40, 0.50, 1, Vectors.dense(1.5, 0.2, -1.2)),
      (0, 4, 0.78, 0.99, 2, Vectors.dense(1.3, 0.3, -1.1)),
      (1, 5, 0.12, 0.34, 3, Vectors.sparse(3, Seq((0, 1.0), (2, 2.0)))),
      (0, 1, 0.50, 0.60, 0, Vectors.dense(1.0, 0.4, -1.23)),
      (1, 3, 0.40, 0.50, 1, Vectors.dense(1.1, 0.5, -1.024)),
      (0, 3, 0.78, 0.99, 2, Vectors.dense(1.0, 0.1, -1.22)),
      (1, 4, 0.12, 0.34, 3, Vectors.dense(Double.NaN, 0.2, -1.23)),
      (0, 0, 0.50, 0.60, 0, Vectors.dense(0.5, 0.3, 1.0)),
      (1, 2, 0.40, 0.50, 1, Vectors.dense(1.0, 0.4, -1.2)),
      (0, 3, 0.78, 0.99, 2, Vectors.dense(0.7, 0.5, -1.1)),
      (1, 4, 0.12, 0.34, 3, Vectors.dense(1.8, 0.1, 2.02))))
      .toDF(mockLabelCol, "col1", "col2", "col3", "col4", "col5")

    val logisticRegressor = createLR.setLabelCol(mockLabelCol)
    trainScoreDataset(mockLabelCol, dataset, logisticRegressor)
  }

  verifyMultiClassCSV("abalone.csv", "Rings", 2, true)
  // Has multiple columns with the same name.  Spark doesn't seem to be able to handle that yet.
  // verifyLearnerOnMulticlassCsvFile("arrhythmia.csv",               "Arrhythmia")
  verifyMultiClassCSV("BreastTissue.csv", "Class", 2, true)
  verifyMultiClassCSV("CarEvaluation.csv", "Col7", 2, true)
  // Getting "code generation" exceeded max size limit error
  // verifyLearnerOnMulticlassCsvFile("mnist.train.csv",              "Label")
  // This works with 2.0.0, but on 2.1.0 it looks like it loops infinitely while leaking memory
  // verifyLearnerOnMulticlassCsvFile("au3_25000.csv",                "class", 2, true)
  // This takes way too long for a gated build.  Need to make it something like a p3 test case.
  // verifyLearnerOnMulticlassCsvFile("Seattle911.train.csv",         "Event Clearance Group")

  verifyBinaryCSV("PimaIndian.csv", "Diabetes mellitus", 2, true)
  verifyBinaryCSV("data_banknote_authentication.csv", "class", 2, false)
  verifyBinaryCSV("task.train.csv", "TaskFailed10", 2, true)
  verifyBinaryCSV("breast-cancer.train.csv", "Label", 2, true)
  verifyBinaryCSV("random.forest.train.csv", "#Malignant", 2, true)
  verifyBinaryCSV("transfusion.csv", "Donated", 2, true)
  // verifyLearnerOnBinaryCsvFile("au2_10000.csv",                    "class", 1)
  verifyBinaryCSV("breast-cancer-wisconsin.csv", "Class", 2, true)
  verifyBinaryCSV("fertility_Diagnosis.train.csv", "Diagnosis", 2, false)
  verifyBinaryCSV("bank.train.csv", "y", 2, false)
  verifyBinaryCSV("TelescopeData.csv", " Class", 2, false)

  test("Compare benchmark results file to generated file", TestBase.Extended) {
    verifyBenchmarks()
  }

  def verifyBinaryCSV(fileName: String,
                      labelCol: String,
                      decimals: Int,
                      includeNaiveBayes: Boolean): Unit = {
    test("Verify classifier can be trained and scored on " + fileName, TestBase.Extended) {
      val fileLocation = binaryTrainFile(fileName).toString
      val results = readAndScoreDataset(fileName, labelCol, fileLocation, true, includeNaiveBayes)
      results.foreach { case (name, result) =>
        val probCol = if (Set(GBTName, MLPName)(name)) {
          SchemaConstants.ScoredLabelsColumn
        } else {
          SchemaConstants.ScoresColumn
        }
        evalAUC(name + s"_${fileName}_", result, labelCol, probCol, decimals)
      }
    }
  }

  def verifyMultiClassCSV(fileName: String,
                          labelCol: String,
                          decimals: Int,
                          includeNaiveBayes: Boolean): Unit = {
    test("Verify classifier can be trained and scored on multiclass " + fileName, TestBase.Extended) {
      val fileLocation = multiclassTrainFile(fileName).toString
      val results =
        readAndScoreDataset(fileName, labelCol, fileLocation, false, includeNaiveBayes)

      val multiClassModelResults = results.filter(Set(LRName, DTName, RFName, NBName))
      multiClassModelResults.foreach { case (name, result) =>
        evalMulticlass(name + s"_${fileName}_", result,
          labelCol, SchemaConstants.ScoredLabelsColumn, decimals)
      }

    }
  }

  def readAndScoreDataset(fileName: String,
                          labelColumnName: String,
                          fileLocation: String,
                          includeNonProb: Boolean,
                          includeNaiveBayes: Boolean)
  : Map[String, DataFrame] = {
    // TODO: Add other file types for testing
    // TODO move this to new CSV reader
    val dataset: DataFrame =
      session.read.format("com.databricks.spark.csv")
        .option("header", "true").option("inferSchema", "true")
        .option("treatEmptyValuesAsNulls", "false")
        .option("delimiter", if (fileName.endsWith(".csv")) "," else "\t")
        .load(fileLocation)

    val modelsToExclude = List(
      if (!includeNaiveBayes) List(NBName) else Nil,
      if (!includeNonProb) List(GBTName, MLPName) else Nil)
      .flatten.toSet

    Map(LRName -> createLR, DTName -> createDT,
      GBTName -> createGBT, RFName -> createRF,
      MLPName -> createMLP, NBName -> createNB)
      .mapValues(creator => creator.setLabelCol(labelColumnName))
      .filterKeys(!modelsToExclude(_))
      .mapValues(trainScoreDataset(labelColumnName, dataset, _))
  }

  /** Get the auc and area over PR for the scored dataset.
    *
    * @param scoredDataset    The scored dataset to evaluate.
    * @param labelColumn      The label column.
    * @param predictionColumn The prediction column.
    * @return The AUC for the scored dataset.
    */
  def evalAUC(name: String,
              scoredDataset: DataFrame,
              labelColumn: String,
              predictionColumn: String,
              decimals: Int): Unit = {
    // Get levels if categorical
    val levels = getLevels(scoredDataset.schema, labelColumn)
    if (levels.isEmpty) throw new Exception("Test unexpectedly received empty levels")
    val levelsToIndex: Map[Any, Double] = levels.get.zipWithIndex.map(t => t._1 -> t._2.toDouble).toMap

    val scoreAndLabels = scoredDataset
      .select(col(predictionColumn), col(labelColumn))
      .na.drop().rdd.map {
      case Row(prediction: Vector, label) => (prediction(1), levelsToIndex(label))
      case Row(prediction: Double, label) => (prediction, levelsToIndex(label))
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val result = (metrics.areaUnderROC(), metrics.areaUnderPR())

    addBenchmark(name + "AUROC", result._1, decimals)
    addBenchmark(name + "AUPR", result._2, decimals)
    metrics.unpersist()
  }

  /** Get the accuracy and f1-score from multiclass data.
    *
    * @param scoredDataset    The scored dataset to evaluate.
    * @param labelColumn      The label column.
    * @param predictionColumn The prediction column.
    * @return The AUC for the scored dataset.
    */
  def evalMulticlass(name: String,
                     scoredDataset: DataFrame,
                     labelColumn: String,
                     predictionColumn: String,
                     decimals: Int): Unit = {

    // Get levels if categorical
    val levels = getLevels(scoredDataset.schema, labelColumn)
    if (levels.isEmpty) throw new Exception("Test unexpectedly received empty levels")
    val levelsToIndexMap: Map[Any, Double] = levels.get.zipWithIndex.map(t => t._1 -> t._2.toDouble).toMap

    val scoreAndLabels =
      scoredDataset.select(col(predictionColumn), col(labelColumn)).na.drop().rdd.map {
        case Row(prediction: Vector, label) => (prediction(1), levelsToIndexMap(label))
        case Row(prediction: Double, label) => (prediction, levelsToIndexMap(label))
      }
    val metrics = new MulticlassMetrics(scoreAndLabels)
    addBenchmark(name + "Accuracy", metrics.accuracy, decimals)
    addBenchmark(name + "WeightedFMeasure", metrics.weightedFMeasure, decimals)
  }

  override def testObjects(): Seq[TestObject[TrainClassifier]] =
    List(new TestObject(new TrainClassifier().setModel(
      new LogisticRegression()).setLabelCol(mockLabelCol), createMockDataset))

  override def reader: MLReadable[_] = TrainClassifier

  override def modelReader: MLReadable[_] = TrainedClassifierModel
}

/** Test helper methods for Train Classifier module. */
object TrainClassifierTestUtilities {
  val defaultFeaturesCol = "mlfeatures"

  //TODO none of these functions should require a label name,
  //TODO thats the whole point of the .setLabelColumn call
  def createLR: TrainClassifier = {
    wrap(new LogisticRegression()
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setMaxIter(10))
  }

  def createDT: TrainClassifier = {
    wrap(new DecisionTreeClassifier()
      .setMaxBins(32)
      .setMaxDepth(5)
      .setMinInfoGain(0.0)
      .setMinInstancesPerNode(1)
      .setSeed(0L))
  }

  def createGBT: TrainClassifier = {
    wrap(new GBTClassifier()
      .setMaxBins(32)
      .setMaxDepth(5)
      .setMaxIter(20)
      .setMinInfoGain(0.0)
      .setMinInstancesPerNode(1)
      .setStepSize(0.1)
      .setSubsamplingRate(1.0)
      .setSeed(0L))
  }

  def createRF: TrainClassifier = {
    wrap(new RandomForestClassifier()
      .setMaxBins(32)
      .setMaxDepth(5)
      .setMinInfoGain(0.0)
      .setMinInstancesPerNode(1)
      .setNumTrees(20)
      .setSubsamplingRate(1.0)
      .setSeed(0L))
  }

  def createMLP: TrainClassifier = {
    wrap(new MultilayerPerceptronClassifier()
      .setLayers(Array[Int](2, 5, 2))
      .setBlockSize(1)
      .setMaxIter(1)
      .setTol(1e-6)
      .setSeed(0L))
  }

  def createNB: TrainClassifier = {
    wrap(new NaiveBayes())
  }

  private def wrap(est: Estimator[_ <: Model[_]]): TrainClassifier = {
    new TrainClassifier()
      .setModel(est)
      .setFeaturesCol(defaultFeaturesCol)
  }

  def trainScoreDataset(labelColumn: String, dataset: DataFrame,
                        trainClassifier: Estimator[TrainedClassifierModel]): DataFrame = {
    val data = dataset.randomSplit(Seq(0.6, 0.4).toArray, 42)
    trainClassifier.fit(data(0))
      .transform(data(1))
  }

}
