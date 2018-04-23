// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts._
import com.microsoft.ml.spark.core.metrics.{MetricConstants, MetricUtils}
import com.microsoft.ml.spark.core.schema.{SchemaConstants, SparkSchema}
import com.microsoft.ml.spark.schema.CategoricalUtilities
import org.apache.log4j.Logger
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, RegressionMetrics}
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object ComputeModelStatistics extends DefaultParamsReadable[ComputeModelStatistics]

trait ComputeModelStatisticsParams extends MMLParams
  with HasLabelCol with HasScoresCol with HasScoredLabelsCol with HasEvaluationMetric {
  /** Param "evaluationMetric" is the metric to evaluate the models with. Default is "all"
    *
    * The metrics that can be chosen are:
    *
    *   For binary classification:
    *     - AreaUnderROC
    *     - AUC
    *     - accuracy
    *     - precision
    *     - recall
    *
    *   For regression:
    *     - mse
    *     - rmse
    *     - r2
    *     - mae
    *
    *   Or, for either:
    *     - all - This will report all the relevant metrics
    *
    *   If using a native Spark ML model, you will need to specify either "classifier" or "regressor"
    *     - classifier
    *     - regressor
    *
    * @group param
    */
  setDefault(evaluationMetric -> MetricConstants.AllSparkMetrics)
}

/** Evaluates the given scored dataset. */
class ComputeModelStatistics(override val uid: String) extends Transformer with ComputeModelStatisticsParams {
  def this() = this(Identifiable.randomUID("ComputeModelStatistics"))

  /** The ROC curve evaluated for a binary classifier. */
  var rocCurve: DataFrame = null

  lazy val metricsLogger = new MetricsLogger(uid)

  /** Calculates the metrics for the given dataset and model.
    * @param dataset
    * @return DataFrame whose columns contain the calculated metrics
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val (modelName, labelColumnName, scoreValueKind) =
      MetricUtils.getSchemaInfo(
        dataset.schema,
        if (isDefined(labelCol)) Some(getLabelCol) else None,
        getEvaluationMetric)

    // For creating the result dataframe in classification or regression case
    val spark = dataset.sparkSession
    import spark.implicits._

    if (scoreValueKind == SchemaConstants.ClassificationKind) {

      var resultDF: DataFrame =
        Seq(MetricConstants.ClassificationEvaluationType)
          .toDF(MetricConstants.EvaluationType)
      val scoredLabelsColumnName =
        if (isDefined(scoredLabelsCol)) getScoredLabelsCol
        else SparkSchema.getScoredLabelsColumnName(dataset.schema, modelName)

      // Get levels for label column if categorical
      val levels = CategoricalUtilities.getLevels(dataset.schema, labelColumnName)

      val levelsExist = levels.isDefined

      lazy val levelsToIndexMap: Map[Any, Double] = getLevelsToIndexMap(levels.get)

      lazy val predictionAndLabels =
        if (levelsExist)
          getPredictionAndLabels(dataset, labelColumnName, scoredLabelsColumnName, levelsToIndexMap)
        else
          selectAndCastToRDD(dataset, scoredLabelsColumnName, labelColumnName)

      lazy val scoresAndLabels = {
        val scoresColumnName =
          if (isDefined(scoresCol)) getScoresCol
          else SparkSchema.getScoresColumnName(dataset.schema, modelName)
        if (scoresColumnName == null) predictionAndLabels
        else if (levelsExist) getScoresAndLabels(dataset, labelColumnName, scoresColumnName, levelsToIndexMap)
        else getScalarScoresAndLabels(dataset, labelColumnName, scoresColumnName)
      }

      lazy val (labels: Array[Double], confusionMatrix: Matrix) = createConfusionMatrix(predictionAndLabels)

      // If levels exist, use the extra information they give to get better performance
      getEvaluationMetric match {
        case allMetrics if allMetrics == MetricConstants.AllSparkMetrics ||
                           allMetrics == MetricConstants.ClassificationMetrics => {
          resultDF = addConfusionMatrixToResult(labels, confusionMatrix, resultDF)
          resultDF = addAllClassificationMetrics(
              modelName, dataset, labelColumnName, predictionAndLabels,
              confusionMatrix, scoresAndLabels, resultDF)
        }
        case simpleMetric if simpleMetric == MetricConstants.AccuracySparkMetric ||
                             simpleMetric == MetricConstants.PrecisionSparkMetric ||
                             simpleMetric == MetricConstants.RecallSparkMetric => {
          resultDF = addSimpleMetric(simpleMetric, predictionAndLabels, resultDF)
        }
        case MetricConstants.AucSparkMetric => {
          val numLevels = if (levelsExist) levels.get.length
          else confusionMatrix.numRows
          if (numLevels <= 2) {
            // Add the AUC
            val auc: Double = getAUC(modelName, dataset, labelColumnName, scoresAndLabels)
            resultDF = resultDF.withColumn(MetricConstants.AucColumnName, lit(auc))
          } else {
            throw new Exception("Error: AUC is not available for multiclass case")
          }
        }
        case default => {
          throw new Exception(s"Error: $default is not a classification metric")
        }
      }
      resultDF
    } else if (scoreValueKind == SchemaConstants.RegressionKind) {
      val scoresColumnName =
        if (isDefined(scoresCol)) getScoresCol
        else SparkSchema.getScoresColumnName(dataset.schema, modelName)

      val scoresAndLabels = selectAndCastToRDD(dataset, scoresColumnName, labelColumnName)

      val regressionMetrics = new RegressionMetrics(scoresAndLabels)

      // get all spark metrics possible: "mse", "rmse", "r2", "mae"
      val mse  = regressionMetrics.meanSquaredError
      val rmse = regressionMetrics.rootMeanSquaredError
      val r2   = regressionMetrics.r2
      val mae  = regressionMetrics.meanAbsoluteError

      metricsLogger.logRegressionMetrics(mse, rmse, r2, mae)

      Seq((mse, rmse, r2, mae)).toDF(MetricConstants.MseColumnName,
                                     MetricConstants.RmseColumnName,
                                     MetricConstants.R2ColumnName,
                                     MetricConstants.MaeColumnName)
    } else {
      throwOnInvalidScoringKind(scoreValueKind)
    }
  }

  private def addSimpleMetric(simpleMetric: String,
                              predictionAndLabels: RDD[(Double, Double)],
                              resultDF: DataFrame): DataFrame = {
    val (labels: Array[Double], confusionMatrix: Matrix) = createConfusionMatrix(predictionAndLabels)
    // Compute metrics for binary classification
    if (confusionMatrix.numCols == 2) {
      val (accuracy: Double, precision: Double, recall: Double) =
        getBinaryAccuracyPrecisionRecall(confusionMatrix)
      metricsLogger.logClassificationMetrics(accuracy, precision, recall)
      // Add the metrics to the DF
      simpleMetric match {
        case MetricConstants.AccuracySparkMetric =>
          resultDF.withColumn(MetricConstants.AccuracyColumnName, lit(accuracy))
        case MetricConstants.PrecisionSparkMetric =>
          resultDF.withColumn(MetricConstants.PrecisionColumnName, lit(precision))
        case MetricConstants.RecallSparkMetric =>
          resultDF.withColumn(MetricConstants.RecallColumnName, lit(recall))
        case default => resultDF
      }
    } else {
      val (microAvgAccuracy: Double, microAvgPrecision: Double, microAvgRecall: Double, _, _, _) =
        getMulticlassMetrics(predictionAndLabels, confusionMatrix)
      metricsLogger.logClassificationMetrics(microAvgAccuracy, microAvgPrecision, microAvgRecall)
      // Add the metrics to the DF
      simpleMetric match {
        case MetricConstants.AccuracySparkMetric =>
          resultDF.withColumn(MetricConstants.AccuracyColumnName, lit(microAvgAccuracy))
        case MetricConstants.PrecisionSparkMetric =>
          resultDF.withColumn(MetricConstants.PrecisionColumnName, lit(microAvgPrecision))
        case MetricConstants.RecallSparkMetric =>
          resultDF.withColumn(MetricConstants.RecallColumnName, lit(microAvgRecall))
        case default => resultDF
      }
    }
  }

  private def addAllClassificationMetrics(modelName: String,
                                          dataset: Dataset[_],
                                          labelColumnName: String,
                                          predictionAndLabels: RDD[(Double, Double)],
                                          confusionMatrix: Matrix,
                                          scoresAndLabels: RDD[(Double, Double)],
                                          resultDF: DataFrame): DataFrame = {
    // Compute metrics for binary classification
    if (confusionMatrix.numCols == 2) {
      val (accuracy: Double, precision: Double, recall: Double)
          = getBinaryAccuracyPrecisionRecall(confusionMatrix)
      metricsLogger.logClassificationMetrics(accuracy, precision, recall)
      // Add the AUC
      val auc: Double = getAUC(modelName, dataset, labelColumnName, scoresAndLabels)
      metricsLogger.logAUC(auc)
      // Add the metrics to the DF
      resultDF
        .withColumn(MetricConstants.AccuracyColumnName, lit(accuracy))
        .withColumn(MetricConstants.PrecisionColumnName, lit(precision))
        .withColumn(MetricConstants.RecallColumnName, lit(recall))
        .withColumn(MetricConstants.AucColumnName, lit(auc))
    } else {
      val (microAvgAccuracy: Double,
           microAvgPrecision: Double,
           microAvgRecall: Double,
           averageAccuracy: Double,
           macroAveragedPrecision: Double,
           macroAveragedRecall: Double)
          = getMulticlassMetrics(predictionAndLabels, confusionMatrix)
      metricsLogger.logClassificationMetrics(microAvgAccuracy, microAvgPrecision, microAvgRecall)
      resultDF
        .withColumn(MetricConstants.AccuracyColumnName, lit(microAvgAccuracy))
        .withColumn(MetricConstants.PrecisionColumnName, lit(microAvgPrecision))
        .withColumn(MetricConstants.RecallColumnName, lit(microAvgRecall))
        .withColumn(MetricConstants.AverageAccuracy, lit(averageAccuracy))
        .withColumn(MetricConstants.MacroAveragedPrecision, lit(macroAveragedPrecision))
        .withColumn(MetricConstants.MacroAveragedRecall, lit(macroAveragedRecall))
    }
  }

  private def addConfusionMatrixToResult(labels: Array[Double],
                                         confusionMatrix: Matrix,
                                         resultDF: DataFrame): DataFrame = {
    var resultDFModified = resultDF
    for (col: Int <- 0 until confusionMatrix.numCols;
         row: Int <- 0 until confusionMatrix.numRows) {
      resultDFModified = resultDFModified
        .withColumn(s"predicted_class_as_${labels(col).toString}_actual_is_${labels(row).toString}",
          lit(confusionMatrix(row, col)))
    }
    resultDFModified
  }

  private def selectAndCastToDF(dataset: Dataset[_],
                                predictionColumnName: String,
                                labelColumnName: String): DataFrame = {
    // TODO: We call cache in order to avoid a bug with catalyst where CMS seems to get stuck in a loop
    // For future spark upgrade past 2.2.0, we should try to see if the cache() call can be removed
    dataset.select(col(predictionColumnName), col(labelColumnName).cast(DoubleType))
      .cache()
      .na
      .drop(Array(predictionColumnName, labelColumnName))
  }

  private def selectAndCastToRDD(dataset: Dataset[_],
                                 predictionColumnName: String,
                                 labelColumnName: String): RDD[(Double, Double)] = {
    selectAndCastToDF(dataset, predictionColumnName, labelColumnName)
      .rdd
      .map {
        case Row(prediction: Double, label: Double) => (prediction, label)
        case default => throw new Exception(s"Error: prediction and label columns invalid or missing")
      }
  }

  private def getPredictionAndLabels(dataset: Dataset[_],
                                     labelColumnName: String,
                                     scoredLabelsColumnName: String,
                                     levelsToIndexMap: Map[Any, Double]): RDD[(Double, Double)] = {
    // Calculate confusion matrix and output it as DataFrame
    // TODO: We call cache in order to avoid a bug with catalyst where CMS seems to get stuck in a loop
    // For future spark upgrade past 2.2.0, we should try to see if the cache() call can be removed
    dataset.select(col(scoredLabelsColumnName), col(labelColumnName))
      .cache()
      .na
      .drop(Array(scoredLabelsColumnName, labelColumnName))
      .rdd
      .map {
        case Row(prediction: Double, label) => (prediction, levelsToIndexMap(label))
        case default => throw new Exception(s"Error: prediction and label columns invalid or missing")
    }
  }

  private def getScalarScoresAndLabels(dataset: Dataset[_],
                                       labelColumnName: String,
                                       scoresColumnName: String): RDD[(Double, Double)] = {
    selectAndCastToDF(dataset, scoresColumnName, labelColumnName)
      .rdd
      .map {
        case Row(prediction: Vector, label: Double) => (prediction(1), label)
        case default => throw new Exception(s"Error: prediction and label columns invalid or missing")
      }
  }

  private def getScoresAndLabels(dataset: Dataset[_],
                         labelColumnName: String,
                         scoresColumnName: String,
                         levelsToIndexMap: Map[Any, Double]): RDD[(Double, Double)] = {
    // TODO: We call cache in order to avoid a bug with catalyst where CMS seems to get stuck in a loop
    // For future spark upgrade past 2.2.0, we should try to see if the cache() call can be removed
    dataset.select(col(scoresColumnName), col(labelColumnName))
      .cache()
      .na
      .drop(Array(scoresColumnName, labelColumnName))
      .rdd
      .map {
        case Row(prediction: Vector, label) => (prediction(1), levelsToIndexMap(label))
        case default => throw new Exception(s"Error: prediction and label columns invalid or missing")
      }
  }

  private def getLevelsToIndexMap(levels: Array[_]): Map[Any, Double] = {
    levels.zipWithIndex.map(t => t._1 -> t._2.toDouble).toMap
  }

  private def getMulticlassMetrics(predictionAndLabels: RDD[(Double, Double)],
                                   confusionMatrix: Matrix): (Double, Double, Double, Double, Double, Double) = {
    // Compute multiclass metrics based on paper "A systematic analysis
    // of performance measure for classification tasks", Sokolova and Lapalme
    var tpSum: Double = 0.0
    for (diag: Int <- 0 until confusionMatrix.numCols) {
      tpSum += confusionMatrix(diag, diag)
    }
    val totalSum = predictionAndLabels.count()

    val microAvgAccuracy = tpSum / totalSum
    val microAvgPrecision = microAvgAccuracy
    val microAvgRecall = microAvgAccuracy

    // Compute class counts - these are the row and column sums of the matrix, used to calculate the
    // average accuracy, macro averaged precision and macro averaged recall
    val actualClassCounts = new Array[Double](confusionMatrix.numCols)
    val predictedClassCounts = new Array[Double](confusionMatrix.numRows)
    val truePositives = new Array[Double](confusionMatrix.numRows)
    for (rowIndex: Int <- 0 until confusionMatrix.numRows) {
      for (colIndex: Int <- 0 until confusionMatrix.numCols) {
        actualClassCounts(rowIndex) += confusionMatrix(rowIndex, colIndex)
        predictedClassCounts(colIndex) += confusionMatrix(rowIndex, colIndex)

        if (rowIndex == colIndex) {
          truePositives(rowIndex) += confusionMatrix(rowIndex, colIndex)
        }
      }
    }

    val (totalAccuracy, totalPrecision, totalRecall)
        = (0 until confusionMatrix.numCols).foldLeft((0.0,0.0,0.0)) {
      case ((acc, prec, rec), classIndex) =>
        (// compute the class accuracy as:
         // (true positive + true negative) / total =>
         // (true positive + (total - (actual + predicted - true positive))) / total =>
         // 2 * true positive + (total - (actual + predicted)) / total
         acc + (2 * truePositives(classIndex) +
                  (totalSum - (actualClassCounts(classIndex) + predictedClassCounts(classIndex)))) / totalSum,
         // compute the class precision as:
         // true positive / predicted as positive (=> tp + fp)
         prec + truePositives(classIndex) / predictedClassCounts(classIndex),
         // compute the class recall as:
         // true positive / actual positive (=> tp + fn)
         rec + truePositives(classIndex) / actualClassCounts(classIndex))
    }
    val averageAccuracy = totalAccuracy / confusionMatrix.numCols
    val macroAveragedPrecision = totalPrecision / confusionMatrix.numCols
    val macroAveragedRecall = totalRecall / confusionMatrix.numCols
    (microAvgAccuracy, microAvgPrecision, microAvgRecall, averageAccuracy, macroAveragedPrecision, macroAveragedRecall)
  }

  private def getAUC(modelName: String,
             dataset: Dataset[_],
             labelColumnName: String,
             scoresAndLabels: RDD[(Double, Double)]): Double = {
    val binaryMetrics = new BinaryClassificationMetrics(scoresAndLabels,
      MetricConstants.BinningThreshold)

    val spark = dataset.sparkSession
    import spark.implicits._

    rocCurve = binaryMetrics.roc()
      .toDF(MetricConstants.FpRateROCColumnName, MetricConstants.TpRateROCColumnName)
    metricsLogger.logROC(rocCurve)
    val auc = binaryMetrics.areaUnderROC()
    metricsLogger.logAUC(auc)
    auc
  }

  private def getBinaryAccuracyPrecisionRecall(confusionMatrix: Matrix): (Double, Double, Double) = {
    val TP: Double = confusionMatrix(1, 1)
    val FP: Double = confusionMatrix(0, 1)
    val TN: Double = confusionMatrix(0, 0)
    val FN: Double = confusionMatrix(1, 0)

    val accuracy: Double = (TP + TN) / (TP + TN + FP + FN)
    val precision: Double = TP / (TP + FP)
    val recall: Double = TP / (TP + FN)
    (accuracy, precision, recall)
  }

  private def createConfusionMatrix(predictionAndLabels: RDD[(Double, Double)]): (Array[Double], Matrix) = {
    val metrics = new MulticlassMetrics(predictionAndLabels)
    var labels = metrics.labels
    var confusionMatrix = metrics.confusionMatrix

    val numCols = confusionMatrix.numCols
    val numRows = confusionMatrix.numRows

    // Reformat the confusion matrix if less than binary size
    if (numCols < 2 && numRows < 2) {
      val values = Array.ofDim[Double](2 * 2)
      for (col: Int <- 0 until confusionMatrix.numCols;
           row: Int <- 0 until confusionMatrix.numRows) {
        // We need to interpret the actual label value
        val colLabel = if (labels(col) > 0) 1 else 0
        val rowLabel = if (labels(row) > 0) 1 else 0
        values(colLabel + rowLabel * 2) =
          confusionMatrix(row, col)
      }
      confusionMatrix = Matrices.dense(2, 2, values)
      labels = Array(0, 1)
    }
    (labels, confusionMatrix)
  }

  override def copy(extra: ParamMap): Transformer = new ComputeModelStatistics()

  override def transformSchema(schema: StructType): StructType = {
    val (_, _, scoreValueKind) =
      MetricUtils.getSchemaInfo(
        schema,
        if (isDefined(labelCol)) Some(getLabelCol) else None,
        getEvaluationMetric)
    val columns =
      if (scoreValueKind == SchemaConstants.ClassificationKind) MetricConstants.classificationColumns
      else if (scoreValueKind == SchemaConstants.RegressionKind) MetricConstants.regressionColumns
      else throwOnInvalidScoringKind(scoreValueKind)
    getTransformedSchema(columns, scoreValueKind)

  }

  private def throwOnInvalidScoringKind(scoreValueKind: String) = {
    throw new Exception(s"Error: unknown scoring kind $scoreValueKind")
  }

  private def getTransformedSchema(columns: List[String], metricType: String) = {
    getEvaluationMetric match {
      case allMetrics if allMetrics == MetricConstants.AllSparkMetrics ||
                         allMetrics == MetricConstants.ClassificationMetrics ||
                         allMetrics == MetricConstants.RegressionMetrics =>
        StructType(columns.map(StructField(_, DoubleType)))
      case metric: String if MetricConstants.metricToColumnName.contains(metric) &&
                             columns.contains(MetricConstants.metricToColumnName(metric)) =>
        StructType(Array(StructField(MetricConstants.metricToColumnName(metric), DoubleType)))
      case default =>
        throw new Exception(s"Error: $default is not a $metricType metric")
    }
  }
}

/** Helper class for logging metrics to log4j.
  * @param uid The unique id of the parent module caller.
  */
class MetricsLogger(uid: String) {

  lazy val logger = Logger.getLogger(this.getClass.getName)

  def logClassificationMetrics(accuracy: Double, precision: Double, recall: Double): Unit = {
    val metrics = MetricData.create(
        Map(MetricConstants.AccuracyColumnName -> accuracy,
            MetricConstants.PrecisionColumnName -> precision,
            MetricConstants.RecallColumnName -> recall),
        "Classification Metrics", uid)
    logger.info(metrics)
  }

  def logRegressionMetrics(mse: Double, rmse: Double, r2: Double, mae: Double): Unit = {
    val metrics = MetricData.create(
        Map(MetricConstants.MseColumnName -> mse,
            MetricConstants.RmseColumnName -> rmse,
            MetricConstants.R2ColumnName -> r2,
            MetricConstants.MaeColumnName -> mae),
        "Regression Metrics", uid)
    logger.info(metrics)
  }

  def logAUC(auc: Double): Unit = {
    val metrics = MetricData.create(Map(MetricConstants.AucColumnName -> auc), "AUC Metric", uid)
    logger.info(metrics)
  }

  def logROC(roc: DataFrame): Unit = {
    val metrics = MetricData.createTable(
      Map(MetricConstants.TpRateROCLog ->
            roc.select(MetricConstants.TpRateROCColumnName)
               .collect()
               .map(row => row(0).asInstanceOf[Double])
               .toSeq,
          MetricConstants.FpRateROCLog ->
            roc.select(MetricConstants.FpRateROCColumnName)
               .collect()
               .map(row => row(0).asInstanceOf[Double])
               .toSeq),
      "ROC Metric", uid)
    logger.info(metrics)
  }

}
