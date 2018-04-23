// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.metrics

/** Contains constants used by modules for metrics. */
object MetricConstants {
  // Regression metrics
  val MseSparkMetric  = "mse"
  val RmseSparkMetric = "rmse"
  val R2SparkMetric   = "r2"
  val MaeSparkMetric  = "mae"
  val RegressionMetrics = "regression"

  val regressionMetrics = Set(MseSparkMetric, RmseSparkMetric, R2SparkMetric,
    MaeSparkMetric, RegressionMetrics)

  // Binary Classification metrics
  val AreaUnderROCMetric   = "areaUnderROC"
  val AucSparkMetric       = "AUC"
  val AccuracySparkMetric  = "accuracy"
  val PrecisionSparkMetric = "precision"
  val RecallSparkMetric    = "recall"
  val ClassificationMetrics = "classification"

  val classificationMetrics = Set(AreaUnderROCMetric, AucSparkMetric, AccuracySparkMetric,
    PrecisionSparkMetric, RecallSparkMetric, ClassificationMetrics)

  val AllSparkMetrics      = "all"

  // Regression column names
  val MseColumnName  = "mean_squared_error"
  val RmseColumnName = "root_mean_squared_error"
  val R2ColumnName   = "R^2"
  val MaeColumnName  = "mean_absolute_error"

  // Binary Classification column names
  val AucColumnName = "AUC"

  // Binary and Multiclass (micro-averaged) column names
  val PrecisionColumnName = "precision"
  val RecallColumnName    = "recall"
  val AccuracyColumnName  = "accuracy"

  // Multiclass Classification column names
  val AverageAccuracy        = "average_accuracy"
  val MacroAveragedRecall    = "macro_averaged_recall"
  val MacroAveragedPrecision = "macro_averaged_precision"

  // Metric to column name
  val metricToColumnName = Map(AccuracySparkMetric -> AccuracyColumnName,
    PrecisionSparkMetric -> PrecisionColumnName,
    RecallSparkMetric    -> RecallColumnName,
    MseSparkMetric       -> MseColumnName,
    RmseSparkMetric      -> RmseColumnName,
    R2SparkMetric        -> R2ColumnName,
    MaeSparkMetric       -> MaeColumnName)

  val classificationColumns = List(AccuracyColumnName, PrecisionColumnName, RecallColumnName)

  val regressionColumns = List(MseColumnName, RmseColumnName, R2ColumnName, MaeColumnName)

  val ClassificationEvaluationType = "Classification"
  val EvaluationType = "evaluation_type"

  val FpRateROCColumnName = "false_positive_rate"
  val TpRateROCColumnName = "true_positive_rate"

  val FpRateROCLog = "fpr"
  val TpRateROCLog = "tpr"

  val BinningThreshold = 1000

  // Regression per instance metrics
  val L1LossMetric  = "L1_loss"
  val L2LossMetric  = "L2_loss"

  val regressionPerInstanceMetrics = Set(RegressionMetrics)

  // Classification per instance metrics
  val LogLossMetric = "log_loss"

  val classificationPerInstanceMetrics = Set(ClassificationMetrics)

  val findBestModelMetrics =
    Set(MetricConstants.MseSparkMetric,
        MetricConstants.RmseSparkMetric,
        MetricConstants.R2SparkMetric,
        MetricConstants.MaeSparkMetric,
        MetricConstants.AccuracySparkMetric,
        MetricConstants.PrecisionSparkMetric,
        MetricConstants.RecallSparkMetric,
        MetricConstants.AucSparkMetric)

}
