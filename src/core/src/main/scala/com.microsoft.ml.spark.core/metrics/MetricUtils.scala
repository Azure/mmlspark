// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.metrics

import com.microsoft.ml.spark.core.schema.{SchemaConstants, SparkSchema}
import org.apache.spark.sql.types.{Metadata, MetadataUtilities, StructField, StructType}
import com.microsoft.ml.spark.core.schema.SchemaConstants.MMLTag

/** Utilities used by modules for metrics. */
object MetricUtils {

  def isClassificationMetric(metric: String): Boolean = {
    if (MetricConstants.regressionMetrics.contains(metric)) false
    else if (MetricConstants.classificationMetrics.contains(metric)) true
    else throw new Exception("Invalid metric specified")
  }

  def getSchemaInfo(schema: StructType, labelCol: Option[String],
                    evaluationMetric: String): (String, String, String) = {
    val schemaInfo = tryGetSchemaInfo(schema)
    if (schemaInfo.isDefined) {
      schemaInfo.get
    } else {
      if (labelCol.isEmpty || evaluationMetric == MetricConstants.AllSparkMetrics) {
        throw new Exception("Please score the model prior to evaluating")
      }
      ("custom model", labelCol.get,
        if (isClassificationMetric(evaluationMetric))
          SchemaConstants.ClassificationKind
        else SchemaConstants.RegressionKind)
    }
  }

  private def tryGetSchemaInfo(schema: StructType): Option[(String, String, String)] = {
    // TODO: evaluate all models; for now, get first model name found
    val firstModelName = schema.collectFirst {
      case StructField(c, t, _, m) if getFirstModelName(m) != null && !getFirstModelName(m).isEmpty => {
        getFirstModelName(m).get
      }
    }
    if (firstModelName.isEmpty) None
    else {
      val modelName = firstModelName.get
      val labelColumnName = SparkSchema.getLabelColumnName(schema, modelName)
      val scoreValueKind = SparkSchema.getScoreValueKind(schema, modelName, labelColumnName)
      Option((modelName, labelColumnName, scoreValueKind))
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

}
