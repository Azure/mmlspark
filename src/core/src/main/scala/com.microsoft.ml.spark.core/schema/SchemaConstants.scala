// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

/** Contains constants used by modules for schema. */
object SchemaConstants {

  val ScoreColumnKind           = "ScoreColumnKind"
  val ScoreValueKind            = "ScoreValueKind"

  val TrueLabelsColumn          = "true_labels"
  val ScoredLabelsColumn        = "scored_labels"
  val ScoresColumn              = "scores"
  val ScoredProbabilitiesColumn = "scored_probabilities"

  val ScoreModelPrefix          = "score_model"
  val MMLTag                    = "mml"      // MML metadata tag
  val MLlibTag                  = "ml_attr"  // MLlib metadata tag, see org.apache.spark.ml.attribute.AttributeKeys

  /** The following tags are used in Metadata representation of categorical data
    * do not change them or use them directly
    * (see org.apache.spark.ml.attribute.AttributeKeys for the first three)
    */
  val Ordinal                   = "ord"        // common tag for both MLlib and MML
  val MLlibTypeTag              = "type"       // MLlib tag for the attribute types
  val ValuesString              = "vals"       // common tag for both MLlib and MML
  val ValuesInt                 = "vals_int"
  val ValuesLong                = "vals_long"
  val ValuesDouble              = "vals_double"
  val ValuesBool                = "vals_bool"
  val HasNullLevels             = "null_exists"

  // Score value kinds, or types of ML:
  val ClassificationKind        = "Classification"
  val RegressionKind            = "Regression"

  // Spark native column names
  val SparkPredictionColumn     = "prediction"
  val SparkRawPredictionColumn  = "rawPrediction"
  val SparkProbabilityColumn    = "probability"

}
