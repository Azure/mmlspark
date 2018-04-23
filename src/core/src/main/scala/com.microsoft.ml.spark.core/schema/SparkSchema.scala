// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import SchemaConstants._
import com.microsoft.ml.spark.schema.CategoricalColumnInfo

import scala.reflect.ClassTag

/** Schema modification and information retrieval methods. */
object SparkSchema {

  /** Sets the label column name.
    *
    * @param dataset    The dataset to set the label column name on.
    * @param modelName  The model name.
    * @param columnName The column name to set as the label.
    * @param scoreValueKindModel The model type.
    * @return The modified dataset.
    */
  def setLabelColumnName: (DataFrame, String, String, String) => DataFrame =
    setColumnName(TrueLabelsColumn)

  /** Sets the scored labels column name.
    *
    * @param dataset    The dataset to set the scored labels column name on.
    * @param modelName  The model name.
    * @param columnName The column name to set as the scored label.
    * @param scoreValueKindModel The model type.
    * @return The modified dataset.
    */
  def setScoredLabelsColumnName: (DataFrame, String, String, String) => DataFrame =
    setColumnName(ScoredLabelsColumn)

  /** Sets the scored probabilities column name.
    *
    * @param dataset    The dataset to set the scored probabilities column name on.
    * @param modelName  The model name.
    * @param columnName The column name to set as the scored probability.
    * @param scoreValueKindModel The model type.
    * @return The modified dataset.
    */
  def setScoredProbabilitiesColumnName: (DataFrame, String, String, String) => DataFrame =
    setColumnName(ScoredProbabilitiesColumn)

  /** Sets the scores column name.
    *
    * @param dataset    The dataset to set the scores column name on.
    * @param modelName  The model name.
    * @param columnName The column name to set as the scores.
    * @param scoreValueKindModel The model type.
    * @return The modified dataset.
    */
  def setScoresColumnName: (DataFrame, String, String, String) => DataFrame =
    setColumnName(ScoresColumn)

  /** Gets the label column name.
    *
    * @param dataset   The dataset to get the label column from.
    * @param modelName The model to retrieve the label column from.
    * @return The label column name.
    */
  def getLabelColumnName(dataset: DataFrame, modelName: String): String =
    getScoreColumnKindColumn(TrueLabelsColumn)(dataset.schema, modelName)

  /** Gets the scored labels column name.
    *
    * @param dataset   The dataset to get the scored labels column from.
    * @param modelName The model to retrieve the scored labels column from.
    * @return The scored labels column name.
    */
  def getScoredLabelsColumnName(dataset: DataFrame, modelName: String): String =
    getScoreColumnKindColumn(ScoredLabelsColumn)(dataset.schema, modelName)

  /** Gets the scores column name.
    *
    * @param dataset   The dataset to get the scores column from.
    * @param modelName The model to retrieve the scores column from.
    * @return The scores column name.
    */
  def getScoresColumnName(dataset: DataFrame, modelName: String): String =
    getScoreColumnKindColumn(ScoresColumn)(dataset.schema, modelName)

  /** Gets the scored probabilities column name.
    *
    * @param dataset   The dataset to get the scored probabilities column from.
    * @param modelName The model to retrieve the scored probabilities column from.
    * @return The scored probabilities column name.
    */
  def getScoredProbabilitiesColumnName(dataset: DataFrame, modelName: String): String =
    getScoreColumnKindColumn(ScoredProbabilitiesColumn)(dataset.schema, modelName)

  /** Gets the label column name.
    *
    * @param dataset   The dataset to get the label column from.
    * @param modelName The model to retrieve the label column from.
    * @return The label column name.
    */
  def getLabelColumnName: (StructType, String) => String =
    getScoreColumnKindColumn(TrueLabelsColumn)

  /** Gets the scored labels column name.
    *
    * @param dataset   The dataset to get the scored labels column from.
    * @param modelName The model to retrieve the scored labels column from.
    * @return The scored labels column name.
    */
  def getScoredLabelsColumnName: (StructType, String) => String =
    getScoreColumnKindColumn(ScoredLabelsColumn)

  /** Gets the scores column name.
    *
    * @param dataset   The dataset to get the scores column from.
    * @param modelName The model to retrieve the scores column from.
    * @return The scores column name.
    */
  def getScoresColumnName: (StructType, String) => String =
    getScoreColumnKindColumn(ScoresColumn)

  /** Gets the scored probabilities column name.
    *
    * @param dataset   The dataset to get the scored probabilities column from.
    * @param modelName The model to retrieve the scored probabilities column from.
    * @return The scored probabilities column name.
    */
  def getScoredProbabilitiesColumnName: (StructType, String) => String =
    getScoreColumnKindColumn(ScoredProbabilitiesColumn)

  /** Gets the score value kind or null if it does not exist from a dataset.
    *
    * @param scoreColumnKindColumn The score column kind to retrieve.
    * @param dataset   The dataset to get the score column kind column name from.
    * @param modelName The model to retrieve the score column kind column name from.
    * @param columnName The column to retrieve the score value kind from.
    * @return
    */
  def getScoreValueKind(dataset: DataFrame, modelName: String, columnName: String): String = {
    getScoreValueKind(dataset.schema, modelName, columnName)
  }

  /** Gets the score value kind or null if it does not exist from the schema.
    *
    * @param scoreColumnKindColumn The score column kind to retrieve.
    * @param schema   The schema to get the score column kind column name from.
    * @param modelName The model to retrieve the score column kind column name from.
    * @param columnName The column to retrieve the score value kind from.
    * @return
    */
  def getScoreValueKind(schema: StructType, modelName: String, columnName: String): String = {
    val metadata = schema(columnName).metadata
    if (metadata == null) return null
    getMetadataFromModule(metadata, modelName, ScoreValueKind)
  }

  /** Sets the score column kind.
    *
    * @param scoreColumnKindColumn The score column kind column.
    * @param dataset               The dataset to set the score column kind on.
    * @param modelName             The model name.
    * @param columnName            The column name to set as the specified score column kind.
    * @param scoreValueKindModel   The model type.
    * @return
    */
  private def setColumnName(scoreColumnKindColumn: String)
                           (dataset: DataFrame, modelName: String,
                            columnName: String, scoreValueKindModel: String): DataFrame = {
    dataset.withColumn(columnName,
      dataset.col(columnName).as(columnName,
        updateMetadata(dataset.schema(columnName).metadata,
          scoreColumnKindColumn, scoreValueKindModel, modelName)))
  }

  /** Gets the score column kind column name or null if it does not exist.
    *
    * @param scoreColumnKindColumn The score column kind to retrieve.
    * @param schema   The schema to get the score column kind column name from.
    * @param modelName The model to retrieve the score column kind column name from.
    * @return
    */
  private def getScoreColumnKindColumn(scoreColumnKindColumn: String)
                                      (schema: StructType, modelName: String): String = {
    val structField = schema.find {
      case StructField(_, _, _, metadata) =>
        getMetadataFromModule(metadata, modelName, ScoreColumnKind) == scoreColumnKindColumn
    }
    if (structField.isEmpty) null else structField.get.name
  }

  private def updateMetadata(metadata: Metadata, scoreColumnKindColumn: String,
                             scoreValueKindModel: String, moduleName: String): Metadata = {
    val mmltagMetadata =
      if (metadata.contains(MMLTag)) metadata.getMetadata(MMLTag)
      else null
    val moduleNameMetadata =
      if (mmltagMetadata != null && mmltagMetadata.contains(moduleName))
        mmltagMetadata.getMetadata(moduleName)
      else null

    val moduleMetadataBuilder = new MetadataBuilder()
    if (mmltagMetadata != null && moduleNameMetadata != null) {
      moduleMetadataBuilder.withMetadata(moduleNameMetadata)
    }
    moduleMetadataBuilder.putString(ScoreColumnKind, scoreColumnKindColumn)
    moduleMetadataBuilder.putString(ScoreValueKind, scoreValueKindModel)

    val moduleBuilder = new MetadataBuilder()
    if (mmltagMetadata != null) {
      moduleBuilder.withMetadata(mmltagMetadata)
    }
    moduleBuilder.putMetadata(moduleName, moduleMetadataBuilder.build())

    new MetadataBuilder()
      .withMetadata(metadata)
      .putMetadata(MMLTag, moduleBuilder.build())
      .build()
  }

  private def getMetadataFromModule(colMetadata: Metadata, moduleName: String, tag: String): String = {
    if (!colMetadata.contains(MMLTag)) return null
    val mlTagMetadata = colMetadata.getMetadata(MMLTag)
    if (!mlTagMetadata.contains(moduleName)) return null
    val modelMetadata = mlTagMetadata.getMetadata(moduleName)
    if (!modelMetadata.contains(tag)) return null
    modelMetadata.getString(tag)
  }

  /** Find if the given column is a string */
  def isString(df: DataFrame, column: String): Boolean = {
    df.schema(column).dataType == DataTypes.StringType
  }

  /** Find if the given column is numeric */
  def isNumeric(df: DataFrame, column: String): Boolean = {
    df.schema(column).dataType.isInstanceOf[NumericType]
  }

  /** Find if the given column is boolean */
  def isBoolean(df: DataFrame, column: String): Boolean = {
    df.schema(column).dataType.isInstanceOf[BooleanType]
  }

  /** Find if the given column is Categorical; use CategoricalColumnInfo for more details */
  def isCategorical(df: DataFrame, column: String): Boolean = {
    val info = new CategoricalColumnInfo(df, column)
    info.isCategorical
  }

}
