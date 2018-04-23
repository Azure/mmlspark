// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.datagen

import com.microsoft.ml.spark.core.test.datagen.ColumnOptions.ColumnOptions
import com.microsoft.ml.spark.core.test.datagen.DataOptions.DataOptions

/** Specifies the column types supported in spark dataframes and modules. */
object ColumnOptions extends Enumeration {
  type ColumnOptions = Value
  // TODO: add Categorical, DenseVector, SparseVector
  val Scalar = Value
}

/** Specifies the data types supported in spark dataframes and modules. */
object DataOptions extends Enumeration {
  type DataOptions = Value
  val String, Int, Double, Boolean, Date, Timestamp, Byte, Short = Value
}

/** Options used to specify how a dataset will be generated.
  * This contains information on what the data and column types
  * (specified as flags) for generating a dataset will be limited to.
  * It also contain options for all possible missing values generation
  * and options for how values will be generated.
  */
case class DatasetOptions(columnTypes: ColumnOptions.ValueSet,
                          dataTypes: DataOptions.ValueSet,
                          missingValuesOptions: DatasetMissingValuesGenerationOptions)

object DatasetOptions {
  def apply(columnOptions: ColumnOptions.ValueSet, dataOptions: DataOptions.ValueSet): DatasetOptions = {
    val missingValueOptions = DatasetMissingValuesGenerationOptions(0.0, columnOptions, dataOptions)
    new DatasetOptions(columnOptions, dataOptions, missingValueOptions)
  }

  def apply(columnOption: ColumnOptions, dataOption: DataOptions): DatasetOptions = {
    val colOptions = ColumnOptions.ValueSet(columnOption)
    val dataOptions = DataOptions.ValueSet(dataOption)
    val missingValueOptions = DatasetMissingValuesGenerationOptions(0.0, colOptions, dataOptions)
    new DatasetOptions(colOptions, dataOptions, missingValueOptions)
  }
}

case class DatasetMissingValuesGenerationOptions(percentMissing: Double,
                                                 columnTypesWithMissings: ColumnOptions.ValueSet,
                                                 dataTypesWithMissings: DataOptions.ValueSet) {
  def hashMissing(): Boolean = {
    !columnTypesWithMissings.isEmpty && !dataTypesWithMissings.isEmpty
  }
}
