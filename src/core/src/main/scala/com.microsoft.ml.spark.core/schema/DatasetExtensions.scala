// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.ScalaReflection.Schema
import org.apache.spark.sql.types.StructType

import scala.collection.mutable

/** Contains methods for manipulating spark dataframes and datasets. */
object DatasetExtensions {

  implicit class MMLDataFrame(val df: DataFrame) extends AnyVal {
    /** Finds an unused column name given initial column name in the given schema.
      * The unused column name will be given prefix with a number appended to it, eg "testColumn_5".
      * There will be an underscore between the column name and the number appended.
      *
      * @return The unused column name.
      */
    def withDerivativeCol(prefix: String): String = {
      val columnNamesSet = mutable.HashSet(df.columns: _*)
      findUnusedColumnName(prefix)(columnNamesSet)
    }

    /** Gets the column values as the given type.
      * @param colname The column name to retrieve from.
      * @tparam T The type to retrieve.
      * @return The sequence of values in the column.
      */
    def getColAs[T](colname: String): Seq[T] = {
      df.select(colname).collect.map(_.getAs[T](0))
    }

    /** Gets the spark sparse vector column.
      * @return The spark sparse vector column.
      */
    def getSVCol: String => Seq[SparseVector] = getColAs[SparseVector] _

    /** Gets the spark dense vector column.
      * @return The spark dense vector column.
      */
    def getDVCol: String => Seq[DenseVector] = getColAs[DenseVector] _
  }

  /** Finds an unused column name given initial column name and a list of existing column names.
    * The unused column name will be given prefix with a number appended to it, eg "testColumn_5".
    * There will be an underline between the column name and the number appended.
    *
    * @return The unused column name.
    */
  def findUnusedColumnName(prefix: String)(columnNames: scala.collection.Set[String]): String = {
    var counter = 2
    var unusedColumnName = prefix
    while (columnNames.contains(unusedColumnName)) {
      unusedColumnName += "_" + counter
      counter += 1
    }
    unusedColumnName
  }

  def findUnusedColumnName(prefix: String, schema: StructType): String = {
    findUnusedColumnName(prefix)(schema.fieldNames.toSet)
  }

  def findUnusedColumnName(prefix: String, df: DataFrame): String = {
    findUnusedColumnName(prefix, df.schema)
  }

}
