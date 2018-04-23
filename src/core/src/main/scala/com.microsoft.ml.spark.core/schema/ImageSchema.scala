// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

object ImageSchema {

  /** Schema for the image column: Row(String, Int, Int, Int, Array[Byte]) */
  val columnSchema = StructType(
    StructField("path",   StringType,  true) ::
    StructField("height", IntegerType, true) ::
    StructField("width",  IntegerType, true) ::
    // OpenCV type: CV_8U in most cases
    StructField("type", IntegerType, true) ::
    // OpenCV bytes: row-wise BGR in most cases
    StructField("bytes",  BinaryType, true) :: Nil)

  // single column of images named "image"
  val schema = StructType(StructField("image", columnSchema, true) :: Nil)

  def getPath  (row: Row): String = row.getString(0)
  def getHeight(row: Row): Int    = row.getInt(1)
  def getWidth (row: Row): Int    = row.getInt(2)
  def getType  (row: Row): Int    = row.getInt(3)
  def getBytes (row: Row): Array[Byte] = row.getAs[Array[Byte]](4)

  /** Check if the dataframe column contains images (i.e. has imageSchema)
    *
    * @param df
    * @param column
    * @return
    */
  def isImage(df: DataFrame, column: String): Boolean =
    df.schema(column).dataType == columnSchema

}
