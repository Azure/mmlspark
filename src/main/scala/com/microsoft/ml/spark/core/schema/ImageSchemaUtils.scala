// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.types._

object ImageSchemaUtils {

  val ColumnSchemaNullable: StructType = {
    StructType(
      StructField("origin", StringType, true) ::
        StructField("height", IntegerType, true) ::
        StructField("width", IntegerType, true) ::
        StructField("nChannels", IntegerType, true) ::
        // OpenCV-compatible type: CV_8UC3 in most cases
        StructField("mode", IntegerType, true) ::
        // Bytes in OpenCV-compatible order: row-wise BGR in most cases
        StructField("data", BinaryType, true) :: Nil)
  }

  val ImageSchemaNullable: StructType = StructType(StructField("image", ColumnSchemaNullable, true) :: Nil)

  def isImage(dataType: DataType): Boolean = {
    dataType == ImageSchema.columnSchema ||
      dataType == ColumnSchemaNullable
  }

  def isImage(dataType: StructField): Boolean = {
    isImage(dataType.dataType)
  }
}
