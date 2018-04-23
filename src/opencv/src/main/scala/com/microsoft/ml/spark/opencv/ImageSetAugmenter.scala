// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.opencv

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.schema.ImageSchema
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql._
import org.apache.spark.sql.types._

object ImageSetAugmenter extends DefaultParamsReadable[ImageSetAugmenter]

class ImageSetAugmenter(val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("ImageSetAugmenter"))

  // if the problem is symmetric, the training dataset is supplemented with the "flipped" images
  // by default, we assume the left-right symmetry, but not up-down
  val flipLeftRight = BooleanParam(this, "flipLeftRight", "Symmetric Left-Right", false)

  def setFlipLeftRight(value: Boolean): this.type = set(flipLeftRight, value)

  def getFlipLeftRight: Boolean = $(flipLeftRight)

  val flipUpDown = BooleanParam(this, "flipUpDown", "Symmetric Up-Down", false)

  def setFlipUpDown(value: Boolean): this.type = set(flipUpDown, value)

  def getFlipUpDown: Boolean = $(flipUpDown)

  setDefault(flipLeftRight -> true, flipUpDown -> false,
    outputCol -> (uid + "_output"), inputCol -> "image")

  //flip images in a given column, keep the rest of the columns intact
  private def flipImages(dataset: Dataset[_], inCol: String, outCol: String, flipCode: Int): DataFrame = {
    val tr = new ImageTransformer()
      .flip(flipCode)
      .setInputCol(inCol)
      .setOutputCol(outCol)

    tr.transform(dataset)
  }

  def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF
    val dfID = df.withColumn(getOutputCol, new Column(getInputCol))

    val dfLR: Option[DataFrame] =
      if (!getFlipLeftRight) None
      else Some(flipImages(df, getInputCol, getOutputCol, Flip.flipLeftRight))

    val dfUD: Option[DataFrame] =
      if (!getFlipUpDown) None
      else Some(flipImages(df, getInputCol, getOutputCol, Flip.flipUpDown))

    List(dfLR, dfUD).flatten(x => x).foldLeft(dfID) { case (dfl, tdr) => dfl.union(tdr) }

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, ImageSchema.columnSchema)
  }

}
