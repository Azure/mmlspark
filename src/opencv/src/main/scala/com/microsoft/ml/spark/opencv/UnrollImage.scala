// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.opencv

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, MMLParams}
import com.microsoft.ml.spark.core.schema.ImageSchema._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

object UnrollImage extends DefaultParamsReadable[UnrollImage]{

  private def unroll(row: Row): DenseVector = {
    val width = getWidth(row)
    val height = getHeight(row)
    val bytes = getBytes(row)

    val area = width*height
    require(area >= 0 && area < 1e8, "image has incorrect dimensions" )
    require(bytes.length == width*height*3, "image has incorrect nuber of bytes" )

    var rearranged =  Array.fill[Double](area*3)(0.0)
    var count = 0
    for (c <- 0 until 3) {
      for (h <- 0 until height) {
        val offset = h * width * 3
        for (w <- 0 until width) {
          val b = bytes(offset + w*3 + c).toDouble

          //TODO: is there a better way to convert to unsigned byte?
          rearranged(count) =  if (b>0) b else b + 256.0
          count += 1
        }
      }
    }
    new DenseVector(rearranged)
  }
}

/** Converts the representation of an m X n pixel image to an m * n vector of Doubles
  *
  * The input column name is assumed to be "image", the output column name is "<uid>_output"
  *
  * @param uid The id of the module
  */
class UnrollImage(val uid: String) extends Transformer with HasInputCol with HasOutputCol with MMLParams{

  def this() = this(Identifiable.randomUID("UnrollImage"))

  setDefault(inputCol -> "image", outputCol -> (uid + "_output"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF
    assert(isImage(df, $(inputCol)), "input column should have Image type")

    val unrollUDF = udf(UnrollImage.unroll _)
    df.withColumn($(outputCol), unrollUDF(df($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add($(outputCol), VectorType)
  }

}
