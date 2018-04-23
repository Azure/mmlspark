// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.SparkException
import org.apache.spark.ml.feature.{FastVectorAssembler, StringIndexer}
import org.apache.spark.sql.DataFrame

/** Verifies the fast vector assembler, which only keeps categorical metadata and removes all other metadata.
  * TODO: Move this to core/spark and remove MML dependencies for the verification
  */
class VerifyFastVectorAssembler extends TestBase {

  val invalidExceptionError = "Could not catch correct exception"

  val inputCols = Array("a", "b", "c", "d", "e")
  val outputCol = "testCol"
  val mockDataset = session.createDataFrame(Seq(
    (0, 2, 0.5, 0.6, 0),
    (1, 3, 0.4, 0.5, 1),
    (2, 4, 0.78, 0.99, 2),
    (3, 5, 0.12, 0.34, 3)
  )).toDF(inputCols: _*)

  test("Verify fast vector assembler does not keep metadata for non-categorical columns") {
    val fastAssembler = new FastVectorAssembler().setInputCols(inputCols).setOutputCol(outputCol)
    val transformedDataset = fastAssembler.transform(mockDataset)
    // Assert metadata is empty
    assert(transformedDataset.schema(outputCol).metadata.toString() ==
      "{\"ml_attr\":{\"attrs\":{},\"num_attrs\":0}}")
  }

  test("Verify fast vector assembler throws when the first column is not categorical") {

    val (inputCols: Array[String], catColumn: String, categoricalData: DataFrame) = createCategoricalData

    val outputCol = "testCol"

    val fastAssembler = new FastVectorAssembler()
      .setInputCols((inputCols.toList.drop(1) ::: (List(catColumn))).toArray)
      .setOutputCol(outputCol)

    var caughtException: Boolean = false
    try {
      val transformedDataset = fastAssembler.transform(categoricalData)
    }
    catch {
      case exception: SparkException => {
        caughtException = true
        exception.getMessage.contains("Categorical columns must precede all others")
      }
      case _: Throwable => throw new Exception(invalidExceptionError)
    }

    if (!caughtException)
      throw new Exception(invalidExceptionError)
  }

  test("Verify fast vector assembler works when the first column is a categorical column") {

    val (inputCols: Array[String], catColumn: String, categoricalData: DataFrame) = createCategoricalData

    val outputCol = "testCol"

    val fastAssembler2 = new FastVectorAssembler()
      .setInputCols((catColumn :: inputCols.toList.drop(1)).toArray)
      .setOutputCol(outputCol)
    val transformedDataset2 = fastAssembler2.transform(categoricalData)

    // Assert metadata is not empty
    val mlattrData = transformedDataset2.schema(outputCol).metadata.getMetadata(SchemaConstants.MLlibTag)
    // assert the metadata is equal to: "{\"ml_attr\":{\"attrs\":{\"nominal\":[{\"vals\":[\"are\",\"how\",
    // \"hello\",\"you\"],\"idx\":0,\"name\":\"cat\"}]},\"num_attrs\":1}}"
    val attrsTag = "attrs"
    assert(mlattrData.contains(attrsTag))
    val attrsData = mlattrData.getMetadata(attrsTag)
    val nominalTag = "nominal"
    assert(attrsData.contains(nominalTag))
    val nominalData = attrsData.getMetadataArray(nominalTag)
    val valsTag = "vals"
    assert(nominalData(0).contains(valsTag))
    assert(nominalData(0).getStringArray(valsTag).contains("are"))
    assert(nominalData(0).getStringArray(valsTag).contains("how"))
    assert(nominalData(0).getStringArray(valsTag).contains("hello"))
    assert(nominalData(0).getStringArray(valsTag).contains("you"))

  }

  def createCategoricalData: (Array[String], String, DataFrame) = {
    val inputCols = Array("a", "b", "c", "d", "e")

    val dataset = session.createDataFrame(Seq(
      ("hello", 2, 0.5, 0.6, 0),
      ("how", 3, 0.4, 0.5, 1),
      ("are", 4, 0.78, 0.99, 2),
      ("you", 5, 0.12, 0.34, 3)
    )).toDF(inputCols: _*)

    val catColumn = "cat"
    val indexer = new StringIndexer().setInputCol("a").setOutputCol(catColumn).fit(dataset)
    val categoricalData = indexer.transform(dataset).toDF()
    (inputCols, catColumn, categoricalData)
  }

}
