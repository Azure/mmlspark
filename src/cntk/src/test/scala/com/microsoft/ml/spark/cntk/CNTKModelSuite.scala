// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import com.microsoft.ml.spark.core.test.base.LinuxOnly
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.SparkException
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

class CNTKModelSuite extends LinuxOnly with CNTKTestUtils with TransformerFuzzing[CNTKModel]{

  // TODO: Move away from getTempDirectoryPath and have TestBase provide one

  def testModel(minibatchSize: Int = 10): CNTKModel = {
    new CNTKModel()
      .setModelLocation(session, modelPath)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setMiniBatchSize(minibatchSize)
      .setOutputNodeIndex(3)
  }
  val images = testImages(session)

  private def checkParameters(minibatchSize: Int) = {
    val model  = testModel(minibatchSize)
    val result = model.transform(images)
    compareToTestModel(result)
  }

  test("A CNTK model should be able to support setting the input and output node") {
    val model = testModel().setInputNode(0)
    val data = makeFakeData(session, 3, featureVectorLength)
    val result = model.transform(data)
    assert(result.select(outputCol).count() == 3)
  }

  test("A CNTK model should support finding a node by name") {
    val model = new CNTKModel()
      .setModelLocation(session, modelPath)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setOutputNodeName("z")

    val data   = makeFakeData(session, 3, featureVectorLength)
    val result = model.transform(data)
    assert(result.select(outputCol).collect()(0).getAs[DenseVector](0).size == 10)
    assert(result.select(outputCol).count() == 3)
  }

  test("throws useful exception when invalid node name is given") {
    val model = new CNTKModel()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setOutputNodeName("nonexistant-node")
      .setModelLocation(session, modelPath)

    val data = makeFakeData(session, 3, featureVectorLength)
    val se = intercept[SparkException] { model.transform(data).collect() }
    assert(se.getCause.isInstanceOf[IllegalArgumentException])
  }

  test("A CNTK model should work on doubles") {
    val model = testModel()
    val data   = makeFakeData(session, 3, featureVectorLength, outputDouble = true)
    val result = model.transform(data)
    assert(result.select(outputCol).collect()(0).getAs[DenseVector](0).size == 10)
    assert(result.count() == 3)
  }

  test("A CNTK model should output Vectors and interop with other estimators") {
    val model = testModel()
    val data   = makeFakeData(session, 3, featureVectorLength, outputDouble = true)
    val result = model.transform(data)
    assert(result.select(outputCol).schema.fields(0).dataType == VectorType)

    val predictions = new LogisticRegression()
      .setFeaturesCol(outputCol)
      .setLabelCol(labelCol)
      .fit(result)
      .transform(result)
    assert(predictions.select("prediction").collect().length == 3)
  }

  test("A CNTK model should have a default minibatch size") {
    val model = testModel()
    val result = model.transform(images)
    compareToTestModel(result)
  }

  test("A CNTK model should work on resized batches") {
    val model = testModel()
    val result = model.transform(images.repartition(1))
    compareToTestModel(result)
  }

  test("A CNTK model should work on an empty dataframe") {
    val images = session.createDataFrame(sc.emptyRDD[Row],
                                         StructType(
                                           StructField(inputCol, ArrayType(FloatType, false)) ::
                                             Nil))
    val model = testModel()
    val result = model.transform(images)
    assert(result.count == 0)
  }

  test("A CNTK Model should process images") {
    checkParameters(1)
    checkParameters(10)
    checkParameters(100)
  }

  test("A CNTK Model should be saveable") {
    val model = testModel()
    model.write.overwrite().save(saveFile)
    val modelLoaded = CNTKModel.load(saveFile)
    val result      = modelLoaded.transform(images)
    compareToTestModel(result)
  }

  test("A CNTK Model should be pipeline compatible") {
    val model  = testModel()
    val pipe   = new Pipeline().setStages(Array(model)).fit(images)
    pipe.write.overwrite().save(saveFile)
    val pipeLoaded = PipelineModel.load(saveFile)
    val result     = pipeLoaded.transform(images)
    compareToTestModel(result)
  }

  test("useful error message if invalid column name is given") {
    val model  = testModel().setInputCol("images")
    val pipe   = new Pipeline().setStages(Array(model)).fit(images)
    pipe.write.overwrite().save(saveFile)
    val pipeLoaded = PipelineModel.load(saveFile)
    assertThrows[IllegalArgumentException] {
      pipeLoaded.transform(images)
    }
  }

  val reader: MLReadable[_] = CNTKModel
  override def testObjects(): Seq[TestObject[CNTKModel]] =
    Seq(new TestObject[CNTKModel](testModel(), images))

}
