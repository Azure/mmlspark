// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import java.io.File
import java.net.URL

import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.test.base.LinuxOnly
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._

import scala.util.Random

class CNTKModelSuite extends LinuxOnly with CNTKTestUtils with TransformerFuzzing[CNTKModel] {

  // TODO: Move away from getTempDirectoryPath and have TestBase provide one

  def testModel(minibatchSize: Int = 10): CNTKModel = {
    spark // make sure session is loaded
    new CNTKModel()
      .setModelLocation(modelPath)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setMiniBatchSize(minibatchSize)
      .setOutputNodeIndex(3)
  }

  lazy val doubleModelFile: File = {
    val f = FileUtilities.join(BuildInfo.datasetDir, "CNTKModel", "CNNDouble_untrained.model")
    if (!f.exists()) {
      FileUtils.copyURLToFile(new URL(
        "https://mmlspark.blob.core.windows.net/datasets/CNTKModels/CNNDouble_untrained.model"), f)
    }
    f
  }

  def testModelDouble(minibatchSize: Int = 10): CNTKModel = {
    spark // make sure session is loaded
    new CNTKModel()
      .setModelLocation(doubleModelFile.toString)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setMiniBatchSize(minibatchSize)
      .setOutputNodeIndex(0)
  }

  lazy val images: DataFrame = testImages(spark)

  import spark.implicits._

  private def checkParameters(minibatchSize: Int) = {
    val model = testModel(minibatchSize)
    val result = model.transform(images)
    compareToTestModel(result)
  }

  test("A CNTK model should be able to support setting the input and output node") {
    val model = testModel().setInputNodeIndex(0)
    val data = makeFakeData(spark, 30, featureVectorLength)
    val result = model.transform(data)
    assert(result.select(outputCol).count() == 30)
  }

  test("A CNTK model should support finding a node by name") {
    val model = new CNTKModel()
      .setModelLocation(modelPath)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setOutputNode("z")

    val data = makeFakeData(spark, 10, featureVectorLength).coalesce(1)
    val result = model.transform(data)
    assert(result.select(outputCol).collect()(0).getAs[DenseVector](0).size == 10)
    assert(result.select(outputCol).count() == 10)
  }

  test("throws useful exception when invalid node name is given") {
    val model = new CNTKModel()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setOutputNode("nonexistant-node")
      .setModelLocation(modelPath)

    val data = makeFakeData(spark, 3, featureVectorLength)
    intercept[IllegalArgumentException] {
      model.transform(data).collect()
    }
  }

  def testCNN(model: CNTKModel, doubleInput: Boolean, shape: Int = featureVectorLength): Unit = {
    val data = makeFakeData(spark, 3, shape, doubleInput)
    val result = model.transform(data)
    assert(result.select(outputCol).collect()(0).getAs[DenseVector](0).size == 10)
    assert(result.count() == 3)
  }

  test("getters") {
    val m = testModel()
    assert(m.getBatchInput === true)
    assert(m.getFeedDict === Map("ARGUMENT_0" -> inputCol))
    assert(m.getFetchDict === Map(outputCol -> "OUTPUT_3"))
    assert(m.getConvertOutputToDenseVector === true)
    assert(m.getInputNode === "ARGUMENT_0")
    assert(m.getInputNodeIndex === 0)
    assert(m.getInputShapes.map(_.toList) === List(List(32, 32, 3), List(10)))
    assert(m.getOutputNode === "OUTPUT_3")
    assert(m.getOutputNodeIndex === 3)
    assert(m.getShapeOutput === false)
    assert(m.getInputCol === inputCol)
    assert(m.getOutputCol === outputCol)
  }

  test("should work on floats") {
    testCNN(testModel(), false)
  }

  test("should work on doubles") {
    testCNN(testModel(), true)
  }

  test("double model float input") {
    testCNN(testModelDouble(), false, shape = 784)
  }

  test("double model double input") {
    testCNN(testModelDouble(), true, shape = 784)
  }

  test("A CNTK model should output Vectors and interop with other estimators") {
    val model = testModel()
    val data = makeFakeData(spark, 3, featureVectorLength, outputDouble = true)
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
    val images = spark.createDataFrame(
      sc.emptyRDD[Row], new StructType().add(inputCol, ArrayType(FloatType, false)))
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
    val result = modelLoaded.transform(images)
    compareToTestModel(result)
  }

  test("A CNTK Model should be pipeline compatible") {
    val model = testModel()
    val pipe = new Pipeline().setStages(Array(model)).fit(images)
    pipe.write.overwrite().save(saveFile)
    val pipeLoaded = PipelineModel.load(saveFile)
    val result = pipeLoaded.transform(images)
    compareToTestModel(result)
  }

  test("useful error message if invalid column name is given") {
    assertThrows[IllegalArgumentException] {
      testModel().setInputCol("images").transform(images)
    }
  }

  test("multi in multi out model") {
    val df = Seq.fill(10) {
      (Seq.fill(32 * 32 * 3) {
        Random.nextFloat()
      }, Seq.fill(10) {
        Random.nextFloat()
      })
    }.toDF("input0", "input1").coalesce(1)
    val model = new CNTKModel()
      .setModelLocation(modelPath)
      .setFeedDict(Map("ARGUMENT_0" -> "input0", "ARGUMENT_1" -> "input1"))
      .setFetchDict(Map("foo" -> "OUTPUT_3"))
    val results = model.transform(df)
    assert(results.schema.fieldNames.length == 3)
    assert(results.collect().length == 10)
  }

  val reader: MLReadable[_] = CNTKModel

  override def testObjects(): Seq[TestObject[CNTKModel]] =
    Seq(new TestObject[CNTKModel](testModel(), images))

}
