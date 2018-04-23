// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import java.net.URI

import com.microsoft.ml.spark.Readers.implicits._
import com.microsoft.ml.spark.core.env.FileUtilities.File
import com.microsoft.ml.spark.core.schema.ImageSchema
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import com.microsoft.ml.spark.downloader.{ModelDownloader, ModelSchema}
import com.microsoft.ml.spark.IO.binary.FileReaderUtils
import org.apache.spark.image.ImageFileFormat
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class ImageFeaturizerSuite extends CNTKTestUtils with FileReaderUtils
  with TransformerFuzzing[ImageFeaturizer]{

  val images: DataFrame = session.readImages(imagePath, true).withColumnRenamed("image", inputCol)

  val modelDir = new File(filesRoot, "CNTKModel")
  val modelDownloader = new ModelDownloader(session, modelDir.toURI)

  lazy val resNetUri: URI = new File(modelDir, "ResNet50_ImageNet.model").toURI
  lazy val resNet: ModelSchema = modelDownloader.downloadByName("ResNet50")

  test("Image featurizer should reproduce the CIFAR10 experiment") {
    val model = new ImageFeaturizer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setModelLocation(session, s"${sys.env("DATASETS_HOME")}/CNTKModel/ConvNet_CIFAR10.model")
      .setCutOutputLayers(0)
      .setLayerNames(Array("z"))
    val result = model.transform(images)
    compareToTestModel(result)
  }

  test("structured streaming") {

    val model = new ImageFeaturizer()
      .setInputCol("image")
      .setOutputCol(outputCol)
      .setModelLocation(session, s"${sys.env("DATASETS_HOME")}/CNTKModel/ConvNet_CIFAR10.model")
      .setCutOutputLayers(0)
      .setLayerNames(Array("z"))

    val imageDF = session
      .readStream
      .format(classOf[ImageFileFormat].getName)
      .schema(ImageSchema.schema)
      .load(cifarDirectory)

    val resultDF = model.transform(imageDF)

    val q1 = resultDF.writeStream
      .format("memory")
      .queryName("images")
      .start()

    tryWithRetries(){ () =>
      assert(session.sql("select * from images").count() == 6)
    }
  }

  def resNetModel(): ImageFeaturizer = new ImageFeaturizer()
    .setInputCol(inputCol)
    .setOutputCol(outputCol)
    .setModel(session, resNet)

  test("the Image feature should work with the modelSchema") {
    val result = resNetModel().setCutOutputLayers(0).transform(images)
    compareToTestModel(result)
  }

  test("the Image feature should work with the modelSchema + new images") {
    val newImages = session.read
      .format(classOf[ImageFileFormat].getName)
      .load(cifarDirectory)
      .withColumnRenamed("image","cntk_images")

    val result = resNetModel().setCutOutputLayers(0).transform(newImages)
    compareToTestModel(result)
  }

  test("Image featurizer should work with ResNet50", TestBase.Extended) {
    val result = resNetModel().transform(images)
    val resVec = result.select(outputCol).collect()(0).getAs[DenseVector](0)
    assert(resVec.size == 1000)
  }

  test("test layers of network", TestBase.Extended) {
    (0 to 9).foreach({ i =>
      val model = new ImageFeaturizer()
        .setModel(session, resNet)
        .setInputCol(inputCol)
        .setOutputCol(outputCol)
        .setCutOutputLayers(i)
      val result = model.transform(images)
    })
  }

  val reader: MLReadable[_] = ImageFeaturizer
  override def testObjects(): Seq[TestObject[ImageFeaturizer]] = Seq(
    new TestObject(resNetModel(), images)
  )
}
