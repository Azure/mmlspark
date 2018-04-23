// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.opencv

import java.awt.GridLayout
import java.nio.file.Paths
import javax.swing._

import com.microsoft.ml.spark.Readers.implicits._
import com.microsoft.ml.spark.core.test.base.LinuxOnly
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

trait ImageTestUtils {
  lazy val groceriesDirectory = "/Images/Grocery/"
  lazy protected val fileLocation = s"${sys.env("DATASETS_HOME")}/$groceriesDirectory"

  protected def selectTestImageBytes(images: DataFrame): Array[Byte] = {
    images.filter(row => row.getString(4).endsWith("negative/5.jpg"))
      .head.getAs[Array[Byte]](3)
  }

  protected def selectImageCols(images: DataFrame): DataFrame = {
    images.select(images("out.height"),
      images("out.width"),
      images("out.type"),
      images("out.bytes"),
      images("out.path"))
  }

  protected def displayImages(images: DataFrame): Unit = {
    val (jframe, panel) = createScrollingFrame(images.count())
    images.collect().foreach(
      (row:Row) => {
        val img = new Mat(row.getInt(0), row.getInt(1), row.getInt(2))
        img.put(0,0,row.getAs[Array[Byte]](3))
        // Have to do the MatOfByte dance here
        val matOfByte = new MatOfByte()
        Imgcodecs.imencode(".jpg", img, matOfByte)
        val icon = new ImageIcon(matOfByte.toArray)
        val label: JLabel = new JLabel()
        label.setIcon(icon)
        panel.add(label)
        ()
      }
    )
    jframe.pack()
    jframe.setVisible(true)
    Thread.sleep(10000)
  }

  protected def createScrollingFrame(count: Long): (JFrame, JPanel) = {
    val jframe: JFrame = new JFrame("images")
    jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    val panel: JPanel = new JPanel()
    panel.setLayout(new GridLayout(count.toInt, 1))
    val scrPane: JScrollPane = new JScrollPane(panel)
    jframe.getContentPane.add(scrPane)
    (jframe, panel)
  }

  protected val firstBytes = Map(
    "00001.png" -> Array(235.0, 231.0, 232.0, 232.0, 232.0, 232.0, 232.0, 232.0, 232.0, 232.0),
    "00002.png" -> Array(222.0, 218.0, 194.0, 186.0, 222.0, 236.0, 238.0, 241.0, 243.0, 245.0),
    "00000.png" -> Array(49.0, 47.0, 51.0, 53.0, 46.0, 41.0, 47.0, 45.0, 44.0, 41.0),
    "00004.png" -> Array(50.0, 64.0, 46.0, 30.0, 22.0, 36.0, 55.0, 57.0, 59.0, 54.0),
    "00005.png" -> Array(83.0, 61.0, 26.0, 36.0, 65.0, 67.0, 58.0, 54.0, 63.0, 65.0),
    "00003.png" -> Array(149.0, 187.0, 193.0, 205.0, 202.0, 183.0, 181.0, 180.0, 182.0, 189.0)
  )

  protected def compareArrays(x: Array[Double], y:Array[Double]): Boolean = {
    val length = Math.min(x.length, y.length)
    for (i <- 0 until length) {
      if (Math.abs(x(i) - y(i)) > 1e-5) return false
    }
    true
  }
}

class UnrollImageSuite extends LinuxOnly
  with TransformerFuzzing[UnrollImage] with ImageTestUtils {

  lazy val filesRoot = s"${sys.env("DATASETS_HOME")}/"
  lazy val imagePath = s"$filesRoot/Images/CIFAR"
  lazy val images: DataFrame = session.readImages(imagePath, recursive = true)

  test("unroll") {
    assert(images.count() == 6)

    val unroll = new UnrollImage().setOutputCol("result")
    val unrolled = unroll.transform(images).select("image.path","result").collect

    unrolled.foreach(row => {
      val path = Paths.get(row.getString(0))
      val expected = firstBytes(path.getFileName.toString)
      val result = row(1).asInstanceOf[DenseVector].toArray

      val length =result.length
      if (length != 3072) throw new Exception(s"array length should be 3072, not $length ")

      if (!compareArrays(expected, result)) {
        println(path)
        println("result:   " + result.slice(0,10).deep.toString)
        println("expected: " + expected.deep.toString)
        throw new Exception("incorrect numeric value for flattened image")
      }
    })
  }

  override def testObjects(): Seq[TestObject[UnrollImage]] =
    Seq(new TestObject(new UnrollImage().setOutputCol("result"), images))

  override def reader: UnrollImage.type = UnrollImage
}

class ImageTransformerSuite extends LinuxOnly
  with TransformerFuzzing[ImageTransformer] with ImageTestUtils{

  test("general workflow") {
    val images = session.readImages(fileLocation, recursive = true)
    assert(images.count() == 30)

    val size = (224,200)
    val tr = new ImageTransformer()
      .setOutputCol("out")
      .resize(height = size._1, width = size._2)
      .crop(x = 0, y = 0, height = 22, width = 26)
      .resize(height = 15, width = 10)

    val preprocessed = tr.transform(images)

    val out_sizes = preprocessed.select(preprocessed("out.height"), preprocessed("out.width")).collect

    out_sizes.foreach(
      (row:Row) => {
        assert(row.getInt(0) == 15 && row.getInt(1) == 10, "output images have incorrect size")
      }
    )

    val unroll = new UnrollImage()
      .setInputCol(tr.getOutputCol)
      .setOutputCol("final")

    val result = unroll.transform(preprocessed).select("final")
    result.collect().foreach(
      row => assert(row(0).asInstanceOf[DenseVector].toArray.length == 10*15*3, "unrolled image is incorrect"))

  }

  test("to parquet") {

    val filename = "test_images_parquet"
    try {
      val images = session.readImages(fileLocation, recursive = true)
      images.write.mode(SaveMode.Overwrite).parquet(filename)

      val images1 = session.sqlContext.read.parquet(filename)
      assert(images1.count() == images.count())
    } finally {
      FileUtils.forceDelete(new java.io.File(filename))
    }
  }

  test("binary file input") {

    val images = session.readBinaryFiles(fileLocation, recursive = true)
    assert(images.count() == 31)

    val tr = new ImageTransformer()
      .setInputCol("value")
      .setOutputCol("out")
      .resize(height = 15, width = 10)

    val preprocessed = tr.transform(images).na.drop
    assert(preprocessed.count() == 30)

    val out_sizes = preprocessed.select(preprocessed("out.height"), preprocessed("out.width")).collect

    out_sizes.foreach(
      (row:Row) => {
        assert(row.getInt(0) == 15 && row.getInt(1) == 10, "output images have incorrect size")
      }
    )
  }

  test("crop") {

    val images = session.readImages(fileLocation, recursive = true)

    val tr = new ImageTransformer()
      .setOutputCol("out")
      .resize(height = 100, width = 200)
      .crop(x = 0, y = 0, height = 22, width = 26)

    val preprocessed = tr.transform(images)

    val out_sizes = preprocessed.select(preprocessed("out.height"), preprocessed("out.width")).collect

    out_sizes.foreach(
      (row:Row) => {
        assert(row.getInt(0) == 22 && row.getInt(1) == 26, "output images have incorrect size")
      }
    )
  }

  test("color format") {

    val images = session.readImages(fileLocation, recursive = true)

    val tr = new ImageTransformer()
      .setOutputCol("out")
      .colorFormat(Imgproc.COLOR_BGR2GRAY)

    val preprocessed = tr.transform(images)

    val grayImages = selectImageCols(preprocessed)

    // For visual debugging uncomment:
    // displayImages(grayImages)
    val bytes = Array(10, 1, 3, 9, 6, 16, 11, 7, 8, 6, 26, 40, 57, 50)
    // Validate first image first few bytes have been transformed correctly
    val firstImageBytes = selectTestImageBytes(grayImages)
    for (i <- 0 until bytes.length) {
      assert(firstImageBytes(i) == bytes(i))
    }
  }

  test("verify blur") {

    val images = session.readImages(fileLocation, recursive = true)

    val tr = new ImageTransformer()
      .setOutputCol("out")
      .blur(100, 100)

    val preprocessed = tr.transform(images)

    val blurImages = selectImageCols(preprocessed)

    // For visual debugging uncomment:
    // displayImages(grayImages)
    val bytes = Array(15, 28, 26, 15, 28, 26, 15, 28, 26, 15, 28, 26, 15, 28, 26, 15)
    // Validate first image first few bytes have been transformed correctly
    val firstImageBytes = selectTestImageBytes(blurImages)
    for (i <- 0 until bytes.length) {
      assert(firstImageBytes(i) == bytes(i))
    }
  }

  test("verify thresholding") {

    val images = session.readImages(fileLocation, recursive = true)

    val tr = new ImageTransformer()
      .setOutputCol("out")
      .threshold(100, 100, Imgproc.THRESH_BINARY)

    val preprocessed = tr.transform(images)

    val thresholdedImages = selectImageCols(preprocessed)

    // For visual debugging uncomment:
    // displayImages(thresholdedImages)
    // Validate first image first few bytes have been transformed correctly
    thresholdedImages.foreach(
      (row:Row) => {
        if (!row.getAs[Array[Byte]](3).forall(b => b == 100 || b == 0)) {
          throw new Exception("threshold did not result in binary values")
        }
      }
    )
  }

  test("verify application of gaussian kernel (has blur effect)") {

    val images = session.readImages(fileLocation, recursive = true)

    val tr = new ImageTransformer()
      .setOutputCol("out")
      .gaussianKernel(20, 10)

    val preprocessed = tr.transform(images)

    val gaussianImages = selectImageCols(preprocessed)

    // For visual debugging uncomment:
    // displayImages(gaussianImages)
    val firstImageBytes = selectTestImageBytes(gaussianImages)
    // Validate first image first few bytes have been transformed correctly
    val bytes = Array(8, 14, 14, 4, 8, 7, 4, 5, 5, 4, 5, 6, 5, 9, 8, 3, 8, 7, 7, 13, 12, 8, 12)
    // Validate first image first few bytes have been transformed correctly
    for (i <- 0 until bytes.length) {
      assert(firstImageBytes(i) == bytes(i))
    }
  }

  override def testObjects(): Seq[TestObject[ImageTransformer]] =
    Seq(new TestObject[ImageTransformer](new ImageTransformer()
    .setOutputCol("out")
    .gaussianKernel(20, 10),
    session.readImages(fileLocation, recursive = true)))

  override def reader: ImageTransformer.type = ImageTransformer
}
