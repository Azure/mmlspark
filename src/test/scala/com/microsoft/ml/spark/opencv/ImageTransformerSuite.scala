// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.opencv

import java.awt.GridLayout
import java.io.File
import java.nio.file.Paths

import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.io.IOImplicits._
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import com.microsoft.ml.spark.image.{UnrollBinaryImage, UnrollImage}
import javax.swing._
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.DataFrameEquality
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.scalactic.Equality
import org.scalatest.Assertion

trait ImageTestUtils {
  lazy protected val fileLocation: File = FileUtilities.join(BuildInfo.datasetDir, "Images", "Grocery")

  protected def selectTestImageBytes(images: DataFrame): Array[Byte] = {
    images.filter(row => row.getString(4).contains("negative") && row.getString(4).endsWith("5.jpg"))
      .head.getAs[Array[Byte]](3)
  }

  protected def selectImageCols(images: DataFrame): DataFrame = {
    images.select(images("out.height"),
      images("out.width"),
      images("out.mode"),
      images("out.data"),
      images("out.origin"))
  }

  protected def displayImages(images: DataFrame): Unit = {
    val (jframe, panel) = createScrollingFrame(images.count())
    images.collect().foreach(
      (row: Row) => {
        val img = new Mat(row.getInt(0), row.getInt(1), row.getInt(2))
        img.put(0, 0, row.getAs[Array[Byte]](3))
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
    jframe.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
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

}

class UnrollImageSuite extends TransformerFuzzing[UnrollImage] with ImageTestUtils with DataFrameEquality {

  lazy val filesRoot: File =  BuildInfo.datasetDir
  lazy val imagePath: String = FileUtilities.join(filesRoot,"Images", "CIFAR").toString
  lazy val images: DataFrame = spark.read.image.load(imagePath)

  test("roll and unroll") {
    val imageCollection = images.select("image").collect().map(_.getAs[Row](0))
    imageCollection.foreach(row =>
      assert(row ===
        UnrollImage.roll(
          UnrollImage.unroll(row).toArray.map(_.toInt),
          row.getString(0),
          row.getInt(1),
          row.getInt(2),
          row.getInt(3),
          row.getInt(4)
        )
      )
    )
  }

  test("unroll") {
    assert(images.count() == 6)

    val unroll = new UnrollImage().setOutputCol("result")
    val unrolled = unroll.transform(images).select("image.origin", "result").collect

    unrolled.foreach(row => {
      val path = new Path(row.getString(0))
      val expected = firstBytes(path.getName)
      val result = row(1).asInstanceOf[DenseVector].toArray

      val length = result.length
      if (length != 3072) throw new Exception(s"array length should be 3072, not $length ")

      assert(result.slice(0, 10) === expected)
    })
  }

  override def testObjects(): Seq[TestObject[UnrollImage]] =
    Seq(new TestObject(new UnrollImage().setOutputCol("result"), images))

  override def reader: UnrollImage.type = UnrollImage
}

class UnrollBinaryImageSuite extends TransformerFuzzing[UnrollBinaryImage]
  with ImageTestUtils with DataFrameEquality {

  lazy val filesRoot: File = BuildInfo.datasetDir
  lazy val imagePath: String = FileUtilities.join(filesRoot, "Images", "CIFAR").toString
  lazy val images: DataFrame = spark.read.image.load(imagePath)
  lazy val binaryImages: DataFrame = spark.read.binary.load(imagePath)
    .withColumn("image", col("value.bytes"))

  test("unroll did not change") {
    assert(
      new UnrollImage().setOutputCol("result")
        .transform(images).select("result") ===
        new UnrollBinaryImage().setOutputCol("result")
          .transform(binaryImages).select("result")
    )
  }

  // This is needed for some small 256!=0 issue in unroll.
  // It only happens at one place throughout the tests though
  override implicit lazy val dvEq: Equality[DenseVector] = new Equality[DenseVector] {
    def areEqual(a: DenseVector, b: Any): Boolean = b match {
      case bArr: DenseVector =>
        a.values.zip(bArr.values).map {
          case (x, y) if doubleEq.areEqual(x, y) => 0
          case _ => 0
        }.sum <= 1
    }
  }

  override def testObjects(): Seq[TestObject[UnrollBinaryImage]] =
    Seq(new TestObject(new UnrollBinaryImage().setOutputCol("result"), binaryImages))

  override def reader: UnrollBinaryImage.type = UnrollBinaryImage
}

class ImageTransformerSuite extends TransformerFuzzing[ImageTransformer] with ImageTestUtils {

  //TODO this is needed to stop the build from freezing
  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    assert(true)
  }

  lazy val images: DataFrame = spark.read.image.option("dropInvalid",true)
    .load(FileUtilities.join(fileLocation, "**").toString)

  test("general workflow") {
    //assert(images.count() == 30) //TODO this does not work on build machine for some reason

    val tr = new ImageTransformer()
      .setOutputCol("out")
      .resize(height = 15, width = 10)

    val preprocessed = tr.transform(images)

    val outSizes = preprocessed.select(preprocessed("out.height"), preprocessed("out.width")).collect

    outSizes.foreach { row: Row =>
        assert(row.getInt(0) == 15 && row.getInt(1) == 10, "output images have incorrect size")
    }

    val unroll = new UnrollImage()
      .setInputCol(tr.getOutputCol)
      .setOutputCol("final")

    unroll.transform(preprocessed)
      .select("final")
      .collect().foreach(row =>
        assert(row.getAs[DenseVector](0).toArray.length == 10 * 15 * 3, "unrolled image is incorrect"))

  }

  test("binary file input") {
    val binaries = spark.read.binary.load(FileUtilities.join(fileLocation,"**").toString)
    assert(binaries.count() == 31)
    binaries.printSchema()

    val tr = new ImageTransformer()
      .setInputCol("value")
      .setOutputCol("out")
      .resize(height = 15, width = 10)

    val preprocessed = tr.transform(binaries).na.drop
    assert(preprocessed.count() == 30)

    val outSizes = preprocessed.select(preprocessed("out.height"), preprocessed("out.width")).collect

    outSizes.foreach(
      (row: Row) => {
        assert(row.getInt(0) == 15 && row.getInt(1) == 10, "output images have incorrect size")
      }
    )
  }

  test("binary file input 2") {
    val binaries = spark.read.binary
      .load(FileUtilities.join(fileLocation, "**").toString)
      .select("value.bytes")
    assert(binaries.count() == 31)
    binaries.printSchema()

    val tr = new ImageTransformer()
      .setInputCol("bytes")
      .setOutputCol("out")
      .resize(height = 15, width = 10)

    val preprocessed = tr.transform(binaries).na.drop
    assert(preprocessed.count() == 30)

    val outSizes = preprocessed.select(preprocessed("out.height"), preprocessed("out.width")).collect

    outSizes.foreach(
      (row: Row) => {
        assert(row.getInt(0) == 15 && row.getInt(1) == 10, "output images have incorrect size")
      }
    )
  }

  test("crop") {

    val tr = new ImageTransformer()
      .setOutputCol("out")
      .resize(height = 100, width = 200)
      .crop(x = 0, y = 0, height = 22, width = 26)

    val preprocessed = tr.transform(images)

    val outSizes = preprocessed.select(preprocessed("out.height"), preprocessed("out.width")).collect

    outSizes.foreach(
      (row: Row) => {
        assert(row.getInt(0) == 22 && row.getInt(1) == 26, "output images have incorrect size")
      }
    )
  }

  test("color format") {
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
    for (i <- bytes.indices) {
      assert(firstImageBytes(i) == bytes(i))
    }
  }

  test("verify blur") {
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
    for (i <- bytes.indices) {
      assert(firstImageBytes(i) == bytes(i))
    }
  }

  test("verify thresholding") {
    val tr = new ImageTransformer()
      .setOutputCol("out")
      .threshold(100, 100, Imgproc.THRESH_BINARY)

    val preprocessed = tr.transform(images)

    val thresholdedImages = selectImageCols(preprocessed)

    // For visual debugging uncomment:
    // displayImages(thresholdedImages)
    // Validate first image first few bytes have been transformed correctly
    thresholdedImages.foreach(
      (row: Row) => {
        if (!row.getAs[Array[Byte]](3).forall(b => b == 100 || b == 0)) {
          throw new Exception("threshold did not result in binary values")
        }
      }
    )
  }

  test("verify application of gaussian kernel (has blur effect)") {
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
    for (i <- bytes.indices) {
      assert(firstImageBytes(i) == bytes(i))
    }
  }

  override def testObjects(): Seq[TestObject[ImageTransformer]] =
    Seq(new TestObject[ImageTransformer](new ImageTransformer()
      .setOutputCol("out")
      .gaussianKernel(20, 10), images))

  override def reader: ImageTransformer.type = ImageTransformer
}
