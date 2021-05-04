// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.split1

import java.io.{File, FileInputStream}

import com.microsoft.ml.spark.cognitive.OsUtils
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.schema.ImageSchemaUtils
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.io.image.ImageUtils
import org.apache.commons.codec.binary.Base64
import org.apache.commons.io.IOUtils
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.source.image.PatchedImageFileFormat
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, to_json, udf}
import org.apache.spark.sql.types.StringType

class ImageReaderSuite extends TestBase with FileReaderUtils {

  val imageFormat: String = classOf[PatchedImageFileFormat].getName

  test("image dataframe") {
    val images = spark.read.format(imageFormat)
      .option("dropInvalid", true)
      .load(FileUtilities.join(groceriesDirectory, "**").toString)
    println(time {
      images.count
    })

    // make sure the column "images" exists and has the right type
    assert(ImageSchemaUtils.isImage(images.schema("image").dataType))

    //make sure that SQL has access to the sub-fields
    val paths = images.select("image.origin")

    assert(paths.count == 30)
    //more complicated SQL statement
    val areas = images.select(images("image.width") * images("image.height"))
    println(s"   area of image 1 ${areas.take(1)(0)}")
  }

  test("read images with subsample") {
    val imageDF = spark
      .read
      .format(imageFormat)
      .load(cifarDirectory)
      .sample(.5,0)
    assert(Set(2,3,4)(imageDF.count().toInt))
  }

  object UDFs extends Serializable {
    val CifarDirectoryVal: String = cifarDirectory
    val Rename: UserDefinedFunction = UDFUtils.oldUdf({ x: String => x.split("/").last }, StringType)
  }

  def recursiveListFiles(f: File): Array[File] = {
    val these = f.listFiles
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
  }

  test("read images from files") {
    val prefix = if (OsUtils.IsWindows) "" else "file://"
    val files = recursiveListFiles(new File(cifarDirectory))
      .toSeq.map(f => Tuple1(prefix + f.toString))
    val df = spark
      .createDataFrame(files)
      .toDF("filenames")
    val imageDF = ImageUtils.readFromPaths(df, "filenames")
    assert(imageDF.select("image.data").collect().map(_.getAs[Array[Byte]](0)).length == 6)
  }

  test("read images from strings") {
    val base64Strings = recursiveListFiles(new File(cifarDirectory)).toSeq.map(f =>
      (f.getAbsolutePath, new Base64().encodeAsString(
        IOUtils.toByteArray(new FileInputStream(f))))
    )
    val df = spark
      .createDataFrame(base64Strings)
      .toDF("filenames","bytes")
    val imageDF = ImageUtils.readFromStrings(df, "bytes")
    assert(imageDF.select("image.data").collect().map(_.getAs[Array[Byte]](0)).length == 6)
  }

  test("read from strings 2") {
    import spark.implicits._

    val df = sc.parallelize(Seq(Tuple1("" +
      "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")))
      .toDF("bytes")
    assert(ImageUtils.readFromStrings(df,"bytes")
      .withColumn("parsed",to_json(col("image")))
      .select("parsed")
      .collect()(0).getString(0).length() == 4154)
  }

  test("write images with subsample function 2") {
    val imageDF = spark
      .read
      .format(imageFormat)
      .load(cifarDirectory)
      .sample(.5,0)
      .withColumn("filenames", UDFs.Rename(col("image.origin")))

    imageDF.printSchema()
    assert(Set(2,3,4)(imageDF.count().toInt))
    val saveDir = new File(tmpDir.toFile, "images").toString
    imageDF.write.mode("overwrite")
      .format(classOf[PatchedImageFileFormat].getName)
      .option("pathCol", "filenames")
      .save(saveDir)

    val newImageDf = spark
      .read
      .format(imageFormat)
      .load(saveDir)
    assert(Set(2,3,4)(newImageDf.count().toInt))
    assert(ImageSchemaUtils.isImage(newImageDf.schema("image").dataType))
  }

  test("structured streaming with images") {
    val schema = ImageSchema.imageSchema
    val imageDF = spark
      .readStream
      .format(imageFormat)
      .schema(schema)
      .load(cifarDirectory)

    val q1 = imageDF.select("image.origin").writeStream
      .format("memory")
      .queryName("images")
      .start()

    tryWithRetries() { () =>
      val df = spark.sql("select * from images")
      assert(df.count() == 6)
      println("success")
    }
    q1.stop()
  }

}
