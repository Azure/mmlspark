// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.split1

import java.io.{File, FileOutputStream}
import java.net.URI

import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.env.FileUtilities.zipFolder
import com.microsoft.ml.spark.core.schema.BinaryFileSchema
import com.microsoft.ml.spark.core.schema.BinaryFileSchema.isBinaryFile
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.io.binary.Binary.implicits.implicitSession
import com.microsoft.ml.spark.io.binary.{BinaryFileFormat, BinaryFileReader}
import org.apache.commons.io.{FileUtils, IOUtils}
import org.apache.hadoop.fs.Path
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.param.DataFrameEquality
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType

trait FileReaderUtils {
  val imagesDirectory: File = FileUtilities.join(BuildInfo.datasetDir, "Images")
  val groceriesDirectory: String = FileUtilities.join(imagesDirectory, "Grocery").toString
  val cifarDirectory: String = FileUtilities.join(imagesDirectory, "CIFAR").toString
}

class BinaryFileReaderSuite extends TestBase with FileReaderUtils with DataFrameEquality {

  def createZip(directory: String, saveDir: File): Unit ={
    val dir = new File(directory)
    val zipfile = new File(saveDir, dir.getName + ".zip")
    if (!zipfile.exists()) zipFolder(dir, zipfile)
  }

  object UDFs extends Serializable {
    val CifarDirectoryVal: String = cifarDirectory
    val Rename: UserDefinedFunction = UDFUtils.oldUdf({ x: String => new Path(x).getName}, StringType)
  }

  test("binary dataframe") {
    val data = spark.readBinaryFiles(groceriesDirectory, recursive = true)
    println(time { data.count })
    assert(isBinaryFile(data.schema("value")))
    val paths = data.select("value.path") //make sure that SQL has access to the sub-fields
    assert(paths.count == 31)             //note that text file is also included
  }

  test("reads the right values"){
    val data = spark.readBinaryFiles(groceriesDirectory, recursive = true)
      .limit(1).select("value.*")
    val data2 = BinaryFileReader.readFromPaths(data.select(
      "path"), "path", "bytes", 2, 600000)

    val path = data.collect().head.getString(0)
    val bytes1 = data.collect().head.getAs[Array[Byte]](1)
    val bytes2 = data2.collect().head.getAs[Array[Byte]](1)
    val trueBytes = IOUtils.toByteArray(new URI(path))
    assert(bytes1 === trueBytes)
    assert(bytes2 === trueBytes)
  }

  test("read from paths yields same values") {
    val data = spark.readBinaryFiles(groceriesDirectory, recursive = true).limit(1)
    val df2 = BinaryFileReader.readFromPaths(data.select(
      "value.path"), "path", "bytes", 2, 600000)
    assert(df2 === data.select("value.*"))
  }

  test("sample ratio test") {
    val all = spark.readBinaryFiles(groceriesDirectory, recursive = true, sampleRatio = 1.0)
    val sampled = spark.readBinaryFiles(groceriesDirectory, recursive = true, sampleRatio = 0.5)
    val count = sampled.count
    assert(count > 0 && count < all.count, "incorrect sampling behavior")
  }

  test("with zip file") {
    /* remove when datasets/Images is updated */
    val saveDir = new File(tmpDir.toFile, "zips")
    if (!saveDir.exists()) saveDir.mkdirs()

    createZip(cifarDirectory, saveDir)
    createZip(groceriesDirectory, saveDir)

    val images = spark.readBinaryFiles(saveDir.getAbsolutePath, recursive = true)
    assert(images.count == 37)

    val images1 = spark.readBinaryFiles(saveDir.getAbsolutePath, recursive = true, inspectZip = false)
    assert(images1.count == 2)
  }

  test("handle folders with spaces") {
    /* remove when datasets/Images is updated */
    val newDirTop = new File(tmpDir.toFile, "foo bar")
    val newDirMid = new File(newDirTop, "fooey barey")
    FileUtils.forceMkdir(newDirMid)
    try {
      val fos = new FileOutputStream(new File(newDirMid, "foo.txt"))
      try {
        fos.write((1 to 10).map(_.toByte).toArray)
      } finally {
        fos.close()
      }
      val files = spark.readBinaryFiles(newDirTop.getAbsolutePath, recursive = true)
      assert(files.count == 1)
    } finally {
      FileUtils.forceDelete(newDirTop)
    }
  }

  test("binary files should allow recursion") {
    val df = spark
      .read
      .format(classOf[BinaryFileFormat].getName)
      .load(FileUtilities.join(groceriesDirectory,"**").toString)
    assert(df.count()==31)
    df.printSchema()
  }

  test("static load with new reader") {
    val df = spark
      .read
      .format(classOf[BinaryFileFormat].getName)
      .option("subsample", .5)
      .load(cifarDirectory)
    df.show()
    assert(df.count() > 0 && df.count() <= 6)
  }

  test("structured streaming with binary files") {
    val imageDF = spark
      .readStream
      .format(classOf[BinaryFileFormat].getName)
      .schema(BinaryFileSchema.Schema)
      .load(cifarDirectory)

    val q1 = imageDF.writeStream
      .format("memory")
      .queryName("images")
      .start()

    tryWithRetries(){ () =>
      val df = spark.sql("select * from images")
      assert(df.count() == 6)
    }
    q1.stop()
  }

  test("write binary files") {
    val imageDF = spark
      .read
      .format(classOf[BinaryFileFormat].getName)
      .load(cifarDirectory)
      .sample(.5,0)
      .select("value.*")
      .withColumn("path", UDFs.Rename(col("path")))
    imageDF.printSchema()
    val count = imageDF.count()
    assert(count > 0 && count <= 6)
    val saveDir = new File(tmpDir.toFile, "binaries").toString
    imageDF.write.mode("overwrite")
      .format(classOf[BinaryFileFormat].getName)
      .option("pathCol", "path")
      .save(saveDir)

    val newImageDf = spark
      .read
      .format(classOf[BinaryFileFormat].getName)
      .load(saveDir)
    assert(newImageDf.count() == count)
  }

}
