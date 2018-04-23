// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.image

import java.io.ByteArrayInputStream

import com.microsoft.ml.spark.NativeLoader
import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.core.schema.ImageSchema
import com.microsoft.ml.spark.IO.binary.BinaryFileReader
import org.apache.commons.codec.binary.Base64
import org.apache.commons.io.IOUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.image.{ConfUtils, ImageFileFormat}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.opencv.core.{CvException, Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs

object ImageReader {

  /** This object will load the openCV binaries when the object is referenced
    * for the first time, subsequent references will not re-load the binaries.
    * In spark, this loads one copy for each running jvm, instead of once per partition.
    * This technique is similar to that used by the cntk_jni jar,
    * but in the case where microsoft cannot edit the jar
    */
  object OpenCVLoader {

    import org.opencv.core.Core

    new NativeLoader("/nu/pattern/opencv").loadLibraryByName(Core.NATIVE_LIBRARY_NAME)
  }

  private[spark] def loadOpenCVFunc[A](it: Iterator[A]) = {
    OpenCVLoader
    it
  }

  private[spark] def loadOpenCV(df: DataFrame): DataFrame = {
    val encoder = RowEncoder(df.schema)
    df.mapPartitions(loadOpenCVFunc)(encoder)
  }

  def decode(filename: String, bytes: Array[Byte]): Option[Row] = {
    decode(Some(filename), bytes)
  }

    /** Convert the image from compressd (jpeg, etc.) into OpenCV representation and store it in Row
    * See ImageSchema for details.
    *
    * @param filename arbitrary string
    * @param bytes    image bytes (for example, jpeg)
    * @return returns None if decompression fails
    */
  def decode(filename: Option[String], bytes: Array[Byte]): Option[Row] = {
    val mat = new MatOfByte(bytes: _*)
    val decodedOpt = try {
      Some(Imgcodecs.imdecode(mat, Imgcodecs.CV_LOAD_IMAGE_COLOR))
    } catch {
      case _: CvException => None
    }

    decodedOpt match {
      case Some(decoded) if !decoded.empty() =>
        val ocvBytes = new Array[Byte](decoded.total.toInt * decoded.elemSize.toInt)
        // extract OpenCV bytes
        decoded.get(0, 0, ocvBytes)
        // type: CvType.CV_8U
        Some(Row(filename.orNull, decoded.height, decoded.width, decoded.`type`, ocvBytes))
      case _ => None
    }
  }

  /** Read the directory of images from the local or remote source
    *
    * @param path      Path to the image directory
    * @param recursive Recursive search flag
    * @return DataFrame with a single column of "images", see "columnSchema" for details
    */
  def read(path: String, recursive: Boolean, spark: SparkSession,
           sampleRatio: Double = 1, inspectZip: Boolean = true, seed: Long = 0L): DataFrame = {
    val p = new Path(path)
    val globs = if (recursive) {
      BinaryFileReader.recursePath(p.getFileSystem(spark.sparkContext.hadoopConfiguration), p, { fs => fs.isDirectory })
        .map(g => g) ++ Array(p)
    } else {
      Array(p)
    }
    spark.read.format(classOf[ImageFileFormat].getName)
      .option("subsample", sampleRatio)
      .option("seed", seed)
      .option("inspectZip", inspectZip).load(globs.map(_.toString): _*)
  }

  /** Read the directory of image files from the local or remote source
    *
    * @param path Path to the directory
    * @return DataFrame with a single column of "imageFiles", see "columnSchema" for details
    */
  def stream(path: String, spark: SparkSession,
             sampleRatio: Double = 1, inspectZip: Boolean = true, seed: Long = 0L): DataFrame = {
    val p = new Path(path)
    spark.readStream.format(classOf[ImageFileFormat].getName)
      .option("subsample", sampleRatio)
      .option("seed", seed)
      .option("inspectZip", inspectZip).schema(ImageSchema.schema).load(p.toString)
  }

  def readFromPaths(df: DataFrame, pathCol: String, imageCol: String = "image"): DataFrame = {
    val outputSchema = df.schema.add(imageCol, ImageSchema.columnSchema)
    val encoder = RowEncoder(outputSchema)
    val hconf = ConfUtils.getHConf(df)
    df.mapPartitions { rows =>
      ImageReader.OpenCVLoader
      rows.map { row =>
        val path = new Path(row.getAs[String](pathCol))
        val fs = path.getFileSystem(hconf.value)
        val bytes = StreamUtilities.using(fs.open(path)) {is => IOUtils.toByteArray(is)}.get
        val imageRow = ImageReader.decode(path.toString, bytes).getOrElse(Row(None))
        val ret = Row.merge(Seq(row, Row(imageRow)): _*)
        ret
      }
    }(encoder)
  }

  def readFromBytes(df: DataFrame, pathCol: String, bytesCol:String, imageCol: String = "image"): DataFrame = {
    val outputSchema = df.schema.add(imageCol, ImageSchema.columnSchema)
    val encoder = RowEncoder(outputSchema)
    df.mapPartitions { rows =>
      ImageReader.OpenCVLoader
      rows.map { row =>
        val path = row.getAs[String](pathCol)
        val bytes = row.getAs[Array[Byte]](bytesCol)
        val imageRow = ImageReader.decode(path, bytes).getOrElse(Row(None))
        val ret = Row.merge(Seq(row, Row(imageRow)): _*)
        ret
      }
    }(encoder)
  }

  def readFromStrings(df: DataFrame, bytesCol:String,
                      imageCol: String = "image", dropPrefix: Boolean = false): DataFrame = {
    val outputSchema = df.schema.add(imageCol, ImageSchema.columnSchema)
    val encoder = RowEncoder(outputSchema)
    df.mapPartitions { rows =>
      ImageReader.OpenCVLoader
      rows.map { row =>
        val path = None
        val encoded = row.getAs[String](bytesCol)
        val bytes = new Base64().decode(
          if (dropPrefix) encoded.split(",")(1) else encoded
        )
        val imageRow = ImageReader.decode(path, bytes).getOrElse(Row(None))
        val ret = Row.merge(Seq(row, Row(imageRow)): _*)
        ret
      }
    }(encoder)
  }

}

object ImageWriter {

  def encode(row: InternalRow, extension: String): Array[Byte] = {
    val mat = new Mat(row.getInt(1),row.getInt(2),row.getInt(3))
    mat.put(0, 0, row.getBinary(4))
    val out = new MatOfByte()
    Imgcodecs.imencode(extension, mat, out)
    out.toArray
  }

  def encode(row: Row, extension: String): Array[Byte] = {
    val mat = new Mat(row.getInt(1),row.getInt(2),row.getInt(3))
    mat.put(0, 0, row.getAs[Array[Byte]](4))
    val out = new MatOfByte()
    Imgcodecs.imencode(extension, mat, out)
    out.toArray
  }

  def write(df: DataFrame,
            basePath: String,
            pathCol:String = "filenames",
            imageCol:String = "image",
            encoding: String=".png"): Unit = {

    val hconf = ConfUtils.getHConf(df)
    df.select(imageCol, pathCol).foreachPartition { rows =>
      val fs = FileSystem.get(new Path(basePath).toUri, hconf.value)
      ImageReader.OpenCVLoader
      rows.foreach {row =>
        val rowInternals = row.getStruct(0)
        val bytes = encode(rowInternals, encoding)
        val outputPath = new Path(basePath,row.getString(1))
        val os = fs.create(outputPath)
        val is = new ByteArrayInputStream(bytes)
        try
          os.write(IOUtils.toByteArray(is))
        finally {
          os.close()
          is.close()
        }
      }
    }
  }

}

/** Implicit conversion allows sparkSession.readImages(...) syntax
  * Example:
  *     import com.microsoft.ml.spark.Readers.implicits._
  *     sparkSession.readImages(path, recursive = false)
  */
object Image {

  object implicits {
    import scala.language.implicitConversions

    class Session(sparkSession: SparkSession) {

      /** Read the directory of images from the local or remote source
        *
        * @param path         Path to the image directory
        * @param recursive    Recursive path search flag
        * @param sampleRatio  Fraction of the files loaded
        * @param inspectZip   Whether zip files are treated as directories
        * @return Dataframe with a single column "image" of images, see ImageSchema for details
        */
      def readImages(path: String, recursive: Boolean,
                     sampleRatio: Double = 1, inspectZip: Boolean = true, seed: Long = 0L): DataFrame =
        ImageReader.read(path, recursive, sparkSession, sampleRatio, inspectZip, seed)
    }

    implicit def ImplicitSession(sparkSession: SparkSession):Session = new Session(sparkSession)

  }
}
