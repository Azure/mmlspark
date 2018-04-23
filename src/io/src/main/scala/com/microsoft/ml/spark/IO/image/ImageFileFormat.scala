// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.image

import java.io.ByteArrayInputStream

import com.microsoft.ml.spark.core.schema.ImageSchema
import com.microsoft.ml.spark.IO.image.{ImageReader, ImageWriter}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.hadoop.io.{IOUtils => HUtils}
import org.apache.hadoop.mapreduce._
import org.apache.spark.TaskContext
import org.apache.spark.binary.{BinaryFileFormat, HadoopFileReader}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.util.SerializableConfiguration

class ImageFileFormat extends TextBasedFileFormat with DataSourceRegister with Serializable {

  override def isSplitable(sparkSession: SparkSession,
                           options: Map[String, String],
                           path: Path): Boolean = false

  override def shortName(): String = "image"

  override def inferSchema(sparkSession: SparkSession,
                           options: Map[String, String],
                           files: Seq[FileStatus]): Option[StructType] = {
    Some(ImageSchema.schema)
  }

  private def verifySchema(schema: StructType): Unit = {
    val target = ImageSchema.schema.add("filenames",StringType)
    if (schema != target) {
      throw new IllegalArgumentException(
        s"Image data source supports $target, and you have $schema.")
    }
  }

  override def prepareWrite(sparkSession: SparkSession,
                            job: Job,
                            options: Map[String, String],
                            dataSchema: StructType): OutputWriterFactory = {
    verifySchema(dataSchema)
    new OutputWriterFactory {
      override def newInstance(
                                path: String,
                                dataSchema: StructType,
                                context: TaskAttemptContext): OutputWriter = {
        new ImageOutputWriter(path, dataSchema, context,
          sparkSession, options.getOrElse("extension", ".png"))
      }

      override def getFileExtension(context: TaskAttemptContext): String = {
        "" //TODO make this work will many types of image formats
      }
    }
  }

  override def buildReader(sparkSession: SparkSession,
                           dataSchema: StructType,
                           partitionSchema: StructType,
                           requiredSchema: StructType,
                           filters: Seq[Filter],
                           options: Map[String, String],
                           hadoopConf: Configuration): PartitionedFile => Iterator[InternalRow] = {

    val broadcastedHadoopConf =
      sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))

    val subsample = options.getOrElse("subsample","1.0").toDouble
    assert(subsample>=0.0 & subsample <=1.0)
    val inspectZip = options.getOrElse("inspectZip", "false").toBoolean
    val seed = options.getOrElse("seed", "0").toLong

    (file: PartitionedFile) => {
      val fileReader = new HadoopFileReader(file, broadcastedHadoopConf.value.value, subsample, inspectZip, seed)
      Option(TaskContext.get()).foreach(_.addTaskCompletionListener(_ => fileReader.close()))
      fileReader.flatMap { record =>
        val recordPath = record._1
        val byteArray = record._2.getBytes
        ImageReader.OpenCVLoader
        val rowOpt = ImageReader.decode(file.filePath, byteArray)

        rowOpt match {
          case None => None
          case Some(row) =>
            val imGenRow = new GenericInternalRow(1)
            val genRow = new GenericInternalRow(ImageSchema.columnSchema.fields.length)
            genRow.update(0, UTF8String.fromString(recordPath))
            genRow.update(1, row.getInt(1))
            genRow.update(2, row.getInt(2))
            genRow.update(3, row.getInt(3))
            genRow.update(4, row.getAs[Array[Byte]](4))
            imGenRow.update(0, genRow)
            Some(imGenRow)
        }
      }
    }
  }

  override def toString: String = "Image"

  override def hashCode(): Int = getClass.hashCode()

  override def equals(other: Any): Boolean = other.isInstanceOf[BinaryFileFormat]

}

class ImageOutputWriter(val path: String,
                        val dataSchema: StructType,
                        val context: TaskAttemptContext,
                        val sparkSession: SparkSession,
                        val extension: String)
  extends OutputWriter {

  private val hconf = context.getConfiguration

  private val fs = new Path(path).getFileSystem(hconf)

  override def write(row: InternalRow): Unit = {
    ImageReader.OpenCVLoader
    val rowInternals = row.getStruct(0, ImageSchema.columnSchema.fields.length)
    val bytes = ImageWriter.encode(rowInternals, extension)
    val outputPath = new Path(path, row.getString(1))
    val os = fs.create(outputPath)
    val is = new ByteArrayInputStream(bytes)
    try {
      HUtils.copyBytes(is, os, hconf)
    } finally {
      os.close()
      is.close()
    }
  }

  override def close(): Unit = {
    fs.close()
  }
}

object ConfUtils {

  def getHConf(df: DataFrame): SerializableConfiguration ={
    new SerializableConfiguration(df.sparkSession.sparkContext.hadoopConfiguration)
  }

}
