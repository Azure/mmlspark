// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.source.image

import com.google.common.io.{ByteStreams, Closeables}
import com.microsoft.ml.spark.core.schema.ImageSchemaUtils
import com.microsoft.ml.spark.io.image.ImageUtils
import javax.imageio.ImageIO
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.util.SerializableConfiguration

class PatchedImageFileFormat extends ImageFileFormat with Serializable with Logging {

  override def shortName(): String = "patchedImage"

  private def verifySchema(schema: StructType): Unit = {
    val target = ImageSchema.imageSchema.add("filenames", StringType)
    val targetNullable = ImageSchemaUtils.ImageSchemaNullable.add("filenames", StringType)
    if (schema != target && schema != targetNullable) {
      throw new IllegalArgumentException(
        s"Image data source supports: " +
          s"\n\t$target" +
          s"\n\tyou have :" +
          s"\n\t$schema.")
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
        new ImageOutputWriter(
          path,
          dataSchema.fieldIndex(options.getOrElse("imageCol", "image")),
          dataSchema.fieldIndex(options.getOrElse("pathCol", "filenames")),
          context)
      }

      override def getFileExtension(context: TaskAttemptContext): String = {
        ""
      }
    }
  }

  //This is needed due to a multi-threading bug n the jvm
  private def catchFlakiness[T](times: Int)(f: => Option[T]): Option[T] = {
    try {
      f
    } catch {
      case e: NullPointerException if times >= 1 =>
        logWarning("caught null pointer exception due to jvm bug", e)
        catchFlakiness(times - 1)(f)
      case _: Exception =>
        None
    }
  }

  override protected def buildReader(sparkSession: SparkSession,
                                     dataSchema: StructType,
                                     partitionSchema: StructType,
                                     requiredSchema: StructType,
                                     filters: Seq[Filter],
                                     options: Map[String, String],
                                     hadoopConf: Configuration): PartitionedFile => Iterator[InternalRow] = {
    assert(
      requiredSchema.length <= 1,
      "Image data source only produces a single data column named \"image\".")

    val broadcastedHadoopConf =
      sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))

    val imageSourceOptions = new ImageOptions(options)

    (file: PartitionedFile) => {
      val emptyUnsafeRow = new UnsafeRow(0)
      if (!imageSourceOptions.dropInvalid && requiredSchema.isEmpty) {
        Iterator(emptyUnsafeRow)
      } else {
        val origin = file.filePath
        val path = new Path(origin)
        val fs = path.getFileSystem(broadcastedHadoopConf.value.value)
        val stream = fs.open(path)
        val bytes = try {
          ByteStreams.toByteArray(stream)
        } finally {
          Closeables.close(stream, true)
        }
        val resultOpt = catchFlakiness(5)(ImageSchema.decode(origin, bytes))
        val filteredResult = if (imageSourceOptions.dropInvalid) {
          resultOpt.toIterator
        } else {
          Iterator(resultOpt.getOrElse(ImageSchema.invalidImageRow(origin)))
        }

        if (requiredSchema.isEmpty) {
          filteredResult.map(_ => emptyUnsafeRow)
        } else {
          val converter = RowEncoder(requiredSchema)
          filteredResult.map(row => converter.createSerializer()(row))
        }
      }
    }
  }

  override def toString: String = "PatchedImageFileFormat"
}

class ImageOutputWriter(val path: String,
                        val imageCol: Int,
                        val pathCol: Int,
                        val context: TaskAttemptContext)
  extends OutputWriter {

  private val hconf = context.getConfiguration

  private val fs = new Path(path).getFileSystem(hconf)

  override def write(row: InternalRow): Unit = {
    val imgRow = row.getStruct(imageCol, 6)
    val bImg = ImageUtils.toBufferedImage(imgRow)
    val nonTempPath = new Path(path).getParent.getParent.getParent.getParent.getParent
    val outputPath = new Path(nonTempPath, row.getString(pathCol))
    val os = fs.create(outputPath)
    try {
      val codec = outputPath.toString.split(".".charAt(0)).last
      val success = ImageIO.write(bImg, codec, os)
      assert(success, s"codec failed: $codec")
    } finally {
      os.close()
    }
  }

  override def close(): Unit = {
    fs.close()
  }
}
