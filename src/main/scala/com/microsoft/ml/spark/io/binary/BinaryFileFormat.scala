// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.binary

import java.io.{Closeable, InputStream}
import java.net.URI

import com.microsoft.ml.spark.core.env.StreamUtilities.ZipIterator
import com.microsoft.ml.spark.core.schema.BinaryFileSchema
import org.apache.commons.io.{FilenameUtils, IOUtils}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.lib.input.FileSplit
import org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl
import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.util.SerializableConfiguration

import scala.util.Random

/** Actually reads the records from files
  *
  * @param subsample  what ratio to subsample
  * @param inspectZip whether to inspect zip files
  */
private[spark] class BinaryRecordReader(val subsample: Double, val inspectZip: Boolean, val seed: Long)
  extends RecordReader[String, Array[Byte]] {

  private var done: Boolean = false
  private var inputStream: InputStream = _
  private var filename: String = _
  private var recordValue: Array[Byte] = _
  private var progress: Float = 0.0F
  private val rng: Random = new Random()
  private var zipIterator: ZipIterator = _

  override def close(): Unit = {
    if (inputStream != null) {
      inputStream.close()
    }
  }

  override def getCurrentKey: String = {
    filename
  }

  override def getCurrentValue: Array[Byte] = {
    recordValue
  }

  override def getProgress: Float = {
    progress
  }

  override def initialize(inputSplit: InputSplit, context: TaskAttemptContext): Unit = {
    // the file input
    val fileSplit = inputSplit.asInstanceOf[FileSplit]

    val file = fileSplit.getPath        // the actual file we will be reading from
    val conf = context.getConfiguration // job configuration
    val fs = file.getFileSystem(conf)   // get the filesystem
    filename = file.toString            // open the File

    inputStream = fs.open(file)
    rng.setSeed(filename.hashCode.toLong ^ seed)
    if (inspectZip && FilenameUtils.getExtension(filename) == "zip") {
        zipIterator = new ZipIterator(inputStream, filename, rng, subsample)
    }
  }

  def markAsDone(): Unit = {
    done = true
    progress = 1.0F
  }

  override def nextKeyValue(): Boolean = {
    if (done) {
      false
    } else if (zipIterator != null) {
      if (zipIterator.hasNext) {
        val (fn, barr) = zipIterator.next
        filename = fn
        recordValue = barr
        true
      } else {
        markAsDone()
        false
      }
    } else {
      if (rng.nextDouble() <= subsample) {
        val barr = IOUtils.toByteArray(inputStream)
        recordValue = barr
        markAsDone()
        true
      } else {
        markAsDone()
        false
      }
    }
  }
}

/** File format used for structured streaming of binary files */
class BinaryFileFormat extends TextBasedFileFormat with DataSourceRegister {

  override def isSplitable(sparkSession: SparkSession,
                           options: Map[String, String],
                           path: Path): Boolean = false

  override def shortName(): String = "binary"

  override def inferSchema(
                            sparkSession: SparkSession,
                            options: Map[String, String],
                            files: Seq[FileStatus]): Option[StructType] = {
    Some(BinaryFileSchema.Schema)
  }

  override def prepareWrite(sparkSession: SparkSession,
                            job: Job,
                            options: Map[String, String],
                            dataSchema: StructType): OutputWriterFactory = {
    new OutputWriterFactory {
      override def newInstance(
                                path: String,
                                dataSchema: StructType,
                                context: TaskAttemptContext): OutputWriter = {
        new BinaryOutputWriter(
          path,
          dataSchema.fieldIndex(options.getOrElse("bytesCol", "bytes")),
          dataSchema.fieldIndex(options.getOrElse("pathCol", "path")),
          context)
      }

      override def getFileExtension(context: TaskAttemptContext): String = {
        ""
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

    val subsample = options.getOrElse("subsample", "1.0").toDouble
    val inspectZip = options.getOrElse("inspectZip", "false").toBoolean
    val seed = options.getOrElse("seed", "0").toLong

    assert(subsample >= 0.0 & subsample <= 1.0)
    (file: PartitionedFile) => {
      val fileReader = new HadoopFileReader(file, broadcastedHadoopConf.value.value, subsample, inspectZip, seed)
      Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => fileReader.close()))
      fileReader.map { record =>
        val recordPath = record._1
        val bytes = record._2
        val row = new GenericInternalRow(2)
        row.update(0, UTF8String.fromString(recordPath))
        row.update(1, bytes)
        val outerRow = new GenericInternalRow(1)
        outerRow.update(0, row)
        outerRow
      }
    }
  }

  override def toString: String = "Binary"

  override def hashCode(): Int = getClass.hashCode()

  override def equals(other: Any): Boolean = other.isInstanceOf[BinaryFileFormat]
}

/** Thin wrapper class analogous to others in the spark ecosystem */
private[spark] class HadoopFileReader(file: PartitionedFile,
                                      conf: Configuration,
                                      subsample: Double,
                                      inspectZip: Boolean,
                                      seed: Long)
  extends Iterator[(String, Array[Byte])] with Closeable {

  private val iterator = {
    val fileSplit = new FileSplit(
      new Path(new URI(file.filePath)),
      file.start,
      file.length,
      Array.empty)
    val attemptId = new TaskAttemptID(new TaskID(new JobID(), TaskType.MAP, 0), 0)
    val hadoopAttemptContext = new TaskAttemptContextImpl(conf, attemptId)
    val reader = new BinaryRecordReader(subsample, inspectZip, seed)
    reader.initialize(fileSplit, hadoopAttemptContext)
    new KeyValueReaderIterator(reader)
  }

  override def hasNext: Boolean = iterator.hasNext

  override def next(): (String, Array[Byte]) = iterator.next()

  override def close(): Unit = iterator.close()

}

class BinaryOutputWriter(val path: String,
                        val bytesCol: Int,
                        val pathCol: Int,
                        val context: TaskAttemptContext)
  extends OutputWriter {

  private val hconf = context.getConfiguration

  private val fs = new Path(path).getFileSystem(hconf)

  override def write(row: InternalRow): Unit = {
    val bytes = row.getBinary(bytesCol)
    val filename = row.getString(pathCol)
    val nonTempPath = new Path(path).getParent.getParent.getParent.getParent.getParent
    val outputPath = new Path(nonTempPath, filename)
    val os = fs.create(outputPath)
    try {
      IOUtils.write(bytes, os)
    } finally {
      os.close()
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
