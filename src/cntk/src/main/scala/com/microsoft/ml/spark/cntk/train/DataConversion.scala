// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk.train

import java.io.PrintWriter
import java.net.URI

import com.microsoft.ml.spark.core.env.EnvironmentUtils
import com.microsoft.ml.spark.core.env.FileUtilities._
import com.microsoft.ml.spark.core.hadoop.HadoopUtils
import com.microsoft.ml.spark.stages.featurize.Featurize
import org.apache.hadoop.conf
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.slf4j.Logger

/** Utilities for reducing data to CNTK format and generating the text file output to disk.
  */
object DataTransferUtils {

  def toText(form: String)(value: Any): String = {
    value match {
      case v: Vector => convertVectorToText(v, form)
      case d: Double => d.toString
      case f: Float => f.toString
    }
  }

  def toVec(value: Any): Vector = {
    value match {
      case v: Vector => v
      case d: Double => new DenseVector(Array(d))
      case f: Float => new DenseVector(Array(f.toDouble))
      case i: Integer => new DenseVector(Array(i.toDouble))
      case l: Long => new DenseVector(Array(l.toDouble))
    }
  }

  def convertVectorToText(v: Vector, form: String): String = {
    val heuristicBloat = 8
    val sb = new StringBuilder(v.numActives * heuristicBloat)
    v match {
      case sv: SparseVector =>
        if (form == CNTKLearner.sparseForm) {
          sv.foreachActive { (idx, value) =>
            sb.append(idx).append(":").append(value).append(" ")
            ()
          }
        } else if (form == CNTKLearner.denseForm) {
          sb.append(convertVectorToText(sv.toDense, form))
        } else {
          throw new Exception(s"Unknown vector form $form")
        }
      case dv: DenseVector =>
        if (form == CNTKLearner.denseForm) {
          dv.values.foreach(value => sb.append(value).append(" "))
        } else if (form == CNTKLearner.sparseForm) {
          for (i <- dv.values.indices) {
            sb.append(i).append(":").append(dv.values(i)).append(" ")
          }
        } else {
          throw new Exception(s"Unknown vector form $form")
        }
    }
    sb.toString
  }

  private def col2vec: UserDefinedFunction  = udf(toVec _)
  private def col2str(form: String): UserDefinedFunction = udf(toText(form) _)

  def reduceAndAssemble(data: Dataset[_],
                        label: String,
                        outputVecName: String,
                        features: Int): DataFrame = {
    val tempFeaturizer = new Featurize()
      .setFeatureColumns(Map(outputVecName -> data.columns.filter(_ != label)))
      .setNumberOfFeatures(features)
      .setOneHotEncodeCategoricals(true)
      .setAllowImages(true)
      .fit(data)
    val reduced = tempFeaturizer.transform(data)
    reduced.select(col2vec(reduced(label)).as(label), reduced(outputVecName))
  }

  def convertDatasetToCNTKTextFormat(data: Dataset[_],
                                     label: String,
                                     feats: String,
                                     labelForm: String,
                                     featureForm: String): DataFrame = {
    val labelStrCol = col2str(labelForm)(data(label))
    val featStrCol = col2str(featureForm)(data(feats))
    val uberCol = concat(
        lit(s"|$label "),
        labelStrCol,
        lit(" "),
        lit(s"|$feats "),
        featStrCol)
    data.select(uberCol.as('value))
  }

  def toFloat(data: Array[Double]): Array[Float] = data.map(_.toFloat)

  def convertDatasetToCNTKParquetFormat(data: Dataset[_],
                                        label: String,
                                        feats: String,
                                        labelForm: String,
                                        featureForm: String,
                                        weightPrecision: String): DataFrame = {
    // Dense conversion functions
    val toDenseArray =
      if (weightPrecision == CNTKLearner.floatPrecision) {
        udf({vector: Vector => vector match {
               case dv: DenseVector => toFloat(dv.toArray)
               case sv: SparseVector => toFloat(sv.toDense.toArray)
             }})
      } else {
        udf({vector: Vector => vector match {
               case dv: DenseVector => dv.toArray
               case sv: SparseVector => sv.toDense.toArray
             }})
      }
    // Sparse format conversion functions
    val getNZs =
      udf({vector: Vector => vector match {
             case dv: DenseVector => dv.toSparse.numNonzeros
             case sv: SparseVector => sv.numNonzeros
           }})
    val getIdxes =
      udf({vector: Vector => vector match {
             case dv: DenseVector => dv.toSparse.indices
             case sv: SparseVector => sv.indices
           }})
    val getVals =
      if (weightPrecision == CNTKLearner.floatPrecision) {
        udf({vector: Vector => vector match {
               case dv: DenseVector => toFloat(dv.toSparse.values)
               case sv: SparseVector => toFloat(sv.values)
             }})
      } else {
        udf({vector: Vector => vector match {
               case dv: DenseVector => dv.toSparse.values
               case sv: SparseVector => sv.values
             }})
      }
    // Note: order matters, features column needs to come first
    val cols = data.select(feats, label).columns.flatMap(name => {
      if ((name == label && labelForm == CNTKLearner.denseForm) ||
        (name == feats && featureForm == CNTKLearner.denseForm)) {
        Seq(toDenseArray(data.col(name)).alias(name))
      } else {
        Seq(getNZs(data.col(name))  .alias(name + "_size"),
            getIdxes(data.col(name)).alias(name + "_indices"),
            getVals(data.col(name)) .alias(name + "_values"))
      }
    })
    data.select(cols: _*)
  }

}

abstract class DataWriter(log: Logger, destPath: String) {
  protected val destUri = new URI(destPath)
  protected val relativeDest: String = destUri.getPath

  protected val partitions: Int

  protected def remapPath(ext: String): String

  def constructedPath: String

  def checkpointToText(data: Dataset[_]): String = {
    val fullPath = constructedPath
    // For local file, collect and write to current driver's file location
    if (partitions == 1 && constructedPath.startsWith("file:")) {
      val localdata = data.collect()

      log.info(s"Writing dataset to $relativeDest")
      val relDest = new File(relativeDest)
      relDest.mkdirs()
      val writer = new PrintWriter(new File(new Path(relativeDest, "part0.txt").toString))
      try {
        localdata.foreach(row => writer.println(row.asInstanceOf[Row].getString(0)))
      } finally {
        writer.close()
      }
    } else {
      log.info(s"Writing dataset to $fullPath")
      data.coalesce(partitions).write.text(fullPath)
    }
    remapPath("txt")
  }

  def checkpointToParquet(data: Dataset[_]): String = {
    val fullPath = constructedPath
    log.info(s"Writing dataset to $fullPath")
    data.coalesce(partitions).write.format("parquet").save(fullPath)
    remapPath("parquet")
  }
}

// This is used when Hadoop creates the actual single part file
// inside the path we've provided - we want that one.
abstract class SingleFileResolver(log: Logger, path: String) extends DataWriter(log, path) {
  protected val remappedRoot: String

  protected def remapPath(extension: String): String = {
    val dir = new File(remappedRoot)
    log.info(s"Probing $dir for single file $constructedPath")
    val file = dir.listFiles.filter(f => f.isFile && f.getName.endsWith(extension)).head
    log.info(s"Resolving single file ${file.getAbsolutePath}")
    file.getAbsolutePath
  }
}

class LocalWriter(log: Logger, path: String) extends SingleFileResolver(log, path) {
  val partitions = 1
  val constructedPath: String = new URI("file", null, relativeDest, null, null).normalize.toString

  // TODO: Move this logic to Apache commons lang helper that already exists
  // And then provide a helper function in FileUtilities. Why doesn't URI normalize properly for new File()?
  val remappedRoot: String = {
    val root = if (EnvironmentUtils.IsWindows) "C:" else ""
    root + relativeDest
  }
}

class HdfsWriter(log: Logger, localMnt: String, parts: Int, path: String, sc: SparkContext)
  extends SingleFileResolver(log, path) {
  val partitions: Int = parts
  val hConf: conf.Configuration = sc.hadoopConfiguration
  val nameNode: String = new HadoopUtils(log, hConf).getActiveNameNode
  val constructedPath: String = new URI("hdfs", nameNode, relativeDest, null, null).toString
  val mountPoint: String = if (localMnt.startsWith("/")) localMnt else "/" + localMnt
  val remappedRoot: String = mountPoint + relativeDest

  override protected def remapPath(extension: String): String = {
    log.info(s"Wrote $extension files to hdfs path: $constructedPath")
    val file =
      if (extension == CNTKLearner.parquetDataFormat) {
        relativeDest
      } else {
        log.info(s"Mount point for process: $mountPoint")
        log.info(s"Remapped root for file: $remappedRoot")
        s"$remappedRoot/merged-input.txt"
      }
    log.info(s"Path to mnt file on GPU VM: $file")
    file
  }

  // The HDFS URL to mount on the edge node
  def getHdfsToMount: String = new URI("hdfs", nameNode, "/", null, null).toString
  // The directory containing the input data in HDFS
  def getHdfsInputDataDir: String = constructedPath
  // The root directory
  def getRootDir: String = remappedRoot
  // The active name node, with port removed at the end
  def getNameNode: String = nameNode.replace(":8020", "")

}
