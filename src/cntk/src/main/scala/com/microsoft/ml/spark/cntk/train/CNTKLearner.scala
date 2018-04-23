// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk.train

import java.net.URI

import com.microsoft.ml.spark.cntk.CNTKModel
import com.microsoft.ml.spark.core.contracts.MMLParams
import com.microsoft.ml.spark.core.env.FileUtilities.File
import com.microsoft.ml.spark.core.env.{EnvironmentUtils, FileUtilities, InternalWrapper}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql._
import org.apache.spark.sql.types._

trait CNTKParams extends MMLParams {
  val labelsColumnName = StringParam(this, "labelsColumnName", "Label column name", "labels")
  def getLabelsColumnName: String = $(labelsColumnName)

  val featuresColumnName = StringParam(this, "featuresColumnName", "Features column name", "features")
  def getFeaturesColumnName: String = $(featuresColumnName)

  // This will go away after the CNTK HDFS Deserializer
  val localHdfsMount = StringParam(this, "localHdfsMount", "Local mount point for hdfs:///")
  def getLocalHdfsMount: String = $(localHdfsMount)

  val dataTransfer = StringParam(this, "dataTransfer", "Transfer strategy", "local")
  def setTransferStrategy(s: String): this.type = set(dataTransfer, s)
  def getDataTransfer: String = $(dataTransfer)

  // TODO: Convert to enum contract shared with CNTK's HDFS Deserializer
  val dataFormat = StringParam(this, "dataFormat", "Transfer format", "text")
  def setDataFormat(s: String): this.type = set(dataFormat, s)
  def getDataFormat: String = $(dataFormat)

  val weightPrecision = StringParam(this, "weightPrecision", "Weights", CNTKLearner.floatPrecision)
  def getWeightPrecision: String = $(weightPrecision)

  val featureCount = IntParam(this, "featureCount", "Number of features for reduction", 1)
  def setFeatureCount(c: Int): this.type = set(featureCount, c)
  def getFeatureCount: Int = $(featureCount)

  val brainScript = StringParam(this, "brainScript", "String of BrainScript config")
  def setBrainScriptText(t: String): this.type = set(brainScript, t)
  def setBrainScriptFile(f: String): this.type = set(brainScript, FileUtilities.readFile(new File(f)))
  def getBrainScript: String = $(brainScript)

  val parallelTrain = BooleanParam(this, "parallelTrain", "Train using an MPI ring", true)
  def setParallelTrain(b: Boolean): this.type = set(parallelTrain, b)
  def getParallelTrain: Boolean = $(parallelTrain)

  val workingDir = StringParam(this, "workingDir", "Working directory for CNTK", "tmp")
  def setWorkingDirectory(d: String): this.type = set(workingDir, d)
  def getWorkingDirectory: String = $(workingDir)

  val gpuMachines = new StringArrayParam(this, "gpuMachines", "GPU machines to train on")
  setDefault(gpuMachines -> Array(s"127.0.0.1,${EnvironmentUtils.GPUCount.getOrElse(1)}"))
  def setGPUMachines(value: Array[String]): this.type = set(gpuMachines, value)
  def getGPUMachines: Array[String] = $(gpuMachines)

  val username = StringParam(this, "username", "Username for the GPU VM", "sshuser")
  def setUserName(value: String): this.type = set(username, value)
  def getUserName: String = $(username)

}

object CNTKLearner extends DefaultParamsReadable[CNTKLearner] {
  val localDataTransfer = "local"
  val hdfsDataTransfer  = "hdfs"
  val textDataFormat    = "text"
  val parquetDataFormat = "parquet"
  val denseForm         = "dense"
  val sparseForm        = "sparse"
  val doublePrecision   = "double"
  val floatPrecision    = "float"
  val rpcPortNumber     = 8020
  val identityLocation  = "wasb:///MML-GPU/identity"
  val localSSH          = ".ssh/MML-GPU"
}

@InternalWrapper
class CNTKLearner(override val uid: String) extends Estimator[CNTKModel] with CNTKParams {

  def this() = this(Identifiable.randomUID("CNTKLearner"))

  override def fit(dataset: Dataset[_]): CNTKModel = {
    val spark = dataset.sparkSession
    val labels = getLabelsColumnName
    val features = getFeaturesColumnName

    // Convert label column to categorical on train, remove rows with missing labels
    val convertedLabelDataset = dataset.na.drop(Seq(labels))

    // The reduction step of Featurize
    val reducedData = DataTransferUtils.reduceAndAssemble(
      convertedLabelDataset,
      labels,
      features,
      getFeatureCount)

    // TODO: we should store vector sizes in schema for quick retrieval in the future
    val feature1 = reducedData.select(features).head.getAs[Vector](0)
    val featureDim = feature1.size
    val featureForm = feature1 match {
      case dv: DenseVector => CNTKLearner.denseForm
      case sv: SparseVector => CNTKLearner.sparseForm
    }

    val label1 = reducedData.select(labels).head.getAs[Vector](0)
    val labelDim = label1.size
    val labelForm = label1 match {
      case dv: DenseVector => CNTKLearner.denseForm
      case sv: SparseVector => CNTKLearner.sparseForm
    }

    val cntkrootURI = new URI(getWorkingDirectory)
    val cntkrootPath = new File(cntkrootURI).getAbsolutePath
    val relativeInPath = new Path(cntkrootPath, s"$uid-inputdata").toString
    logInfo(s"$uid working in $cntkrootPath with relative path $relativeInPath")

    val (writer, hdfsPath) = getWriter(dataset, relativeInPath)

    // Generating the CNTK text file
    val conformedData = getDataFormat match {
      case CNTKLearner.textDataFormat =>
        DataTransferUtils.convertDatasetToCNTKTextFormat(
          reducedData, labels, features, labelForm, featureForm)
      case CNTKLearner.parquetDataFormat =>
        DataTransferUtils.convertDatasetToCNTKParquetFormat(
          reducedData, labels, features, labelForm, featureForm, getWeightPrecision)
    }

    conformedData.persist()

    val remappedInPath = getDataFormat match {
      case CNTKLearner.textDataFormat => writer.checkpointToText(conformedData)
      case CNTKLearner.parquetDataFormat => writer.checkpointToParquet(conformedData)
    }
    val outputDir = s"$uid-outdir"
    val relativeOutRoot = s"$cntkrootPath/$outputDir"

    val config = new BrainScriptBuilder()
      .setOutputRoot(relativeOutRoot)
      // TODO: Refactor to more structured form of converting schema to CNTK config
      .setInputFile(
        remappedInPath,
        getDataFormat,
        Map(features -> InputShape(featureDim, featureForm),
            labels -> InputShape(labelDim, labelForm)))
      .setHDFSPath(hdfsPath)
      .setWeightPrecision(getWeightPrecision)

    // Train the learner
    val cb =
      if (getParallelTrain) new MPICommandBuilder(log, getGPUMachines, hdfsPath, remappedInPath, getUserName)
      else new CNTKCommandBuilder(log)
    cb.setWorkingDir(cntkrootPath)
      .insertBaseConfig(getBrainScript)
      .appendOverrideConfig(config.toOverrideConfig)
      .setOutputDir(outputDir)
      .setSparkSession(spark)
      .setDataFormat(getDataFormat)
      .setModelOutputDir(config.getLocalModelOutputDir)
      .setModelName(config.getModelName)

    val output = cb.runCommand
    logInfo("CNTK training finished")
    conformedData.unpersist()
    // Note: This currently runs only on linux
    new CNTKModel(uid + "-model")
      .setModelLocation(spark, config.getModelPath)
      .setInputCol(features)
      .setOutputCol(labels)
  }

  def getWriter(dataset: Dataset[_], relativeInPath: String): (SingleFileResolver,
    Option[(String, String, String)]) = {
    getDataTransfer match {
      case CNTKLearner.localDataTransfer => (new LocalWriter(log, relativeInPath), None)
      case CNTKLearner.hdfsDataTransfer => {
        val mntpt = if (isDefined(localHdfsMount)) {
          val x = getLocalHdfsMount
          logInfo(s"Using override hdfsm point: $x")
          x
        } else {
          val x = sys.env.getOrElse("HDFS_MOUNTPOINT", "tmp/mnt")
          logInfo(s"Using deduced hdfsm point: $x")
          x
        }
        logInfo(s"hdfs-mount mounted at $mntpt, writing ${dataset.rdd.getNumPartitions} distributed files")
        val hdfsWriter =
          new HdfsWriter(log,
                         mntpt,
                         dataset.rdd.getNumPartitions,
                         relativeInPath,
                         dataset.sparkSession.sparkContext)
        val hdfsPath = Some((hdfsWriter.getHdfsInputDataDir,
                             hdfsWriter.getRootDir,
                             hdfsWriter.getNameNode))
        (hdfsWriter, hdfsPath)
      }
      case _ =>
        throw new Exception(
          s"Only ${CNTKLearner.localDataTransfer} and" +
          s" ${CNTKLearner.hdfsDataTransfer} supported options for data transfer")
    }
  }

  override def copy(extra: ParamMap): Estimator[CNTKModel] = defaultCopy(extra)

  /** Taken from CNTKModel - adds the output column to the input
    * @param schema The input schema
    * @return The transformed schema with output column added
    */
  override def transformSchema(schema: StructType): StructType =
    schema.add(getLabelsColumnName, VectorType)

}
