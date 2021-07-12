// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import java.io._
import java.net.{ServerSocket, Socket}
import java.util.concurrent.Executors

import com.microsoft.ml.lightgbm._
import com.microsoft.ml.spark.core.env.NativeLoader
import com.microsoft.ml.spark.core.utils.ClusterUtil
import com.microsoft.ml.spark.featurize.{Featurize, FeaturizeUtilities}
import com.microsoft.ml.spark.lightgbm.dataset.LightGBMDataset
import com.microsoft.ml.spark.lightgbm.params.TrainParams
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, Dataset}
import org.slf4j.Logger

import ConnectionState._

import scala.collection.immutable.HashSet
import scala.collection.mutable.ListBuffer
import scala.concurrent.duration.{Duration, SECONDS}
import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}

/** Helper utilities for LightGBM learners */
object LightGBMUtils {
  def validate(result: Int, component: String): Unit = {
    if (result == -1) {
      throw new Exception(component + " call failed in LightGBM with error: "
        + lightgbmlib.LGBM_GetLastError())
    }
  }

  def validateArray(result: SWIGTYPE_p_void, component: String): Unit = {
    if (result == null) {
      throw new Exception(component + " call failed in LightGBM with error: "
        + lightgbmlib.LGBM_GetLastError())
    }
  }

  /** Loads the native shared object binaries lib_lightgbm.so and lib_lightgbm_swig.so
    */
  def initializeNativeLibrary(): Unit = {
    val osPrefix = NativeLoader.getOSPrefix
    if (System.getProperty("os.version").toLowerCase.contains("5.4.0")) {
      new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbmcuda")
      new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbmcuda_swig")
    }
    else {
      new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm")
      new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm_swig")
    }
  }

  def getFeaturizer(dataset: Dataset[_], labelColumn: String, featuresColumn: String,
                    weightColumn: Option[String] = None,
                    groupColumn: Option[String] = None,
                    oneHotEncodeCategoricals: Boolean = true): PipelineModel = {
    // Create pipeline model to featurize the dataset
    val featuresToHashTo = FeaturizeUtilities.NumFeaturesTreeOrNNBased
    val featureColumns = dataset.columns.filter(col => col != labelColumn &&
      !weightColumn.contains(col) && !groupColumn.contains(col)).toSeq
    val featurizer = new Featurize()
      .setOutputCol(featuresColumn)
      .setInputCols(featureColumns.toArray)
      .setOneHotEncodeCategoricals(oneHotEncodeCategoricals)
      .setNumFeatures(featuresToHashTo)
    featurizer.fit(dataset)
  }

  def getCategoricalIndexes(df: DataFrame,
                            featuresCol: String,
                            slotNames: Array[String],
                            categoricalColumnIndexes: Array[Int],
                            categoricalColumnSlotNames: Array[String]): Array[Int] = {
    val categoricalIndexes = if(slotNames.nonEmpty) {
      categoricalColumnSlotNames.map(slotNames.indexOf(_))
    } else {
      val categoricalSlotNamesSet = HashSet(categoricalColumnSlotNames: _*)
      val featuresSchema = df.schema(featuresCol)
      val metadata = AttributeGroup.fromStructField(featuresSchema)
      if (metadata.attributes.isEmpty) Array[Int]()
      else {
        metadata.attributes.get.zipWithIndex.flatMap {
          case (null, _) => Iterator()
          case (attr, idx) =>
            if (attr.name.isDefined && categoricalSlotNamesSet.contains(attr.name.get)) {
              Iterator(idx)
            } else {
              attr match {
                case _: NumericAttribute | UnresolvedAttribute => Iterator()
                // Note: it seems that BinaryAttribute is not considered categorical,
                // since all OHE cols are marked with this, but StringIndexed are always Nominal
                case _: BinaryAttribute => Iterator()
                case _: NominalAttribute => Iterator(idx)
              }
            }
        }
      }
    }

    categoricalColumnIndexes.union(categoricalIndexes).distinct
  }

  def sendDataToExecutors(hostAndPorts: ListBuffer[(Socket, String)], allConnections: String): Unit = {
    hostAndPorts.foreach(hostAndPort => {
      val writer = new BufferedWriter(new OutputStreamWriter(hostAndPort._1.getOutputStream))
      writer.write(allConnections + "\n")
      writer.flush()
    })
  }

  def closeConnections(log: Logger, hostAndPorts: ListBuffer[(Socket, String)],
                       driverServerSocket: ServerSocket): Unit = {
    log.info("driver closing all sockets and server socket")
    hostAndPorts.foreach(_._1.close())
    driverServerSocket.close()
  }

  def addSocketAndComm(hostAndPorts: ListBuffer[(Socket, String)], log: Logger,
                       comm: String, driverSocket: Socket): Unit = {
    log.info(s"driver received socket from task: $comm")
    val socketAndComm = (driverSocket, comm)
    hostAndPorts += socketAndComm
  }

  /** Handles the connection to a task from the driver.
    *
    * @param driverServerSocket The driver socket.
    * @param log The log4j logger.
    * @param hostAndPorts A list of host and ports of connected tasks.
    * @return The connection status, can be finished for barrier mode, empty task or connected.
    */
  def handleConnection(driverServerSocket: ServerSocket, log: Logger,
                       hostAndPorts: ListBuffer[(Socket, String)]): ConnectionState = {
    log.info("driver accepting a new connection...")
    val driverSocket = driverServerSocket.accept()
    val reader = new BufferedReader(new InputStreamReader(driverSocket.getInputStream))
    val comm = reader.readLine()
    if (comm == LightGBMConstants.FinishedStatus) {
      log.info("driver received all tasks from barrier stage")
      Finished
    } else if (comm == LightGBMConstants.IgnoreStatus) {
      log.info("driver received ignore status from task")
      EmptyTask
    } else {
      addSocketAndComm(hostAndPorts, log, comm, driverSocket)
      Connected
    }
  }

  /**
    * Opens a socket communications channel on the driver, starts a thread that
    * waits for the host:port from the executors, and then sends back the
    * information to the executors.
    *
    * @param numTasks The total number of training tasks to wait for.
    * @return The address and port of the driver socket.
    */
  def createDriverNodesThread(numTasks: Int, df: DataFrame,
                              log: Logger, timeout: Double,
                              barrierExecutionMode: Boolean,
                              driverServerPort: Int): (String, Int, Future[Unit]) = {
    // Start a thread and open port to listen on
    implicit val context: ExecutionContextExecutor =
      ExecutionContext.fromExecutor(Executors.newSingleThreadExecutor())
    val driverServerSocket = new ServerSocket(driverServerPort)
    // Set timeout on socket
    val duration = Duration(timeout, SECONDS)
    if (duration.isFinite()) {
      driverServerSocket.setSoTimeout(duration.toMillis.toInt)
    }
    val f = Future {
      var emptyTaskCounter = 0
      val hostAndPorts = ListBuffer[(Socket, String)]()
      if (barrierExecutionMode) {
        log.info(s"driver using barrier execution mode")
        def connectToWorkers: Boolean = handleConnection(driverServerSocket, log,
          hostAndPorts) == Finished || connectToWorkers
        connectToWorkers
      } else {
        log.info(s"driver expecting $numTasks connections...")
        while (hostAndPorts.size + emptyTaskCounter < numTasks) {
          val connectionResult = handleConnection(driverServerSocket, log, hostAndPorts)
          if (connectionResult == ConnectionState.EmptyTask) emptyTaskCounter += 1
        }
      }
      // Concatenate with commas, eg: host1:port1,host2:port2, ... etc
      val allConnections = hostAndPorts.map(_._2).mkString(",")
      log.info(s"driver writing back to all connections: $allConnections")
      // Send data back to all tasks and helper tasks on executors
      sendDataToExecutors(hostAndPorts, allConnections)
      closeConnections(log, hostAndPorts, driverServerSocket)
    }
    val host = ClusterUtil.getDriverHost(df)
    val port = driverServerSocket.getLocalPort
    log.info(s"driver waiting for connections on host: $host and port: $port")
    (host, port, f)
  }

  /** Returns an integer ID for the current worker.
    * @return In cluster, returns the executor id.  In local case, returns the partition id.
    */
  def getWorkerId(): Int = {
    val executorId = SparkEnv.get.executorId
    val ctx = TaskContext.get
    val partId = ctx.partitionId
    // If driver, this is only in test scenario, make each partition a separate task
    val id = if (executorId == "driver") partId else executorId
    val idAsInt = id.toString.toInt
    idAsInt
  }

  /** Returns true if spark is run in local mode.
    * @return True if spark is run in local mode.
    */
  def isLocalExecution(): Boolean = {
    val executorId = SparkEnv.get.executorId
    executorId == "driver"
  }

  /** Returns a unique task Id for the current task run on the executor.
    * @return A unique task id.
    */
  def getTaskId(): Long = {
    val ctx = TaskContext.get
    val taskId = ctx.taskAttemptId()
    taskId
  }

  def getNumRowsForChunksArray(numRows: Int, chunkSize: Int): SWIGTYPE_p_int = {
    var numChunks = Math.floorDiv(numRows, chunkSize)
    var leftoverChunk = numRows % chunkSize
    if (leftoverChunk > 0) {
      numChunks += 1
    }
    val numRowsForChunks = lightgbmlib.new_intArray(numChunks)
    (0 until numChunks).foreach({ index: Int =>
      if (index == numChunks - 1 && leftoverChunk > 0) {
        lightgbmlib.intArray_setitem(numRowsForChunks, index, leftoverChunk)
      } else {
        lightgbmlib.intArray_setitem(numRowsForChunks, index, chunkSize)
      }
    })
    numRowsForChunks
  }

  def getDatasetParams(trainParams: TrainParams): String = {
    val datasetParams = s"max_bin=${trainParams.maxBin} is_pre_partition=True " +
      s"bin_construct_sample_cnt=${trainParams.binSampleCount} " +
      s"num_threads=${trainParams.executionParams.numThreads} " +
      (if (trainParams.categoricalFeatures.isEmpty) ""
      else s"categorical_feature=${trainParams.categoricalFeatures.mkString(",")}")
    datasetParams
  }

  def generateDenseDataset(numRows: Int, numCols: Int, featuresArray: doubleChunkedArray,
                           referenceDataset: Option[LightGBMDataset],
                           featureNamesOpt: Option[Array[String]],
                           trainParams: TrainParams, chunkSize: Int): LightGBMDataset = {
    val isRowMajor = 1
    val datasetOutPtr = lightgbmlib.voidpp_handle()
    val datasetParams = getDatasetParams(trainParams)
    val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64
    var data: Option[(SWIGTYPE_p_void, SWIGTYPE_p_double)] = None
    val numRowsForChunks = getNumRowsForChunksArray(numRows, chunkSize)
    try {
      // Generate the dataset for features
      featuresArray.get_last_chunk_add_count()
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromMats(featuresArray.get_chunks_count().toInt,
        featuresArray.data_as_void(), data64bitType,
        numRowsForChunks, numCols,
        isRowMajor, datasetParams, referenceDataset.map(_.datasetPtr).orNull, datasetOutPtr),
        "Dataset create")
    } finally {
      featuresArray.release()
      lightgbmlib.delete_intArray(numRowsForChunks)
    }
    val dataset = new LightGBMDataset(lightgbmlib.voidpp_value(datasetOutPtr))
    dataset.setFeatureNames(featureNamesOpt, numCols)
    dataset
  }

  /** Generates a sparse dataset in CSR format.
    *
    * @param sparseRows The rows of sparse vector.
    * @return
    */
  def generateSparseDataset(sparseRows: Array[SparseVector],
                            referenceDataset: Option[LightGBMDataset],
                            featureNamesOpt: Option[Array[String]],
                            trainParams: TrainParams): LightGBMDataset = {
    val numCols = sparseRows(0).size

    val datasetOutPtr = lightgbmlib.voidpp_handle()
    val datasetParams = getDatasetParams(trainParams)
    // Generate the dataset for features
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromCSRSpark(
      sparseRows.asInstanceOf[Array[Object]],
      sparseRows.length,
      numCols, datasetParams, referenceDataset.map(_.datasetPtr).orNull,
      datasetOutPtr),
      "Dataset create")
    val dataset = new LightGBMDataset(lightgbmlib.voidpp_value(datasetOutPtr))
    dataset.setFeatureNames(featureNamesOpt, numCols)
    dataset
  }
}
