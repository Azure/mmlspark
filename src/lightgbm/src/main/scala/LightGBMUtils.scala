// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.io._
import java.net.{InetAddress, ServerSocket, Socket}
import java.util.concurrent.{ConcurrentLinkedQueue, Executors}

import com.microsoft.ml.lightgbm._
import com.microsoft.ml.spark.schema.SparkSchema
import org.apache.http.conn.util.InetAddressUtils
import org.apache.spark.{BlockManagerUtils, SparkEnv, TaskContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.scheduler.{SparkListener, SparkListenerExecutorAdded, SparkListenerExecutorRemoved}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.slf4j.Logger

import scala.collection.immutable.HashSet
import scala.collection.mutable.ListBuffer
import scala.concurrent.duration.{Duration, SECONDS}
import scala.concurrent.{ExecutionContext, Future}

/** Helper utilities for LightGBM learners */
object LightGBMUtils {
  val AUTOSCALE_INTERRUPT = "autoscale interrupt"
  val CONTINUE_TRAINING = "continue training"

  def validate(result: Int, component: String): Unit = {
    if (result == -1) {
      throw new Exception(component + " call failed in LightGBM with error: "
                            + lightgbmlib.LGBM_GetLastError())
    }
  }

  /** Loads the native shared object binaries lib_lightgbm.so and lib_lightgbm_swig.so
    */
  def initializeNativeLibrary(): Unit = {
    val osPrefix = NativeLoader.getOSPrefix()
    new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm")
    new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm_swig")
  }

  def featurizeData(dataset: Dataset[_], labelColumn: String, featuresColumn: String): PipelineModel = {
    // Create pipeline model to featurize the dataset
    val oneHotEncodeCategoricals = true
    val featuresToHashTo = FeaturizeUtilities.numFeaturesTreeOrNNBased
    val featureColumns = dataset.columns.filter(col => col != labelColumn).toSeq
    val featurizer = new Featurize()
      .setFeatureColumns(Map(featuresColumn -> featureColumns))
      .setOneHotEncodeCategoricals(oneHotEncodeCategoricals)
      .setNumberOfFeatures(featuresToHashTo)
    featurizer.fit(dataset)
  }

  def addListener(dataset: Dataset[_]): (ConcurrentLinkedQueue[String], SparkListener) = {
    val queue = new ConcurrentLinkedQueue[String]()
    val executorListener = new SparkListener() {
      override def onExecutorAdded(executorAdded: SparkListenerExecutorAdded): Unit = {
        queue.add("executor added")
        ()
      }

      override def onExecutorRemoved(executorRemoved: SparkListenerExecutorRemoved): Unit = {
        queue.add("executor removed")
        ()
      }
    }
    dataset.sparkSession.sparkContext.addSparkListener(executorListener)
    (queue, executorListener)
  }

  def getBoosterPtrFromModelString(lgbModelString: String): SWIGTYPE_p_void = {
    val boosterOutPtr = lightgbmlib.voidpp_handle()
    val numItersOut = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterLoadModelFromString(lgbModelString, numItersOut, boosterOutPtr),
      "Booster LoadFromString")
    lightgbmlib.voidpp_value(boosterOutPtr)
  }

  def getCategoricalIndexes(df: DataFrame,
                            featuresCol: String,
                            categoricalColumnIndexes: Array[Int],
                            categoricalColumnSlotNames: Array[String]): Array[Int]  = {
    val categoricalSlotNamesSet = HashSet(categoricalColumnSlotNames: _*)
    val featuresSchema = df.schema(featuresCol)
    val metadata = AttributeGroup.fromStructField(featuresSchema)
    val categoricalIndexes =
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
                case binAttr: BinaryAttribute => Iterator(idx)
                case nomAttr: NominalAttribute => Iterator(idx)
              }
            }
        }
      }
    categoricalColumnIndexes.union(categoricalIndexes).distinct
  }

  /**
    * Opens a socket communications channel on the driver, starts a thread that
    * waits for the host:port from the executors, and then sends back the
    * information to the executors.
    * @param numWorkers The total number of training workers to wait for.
    * @return The address and port of the driver socket.
    */
  def createDriverNodesThread(numWorkers: Int, df: DataFrame,
                              log: Logger, timeout: Double,
                              autoscale: Boolean, iters: Int): (String, Int, Future[Unit], ListBuffer[Int]) = {
    // Start a thread and open port to listen on
    implicit val context = ExecutionContext.fromExecutor(Executors.newSingleThreadExecutor())
    val driverServerSocket = new ServerSocket(0)
    // Set timeout on socket
    val duration = Duration(timeout, SECONDS)
    if (duration.isFinite()) {
      driverServerSocket.setSoTimeout(duration.toMillis.toInt)
    }
    // TODO: this should be passed as param in case we autoscaled earlier
    val (listenerQueue, executorListener) = LightGBMUtils.addListener(df)
    val currentIter = ListBuffer(iters)
    val f = Future {
      val hostAndPorts = ListBuffer[(Socket, String)]()
      log.info(s"driver expecting $numWorkers connections...")
      while (hostAndPorts.size < numWorkers) {
        log.info("driver accepting a new connection...")
        val driverSocket = driverServerSocket.accept()
        val reader = new BufferedReader(new InputStreamReader(driverSocket.getInputStream))
        val comm = reader.readLine()
        log.info(s"driver received socket from worker: $comm")
        val socketAndComm = (driverSocket, comm)
        hostAndPorts += socketAndComm
      }
      // Concatenate with commas, eg: host1:port1,host2:port2, ... etc
      val allConnections = hostAndPorts.map(_._2).mkString(",")
      log.info(s"driver writing back to all connections: $allConnections")
      // Send data back to all threads on executors
      hostAndPorts.foreach(hostAndPort => {
        val writer = new BufferedWriter(new OutputStreamWriter(hostAndPort._1.getOutputStream))
        writer.write(allConnections + "\n")
        writer.flush()
      })
      if (autoscale) {
        log.info(s"Autoscaling enabled, retrieving worker status during training for each iteration")
        var autoscaleDetected = false
        (1 to iters).iterator.takeWhile(_ => !autoscaleDetected).foreach(iter => {
          log.info(s"driver expecting $numWorkers connections for iteration $iter")
          currentIter(0) = iter
          for (worker <- 1 to numWorkers) {
            log.info(s"driver accepting a new connection from $worker")
            val driverSocket = driverServerSocket.accept()
            val reader = new BufferedReader(new InputStreamReader(driverSocket.getInputStream))
            val status = reader.readLine()
            // TODO: If status is finished with early stop, exit without interrupt
            log.info(s"driver received status from worker: $status")
          }
          autoscaleDetected = !listenerQueue.isEmpty
          val driverStatus =
            if (autoscaleDetected) {
              // Tell executors to save learner and exit
              log.info(s"driver detected spark cluster autoscale")
              df.sparkSession.sparkContext.removeSparkListener(executorListener)
              AUTOSCALE_INTERRUPT
            } else {
              CONTINUE_TRAINING
            }
          // Return status
          hostAndPorts.foreach(hostAndPort => {
            val writer = new BufferedWriter(new OutputStreamWriter(hostAndPort._1.getOutputStream))
            writer.write(driverStatus + "\n")
            writer.flush()
          })
        })
      }
      log.info("driver closing all sockets and server socket")
      hostAndPorts.foreach(_._1.close())
      driverServerSocket.close()
    }
    val host = getDriverHost(df)
    val port = driverServerSocket.getLocalPort
    log.info(s"driver waiting for connections on host: ${host} and port: $port")
    (host, port, f, currentIter)
  }

  /** Returns an integer ID for the current node.
    * @return In cluster, returns the executor id.  In local case, returns the worker id.
    */
  def getId(): Int = {
    val executorId = SparkEnv.get.executorId
    val ctx = TaskContext.get
    val partId = ctx.partitionId
    // If driver, this is only in test scenario, make each partition a separate worker
    val id = if (executorId == "driver") partId else executorId
    val idAsInt = id.toString.toInt
    idAsInt
  }

  def getHostToIP(hostname: String): String = {
    if (InetAddressUtils.isIPv4Address(hostname) || InetAddressUtils.isIPv6Address(hostname))
      hostname
    else
      InetAddress.getByName(hostname).getHostAddress
  }

  /** Get number of cores from dummy dataset for 1 executor.
    * Note: all executors have same number of cores,
    * and this is more reliable than getting value from conf.
    * @param dataset The dataset containing the current spark session.
    * @return The number of cores per executor.
    */
  def getNumCoresPerExecutor(dataset: Dataset[_]): Int = {
    val spark = dataset.sparkSession
    try {
      val confCores = spark.sparkContext.getConf
        .get("spark.executor.cores").toInt
      val confTaskCpus = spark.sparkContext.getConf
        .get("spark.task.cpus", "1").toInt
      confCores / confTaskCpus
    } catch {
      case _: NoSuchElementException =>
        // If spark.executor.cores is not defined, get the cores per JVM
        import spark.implicits._
        val numMachineCores = spark.range(0, 1)
          .map(_ => java.lang.Runtime.getRuntime.availableProcessors).collect.head
        numMachineCores
    }
  }

  private def getDriverHost(dataset: Dataset[_]): String = {
    val blockManager = BlockManagerUtils.getBlockManager(dataset)
    blockManager.master.getMemoryStatus.toList.flatMap({ case (blockManagerId, _) =>
      if (blockManagerId.executorId == "driver") Some(getHostToIP(blockManagerId.host))
      else None
    }).head
  }

  /** Returns a list of executor id and host.
    * @param dataset The dataset containing the current spark session.
    * @param numCoresPerExec The number of cores per executor.
    * @return List of executors as an array of (id,host).
    */
  def getExecutors(dataset: Dataset[_], numCoresPerExec: Int): Array[(Int, String)] = {
    val blockManager = BlockManagerUtils.getBlockManager(dataset)
    blockManager.master.getMemoryStatus.toList.flatMap({ case (blockManagerId, _) =>
      if (blockManagerId.executorId == "driver") None
      else Some((blockManagerId.executorId.toInt, getHostToIP(blockManagerId.host)))
    }).toArray
  }

  /** Returns the number of executors * number of cores.
    * @param dataset The dataset containing the current spark session.
    * @param numCoresPerExec The number of cores per executor.
    * @return The number of executors * number of cores.
    */
  def getNumExecutorCores(dataset: Dataset[_], numCoresPerExec: Int): Int = {
    val executors = getExecutors(dataset, numCoresPerExec)
    if (!executors.isEmpty) {
      executors.length * numCoresPerExec
    } else {
      val master = dataset.sparkSession.sparkContext.master
      val rx = "local(?:\\[(\\*|\\d+)(?:,\\d+)?\\])?".r
      master match {
        case rx(null)  => 1
        case rx("*")   => Runtime.getRuntime.availableProcessors()
        case rx(cores) => cores.toInt
        case _         => BlockManagerUtils.getBlockManager(dataset)
                            .master.getMemoryStatus.size
      }
    }
  }

  /** Converts a host,id pair to the lightGBM host:port format.
    * @param hostAndId The host,id.
    * @param defaultListenPort The default listen port.
    * @return The string lightGBM representation of host:port.
    */
  def toAddr(hostAndId: (String, Int), defaultListenPort: Int): String =
    hostAndId._1 + ":" + (defaultListenPort + hostAndId._2)

  /** Returns the executor node ips and ports.
    * @param data The input dataframe.
    * @param defaultListenPort The default listen port.
    * @param numCoresPerExec The number of cores per executor.
    * @return List of nodes as comma separated string and count.
    */
  def getNodes(data: DataFrame, defaultListenPort: Int, numCoresPerExec: Int): Array[(Int, String)] = {
    val nodes = getExecutors(data, numCoresPerExec)
    val getSubsetExecutors = data.rdd.getNumPartitions < nodes.length * numCoresPerExec
    if (nodes.isEmpty) {
      // Running in local[*]
      getNodesFromPartitionsLocal(data, defaultListenPort)
    } else if (getSubsetExecutors) {
      // Special case when num partitions < num executors * numCoresPerExec
      getNodesFromPartitions(data, defaultListenPort, nodes.toMap)
    } else {
      // Running on cluster, include all workers with driver excluded
      nodes
    }
  }

  /** Returns the nodes from mapPartitions.
    * Only run in case when num partitions < num executors.
    * @param processedData The input data.
    * @param defaultListenPort The default listening port.
    * @param executorToHost Map from executor id to host name.
    * @return The list of nodes in host:port format.
    */
  def getNodesFromPartitions(processedData: DataFrame, defaultListenPort: Int,
                             executorToHost: Map[Int, String]): Array[(Int, String)] = {
    import processedData.sparkSession.implicits._
    val nodes =
      processedData.mapPartitions((_: Iterator[Row]) => {
        val id = getId()
        Array((id, executorToHost(id))).toIterator
      }).collect()
    nodes
  }

  /** Returns the nodes from mapPartitions.
    * Only run in local[*] case.
    * @param processedData The input data.
    * @param defaultListenPort The default listening port.
    * @return The list of nodes in host:port format.
    */
  def getNodesFromPartitionsLocal(processedData: DataFrame,
                                  defaultListenPort: Int): Array[(Int, String)] = {
    import processedData.sparkSession.implicits._
    val blockManager = BlockManagerUtils.getBlockManager(processedData)
    val host = blockManager.master.getMemoryStatus.flatMap({ case (blockManagerId, _) =>
      Some(getHostToIP(blockManagerId.host))
    }).head
    val nodes =
      processedData.mapPartitions((_: Iterator[Row]) => {
        // The logic below is to get it to run in local[*] spark context
        val id = getId()
        Array((id, host)).toIterator
      }).collect()
    nodes
  }

  def newDoubleArray(array: Array[Double]): (SWIGTYPE_p_void, SWIGTYPE_p_double) = {
    val data = lightgbmlib.new_doubleArray(array.length)
    array.zipWithIndex.foreach {
      case (value, index) => lightgbmlib.doubleArray_setitem(data, index, value)
    }
    (lightgbmlib.double_to_voidp_ptr(data), data)
  }

  def newIntArray(array: Array[Int]): (SWIGTYPE_p_int32_t, SWIGTYPE_p_int) = {
    val data = lightgbmlib.new_intArray(array.length)
    array.zipWithIndex.foreach {
      case (value, index) => lightgbmlib.intArray_setitem(data, index, value)
    }
    (lightgbmlib.int_to_int32_t_ptr(data), data)
  }

  def intToPtr(value: Int): SWIGTYPE_p_int64_t = {
    val longPtr = lightgbmlib.new_longp()
    lightgbmlib.longp_assign(longPtr, value)
    lightgbmlib.long_to_int64_t_ptr(longPtr)
  }

  def generateData(numRows: Int, rowsAsDoubleArray: Array[Array[Double]]):
      (SWIGTYPE_p_void, SWIGTYPE_p_double) = {
    val numCols = rowsAsDoubleArray.head.length
    val data = lightgbmlib.new_doubleArray(numCols * numRows)
    rowsAsDoubleArray.zipWithIndex.foreach(ri =>
      ri._1.zipWithIndex.foreach(value =>
        lightgbmlib.doubleArray_setitem(data, value._2 + (ri._2 * numCols), value._1)))
    (lightgbmlib.double_to_voidp_ptr(data), data)
  }

  def generateDenseDataset(numRows: Int, rowsAsDoubleArray: Array[Array[Double]]):
      SWIGTYPE_p_void = {
    val numRowsIntPtr = lightgbmlib.new_intp()
    lightgbmlib.intp_assign(numRowsIntPtr, numRows)
    val numRows_int32_tPtr = lightgbmlib.int_to_int32_t_ptr(numRowsIntPtr)
    val numCols = rowsAsDoubleArray.head.length
    val isRowMajor = 1
    val numColsIntPtr = lightgbmlib.new_intp()
    lightgbmlib.intp_assign(numColsIntPtr, numCols)
    val numCols_int32_tPtr = lightgbmlib.int_to_int32_t_ptr(numColsIntPtr)
    val datasetOutPtr = lightgbmlib.voidpp_handle()
    val datasetParams = "max_bin=255 is_pre_partition=True"
    val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64
    var data: Option[(SWIGTYPE_p_void, SWIGTYPE_p_double)] = None
    try {
      data = Some(generateData(numRows, rowsAsDoubleArray))
      // Generate the dataset for features
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromMat(
                               data.get._1, data64bitType,
                               numRows_int32_tPtr, numCols_int32_tPtr,
                               isRowMajor, datasetParams, null, datasetOutPtr),
                             "Dataset create")
    } finally {
      if (data.isDefined) lightgbmlib.delete_doubleArray(data.get._2)
    }
    lightgbmlib.voidpp_value(datasetOutPtr)
  }

  /** Generates a sparse dataset in CSR format.
    * @param sparseRows The rows of sparse vector.
    * @return
    */
  def generateSparseDataset(sparseRows: Array[SparseVector]): SWIGTYPE_p_void = {
    var values: Option[(SWIGTYPE_p_void, SWIGTYPE_p_double)] = None
    var indexes: Option[(SWIGTYPE_p_int32_t, SWIGTYPE_p_int)] = None
    var indptrNative: Option[(SWIGTYPE_p_int32_t, SWIGTYPE_p_int)] = None
    try {
      val valuesArray = sparseRows.flatMap(_.values)
      values = Some(newDoubleArray(valuesArray))
      val indexesArray = sparseRows.flatMap(_.indices)
      indexes = Some(newIntArray(indexesArray))
      val indptr = new Array[Int](sparseRows.length + 1)
      sparseRows.zipWithIndex.foreach {
        case (row, index) => indptr(index + 1) = indptr(index) + row.numNonzeros
      }
      indptrNative = Some(newIntArray(indptr))
      val numCols = sparseRows(0).size

      val datasetOutPtr = lightgbmlib.voidpp_handle()
      val datasetParams = "max_bin=255 is_pre_partition=True"
      val dataInt32bitType = lightgbmlibConstants.C_API_DTYPE_INT32
      val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64

      // Generate the dataset for features
      LightGBMUtils.validate(CSRUtils.LGBM_DatasetCreateFromCSR(
                               indptrNative.get._1, dataInt32bitType,
                               indexes.get._1, values.get._1, data64bitType,
                               intToPtr(indptr.length), intToPtr(valuesArray.length),
                               intToPtr(numCols), datasetParams, null,
                               datasetOutPtr),
                             "Dataset create")
      lightgbmlib.voidpp_value(datasetOutPtr)
    } finally {
      // Delete the input rows
      if (values.isDefined)  lightgbmlib.delete_doubleArray(values.get._2)
      if (indexes.isDefined) lightgbmlib.delete_intArray(indexes.get._2)
      if (indptrNative.isDefined) lightgbmlib.delete_intArray(indptrNative.get._2)
    }
  }

}
