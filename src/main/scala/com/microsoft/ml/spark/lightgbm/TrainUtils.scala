// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import java.io._
import java.net._

import com.microsoft.ml.lightgbm._
import com.microsoft.ml.spark.core.env.StreamUtilities._
import com.microsoft.ml.spark.downloader.FaultToleranceUtils
import org.apache.spark.{BarrierTaskContext, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.slf4j.Logger

import scala.collection.mutable.ListBuffer

case class NetworkParams(defaultListenPort: Int, addr: String, port: Int, barrierExecutionMode: Boolean)
case class ColumnParams(labelColumn: String, featuresColumn: String, weightColumn: Option[String],
                        initScoreColumn: Option[String], groupColumn: Option[String])

private object TrainUtils extends Serializable {

  def generateDataset(rowsIter: Iterator[Row], columnParams: ColumnParams,
                      referenceDataset: Option[LightGBMDataset], schema: StructType,
                      log: Logger, trainParams: TrainParams): Option[LightGBMDataset] = {
    val hrow = rowsIter.next()
    var datasetPtr: Option[LightGBMDataset] = None
    if (hrow.get(schema.fieldIndex(columnParams.featuresColumn)).isInstanceOf[DenseVector]) {
      datasetPtr = aggregateDenseStreamedData(hrow, rowsIter, columnParams, referenceDataset, schema, log, trainParams)
      // Validate generated dataset has the correct number of rows and cols
      datasetPtr.get.validateDataset()
    } else {
      val rows = (Iterator[Row](hrow) ++ rowsIter).toArray
      val numRows = rows.length
      val labels = rows.map(row => row.getDouble(schema.fieldIndex(columnParams.labelColumn)))
      val rowsAsSparse = rows.map(row => row.get(schema.fieldIndex(columnParams.featuresColumn)) match {
        case dense: DenseVector => dense.toSparse
        case sparse: SparseVector => sparse
      })
      val numCols = rowsAsSparse(0).size
      val slotNames = getSlotNames(schema, columnParams.featuresColumn, numCols, trainParams)
      log.info(s"LightGBM task generating sparse dataset with $numRows rows and $numCols columns")
      datasetPtr = Some(LightGBMUtils.generateSparseDataset(rowsAsSparse, referenceDataset, slotNames, trainParams))
      // Validate generated dataset has the correct number of rows and cols
      datasetPtr.get.validateDataset()
      datasetPtr.get.addFloatField(labels, "label", numRows)
      columnParams.weightColumn.foreach { col =>
        val weights = rows.map(row => row.getDouble(schema.fieldIndex(col)))
        datasetPtr.get.addFloatField(weights, "weight", numRows)
      }
      addInitScoreColumn(rows, columnParams.initScoreColumn, datasetPtr, numRows, schema)
      addGroupColumn(rows, columnParams.groupColumn, datasetPtr, numRows, schema, None)
    }
    datasetPtr
  }

  def getRowAsDoubleArray(row: Row, columnParams: ColumnParams, schema: StructType): Array[Double] = {
    row.get(schema.fieldIndex(columnParams.featuresColumn)) match {
      case dense: DenseVector => dense.toArray
      case sparse: SparseVector => sparse.toDense.toArray
    }
  }

  def addFeaturesToChunkedArray(featuresChunkedArrayOpt: Option[doubleChunkedArray], numCols: Int,
                  rowAsDoubleArray: Array[Double]): Unit = {
    featuresChunkedArrayOpt.foreach { featuresChunkedArray =>
      rowAsDoubleArray.foreach { doubleVal =>
        featuresChunkedArray.add(doubleVal)
      }
    }
  }

  def addInitScoreColumnRow(initScoreChunkedArrayOpt: Option[doubleChunkedArray], row: Row,
                            columnParams: ColumnParams, schema: StructType): Unit = {
    columnParams.initScoreColumn.foreach { col =>
      val field = schema.fields(schema.fieldIndex(col))
      if (field.dataType == VectorType) {
        val initScores = row.get(schema.fieldIndex(col)).asInstanceOf[DenseVector]
        // Note: rows * # classes in multiclass case
        initScores.values.foreach { rowValue =>
          initScoreChunkedArrayOpt.get.add(rowValue)
        }
      } else {
        val initScore = row.getDouble(schema.fieldIndex(col))
        initScoreChunkedArrayOpt.get.add(initScore)
      }
    }
  }

  def addGroupColumnRow(row: Row, groupColumnValues: ListBuffer[Row],
                        columnParams: ColumnParams, schema: StructType): Unit = {
    columnParams.groupColumn.foreach { col =>
      val colIdx = schema.fieldIndex(col)
      groupColumnValues.append(Row(row.get(colIdx)))
    }
  }

  def releaseArrays(labelsChunkedArray: floatChunkedArray, weightChunkedArrayOpt: Option[floatChunkedArray],
                    initScoreChunkedArrayOpt: Option[doubleChunkedArray]): Unit = {
    labelsChunkedArray.release()
    weightChunkedArrayOpt.foreach(_.release())
    initScoreChunkedArrayOpt.foreach(_.release())
  }

  def aggregateDenseStreamedData(hrow: Row, rowsIter: Iterator[Row], columnParams: ColumnParams,
                                 referenceDataset: Option[LightGBMDataset], schema: StructType,
                                 log: Logger, trainParams: TrainParams): Option[LightGBMDataset] = {
    var numRows = 0
    val defaultChunkSize = 1000
    val labelsChunkedArray = new floatChunkedArray(defaultChunkSize)
    val weightChunkedArrayOpt = columnParams.weightColumn.map { _ => new floatChunkedArray(defaultChunkSize) }
    val initScoreChunkedArrayOpt = columnParams.initScoreColumn.map { _ => new doubleChunkedArray(defaultChunkSize) }
    var featuresChunkedArrayOpt: Option[doubleChunkedArray] = None
    val groupColumnValues: ListBuffer[Row] = new ListBuffer[Row]()
    try {
      var numCols = 0
      while (rowsIter.hasNext || numRows == 0) {
        val row = if (numRows == 0) hrow else rowsIter.next()
        numRows += 1
        labelsChunkedArray.add(row.getDouble(schema.fieldIndex(columnParams.labelColumn)).toFloat)
        columnParams.weightColumn.map { col =>
          weightChunkedArrayOpt.get.add(row.getDouble(schema.fieldIndex(col)).toFloat)
        }
        val rowAsDoubleArray = getRowAsDoubleArray(row, columnParams, schema)
        numCols = rowAsDoubleArray.length
        if (featuresChunkedArrayOpt.isEmpty) {
          featuresChunkedArrayOpt = Some(new doubleChunkedArray(numCols * defaultChunkSize))
        }
        addFeaturesToChunkedArray(featuresChunkedArrayOpt, numCols, rowAsDoubleArray)
        addInitScoreColumnRow(initScoreChunkedArrayOpt, row, columnParams, schema)
        addGroupColumnRow(row, groupColumnValues, columnParams, schema)
      }

      val slotNames = getSlotNames(schema, columnParams.featuresColumn, numCols, trainParams)
      log.info(s"LightGBM task generating dense dataset with $numRows rows and $numCols columns")
      val datasetPtr = Some(LightGBMUtils.generateDenseDataset(numRows, numCols, featuresChunkedArrayOpt.get,
        referenceDataset, slotNames, trainParams, defaultChunkSize))
      datasetPtr.get.addFloatField(labelsChunkedArray, "label", numRows)

      weightChunkedArrayOpt.foreach(datasetPtr.get.addFloatField(_, "weight", numRows))
      initScoreChunkedArrayOpt.foreach(datasetPtr.get.addDoubleField(_, "init_score", numRows))
      val overrideGroupIndex = Some(0)
      addGroupColumn(groupColumnValues.toArray, columnParams.groupColumn, datasetPtr, numRows, schema,
        overrideGroupIndex)
      datasetPtr
    } finally {
      releaseArrays(labelsChunkedArray, weightChunkedArrayOpt, initScoreChunkedArrayOpt)
    }
  }

  trait CardinalityType[T]

  object CardinalityTypes {

    implicit object LongType extends CardinalityType[Long]

    implicit object IntType extends CardinalityType[Int]

    implicit object StringType extends CardinalityType[String]

  }

  import CardinalityTypes._

  def addInitScoreColumn(rows: Array[Row], initScoreColumn: Option[String],
                         datasetPtr: Option[LightGBMDataset], numRows: Int, schema: StructType): Unit = {
    initScoreColumn.foreach { col =>
      val field = schema.fields(schema.fieldIndex(col))
      if (field.dataType == VectorType) {
        val initScores = rows.map(row => row.get(schema.fieldIndex(col)).asInstanceOf[DenseVector])
        // Calculate # rows * # classes in multiclass case
        val initScoresLength = initScores.length
        val totalLength = initScoresLength * initScores(0).size
        val flattenedInitScores = new Array[Double](totalLength)
        initScores.zipWithIndex.foreach { case (rowVector, rowIndex) =>
          rowVector.values.zipWithIndex.foreach { case (rowValue, colIndex) =>
            flattenedInitScores(colIndex * initScoresLength + rowIndex) = rowValue
          }
        }
        datasetPtr.get.addDoubleField(flattenedInitScores, "init_score", numRows)
      } else {
        val initScores = rows.map(row => row.getDouble(schema.fieldIndex(col)))
        datasetPtr.get.addDoubleField(initScores, "init_score", numRows)
      }
    }
  }

  def validateGroupColumn(groupColumn: Option[String], schema: StructType): Unit = {
    groupColumn.foreach { col =>
      val datatype = schema.fields(schema.fieldIndex(col)).dataType

      if (datatype != org.apache.spark.sql.types.IntegerType
        && datatype != org.apache.spark.sql.types.LongType
        && datatype != org.apache.spark.sql.types.StringType) {
        throw new IllegalArgumentException(
          s"group column $col must be of type Long, Int or String but is ${datatype.typeName}")
      }
    }
  }

  def addGroupColumn(rows: Array[Row], groupColumn: Option[String],
                     datasetPtr: Option[LightGBMDataset], numRows: Int,
                     schema: StructType, overrideIdx: Option[Int]): Unit = {
    validateGroupColumn(groupColumn, schema)
    groupColumn.foreach { col =>
      val datatype = schema.fields(schema.fieldIndex(col)).dataType
      val colIdx = if (overrideIdx.isEmpty) schema.fieldIndex(col) else overrideIdx.get

      // Convert to distinct count (note ranker should have sorted within partition by group id)
      // We use a triplet of a list of cardinalities, last unique value and unique value count
      val groupCardinality = datatype match {
        case org.apache.spark.sql.types.IntegerType => countCardinality(rows.map(row => row.getInt(colIdx)))
        case org.apache.spark.sql.types.LongType => countCardinality(rows.map(row => row.getLong(colIdx)))
        case org.apache.spark.sql.types.StringType => countCardinality(rows.map(row => row.getString(colIdx)))
      }

      datasetPtr.get.addIntField(groupCardinality, "group", groupCardinality.length)
    }
  }

  case class CardinalityTriplet[T](groupCounts: List[Int], currentValue: T, currentCount: Int)

  def countCardinality[T](input: Seq[T])(implicit ev: CardinalityType[T]): Array[Int] = {
    val default: T = null.asInstanceOf[T]

    val cardinalityTriplet = input.foldLeft(CardinalityTriplet(List.empty[Int], default, 0)) {
      case (listValue: CardinalityTriplet[T], currentValue) =>

        if (listValue.groupCounts.isEmpty && listValue.currentCount == 0) {
          // Base case, keep list as empty and set cardinality to 1
          CardinalityTriplet(listValue.groupCounts, currentValue, 1)
        }
        else if (listValue.currentValue == currentValue) {
          // Encountered same value
          CardinalityTriplet(listValue.groupCounts, currentValue, listValue.currentCount + 1)
        }
        else {
          // New value, need to reset counter and add new cardinality to list
          CardinalityTriplet(listValue.currentCount :: listValue.groupCounts, currentValue, 1)
        }
    }

    val groupCardinality = (cardinalityTriplet.currentCount :: cardinalityTriplet.groupCounts).reverse.toArray
    groupCardinality
  }

  def createBooster(trainParams: TrainParams, trainDatasetPtr: Option[LightGBMDataset],
                    validDatasetPtr: Option[LightGBMDataset]): Option[SWIGTYPE_p_void] = {
    // Create the booster
    val boosterOutPtr = lightgbmlib.voidpp_handle()
    val parameters = trainParams.toString()
    LightGBMUtils.validate(lightgbmlib.LGBM_BoosterCreate(trainDatasetPtr.map(_.dataset).get,
      parameters, boosterOutPtr), "Booster")
    val boosterPtr = Some(lightgbmlib.voidpp_value(boosterOutPtr))
    trainParams.modelString.foreach { modelStr =>
      val booster = LightGBMUtils.getBoosterPtrFromModelString(modelStr)
      LightGBMUtils.validate(lightgbmlib.LGBM_BoosterMerge(boosterPtr.get, booster), "Booster Merge")
    }
    validDatasetPtr.foreach { lgbmdataset =>
      LightGBMUtils.validate(lightgbmlib.LGBM_BoosterAddValidData(boosterPtr.get,
        lgbmdataset.dataset), "Add Validation Dataset")
    }
    boosterPtr
  }

  def saveBoosterToString(boosterPtr: Option[SWIGTYPE_p_void], log: Logger): String = {
    val bufferLength = LightGBMConstants.DefaultBufferLength
    val bufferOutLengthPtr = lightgbmlib.new_int64_tp()
    lightgbmlib.LGBM_BoosterSaveModelToStringSWIG(boosterPtr.get, 0, -1, 0, bufferLength, bufferOutLengthPtr)
  }

  def getEvalNames(boosterPtr: Option[SWIGTYPE_p_void]): Array[String] = {
    // Need to keep track of best scores for each metric, see callback.py in lightgbm for reference
    // For debugging, can get metric names
    val stringArrayHandle = lightgbmlib.LGBM_BoosterGetEvalNamesSWIG(boosterPtr.get)
    LightGBMUtils.validateArray(stringArrayHandle, "Booster Get Eval Names")
    val evalNames = lightgbmlib.StringArrayHandle_get_strings(stringArrayHandle)
    lightgbmlib.StringArrayHandle_free(stringArrayHandle)
    evalNames
  }

  def beforeTrainIteration(batchIndex: Int, partitionId: Int, curIters: Int, log: Logger,
                           trainParams: TrainParams, boosterPtr: Option[SWIGTYPE_p_void], hasValid: Boolean): Unit = {
    if (trainParams.delegate.isDefined) {
      trainParams.delegate.get.beforeTrainIteration(batchIndex, partitionId, curIters, log, trainParams, boosterPtr,
        hasValid)
    }
  }

  def afterTrainIteration(batchIndex: Int, partitionId: Int, curIters: Int, log: Logger,
                          trainParams: TrainParams, boosterPtr: Option[SWIGTYPE_p_void], hasValid: Boolean,
                          isFinished: Boolean,
                          trainEvalResults: Option[Map[String, Double]],
                          validEvalResults: Option[Map[String, Double]]): Unit = {
    if (trainParams.delegate.isDefined) {
      trainParams.delegate.get.afterTrainIteration(batchIndex, partitionId, curIters, log, trainParams, boosterPtr,
        hasValid, isFinished, trainEvalResults, validEvalResults)
    }
  }

  def getLearningRate(batchIndex: Int, partitionId: Int, curIters: Int, log: Logger, trainParams: TrainParams,
                      previousLearningRate: Double): Double = {
    trainParams.delegate match {
      case Some(delegate) => delegate.getLearningRate(batchIndex, partitionId, curIters, log, trainParams,
          previousLearningRate)
      case None => previousLearningRate
    }
  }

  def trainCore(batchIndex: Int, trainParams: TrainParams, boosterPtr: Option[SWIGTYPE_p_void],
                log: Logger, hasValid: Boolean): Option[Int] = {
    val isFinishedPtr = lightgbmlib.new_intp()
    var isFinished = false
    var iters = 0
    val evalNames = getEvalNames(boosterPtr)
    val evalCounts = evalNames.length
    val bestScore = new Array[Double](evalCounts)
    val bestScores = new Array[Array[Double]](evalCounts)
    val bestIter = new Array[Int](evalCounts)
    val partitionId = TaskContext.getPartitionId
    var learningRate: Double = trainParams.learningRate
    var bestIterResult: Option[Int] = None
    while (!isFinished && iters < trainParams.numIterations) {
      beforeTrainIteration(batchIndex, partitionId, iters, log, trainParams, boosterPtr, hasValid)
      val newLearningRate = getLearningRate(batchIndex, partitionId, iters, log, trainParams,
        learningRate)
      if (newLearningRate != learningRate) {
        log.info(s"LightGBM task calling LGBM_BoosterResetParameter to reset learningRate" +
          s" (newLearningRate: $newLearningRate)")
        LightGBMUtils.validate(lightgbmlib.LGBM_BoosterResetParameter(boosterPtr.get,
          s"learning_rate=$newLearningRate"), "Booster Reset learning_rate Param")
        learningRate = newLearningRate
      }

      try {
        val result = lightgbmlib.LGBM_BoosterUpdateOneIter(boosterPtr.get, isFinishedPtr)
        LightGBMUtils.validate(result, "Booster Update One Iter")
        isFinished = lightgbmlib.intp_value(isFinishedPtr) == 1
        log.info("LightGBM running iteration: " + iters + " with result: " +
          result + " and is finished: " + isFinished)
      } catch {
        case _: java.lang.Exception =>
          isFinished = true
          log.warn("LightGBM reached early termination on one task," +
            " stopping training on task. This message should rarely occur")
      }

      val trainEvalResults: Option[Map[String, Double]] = if (trainParams.isProvideTrainingMetric && !isFinished) {
        val trainResults = lightgbmlib.new_doubleArray(evalNames.length.toLong)
        val dummyEvalCountsPtr = lightgbmlib.new_intp()
        val resultEval = lightgbmlib.LGBM_BoosterGetEval(boosterPtr.get, 0, dummyEvalCountsPtr, trainResults)
        lightgbmlib.delete_intp(dummyEvalCountsPtr)
        LightGBMUtils.validate(resultEval, "Booster Get Train Eval")

        val results: Array[(String, Double)] = evalNames.zipWithIndex.map { case (evalName, index) =>
          val score = lightgbmlib.doubleArray_getitem(trainResults, index.toLong)
          log.info(s"Train $evalName=$score")
          (evalName, score)
        }

        Option(Map(results:_*))
      } else {
        None
      }

      val validEvalResults: Option[Map[String, Double]] = if (hasValid && !isFinished) {
        val evalResults = lightgbmlib.new_doubleArray(evalNames.length.toLong)
        val dummyEvalCountsPtr = lightgbmlib.new_intp()
        val resultEval = lightgbmlib.LGBM_BoosterGetEval(boosterPtr.get, 1, dummyEvalCountsPtr, evalResults)
        lightgbmlib.delete_intp(dummyEvalCountsPtr)
        LightGBMUtils.validate(resultEval, "Booster Get Valid Eval")
        val results: Array[(String, Double)] = evalNames.zipWithIndex.map { case (evalName, index) =>
          val score = lightgbmlib.doubleArray_getitem(evalResults, index.toLong)
          log.info(s"Valid $evalName=$score")
          val cmp =
            if (evalName.startsWith("auc") || evalName.startsWith("ndcg@") || evalName.startsWith("map@") ||
              evalName.startsWith("average_precision"))
              (x: Double, y: Double, tol: Double) => x - y > tol
            else
              (x: Double, y: Double, tol: Double) => x - y < tol
          if (bestScores(index) == null || cmp(score, bestScore(index), trainParams.improvementTolerance)) {
            bestScore(index) = score
            bestIter(index) = iters
            bestScores(index) = evalNames.indices
              .map(j => lightgbmlib.doubleArray_getitem(evalResults, j.toLong)).toArray
          } else if (iters - bestIter(index) >= trainParams.earlyStoppingRound) {
            isFinished = true
            log.info("Early stopping, best iteration is " + bestIter(index))
            bestIterResult = Some(bestIter(index))
          }

          (evalName, score)
        }

        lightgbmlib.delete_doubleArray(evalResults)

        Option(Map(results:_*))
      } else {
        None
      }

      afterTrainIteration(batchIndex, partitionId, iters, log, trainParams, boosterPtr, hasValid, isFinished,
        trainEvalResults, validEvalResults)

      iters = iters + 1
    }
    bestIterResult
  }

  def getSlotNames(schema: StructType, featuresColumn: String, numCols: Int,
                   trainParams: TrainParams): Option[Array[String]] = {
    if (trainParams.featureNames.nonEmpty) {
      Some(trainParams.featureNames)
    } else {
      val featuresSchema = schema.fields(schema.fieldIndex(featuresColumn))
      val metadata = AttributeGroup.fromStructField(featuresSchema)
      if (metadata.attributes.isEmpty) None
      else if (metadata.attributes.get.isEmpty) None
      else {
        val colnames = (0 until numCols).map(_.toString).toArray
        metadata.attributes.get.foreach(attr =>
          attr.index.foreach(index => colnames(index) = attr.name.getOrElse(index.toString)))
        Some(colnames)
      }
    }
  }

  def beforeGenerateTrainDataset(batchIndex: Int, columnParams: ColumnParams, schema: StructType,
                                 log: Logger, trainParams: TrainParams): Unit = {
    if(trainParams.delegate.isDefined) {
      trainParams.delegate.get.beforeGenerateTrainDataset(batchIndex, TaskContext.getPartitionId, columnParams,
        schema, log, trainParams)
    }
  }

  def afterGenerateTrainDataset(batchIndex: Int, columnParams: ColumnParams, schema: StructType,
                                 log: Logger, trainParams: TrainParams): Unit = {
    if(trainParams.delegate.isDefined) {
      trainParams.delegate.get.afterGenerateTrainDataset(batchIndex, TaskContext.getPartitionId, columnParams,
        schema, log, trainParams)
    }
  }

  def beforeGenerateValidDataset(batchIndex: Int, columnParams: ColumnParams, schema: StructType,
                                 log: Logger, trainParams: TrainParams): Unit = {
    if(trainParams.delegate.isDefined) {
      trainParams.delegate.get.beforeGenerateValidDataset(batchIndex, TaskContext.getPartitionId, columnParams,
        schema, log, trainParams)
    }
  }

  def afterGenerateValidDataset(batchIndex: Int, columnParams: ColumnParams, schema: StructType,
                                log: Logger, trainParams: TrainParams): Unit = {
    if (trainParams.delegate.isDefined) {
      trainParams.delegate.get.afterGenerateValidDataset(batchIndex, TaskContext.getPartitionId, columnParams,
        schema, log, trainParams)
    }
  }

  def translate(batchIndex: Int, columnParams: ColumnParams, validationData: Option[Broadcast[Array[Row]]],
                log: Logger, trainParams: TrainParams, returnBooster: Boolean, schema: StructType,
                inputRows: Iterator[Row]): Iterator[LightGBMBooster] = {
    var trainDatasetPtr: Option[LightGBMDataset] = None
    var validDatasetPtr: Option[LightGBMDataset] = None
    try {
      beforeGenerateTrainDataset(batchIndex, columnParams, schema, log, trainParams)
      trainDatasetPtr = generateDataset(inputRows, columnParams, None, schema, log, trainParams)
      afterGenerateTrainDataset(batchIndex, columnParams, schema, log, trainParams)

      if (validationData.isDefined) {
        beforeGenerateValidDataset(batchIndex, columnParams, schema, log, trainParams)
        validDatasetPtr = generateDataset(validationData.get.value.toIterator, columnParams,
          trainDatasetPtr, schema, log, trainParams)
        afterGenerateValidDataset(batchIndex, columnParams, schema, log, trainParams)
      }

      var boosterPtr: Option[SWIGTYPE_p_void] = None
      try {
        boosterPtr = createBooster(trainParams, trainDatasetPtr, validDatasetPtr)
        val bestIterResult = trainCore(batchIndex, trainParams, boosterPtr, log, validDatasetPtr.isDefined)
        if (returnBooster) {
          val model = saveBoosterToString(boosterPtr, log)
          val booster = new LightGBMBooster(model)
          // Set best iteration on booster if hit early stopping criteria in trainCore
          bestIterResult.foreach(booster.setBestIteration(_))
          Iterator.single(booster)
        } else {
          Iterator.empty
        }
      } finally {
        // Free booster
        boosterPtr.foreach { booster =>
          LightGBMUtils.validate(lightgbmlib.LGBM_BoosterFree(booster), "Finalize Booster")
        }
      }
    } finally {
      // Free datasets
      trainDatasetPtr.foreach(_.close())
      validDatasetPtr.foreach(_.close())
    }
  }

  private def findOpenPort(defaultListenPort: Int, numTasksPerExec: Int, log: Logger): Socket = {
    val basePort = defaultListenPort + (LightGBMUtils.getId * numTasksPerExec)
    if (basePort > LightGBMConstants.MaxPort) {
      throw new Exception(s"Error: port $basePort out of range, possibly due to too many executors or unknown error")
    }
    var localListenPort = basePort
    var foundPort = false
    var taskServerSocket: Socket = null
    while (!foundPort) {
      try {
        taskServerSocket = new Socket()
        taskServerSocket.bind(new InetSocketAddress(localListenPort))
        foundPort = true
      } catch {
        case _: IOException =>
          log.warn(s"Could not bind to port $localListenPort...")
          localListenPort += 1
          if (localListenPort > LightGBMConstants.MaxPort) {
            throw new Exception(s"Error: port $basePort out of range, possibly due to networking or firewall issues")
          }
          if (localListenPort - basePort > 1000) {
            throw new Exception("Error: Could not find open port after 1k tries")
          }
      }
    }
    log.info(s"Successfully bound to port $localListenPort")
    taskServerSocket
  }

  def setFinishedStatus(networkParams: NetworkParams,
                        localListenPort: Int, log: Logger): Unit = {
    using(new Socket(networkParams.addr, networkParams.port)) {
      driverSocket =>
        using(new BufferedWriter(new OutputStreamWriter(driverSocket.getOutputStream))) {
          driverOutput =>
            log.info("sending finished status to driver")
            // If barrier execution mode enabled, create a barrier across tasks
            driverOutput.write(s"${LightGBMConstants.FinishedStatus}\n")
            driverOutput.flush()
        }.get
    }.get
  }

  def getNetworkInitNodes(networkParams: NetworkParams,
                          localListenPort: Int, log: Logger,
                          emptyPartition: Boolean): String = {
    using(new Socket(networkParams.addr, networkParams.port)) {
      driverSocket =>
        usingMany(Seq(new BufferedReader(new InputStreamReader(driverSocket.getInputStream)),
          new BufferedWriter(new OutputStreamWriter(driverSocket.getOutputStream)))) {
          io =>
            val driverInput = io(0).asInstanceOf[BufferedReader]
            val driverOutput = io(1).asInstanceOf[BufferedWriter]
            val taskStatus =
              if (emptyPartition) {
                log.info("send empty status to driver")
                LightGBMConstants.IgnoreStatus
              } else {
                val taskHost = driverSocket.getLocalAddress.getHostAddress
                val taskInfo = s"$taskHost:$localListenPort"
                log.info(s"send current task info to driver: $taskInfo ")
                taskInfo
              }
            // Send the current host:port to the driver
            driverOutput.write(s"$taskStatus\n")
            driverOutput.flush()
            // If barrier execution mode enabled, create a barrier across tasks
            if (networkParams.barrierExecutionMode) {
              val context = BarrierTaskContext.get()
              context.barrier()
              if (context.partitionId() == 0) {
                setFinishedStatus(networkParams, localListenPort, log)
              }
            }
            if (taskStatus != LightGBMConstants.IgnoreStatus) {
              // Wait to get the list of nodes from the driver
              val nodes = driverInput.readLine()
              log.info(s"LightGBM worker got nodes for network init: $nodes")
              nodes
            } else {
              taskStatus
            }
        }.get
    }.get
  }

  def networkInit(nodes: String, localListenPort: Int, log: Logger, retry: Int, delay: Long): Unit = {
    try {
      LightGBMUtils.validate(lightgbmlib.LGBM_NetworkInit(nodes, localListenPort,
        LightGBMConstants.DefaultListenTimeout, nodes.split(",").length), "Network init")
    } catch {
      case ex@(_: Exception | _: Throwable) =>
        log.info(s"NetworkInit failed with exception on local port $localListenPort with exception: $ex")
        Thread.sleep(delay)
        if (retry > 0) {
          log.info(s"Retrying NetworkInit with local port $localListenPort")
          networkInit(nodes, localListenPort, log, retry - 1, delay * 2)
        } else {
          log.info(s"NetworkInit reached maximum exceptions on retry: $ex")
          throw ex
        }
    }
  }

  /**
    * Gets the main node's port that will return the LightGBM Booster.
    * Used to minimize network communication overhead in reduce step.
    * @return The main node's port number.
    */
  def getMainWorkerPort(nodes: String, log: Logger): Int = {
    val nodesList = nodes.split(",")
    if (nodesList.length == 0) {
      throw new Exception("Error: could not split nodes list correctly")
    }
    val mainNode = nodesList(0)
    val hostAndPort = mainNode.split(":")
    if (hostAndPort.length != 2) {
      throw new Exception("Error: could not parse main worker host and port correctly")
    }
    val mainHost = hostAndPort(0)
    val mainPort = hostAndPort(1)
    log.info(s"LightGBM setting main worker host: $mainHost and port: $mainPort")
    mainPort.toInt
  }

  def trainLightGBM(batchIndex: Int, networkParams: NetworkParams, columnParams: ColumnParams,
                    validationData: Option[Broadcast[Array[Row]]], log: Logger,
                    trainParams: TrainParams, numTasksPerExec: Int, schema: StructType)
                   (inputRows: Iterator[Row]): Iterator[LightGBMBooster] = {
    val emptyPartition = !inputRows.hasNext
    // Ideally we would start the socket connections in the C layer, this opens us up for
    // race conditions in case other applications open sockets on cluster, but usually this
    // should not be a problem
    val (nodes, localListenPort) = using(findOpenPort(networkParams.defaultListenPort, numTasksPerExec, log)) {
      openPort =>
        val localListenPort = openPort.getLocalPort
        // Initialize the native library
        LightGBMUtils.initializeNativeLibrary()
        log.info(s"LightGBM task connecting to host: ${networkParams.addr} and port: ${networkParams.port}")
        FaultToleranceUtils.retryWithTimeout() {
          (getNetworkInitNodes(networkParams, localListenPort, log, emptyPartition), localListenPort)
        }
    }.get

    if (emptyPartition) {
      log.warn("LightGBM task encountered empty partition, for best performance ensure no partitions empty")
      List[LightGBMBooster]().toIterator
    } else {
      // Initialize the network communication
      log.info(s"LightGBM task listening on: $localListenPort")
      // Return booster only from main worker to reduce network communication overhead
      val mainWorkerPort = getMainWorkerPort(nodes, log)
      val returnBooster = mainWorkerPort == localListenPort
      try {
        val retries = 3
        val initialDelay = 1000L
        networkInit(nodes, localListenPort, log, retries, initialDelay)
        translate(batchIndex, columnParams, validationData, log, trainParams, returnBooster, schema, inputRows)
      } finally {
        // Finalize network when done
        LightGBMUtils.validate(lightgbmlib.LGBM_NetworkFree(), "Finalize network")
      }
    }
  }

}
