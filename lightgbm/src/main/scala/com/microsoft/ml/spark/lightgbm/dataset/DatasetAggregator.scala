// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm.dataset

import java.util.concurrent.atomic.AtomicLong

import com.microsoft.ml.spark.lightgbm.ColumnParams
import com.microsoft.ml.spark.lightgbm.swig._
import org.apache.spark.sql.Row

import scala.collection.mutable.ListBuffer

private[lightgbm] object ChunkedArrayUtils {
  def copyChunkedArray[T: Numeric](chunkedArray: ChunkedArray[T],
                                   mainArray: BaseSwigArray[T],
                                   threadRowStartIndex: Long,
                                   chunkSize: Long): Unit = {
    val num = implicitly[Numeric[T]]
    val defaultVal = num.fromInt(-1)
    // Copy in parallel on each thread
    // First copy full chunks
    val chunkCount = chunkedArray.getChunksCount() - 1
    for (chunk <- 0L until chunkCount) {
      for (inChunkIdx <- 0L until chunkSize) {
        mainArray.setItem(threadRowStartIndex + chunk * chunkSize + inChunkIdx,
          chunkedArray.getItem(chunk, inChunkIdx, defaultVal))
      }
    }
    // Next copy filled values from last chunk only
    val lastChunkCount = chunkedArray.getLastChunkAddCount()
    for (lastChunkIdx <- 0L until lastChunkCount) {
      mainArray.setItem(threadRowStartIndex + chunkCount * chunkSize + lastChunkIdx,
        chunkedArray.getItem(chunkCount, lastChunkIdx, defaultVal))
    }
  }
}

private[lightgbm] trait AggregatedColumns

private[lightgbm] abstract class BaseChunkedColumns(columnParams: ColumnParams,
                                                    chunkSize: Int) extends AggregatedColumns {
  protected val labels: FloatChunkedArray = new FloatChunkedArray(chunkSize)
  protected val weights: Option[FloatChunkedArray] = columnParams.weightColumn.map {
    _ => new FloatChunkedArray(chunkSize)
  }
  protected val initScores: Option[DoubleChunkedArray] = columnParams.initScoreColumn.map {
    _ => new DoubleChunkedArray(chunkSize)
  }
  protected val groups: ListBuffer[Row] = new ListBuffer[Row]()

  protected var rowCount = 0

  def addRow(): Unit = {
    rowCount += 1
  }

  def release(): Unit = {
    // Clear memory
    labels.delete()
    weights.foreach(_.delete())
    initScores.foreach(_.delete())
  }

  def getRowCount: Long = rowCount
}

private[lightgbm] final class SparseChunkedColumns(columnParams: ColumnParams, chunkSize: Int)
  extends BaseChunkedColumns(columnParams, chunkSize) {

  protected var indexesChunkedArray = new IntChunkedArray(chunkSize)
  protected var valuesChunkedArray = new DoubleChunkedArray(chunkSize)
  protected var indptrChunkedArray = new IntChunkedArray(chunkSize)

  private var numCols = 0

  def setNumCols(numCols: Int): Unit = {
    this.numCols = numCols
  }

  def getNumCols: Int = numCols

  override def release(): Unit = {
    // Clear memory
    super.release()
    indexesChunkedArray.delete()
    valuesChunkedArray.delete()
    indptrChunkedArray.delete()
  }
}

private[lightgbm] final class DenseChunkedColumns(columnParams: ColumnParams,
                                                  chunkSize: Int,
                                                  val numCols: Int)
  extends BaseChunkedColumns(columnParams, chunkSize) {
  var featuresChunkedArray = new DoubleChunkedArray(numCols * chunkSize)

  override def release(): Unit = {
    // Clear memory
    super.release()
    featuresChunkedArray.delete()
  }
}

private[lightgbm] abstract class BaseAggregatedColumns extends AggregatedColumns {
  protected var labelsArray: FloatSwigArray = _
  protected var weightArrayOpt: Option[FloatSwigArray] = None
  protected var initScoreArrayOpt: Option[DoubleSwigArray] = None
  protected var groupColumnValuesArray: Array[Row] = _

  /**
    * Variables for knowing how large full array should be allocated to
    */
  var rowCount = new AtomicLong(0L)
  var initScoreCount = new AtomicLong(0L)

  protected var numCols = 0

  def getNumCols: Int = numCols
}

private[lightgbm] abstract class BaseDenseAggregatedColumns extends BaseAggregatedColumns {
  protected var featuresArray: DoubleSwigArray = _

  def incrementCount(rowCount: Long,
                     initScoreCount: Long): Unit = {
    this.rowCount.addAndGet(rowCount)
    this.initScoreCount.addAndGet(initScoreCount)
  }

  def addRows(labels: FloatChunkedArray,
              weights: Option[FloatChunkedArray],
              initScores: Option[DoubleChunkedArray],
              features: DoubleChunkedArray,
              groups: ListBuffer[Row],
              numCols: Int): Unit

  protected def initializeRows(weights: Option[FloatChunkedArray],
                               initScores: Option[DoubleChunkedArray],
                               numCols: Int): Unit = {
    this.numCols = numCols
    val rowCount = this.rowCount.get()
    val initScoreCount = this.initScoreCount.get()
    labelsArray = new FloatSwigArray(rowCount)
    weightArrayOpt = weights.map(_ => new FloatSwigArray(rowCount))
    initScoreArrayOpt = initScores.map(_ => new DoubleSwigArray(initScoreCount))
    featuresArray = new DoubleSwigArray(numCols * rowCount)
    groupColumnValuesArray = new Array[Row](rowCount.toInt)
  }
}

private[lightgbm] final class DenseAggregatedColumns(chunkSize: Int) extends BaseDenseAggregatedColumns {
  /** Adds the rows to the internal data structure.
    * @param labels The column of label values.
    * @param weights The optional column of weights, if specified.
    * @param initScores The optional column of initial scores, if specified.
    * @param features The features vector.
    * @param groups The column of group values, if in ranking scenario.
    */
  def addRows(labels: FloatChunkedArray,
              weights: Option[FloatChunkedArray],
              initScores: Option[DoubleChunkedArray],
              features: DoubleChunkedArray,
              groups: ListBuffer[Row],
              numCols: Int): Unit = {
    initializeRows(weights, initScores, numCols)
    // Coalesce to main arrays passed to dataset create
    labels.coalesceTo(this.labelsArray)
    weights.foreach(_.coalesceTo(this.weightArrayOpt.get))
    initScores.foreach(_.coalesceTo(this.initScoreArrayOpt.get))
    features.coalesceTo(this.featuresArray)
    groups.copyToArray(groupColumnValuesArray)
  }
}

/** Defines class for aggregating rows to a single structure before creating the native LightGBMDataset.
  * @param chunkSize The chunk size for the chunked arrays.
  */
private[lightgbm] final class DenseSyncAggregatedColumns(chunkSize: Int) extends BaseDenseAggregatedColumns {
  /**
    * Variables for current thread to use in order to update common arrays in parallel
    */
  protected var threadRowStartIndex = new AtomicLong(0L)
  protected var threadInitScoreStartIndex = new AtomicLong(0L)

  /** Adds the rows to the internal data structure.
    * @param labels The column of label values.
    * @param weights The optional column of weights, if specified.
    * @param initScores The optional column of initial scores, if specified.
    * @param features The features vector.
    * @param groups The column of group values, if in ranking scenario.
    */
  def addRows(labels: FloatChunkedArray,
              weights: Option[FloatChunkedArray],
              initScores: Option[DoubleChunkedArray],
              features: DoubleChunkedArray,
              groups: ListBuffer[Row],
              numCols: Int): Unit = {
    parallelInitializeRows(weights, initScores, numCols)
    parallelizedCopy(labels, weights, initScores, features,
      groups, numCols)
  }

  private def parallelInitializeRows(weightChunkedArrayOpt: Option[FloatChunkedArray],
                                     initScoreChunkedArrayOpt: Option[DoubleChunkedArray],
                                     numCols: Int): Unit = {
    // Initialize arrays if they are not defined - first thread to get here does the initialization for all of them
    if (labelsArray == null) {
      this.synchronized {
        if (labelsArray == null) {
          initializeRows(weightChunkedArrayOpt, initScoreChunkedArrayOpt, numCols)
        }
      }
    }
  }

  private def parallelizedCopy(labelsChunkedArray: FloatChunkedArray,
                               weightChunkedArrayOpt: Option[FloatChunkedArray],
                               initScoreChunkedArrayOpt: Option[DoubleChunkedArray],
                               featuresChunkedArray: DoubleChunkedArray,
                               groupColumnValues: ListBuffer[Row],
                               numCols: Int): Unit = {
    // Parallelized copy to common arrays
    var threadRowStartIndex = 0L
    var threadInitScoreStartIndex = 0L
    this.synchronized {
      val labelsSize = labelsChunkedArray.getAddCount()
      threadRowStartIndex = this.threadRowStartIndex.getAndAdd(labelsSize.toInt)
      val initScoreSize = initScoreChunkedArrayOpt.map(_.getAddCount())
      initScoreSize.foreach(size => threadInitScoreStartIndex = this.threadInitScoreStartIndex.getAndAdd(size))
    }
    ChunkedArrayUtils.copyChunkedArray(labelsChunkedArray, this.labelsArray, threadRowStartIndex, chunkSize)
    weightChunkedArrayOpt.foreach {
      weightChunkedArray =>
        ChunkedArrayUtils.copyChunkedArray(weightChunkedArray, this.weightArrayOpt.get, threadRowStartIndex,
          chunkSize)
    }
    initScoreChunkedArrayOpt.foreach {
      initScoreChunkedArray =>
        ChunkedArrayUtils.copyChunkedArray(initScoreChunkedArray, this.initScoreArrayOpt.get,
          threadInitScoreStartIndex, chunkSize)
    }
    ChunkedArrayUtils.copyChunkedArray(featuresChunkedArray, this.featuresArray, threadRowStartIndex * numCols,
      chunkSize)
    groupColumnValues.copyToArray(groupColumnValuesArray, threadRowStartIndex.toInt)
    // rewrite array reference for volatile arrays, see: https://www.javamex.com/tutorials/volatile_arrays.shtml
    this.synchronized {
      groupColumnValuesArray = groupColumnValuesArray
    }
  }
}

private[lightgbm] abstract class BaseSparseAggregatedColumns extends BaseAggregatedColumns {
  protected var indexesArray: IntSwigArray = _
  protected var valuesArray: DoubleSwigArray = _
  protected var indptrArray: IntSwigArray = _

  /**
    * Aggregated variables for knowing how large full array should be allocated to
    */
  protected var indexesCount = new AtomicLong(0L)
  protected var indptrCount = new AtomicLong(0L)

  def setNumCols(numCols: Int): Unit = {
    this.numCols = numCols
  }

  def incrementCount(rowCount: Long,
                     initScoreCount: Long,
                     indexesCount: Long,
                     indptrCount: Long): Unit = {
    this.rowCount.addAndGet(rowCount)
    this.initScoreCount.addAndGet(initScoreCount)
    this.indexesCount.addAndGet(indexesCount)
    this.indptrCount.addAndGet(indptrCount)
  }

  def addRows(labelsChunkedArray: FloatChunkedArray,
              weightChunkedArrayOpt: Option[FloatChunkedArray],
              initScoreChunkedArrayOpt: Option[DoubleChunkedArray],
              indexesChunkedArray: IntChunkedArray,
              valuesChunkedArray: DoubleChunkedArray,
              indptrChunkedArray: IntChunkedArray,
              groupColumnValues: ListBuffer[Row]): Unit

  protected def initializeRows(weightChunkedArrayOpt: Option[FloatChunkedArray],
                             initScoreChunkedArrayOpt: Option[DoubleChunkedArray]): Unit = {
    val rowCount = this.rowCount.get()
    val initScoreCount = this.initScoreCount.get()
    val indexesCount = this.indexesCount.get()
    val indptrCount = this.indptrCount.get()
    labelsArray = new FloatSwigArray(rowCount)
    weightArrayOpt = weightChunkedArrayOpt.map(_ => new FloatSwigArray(rowCount))
    initScoreArrayOpt = initScoreChunkedArrayOpt.map(_ => new DoubleSwigArray(initScoreCount))
    indexesArray = new IntSwigArray(indexesCount)
    valuesArray = new DoubleSwigArray(indexesCount)
    indptrArray = new IntSwigArray(indptrCount)
    indptrArray.setItem(0, 0)
    groupColumnValuesArray = new Array[Row](rowCount.toInt)
  }
}


/** Defines class for aggregating rows to a single structure before creating the native LightGBMDataset.
  * @param chunkSize The chunk size for the chunked arrays.
  */
private[lightgbm] final class SparseAggregatedColumns(chunkSize: Int) extends BaseSparseAggregatedColumns {
  /** Adds the rows to the internal data structure.
    * @param labelsChunkedArray The column of label values.
    * @param weightChunkedArrayOpt The optional column of weights, if specified.
    * @param initScoreChunkedArrayOpt The optional column of initial scores, if specified.
    * @param indexesChunkedArray The feature SparseVector indexes.
    * @param valuesChunkedArray The feature SparseVector values.
    * @param indptrChunkedArray The feature SparseVector indptr.
    * @param groupColumnValues The column of group values, if in ranking scenario.
    */
  def addRows(labelsChunkedArray: FloatChunkedArray,
              weightChunkedArrayOpt: Option[FloatChunkedArray],
              initScoreChunkedArrayOpt: Option[DoubleChunkedArray],
              indexesChunkedArray: IntChunkedArray,
              valuesChunkedArray: DoubleChunkedArray,
              indptrChunkedArray: IntChunkedArray,
              groupColumnValues: ListBuffer[Row]): Unit = {
    initializeRows(weightChunkedArrayOpt, initScoreChunkedArrayOpt)

    // Coalesce to main arrays passed to dataset create
    labelsChunkedArray.coalesceTo(this.labelsArray)
    weightChunkedArrayOpt.foreach(_.coalesceTo(this.weightArrayOpt.get))
    initScoreChunkedArrayOpt.foreach(_.coalesceTo(this.initScoreArrayOpt.get))
    indexesChunkedArray.coalesceTo(this.indexesArray)
    valuesChunkedArray.coalesceTo(this.valuesArray)
    indptrChunkedArray.coalesceTo(this.indptrArray)
    groupColumnValues.copyToArray(groupColumnValuesArray)
  }
}

/** Defines class for aggregating rows to a single structure before creating the native LightGBMDataset.
  * @param chunkSize The chunk size for the chunked arrays.
  */
private[lightgbm] final class SparseSyncAggregatedColumns(chunkSize: Int) extends BaseSparseAggregatedColumns {
  /**
    * Variables for current thread to use in order to update common arrays in parallel
    */
  protected val threadRowStartIndex = new AtomicLong(0L)
  protected val threadInitScoreStartIndex = new AtomicLong(0L)
  protected val threadIndexesStartIndex = new AtomicLong(0L)
  protected val threadIndptrStartIndex = new AtomicLong(1L)

  private def parallelInitializeRows(weightChunkedArrayOpt: Option[FloatChunkedArray],
                                     initScoreChunkedArrayOpt: Option[DoubleChunkedArray]): Unit = {
    // Initialize arrays if they are not defined - first thread to get here does the initialization for all of them
    if (labelsArray == null) {
      this.synchronized {
        if (labelsArray == null) {
          // Add extra 0 for start of indptr in parallel case
          this.indptrCount.addAndGet(1L)
          initializeRows(weightChunkedArrayOpt, initScoreChunkedArrayOpt)
        }
      }
    }
  }

  def parallelizedCopy(labelsChunkedArray: FloatChunkedArray,
                       weightChunkedArrayOpt: Option[FloatChunkedArray],
                       initScoreChunkedArrayOpt: Option[DoubleChunkedArray],
                       indexesChunkedArray: IntChunkedArray,
                       valuesChunkedArray: DoubleChunkedArray,
                       indptrChunkedArray: IntChunkedArray,
                       groupColumnValues: ListBuffer[Row]): Unit = {
    // Parallelized copy to common arrays
    var threadRowStartIndex = 0L
    var threadInitScoreStartIndex = 0L
    var threadIndexesStartIndex = 0L
    var threadIndPtrStartIndex = 0L
    this.synchronized {
      val labelsSize = labelsChunkedArray.getAddCount()
      threadRowStartIndex = this.threadRowStartIndex.getAndAdd(labelsSize.toInt)

      val initScoreSize = initScoreChunkedArrayOpt.map(_.getAddCount())
      initScoreSize.foreach(size => threadInitScoreStartIndex = this.threadInitScoreStartIndex.getAndAdd(size))

      val indexesSize = indexesChunkedArray.getAddCount()
      threadIndexesStartIndex = this.threadIndexesStartIndex.getAndAdd(indexesSize)

      val indPtrSize = indptrChunkedArray.getAddCount()
      threadIndPtrStartIndex = this.threadIndptrStartIndex.getAndAdd(indPtrSize)
    }
    ChunkedArrayUtils.copyChunkedArray(labelsChunkedArray, this.labelsArray, threadRowStartIndex, chunkSize)
    weightChunkedArrayOpt.foreach {
      weightChunkedArray =>
        ChunkedArrayUtils.copyChunkedArray(weightChunkedArray, this.weightArrayOpt.get, threadRowStartIndex,
          chunkSize)
    }
    initScoreChunkedArrayOpt.foreach {
      initScoreChunkedArray =>
        ChunkedArrayUtils.copyChunkedArray(initScoreChunkedArray, this.initScoreArrayOpt.get,
          threadInitScoreStartIndex, chunkSize)
    }
    ChunkedArrayUtils.copyChunkedArray(indexesChunkedArray, this.indexesArray, threadIndexesStartIndex, chunkSize)
    ChunkedArrayUtils.copyChunkedArray(valuesChunkedArray, this.valuesArray, threadIndexesStartIndex, chunkSize)
    ChunkedArrayUtils.copyChunkedArray(indptrChunkedArray, this.indptrArray, threadIndPtrStartIndex, chunkSize)
    groupColumnValues.copyToArray(groupColumnValuesArray, threadRowStartIndex.toInt)
    // rewrite array reference for volatile arrays, see: https://www.javamex.com/tutorials/volatile_arrays.shtml
    this.synchronized {
      groupColumnValuesArray = groupColumnValuesArray
    }
  }

  /** Adds the rows to the internal data structure.
    * @param labelsChunkedArray The column of label values.
    * @param weightChunkedArrayOpt The optional column of weights, if specified.
    * @param initScoreChunkedArrayOpt The optional column of initial scores, if specified.
    * @param indexesChunkedArray The feature SparseVector indexes.
    * @param valuesChunkedArray The feature SparseVector values.
    * @param indptrChunkedArray The feature SparseVector indptr.
    * @param groupColumnValues The column of group values, if in ranking scenario.
    */
  def addRows(labelsChunkedArray: FloatChunkedArray,
              weightChunkedArrayOpt: Option[FloatChunkedArray],
              initScoreChunkedArrayOpt: Option[DoubleChunkedArray],
              indexesChunkedArray: IntChunkedArray,
              valuesChunkedArray: DoubleChunkedArray,
              indptrChunkedArray: IntChunkedArray,
              groupColumnValues: ListBuffer[Row]): Unit = {
    parallelInitializeRows(weightChunkedArrayOpt, initScoreChunkedArrayOpt)
    parallelizedCopy(labelsChunkedArray, weightChunkedArrayOpt, initScoreChunkedArrayOpt, indexesChunkedArray,
      valuesChunkedArray, indptrChunkedArray, groupColumnValues)
  }
}
