// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV}
import org.apache.spark.annotation.Since
import org.apache.spark.ml.linalg.{DenseMatrix, Matrices, Matrix, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.DataType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkException, ml}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.StructType

/**
  * A grid partitioner, which uses a regular grid to partition coordinates.
  *
  * @param rows        Number of rows.
  * @param cols        Number of columns.
  * @param rowsPerPart Number of rows per partition, which may be less at the bottom edge.
  * @param colsPerPart Number of columns per partition, which may be less at the right edge.
  */
private[ml] class GridPartitioner(
  val rows: Int,
  val cols: Int,
  val rowsPerPart: Int,
  val colsPerPart: Int) extends Partitioner {

  require(rows > 0)
  require(cols > 0)
  require(rowsPerPart > 0)
  require(colsPerPart > 0)

  private val rowPartitions = math.ceil(rows * 1.0 / rowsPerPart).toInt
  private val colPartitions = math.ceil(cols * 1.0 / colsPerPart).toInt

  override val numPartitions: Int = rowPartitions * colPartitions

  /**
    * Returns the index of the partition the input coordinate belongs to.
    *
    * @param key The partition id i (calculated through this method for coordinate (i, j) in
    *            `simulateMultiply`, the coordinate (i, j) or a tuple (i, j, k), where k is
    *            the inner index used in multiplication. k is ignored in computing partitions.
    * @return The index of the partition, which the coordinate belongs to.
    */
  override def getPartition(key: Any): Int = {
    key match {
      case i: Int                   => i
      case (i: Int, j: Int)         =>
        getPartitionId(i, j)
      case (i: Int, j: Int, _: Int) =>
        getPartitionId(i, j)
      case _                        =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  private def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    i / rowsPerPart + j / colsPerPart * rowPartitions
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: GridPartitioner =>
        (this.rows == r.rows) && (this.cols == r.cols) &&
          (this.rowsPerPart == r.rowsPerPart) && (this.colsPerPart == r.colsPerPart)
      case _                  =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      rows: java.lang.Integer,
      cols: java.lang.Integer,
      rowsPerPart: java.lang.Integer,
      colsPerPart: java.lang.Integer)
  }
}

private[ml] object GridPartitioner {

  /** Creates a new [[GridPartitioner]] instance. */
  def apply(rows: Int, cols: Int, rowsPerPart: Int, colsPerPart: Int): GridPartitioner = {
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }

  /** Creates a new [[GridPartitioner]] instance with the input suggested number of partitions. */
  def apply(rows: Int, cols: Int, suggestedNumPartitions: Int): GridPartitioner = {
    require(suggestedNumPartitions > 0)
    val scale = 1.0 / math.sqrt(suggestedNumPartitions.toDouble)
    val rowsPerPart = math.round(math.max(scale * rows, 1.0)).toInt
    val colsPerPart = math.round(math.max(scale * cols, 1.0)).toInt
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }
}

/**
  * Represents a distributed matrix in blocks of local matrices.
  *
  * @param blocks       The RDD of sub-matrix blocks ((blockRowIndex, blockColIndex), sub-matrix) that
  *                     form this distributed matrix. If multiple blocks with the same index exist, the
  *                     results for operations like add and multiply will be unpredictable.
  * @param rowsPerBlock Number of rows that make up each block. The blocks forming the final
  *                     rows are not required to have the given number of rows
  * @param colsPerBlock Number of columns that make up each block. The blocks forming the final
  *                     columns are not required to have the given number of columns
  * @param nRows        Number of rows of this matrix. If the supplied value is less than or equal to zero,
  *                     the number of rows will be calculated when `numRows` is invoked.
  * @param nCols        Number of columns of this matrix. If the supplied value is less than or equal to
  *                     zero, the number of columns will be calculated when `numCols` is invoked.
  */
@Since("2.4.0")
class BlockMatrix @Since("2.4.0")(
  @Since("2.4.0") val blocks: Dataset[((Int, Int), Matrix)],
  @Since("2.4.0") val rowsPerBlock: Int,
  @Since("2.4.0") val colsPerBlock: Int,
  private var nRows: Long,
  private var nCols: Long) extends DistributedMatrix {

  private type MatrixBlock = ((Int, Int), Matrix)

  /**
    * Alternate constructor for BlockMatrix without the input of the number of rows and columns.
    *
    * @param blocks       The RDD of sub-matrix blocks ((blockRowIndex, blockColIndex), sub-matrix) that
    *                     form this distributed matrix. If multiple blocks with the same index exist, the
    *                     results for operations like add and multiply will be unpredictable.
    * @param rowsPerBlock Number of rows that make up each block. The blocks forming the final
    *                     rows are not required to have the given number of rows
    * @param colsPerBlock Number of columns that make up each block. The blocks forming the final
    *                     columns are not required to have the given number of columns
    */
  @Since("2.4.0")
  def this(
    blocks: Dataset[((Int, Int), Matrix)],
    rowsPerBlock: Int,
    colsPerBlock: Int) = {
    this(blocks, rowsPerBlock, colsPerBlock, 0L, 0L)
  }

  @Since("2.4.0")
  override def numRows(): Long = {
    if (nRows <= 0L) estimateDim()
    nRows
  }

  @Since("2.4.0")
  override def numCols(): Long = {
    if (nCols <= 0L) estimateDim()
    nCols
  }

  @Since("2.4.0")
  val numRowBlocks: Int = math.ceil(numRows() * 1.0 / rowsPerBlock).toInt
  @Since("2.4.0")
  val numColBlocks: Int = math.ceil(numCols() * 1.0 / colsPerBlock).toInt

  private[ml] def createPartitioner(): GridPartitioner =
    GridPartitioner(numRowBlocks, numColBlocks, suggestedNumPartitions = blocks.rdd.partitions.length)

  import blocks.sparkSession.implicits._

  private lazy val blockInfo: RDD[((Int, Int), (Int, Int))] = blocks.rdd.mapValues(block => (block.numRows, block
    .numCols)).cache()
  private lazy val blockInfoDS = blocks.map(row => (row._1, (row._2.numRows, row._2.numCols))).cache()

  /** Estimates the dimensions of the matrix. */
  private def estimateDim(): Unit = {
    val (rows, cols) = blockInfoDS.map { case ((blockRowIndex, blockColIndex), (m, n)) =>
      (blockRowIndex.toLong * rowsPerBlock + m,
        blockColIndex.toLong * colsPerBlock + n)
    }.reduce { (x0, x1) =>
      (math.max(x0._1, x1._1), math.max(x0._2, x1._2))
    }
    if (nRows <= 0L) nRows = rows
    assert(rows <= nRows, s"The number of rows $rows is more than claimed $nRows.")
    if (nCols <= 0L) nCols = cols
    assert(cols <= nCols, s"The number of columns $cols is more than claimed $nCols.")
  }

  /**
    * Validates the block matrix info against the matrix data (`blocks`) and throws an exception if
    * any error is found.
    */
  @Since("2.4.0")
  def validate(): Unit = {
    println("Validating BlockMatrix...")
    // check if the matrix is larger than the claimed dimensions
    estimateDim()
    println("BlockMatrix dimensions are okay...")

    // Check if there are multiple MatrixBlocks with the same index.
    blockInfo.countByKey().foreach { case (key, cnt) =>
      if (cnt > 1) {
        throw new SparkException(s"Found multiple MatrixBlocks with the indices $key. Please " +
          "remove blocks with duplicate indices.")
      }
    }
    println("MatrixBlock indices are okay...")
    // Check if each MatrixBlock (except edges) has the dimensions rowsPerBlock x colsPerBlock
    // The first tuple is the index and the second tuple is the dimensions of the MatrixBlock
    val dimensionMsg = s"dimensions different than rowsPerBlock: $rowsPerBlock, and " +
      s"colsPerBlock: $colsPerBlock. Blocks on the right and bottom edges can have smaller " +
      s"dimensions. You may use the repartition method to fix this issue."
    blockInfo.foreach { case ((blockRowIndex, blockColIndex), (m, n)) =>
      if ((blockRowIndex < numRowBlocks - 1 && m != rowsPerBlock) ||
        (blockRowIndex == numRowBlocks - 1 && (m <= 0 || m > rowsPerBlock))) {
        throw new SparkException(s"The MatrixBlock at ($blockRowIndex, $blockColIndex) has " +
          dimensionMsg)
      }
      if ((blockColIndex < numColBlocks - 1 && n != colsPerBlock) ||
        (blockColIndex == numColBlocks - 1 && (n <= 0 || n > colsPerBlock))) {
        throw new SparkException(s"The MatrixBlock at ($blockRowIndex, $blockColIndex) has " +
          dimensionMsg)
      }
    }
    println("MatrixBlock dimensions are okay...")
    println("BlockMatrix is valid!")
  }

  /** Caches the underlying Dataset. */
  @Since("2.4.0")
  def cache(): this.type = {
    blocks.cache()
    this
  }

  /** Persists the underlying Dataset with the specified storage level. */
  @Since("2.4.0")
  def persist(storageLevel: StorageLevel): this.type = {
    blocks.persist(storageLevel)
    this
  }

  /** Converts to CoordinateMatrix. */
  @Since("2.4.0")
  def toCoordinateMatrix(): CoordinateMatrix = {
    val entryDS = blocks.flatMap { case ((blockRowIndex, blockColIndex), mat) =>
      val rowStart = blockRowIndex.toLong * rowsPerBlock
      val colStart = blockColIndex.toLong * colsPerBlock
      val entryValues = new ArrayBuffer[MatrixEntry]()
      mat.foreachActive { (i, j, v) =>
        if (v != 0.0) entryValues += new MatrixEntry(rowStart + i, colStart + j, v)
        ()
      }
      entryValues
    }
    new CoordinateMatrix(entryDS, numRows(), numCols())
  }

  /** Converts to IndexedRowMatrix. The number of columns must be within the integer range. */
  @Since("2.4.0")
  def toIndexedRowMatrix(): IndexedRowMatrix = {
    val cols = numCols().toInt

    require(cols < Int.MaxValue, s"The number of columns should be less than Int.MaxValue ($cols).")

    val rows = blocks.flatMap { case (blockRowIdx, blockColIdx) => {
      blockColIdx.rowIter.zipWithIndex.map((vector) =>
        blockRowIdx._1 * rowsPerBlock + vector._2 -> ((blockRowIdx._2, vector._1.asBreeze)))
    }
    }.groupByKey(_._1).mapGroups((rowIndex, vector) => {
      val numberNonZeroPerRow = vector.map((c) => c._2._2.activeSize).sum.toDouble / cols.toDouble
      val wholeVector = if (numberNonZeroPerRow <= 0.1) { // Sparse at 1/10th nnz
        BSV.zeros[Double](cols)
      } else {
        BDV.zeros[Double](cols)
      }
      vector.foreach(c => {
        val offset = colsPerBlock * c._2._1
        wholeVector(offset until Math.min(cols, offset + colsPerBlock)) := c._2._2
      })
      new IndexedVector(rowIndex.toLong, Vectors.fromBreeze(wholeVector))
    })
    new IndexedRowMatrix(rows)
  }

  /**
    * Collect the distributed matrix on the driver as a `DenseMatrix`.
    */
  @Since("2.4.0")
  def toLocalMatrix(): Matrix = {
    require(numRows() < Int.MaxValue, "The number of rows of this matrix should be less than " +
      s"Int.MaxValue. Currently numRows: ${numRows()}")
    require(numCols() < Int.MaxValue, "The number of columns of this matrix should be less than " +
      s"Int.MaxValue. Currently numCols: ${numCols()}")
    require(numRows() * numCols() < Int.MaxValue, "The length of the values array must be " +
      s"less than Int.MaxValue. Currently numRows * numCols: ${numRows() * numCols()}")
    val m = numRows().toInt
    val n = numCols().toInt
    val mem = m * n / 125000
    if (mem > 500) println(s"Storing this matrix will require $mem MB of memory!")
    val localBlocks = blocks.collect()
    val values = new Array[Double](m * n)
    localBlocks.foreach { case ((blockRowIndex, blockColIndex), submat) =>
      val rowOffset = blockRowIndex * rowsPerBlock
      val colOffset = blockColIndex * colsPerBlock
      submat.foreachActive { (i, j, v) =>
        val indexOffset = (j + colOffset) * m + rowOffset + i
        values(indexOffset) = v
      }
    }
    new DenseMatrix(m, n, values)
  }

  /**
    * Transpose this `BlockMatrix`. Returns a new `BlockMatrix` instance sharing the
    * same underlying data. Is a lazy operation.
    */
  @Since("2.4.0")
  def transpose: BlockMatrix = {
    import blocks.sparkSession.implicits._
    val transposedBlocks = blocks.map { case ((blockRowIndex, blockColIndex), mat) =>
      ((blockColIndex, blockRowIndex), mat.transpose)
    }
    new BlockMatrix(transposedBlocks, colsPerBlock, rowsPerBlock, nCols, nRows)
  }

  /** Collects data and assembles a local dense breeze matrix (for test only). */
  private[ml] def toBreeze(): BDM[Double] = {
    val localMat = toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }

  /**
    * For given matrices `this` and `other` of compatible dimensions and compatible block dimensions,
    * it applies a binary function on their corresponding blocks.
    *
    * @param other  The second BlockMatrix argument for the operator specified by `binMap`
    * @param binMap A function taking two breeze matrices and returning a breeze matrix
    * @return A [[BlockMatrix]] whose blocks are the results of a specified binary map on blocks
    *         of `this` and `other`.
    *         Note: `blockMap` ONLY works for `add` and `subtract` methods and it does not support
    *         operators such as (a, b) => -a + b
    *         TODO: Make the use of zero matrices more storage efficient.
    */
  private[ml] def blockMap(
    other: BlockMatrix,
    binMap: (BM[Double], BM[Double]) => BM[Double]): BlockMatrix = {
    require(numRows() == other.numRows(), "Both matrices must have the same number of rows. " +
      s"A.numRows: ${numRows()}, B.numRows: ${other.numRows()}")
    require(numCols() == other.numCols(), "Both matrices must have the same number of columns. " +
      s"A.numCols: ${numCols()}, B.numCols: ${other.numCols()}")
    if (rowsPerBlock == other.rowsPerBlock && colsPerBlock == other.colsPerBlock) {

      //            val value: RDD[((Int, Int), (Iterable[Matrix], Iterable[Matrix]))] = blocks.rdd.cogroup(other
      // .blocks.rdd,
      //              createPartitioner())
      val value2: Dataset[(((Int, Int), Matrix), ((Int, Int), Matrix))] = blocks
        .joinWith(other.blocks, blocks.col("_1") === other.blocks.col("_1"))
      //
      //      value2.map(input => {
      //        val a = Array(input._1._2)
      //        val b = Array(input._2._2)
      //        if (a.size > 1 || b.size > 1) {
      //          throw new SparkException("There are multiple MatrixBlocks with indices: " +
      //            s"($blockRowIndex, $blockColIndex). Please remove them.")
      //        }
      //        if (a.isEmpty) {
      //          val zeroBlock = BM.zeros[Double](b.head.numRows, b.head.numCols)
      //          val result = binMap(zeroBlock, b.head.asBreeze)
      //          new MatrixBlock((blockRowIndex, blockColIndex), Matrices.fromBreeze(result))
      //        } else if (b.isEmpty) {
      //          new MatrixBlock((blockRowIndex, blockColIndex), a.head)
      //        } else {
      //          val result = binMap(a.head.asBreeze, b.head.asBreeze)
      //          new MatrixBlock((blockRowIndex, blockColIndex), Matrices.fromBreeze(result))
      //        }
      //      })
      val rddCogroup: RDD[((Int, Int), (Iterable[Matrix], Iterable[Matrix]))] = blocks.rdd.cogroup(other.blocks.rdd,
        createPartitioner())

      val newBlocks: RDD[((Int, Int), Matrix)] = rddCogroup
        .map { case ((blockRowIndex, blockColIndex), (a, b)) =>
          if (a.size > 1 || b.size > 1) {
            throw new SparkException("There are multiple MatrixBlocks with indices: " +
              s"($blockRowIndex, $blockColIndex). Please remove them.")
          }
          if (a.isEmpty) {
            val zeroBlock = BM.zeros[Double](b.head.numRows, b.head.numCols)
            val result = binMap(zeroBlock, b.head.asBreeze)
            new MatrixBlock((blockRowIndex, blockColIndex), Matrices.fromBreeze(result))
          } else if (b.isEmpty) {
            new MatrixBlock((blockRowIndex, blockColIndex), a.head)
          } else {
            val result = binMap(a.head.asBreeze, b.head.asBreeze)
            new MatrixBlock((blockRowIndex, blockColIndex), Matrices.fromBreeze(result))
          }
        }
      import blocks.sparkSession.implicits._
      new BlockMatrix(newBlocks.toDS(), rowsPerBlock, colsPerBlock, numRows(), numCols())
    } else {
      throw new SparkException("Cannot perform on matrices with different block dimensions")
    }
  }

  /**
    * Adds the given block matrix `other` to `this` block matrix: `this + other`.
    * The matrices must have the same size and matching `rowsPerBlock` and `colsPerBlock`
    * values. If one of the blocks that are being added are instances of `SparseMatrix`,
    * the resulting sub matrix will also be a `SparseMatrix`, even if it is being added
    * to a `DenseMatrix`. If two dense matrices are added, the output will also be a
    * `DenseMatrix`.
    */
  @Since("2.4.0")
  def add(other: BlockMatrix): BlockMatrix =
    blockMap(other, (x: BM[Double], y: BM[Double]) => x + y)

  /**
    * Subtracts the given block matrix `other` from `this` block matrix: `this - other`.
    * The matrices must have the same size and matching `rowsPerBlock` and `colsPerBlock`
    * values. If one of the blocks that are being subtracted are instances of `SparseMatrix`,
    * the resulting sub matrix will also be a `SparseMatrix`, even if it is being subtracted
    * from a `DenseMatrix`. If two dense matrices are subtracted, the output will also be a
    * `DenseMatrix`.
    */
  @Since("2.4.0")
  def subtract(other: BlockMatrix): BlockMatrix =
    blockMap(other, (x: BM[Double], y: BM[Double]) => x - y)

  /** Block (i,j) --> Set of destination partitions */
  private type BlockDestinations = Map[(Int, Int), Set[Int]]

  /**
    * Simulate the multiplication with just block indices in order to cut costs on communication,
    * when we are actually shuffling the matrices.
    * The `colsPerBlock` of this matrix must equal the `rowsPerBlock` of `other`.
    * Exposed for tests.
    *
    * @param other       The BlockMatrix to multiply
    * @param partitioner The partitioner that will be used for the resulting matrix `C = A * B`
    * @return A tuple of [[BlockDestinations]]. The first element is the Map of the set of partitions
    *         that we need to shuffle each blocks of `this`, and the second element is the Map for
    *         `other`.
    */
  private[distributed] def simulateMultiply_old(
    other: BlockMatrix,
    partitioner: GridPartitioner,
    midDimSplitNum: Int): (BlockDestinations, BlockDestinations) = {
    val leftMatrix = blockInfo.keys.collect()
    val rightMatrix = other.blockInfo.keys.collect()

    val rightCounterpartsHelper = rightMatrix.groupBy(_._1).mapValues(_.map(_._2))
    val leftDestinations = leftMatrix.map { case (rowIndex, colIndex) =>
      val rightCounterparts = rightCounterpartsHelper.getOrElse(colIndex, Array.empty[Int])
      val partitions = rightCounterparts.map(b => partitioner.getPartition((rowIndex, b)))
      val midDimSplitIndex = colIndex % midDimSplitNum
      ((rowIndex, colIndex),
        partitions.toSet.map((pid: Int) => pid * midDimSplitNum + midDimSplitIndex))
    }.toMap

    val leftCounterpartsHelper = leftMatrix.groupBy(_._2).mapValues(_.map(_._1))
    val rightDestinations = rightMatrix.map { case (rowIndex, colIndex) =>
      val leftCounterparts = leftCounterpartsHelper.getOrElse(rowIndex, Array.empty[Int])
      val partitions = leftCounterparts.map(b => partitioner.getPartition((b, colIndex)))
      val midDimSplitIndex = rowIndex % midDimSplitNum
      ((rowIndex, colIndex),
        partitions.toSet.map((pid: Int) => pid * midDimSplitNum + midDimSplitIndex))
    }.toMap

    (leftDestinations, rightDestinations)
  }

  private[distributed] def simulateMultiply(
    other: BlockMatrix,
    partitioner: GridPartitioner,
    midDimSplitNum: Int): (BlockDestinations, BlockDestinations) = {
    val leftMatrix = blockInfo.keys
    //.collect()
    val rightMatrix = other.blockInfo.keys //.collect()

    val rightCounterpartsHelper = rightMatrix.groupBy(_._1).mapValues(_.map(_._2)).collect().toMap
    val leftDestinations = leftMatrix.map { case (rowIndex, colIndex) =>
      val rightCounterparts = rightCounterpartsHelper.getOrElse(colIndex, Iterable.empty[Int])
      val partitions = rightCounterparts.map(b => partitioner.getPartition((rowIndex, b)))
      val midDimSplitIndex = colIndex % midDimSplitNum
      ((rowIndex, colIndex),
        partitions.toSet.map((pid: Int) => pid * midDimSplitNum + midDimSplitIndex))
    }.collect.toMap

    val leftCounterpartsHelper = leftMatrix.groupBy(_._2).mapValues(_.map(_._1)).collect().toMap
    val rightDestinations = rightMatrix.map { case (rowIndex, colIndex) =>
      val leftCounterparts = leftCounterpartsHelper.getOrElse(rowIndex, Iterable.empty[Int])
      val partitions = leftCounterparts.map(b => partitioner.getPartition((b, colIndex)))
      val midDimSplitIndex = rowIndex % midDimSplitNum
      ((rowIndex, colIndex),
        partitions.toSet.map((pid: Int) => pid * midDimSplitNum + midDimSplitIndex))
    }.collect.toMap

    (leftDestinations, rightDestinations)
  }

  private[distributed] def simulateMultiply_new(
    other: BlockMatrix,
    partitioner: GridPartitioner,
    midDimSplitNum: Int): (BlockDestinations, BlockDestinations) = {
    val leftMatrix = blockInfoDS.mapPartitions(_.map((_._1)))
    val rightMatrix = other.blockInfoDS.mapPartitions(_.map((_._1)))

    val rightCounterpartsHelper = rightMatrix
      .groupByKey(_._1)
      .flatMapGroups((a, b) => {
        val array = b.map(_._2).toArray
        Array((a, array))
      })
      .collect
      .toMap

    val rightCounterpartsHelperBC = blocks.sparkSession.sparkContext.broadcast(rightCounterpartsHelper)

    val leftDestinations = leftMatrix.mapPartitions(_.map(row => {
      val rowIndex = row._1
      val colIndex = row._2
      val rightCounterparts = rightCounterpartsHelperBC.value.getOrElse(colIndex, Array.empty[Int])
      val partitions = rightCounterparts.map(b => partitioner.getPartition((rowIndex, b)))
      val midDimSplitIndex = colIndex % midDimSplitNum
      ((rowIndex, colIndex),
        partitions.toSet.map((pid: Int) => pid * midDimSplitNum + midDimSplitIndex).toArray)
    })).collect().map(a => (a._1, a._2.toSet)).toMap

    val leftCounterpartsHelper = leftMatrix
      .groupByKey(_._2)
      .flatMapGroups((a, b) => {
        val array = b.map(_._1).toArray
        Array((a, array))
      })
      .collect
      .toMap

    val leftCounterpartsHelperBC = blocks.sparkSession.sparkContext.broadcast(leftCounterpartsHelper)

    val rightDestinations = rightMatrix.mapPartitions(_.map(row => {
      val rowIndex = row._1
      val colIndex = row._2
      val leftCounterparts: Array[Int] = leftCounterpartsHelperBC.value.getOrElse(rowIndex, Array.empty[Int])
      val partitions = leftCounterparts.map(b => partitioner.getPartition((b, colIndex)))
      val midDimSplitIndex = rowIndex % midDimSplitNum
      ((rowIndex, colIndex),
        partitions.toSet.map((pid: Int) => pid * midDimSplitNum + midDimSplitIndex).toArray)
    })).collect().map(a => (a._1, a._2.toSet)).toMap

    (leftDestinations, rightDestinations)
  }

  /**
    * Left multiplies this [[BlockMatrix]] to `other`, another [[BlockMatrix]]. The `colsPerBlock`
    * of this matrix must equal the `rowsPerBlock` of `other`. If `other` contains
    * `SparseMatrix`, they will have to be converted to a `DenseMatrix`. The output
    * [[BlockMatrix]] will only consist of blocks of `DenseMatrix`. This may cause
    * some performance issues until support for multiplying two sparse matrices is added.
    *
    * @note The behavior of multiply has changed in 1.6.0. `multiply` used to throw an error when
    *       there were blocks with duplicate indices. Now, the blocks with duplicate indices will be added
    *       with each other.
    */
  @Since("1.4.0")
  def multiply(other: BlockMatrix): BlockMatrix = {
    multiplyRDD(other, 1)
  }

  def multiply(
    other: BlockMatrix,
    numMidDimSplits: Int = 1,
    rdd: Boolean = true): BlockMatrix = {
    if (rdd) {
      multiplyRDD(other, numMidDimSplits)
    }
    else {
      multiplyDS(other, numMidDimSplits)
    }
  }

  private[ml] def multiplyRDD(
    other: BlockMatrix,
    numMidDimSplits: Int): BlockMatrix = {
    require(numCols() == other.numRows(), "The number of columns of A and the number of rows " +
      s"of B must be equal. A.numCols: ${numCols()}, B.numRows: ${other.numRows()}. If you " +
      "think they should be equal, try setting the dimensions of A and B explicitly while " +
      "initializing them.")
    require(numMidDimSplits > 0, "numMidDimSplits should be a positive integer.")
    import other.blocks.sparkSession.implicits._

    if (colsPerBlock == other.rowsPerBlock) {
      val resultPartitioner = GridPartitioner(numRowBlocks, other.numColBlocks,
        math.max(blocks.rdd.partitions.length, other.blocks.rdd.partitions.length))
      val (leftDestinations, rightDestinations)
      = simulateMultiply(other, resultPartitioner, numMidDimSplits)
      // Each block of A must be multiplied with the corresponding blocks in the columns of B.
      val flatA = blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
        val destinations = leftDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
      }
      // Each block of B must be multiplied with the corresponding blocks in each row of A.
      val flatB = other.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
        val destinations = rightDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
      }
      val intermediatePartitioner = new Partitioner {
        override def numPartitions: Int = resultPartitioner.numPartitions * numMidDimSplits

        override def getPartition(key: Any): Int = key.asInstanceOf[Int]
      }
      val newBlocks = flatA.rdd.cogroup(flatB.rdd, intermediatePartitioner).flatMap { case (pId, (a, b)) =>
        a.flatMap { case (leftRowIndex, leftColIndex, leftBlock) =>
          b.filter(_._1 == leftColIndex).map { case (rightRowIndex, rightColIndex, rightBlock) =>
            val C = rightBlock match {
              case dense: DenseMatrix             => leftBlock.multiply(dense).asBreeze
              case sparse: ml.linalg.SparseMatrix => leftBlock.asBreeze * sparse.asBreeze //leftBlock.multiply(sparse
              // .toDense)
              case _ =>
                throw new SparkException(s"Unrecognized matrix type ${rightBlock.getClass}.")
            }
            ((leftRowIndex, rightColIndex), C)
          }
        }
      }.reduceByKey(resultPartitioner, (a, b) => a + b).mapValues(Matrices.fromBreeze).toDS()
      // TODO: Try to use aggregateByKey instead of reduceByKey to get rid of intermediate matrices
      new BlockMatrix(newBlocks, rowsPerBlock, other.colsPerBlock, numRows(), other.numCols())
    } else {
      throw new SparkException("colsPerBlock of A doesn't match rowsPerBlock of B. " +
        s"A.colsPerBlock: $colsPerBlock, B.rowsPerBlock: ${other.rowsPerBlock}")
    }

  }

  /**
    * Left multiplies this [[BlockMatrix]] to `other`, another [[BlockMatrix]]. The `colsPerBlock`
    * of this matrix must equal the `rowsPerBlock` of `other`. If `other` contains
    * `SparseMatrix`, they will have to be converted to a `DenseMatrix`. The output
    * [[BlockMatrix]] will only consist of blocks of `DenseMatrix`. This may cause
    * some performance issues until support for multiplying two sparse matrices is added.
    * Blocks with duplicate indices will be added with each other.
    *
    * @param other           Matrix `B` in `A * B = C`
    * @param numMidDimSplits Number of splits to cut on the middle dimension when doing
    *                        multiplication. For example, when multiplying a Matrix `A` of
    *                        size `m x n` with Matrix `B` of size `n x k`, this parameter
    *                        configures the parallelism to use when grouping the matrices. The
    *                        parallelism will increase from `m x k` to `m x k x numMidDimSplits`,
    *                        which in some cases also reduces total shuffled data.
    */
  private[ml] def multiplyDS(
    other: BlockMatrix,
    numMidDimSplits: Int): BlockMatrix = {
    require(numCols() == other.numRows(), "The number of columns of A and the number of rows " +
      s"of B must be equal. A.numCols: ${numCols()}, B.numRows: ${other.numRows()}. If you " +
      "think they should be equal, try setting the dimensions of A and B explicitly while " +
      "initializing them.")
    require(numMidDimSplits > 0, "numMidDimSplits should be a positive integer.")
    import blocks.sparkSession.implicits._

    val timeFlag = false

    def time(message: String, ds: Dataset[_]): Unit = {
      if (timeFlag) {
        val start = System.currentTimeMillis()
        println(ds.cache.count)
        val end = System.currentTimeMillis()
        println(message + ": " + (end - start))
      }
      ()
    }

    if (colsPerBlock == other.rowsPerBlock) {
      val resultPartitioner = GridPartitioner(numRowBlocks, other.numColBlocks,
        math.max(blocks.rdd.partitions.length, other.blocks.rdd.partitions.length))
      val (leftDestinations, rightDestinations) = simulateMultiply_new(other, resultPartitioner, numMidDimSplits)

      // Each block of A must be multiplied with the corresponding blocks in the columns of B.
      val flatA = blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
        val destinations = leftDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, blockRowIndex, blockColIndex, block))
      }
      time("FlatA", flatA)

      // Each block of B must be multiplied with the corresponding blocks in each row of A.
      val flatB = other.blocks.flatMap { case ((blockRowIndex, blockColIndex),
      block) =>
        val destinations = rightDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, blockRowIndex, blockColIndex, block))
      }
        .withColumnRenamed("_1", "_5")
        .withColumnRenamed("_2", "_6")
        .withColumnRenamed("_3", "_7")
        .withColumnRenamed("_4", "_8")
      time("FlatA", flatB)

      val dot = udf((left: Matrix, right: Matrix) => {
        Matrices.fromBreeze(left.asBreeze * right.asBreeze)
      })

      val matrixSum = new MatrixSum()

      val leftRowIndex = "_2"
      val leftColIndex = "_3"
      val leftMatrixIndex = "_4"

      val rightRowIndex = "_6"
      val rightColIndex = "_7"
      val rightMatrixIndex = "_8"

      val newBlocks = flatA
        .join(flatB, flatA.col(leftColIndex) === flatB.col(rightRowIndex) && flatA.col("_1") === flatB.col("_5"))
        .groupBy(col("_1"), col(leftRowIndex), col(rightColIndex))
        .agg(matrixSum(dot(col(leftMatrixIndex), col(rightMatrixIndex))))
        .mapPartitions((row: Iterator[Row]) =>
          row.map(row => ((row.getInt(1), row.getInt(2)), row.getAs[Matrix](3))))
      //        .map(row => ((row.getInt(1), row.getInt(2)), row.getAs[Matrix](3)))
      time("newBlocks", newBlocks)

      new BlockMatrix(newBlocks, rowsPerBlock, other.colsPerBlock, numRows(), other.numCols())
    } else {
      throw new SparkException("colsPerBlock of A doesn't match rowsPerBlock of B. " +
        s"A.colsPerBlock: $colsPerBlock, B.rowsPerBlock: ${other.rowsPerBlock}")
    }
  }

}

case class InnerMatrix(i: Int, j: Int, mat: Matrix)

case class MatrixRow(j: Int, tuple: InnerMatrix)

class MatrixSum() extends UserDefinedAggregateFunction {
  def inputSchema: StructType = new StructType().add("matrix", SQLDataTypes.MatrixType)

  def bufferSchema: StructType = new StructType().add("buff", SQLDataTypes.MatrixType)

  def dataType: DataType = SQLDataTypes.MatrixType

  def deterministic: Boolean = true

  def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer.update(0, Matrices.zeros(0, 0))
  }

  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val buff = buffer.getAs[Matrix](0)
    val v = input.getAs[Matrix](0)
    if (buff.numCols == 0) {
      buffer.update(0, v)
    }
    else if (v.numNonzeros > 0) {
      if (buff.numNonzeros > 0) {
        buffer.update(0, Matrices.fromBreeze(buff.asBreeze + v.asBreeze))
      }
      else {
        buffer.update(0, v)
      }
    }
  }

  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val buff1 = buffer1.getAs[Matrix](0)
    val buff2 = buffer2.getAs[Matrix](0)
    if (buff1.numCols == 0) {
      buffer1.update(0, buff2)
    }
    else if (buff2.numNonzeros > 0) {
      if (buff1.numNonzeros > 0) {
        buffer1.update(0, Matrices.fromBreeze(buff1.asBreeze + buff2.asBreeze))
      }
      else {
        buffer1.update(0, buff2)
      }
    }
  }

  def evaluate(buffer: Row): Matrix = buffer.getAs[Matrix](0)
}
