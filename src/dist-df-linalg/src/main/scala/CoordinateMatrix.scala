// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.ml.linalg.{Matrix, SparseMatrix, Vectors => V}
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix => CM}
import org.apache.spark.sql.functions.{col, collect_list}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable
import scala.language.implicitConversions

/**
  * Sparse Matrix wrapped for Spark Dataset of MatrixEntry
  *
  * Based on the requirements for Spark's CoordinateMatrix
  * An RDD that would be used to create a CoordinateMatrix, can be used to create a
  * Sparse Matrix using rdd.toDS when it is an RDD of MatrixEntry.
  *
  * implicit def dsToSM(ds: Dataset[MatrixEntry]): SparseMatrix = new spark.linalg.SparseMatrix(ds)
  * val rddMatrixEntry: RDD[MatrixEntry] = rdd.map(row => new MatrixEntry(row._1, row._2, row._3)
  * val sparseMatrix: SparseMatrix = rddMatrixEntry.toDS
  *
  * Or, a DataFrame of Row, can be mapped to a MatrixEntry.
  *
  * implicit def dsToSM(ds: Dataset[MatrixEntry]): SparseMatrix = new spark.linalg.SparseMatrix(ds)
  * val sparseMatrix: SparseMatrix =
  *     dataframe.map(row => new MatrixEntry(row.getLong(0). row.getLong(1), row.getDouble(2))
  *
  * The SparseMatrix can be easily converted back to the MLlib distributed matricies using the same
  * type of casting methods.
  *
  * val coordinateMatrix: CoordinateMatrix = sparseMatrix.toCoordinateMatrix
  *
  * val blockMatrix: BlockMatrix = sparseMatrix.toBlockMatrix
  *
  * val localMatrix: LocalMatrix = sparseMatrix.toLocalMatrix
  *
  * @param entries Input data, a Dataset of MatrixEntry
  * @param nRows   number of rows, in case padding is needed
  * @param nCols   number of columns, in case padding is needed
  */
class CoordinateMatrix(val entries: Dataset[MatrixEntry],
  private var nRows: Long,
  private var nCols: Long) extends DistributedMatrix with MLLib {

  /**
    * Converts to BlockMatrix. Creates blocks with size 1024 x 1024.
    */
  def toBlockMatrix(rdd: Boolean = true): BlockMatrix = {
    if (rdd)
      toBlockMatrix(1024, 1024)
    else
      toBlockMatrixDS(10, 10)
  }

  /**
    * Converts to BlockMatrix. Blocks may be sparse or dense depending on the sparsity of the rows.
    *
    * @param rowsPerBlock The number of rows of each block. The blocks at the bottom edge may have
    *                     a smaller value. Must be an integer value greater than 0.
    * @param colsPerBlock The number of columns of each block. The blocks at the right edge may have
    *                     a smaller value. Must be an integer value greater than 0.
    * @return a [[BlockMatrix]]
    */
  def toBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    import entries.sparkSession.implicits._

    val m = numRows()
    val n = numCols()

    val blocks = new CM(entries.rdd, m, n)
      .toBlockMatrix(rowsPerBlock, colsPerBlock)
      .blocks
      .mapValues(_.asML)

    val ds: Dataset[((Int, Int), Matrix)] = entries.sparkSession.createDataset(blocks)

    new BlockMatrix(ds, rowsPerBlock, colsPerBlock, m, n)
  }

  def toBlockMatrixDS(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    require(rowsPerBlock > 0,
      s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
    require(colsPerBlock > 0,
      s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")
    import entries.sparkSession.implicits._
    val m = numRows()
    val n = numCols()

    // Since block matrices require an integer row and col index
    require(math.ceil(m.toDouble / rowsPerBlock) <= Int.MaxValue,
      "Number of rows divided by rowsPerBlock cannot exceed maximum integer.")
    require(math.ceil(n.toDouble / colsPerBlock) <= Int.MaxValue,
      "Number of cols divided by colsPerBlock cannot exceed maximum integer.")

    val numRowBlocks = math.ceil(m.toDouble / rowsPerBlock).toInt
    val numColBlocks = math.ceil(n.toDouble / colsPerBlock).toInt
    val partitioner = GridPartitioner(numRowBlocks, numColBlocks, entries.rdd.partitions.length)

    val blocks: Dataset[((Int, Int), Matrix)] = entries
      .mapPartitions(_.map(entry => {
        val blockRowIndex = (entry.i / rowsPerBlock).toInt
        val blockColIndex = (entry.j / colsPerBlock).toInt

        val rowId = entry.i % rowsPerBlock
        val colId = entry.j % colsPerBlock

        ((blockRowIndex, blockColIndex), (rowId.toInt, colId.toInt, entry.value))
      }))
      .groupBy(col("_1"))
      .agg(collect_list(col("_2")))
      .mapPartitions(_.map(r => {
        val blockRowIndex = r.getAs[Row](0).getInt(0)
        val blockColIndex = r.getAs[Row](0).getInt(1)
        val entry = r.getAs[mutable.WrappedArray[Row]](1)
          .map(r => (r.getInt(0), r.getInt(1), r.getDouble(2)))
        val effRows = math.min(m - blockRowIndex.toLong * rowsPerBlock, rowsPerBlock.toLong).toInt
        val effCols = math.min(n - blockColIndex.toLong * colsPerBlock, colsPerBlock.toLong).toInt
        ((blockRowIndex, blockColIndex), SparseMatrix.fromCOO(effRows, effCols, entry))
      }))

    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, m, n)
  }

  private[ml] def toBreeze(): BDM[Double] = {
    val m = numRows().toInt
    val n = numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    entries.collect().foreach { case MatrixEntry(i, j, value) =>
      mat(i.toInt, j.toInt) = value
    }
    mat
  }

  //  def multiply(other: IndexedRowMatrix): DistributedMatrix = multiply(other.toCoordinateMatrix)

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(entries: Dataset[MatrixEntry]) = this(entries, 0L, 0L)

  /**
    * Get Number of Columns in Matrix
    *
    * @return
    */
  override def numCols(): Long = {
    if (nCols <= 0L) {
      computeSize()
    }
    nCols
  }

  /**
    * Get Number of Rows in Matrix
    *
    * @return
    */
  override def numRows(): Long = {
    if (nRows <= 0L) {
      computeSize()
    }
    nRows
  }

  /**
    * Transpose Matrix
    *
    * @return
    */
  def transpose(): CoordinateMatrix = {
    import entries.sparkSession.implicits._
    new CoordinateMatrix(
      entries.mapPartitions(_.map(x => MatrixEntry(x.j, x.i, x.value))),
      nCols, nRows)
  }

  def toRowMatrix(): RowMatrix = {
    toIndexedRowMatrix.toRowMatrix()
  }

  /**
    * Compute the Rows and Columns of the Matrix
    * Used by the cols and rows getters
    */
  private def computeSize(): Unit = {
    // There may be empty columns at the very right and empty rows at the very bottom.
    import entries.sparkSession.implicits._
    val (m1, n1) = entries
      .map(entry => (entry.i, entry.j))
      .reduce((l1, l2) => (math.max(l1._1, l2._1), math.max(l1._2, l2._2)))

    nRows = math.max(nRows, m1 + 1L)
    nCols = math.max(nCols, n1 + 1L)
  }

  /**
    * Multiply two sparse matrices together
    *
    * @param other left matrix for multiplication, in SparseMatrix type
    * @return Product of A * B
    */
  def multiply(other: CoordinateMatrix): CoordinateMatrix = multiply(this, other)

  /**
    * Multiply two sparse matrices together
    *
    * @param other left matrix for multiplication
    * @return Product of A * B
    */
  def multiply(other: Dataset[MatrixEntry]): CoordinateMatrix = {
    multiply(new CoordinateMatrix(entries), new CoordinateMatrix(other))
  }

  case class VectorPair(i: Int, iVec: mutable.WrappedArray[(Int, Double)], j: Int, jVec: mutable.WrappedArray[(Int,
    Double)])

  private[linalg] def multiply(leftMatrix: CoordinateMatrix, rightMatrix: CoordinateMatrix): CoordinateMatrix = {
    leftMatrix.toIndexedRowMatrix()
      .multiply(rightMatrix.toIndexedColMatrix())
  }

  /**
    * Convert from Sparse to Dense format
    *
    * @return
    */
  def toIndexedRowMatrix(): IndexedRowMatrix = {
    import entries.sparkSession.implicits._
    val cols = numCols().toInt
    new IndexedRowMatrix(
      entries
        .groupByKey(_.i)
        .mapGroups((a, iVecSeq) => {
          val vecSeq = iVecSeq.map(matEntry => (matEntry.j.toInt, matEntry.value)).toSeq
          IndexedVector(a, V.sparse(cols, vecSeq).compressed)
        })
    )
  }

  def toIndexedColMatrix(): IndexedColMatrix = {
    import entries.sparkSession.implicits._
    val rows = numRows().toInt
    new IndexedColMatrix(
      entries
        .groupByKey(_.j)
        .mapGroups((a, iVecSeq) => {
          val vecSeq = iVecSeq.map(matEntry => (matEntry.i.toInt, matEntry.value)).toSeq
          IndexedVector(a, V.sparse(rows, vecSeq).compressed)
        })
    )
  }

  private[distributed] def vectorizeRows(indexName: String, colName: String, itemCount: Int) = {
    import entries.sparkSession.implicits._
    entries
      .groupByKey(_.i)
      .mapGroups((a, b) => {
        val vecSeq = b.map(matEntry => (matEntry.j.toInt, matEntry.value)).toSeq
        (a, V.sparse(itemCount, vecSeq).compressed)
      })
      .withColumnRenamed("_1", indexName)
      .withColumnRenamed("_2", colName)
  }

  private[distributed] def vectorizeCols(indexName: String, colName: String, itemCount: Int) = {
    import entries.sparkSession.implicits._
    entries
      .groupByKey(_.j)
      .mapGroups((a, b) => {
        val vecSeq = b.map(matEntry => (matEntry.i.toInt, matEntry.value)).toSeq
        (a, V.sparse(itemCount, vecSeq).compressed)
      })
      .withColumnRenamed("_1", indexName)
      .withColumnRenamed("_2", colName)
  }

  private[distributed] def vectorizeRows(indexName: String, colName: String): DataFrame = {
    import entries.sparkSession.implicits._
    entries
      .map(d => (d.i, (d.j.toInt, d.value)))
      .groupBy($"_1")
      .agg(collect_list($"_2"))
      .withColumnRenamed("_1", "i")
      .withColumnRenamed("collect_list(_2)", colName)
  }

  private[distributed] def vectorizeCols(indexName: String, colName: String): DataFrame = {
    import entries.sparkSession.implicits._
    entries
      .map(d => (d.j, (d.i.toInt, d.value)))
      .groupBy($"_1")
      .agg(collect_list($"_2"))
      .withColumnRenamed("_1", "j")
      .withColumnRenamed("collect_list(_2)", colName)
  }

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.CoordinateMatrix backed by RDD
    *
    * @return
    */
  private[distributed] override def toMLLib: CM = toMLLibCoordinateMatrix(entries)
}

private[distributed] trait MLLib {

  private[distributed] def toMLLib: CM = ???

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.CoordinateMatrix backed by RDD
    *
    * @return
    */
  private[distributed] def toMLLibCoordinateMatrix(entries: Dataset[MatrixEntry]): CM = new CM(entries.rdd)

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.BlockMatrix backed by RDD
    *
    * Takes parameters to specific the row and column block size
    *
    * @param rowsPerBlock number of items to be placed in each row block
    * @param colsPerBlock number of items to be placed in each column block
    * @return
    */
  private[distributed] def toMLLibBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): mllib.linalg.distributed
  .BlockMatrix = {
    toMLLib.toBlockMatrix(rowsPerBlock, colsPerBlock)
  }

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.BlockMatrix backed by RDD
    *
    * Uses default block size, 1024, for columns and rows
    *
    * @return
    */
  private[distributed] def toMLLibBlockMatrix: mllib.linalg.distributed.BlockMatrix = toMLLibBlockMatrix(1024, 1024)

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.LocalMatrix backed by RDD
    *
    * @return
    */
  private[distributed] def toMLLibLocalMatrix: Matrix = toMLLibBlockMatrix.toLocalMatrix.asML

}
