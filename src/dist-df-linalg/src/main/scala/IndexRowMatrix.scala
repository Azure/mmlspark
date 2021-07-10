// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.Dataset

/**
  * ML version of IndexedRow
  *
  * @param index  Index of row
  * @param vector Sparse or Dense form of the vector for the provided index
  */
case class IndexedVector(index: Long, vector: Vector)

/**
  * Dense Matrix wrapped for Spark Dataset of IndexRow
  *
  * Based on the requirements for Spark's IndexedRowMatrix
  *
  * @param rows  list of the index, vector row pairs
  * @param nRows number of rows, in case padding is needed
  * @param nCols number of columns, in case padding is needed
  */
class IndexedRowMatrix(val rows: Dataset[IndexedVector],
  private var nRows: Long,
  private var nCols: Long) extends DistributedMatrix {
  def multiply(other: IndexedColMatrix): CoordinateMatrix = {
    import rows.sparkSession.implicits._

    new CoordinateMatrix(
      rows.crossJoin(other.rows)
        .mapPartitions(_.map((vectors) => {
          val iVec = vectors.getAs[Vector](1)
          val jVec = vectors.getAs[Vector](3)
          val product = (iVec, jVec) match {
            case (DenseVector(vs), DenseVector(vs2)) => {
              val ibdv = new BDV[Double](iVec.toArray)
              val jbdv = new BDV[Double](jVec.toArray)
              (jbdv.asDenseMatrix * ibdv).data(0)
            }
            case (DenseVector(vs), SparseVector(n2, ids2, vs2)) => {
              val ibdv = new BDV[Double](iVec.toArray)
              val jMLSparse = jVec.toSparse
              val jbsv = new BSV[Double](jMLSparse.indices, jMLSparse.values, jMLSparse.size)
              (jbsv.asCscColumn * ibdv).data(0)
            }
            case (SparseVector(n, ids, vs), DenseVector(vs2)) => {
              val iMLSparse = iVec.toSparse
              val ibsv = new BSV[Double](iMLSparse.indices, iMLSparse.values, iMLSparse.size)
              val jbdv = new BDV[Double](jVec.toArray)
              (jbdv.asDenseMatrix * ibsv).data(0)
            }
            case (SparseVector(n, ids, vs), SparseVector(n2, ids2, vs2)) => {
              val iMLSparse = iVec.toSparse
              val ibsv = new BSV[Double](iMLSparse.indices, iMLSparse.values, iMLSparse.size)
              val jMLSparse = jVec.toSparse
              val jbsv = new BSV[Double](jMLSparse.indices, jMLSparse.values, jMLSparse.size)
              (jbsv.asCscRow * ibsv).data(0)
            }
          }
          MatrixEntry(vectors.getAs[Long](0), vectors.getAs[Long](2), product)
        }))
    )
  }

  def transpose(): IndexedColMatrix = {
    new IndexedColMatrix(rows, nCols, nRows)
  }

  /**
    * Converts to BlockMatrix. Creates blocks with size 1024 x 1024.
    */
  def toBlockMatrix(): BlockMatrix = {
    toBlockMatrix(1024, 1024)
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
    require(rowsPerBlock > 0,
      s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
    require(colsPerBlock > 0,
      s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")

    val m = numRows()
    val n = numCols()

    // Since block matrices require an integer row index
    require(math.ceil(m.toDouble / rowsPerBlock) <= Int.MaxValue,
      "Number of rows divided by rowsPerBlock cannot exceed maximum integer.")

    // The remainder calculations only matter when m % rowsPerBlock != 0 or n % colsPerBlock != 0
    val remainderRowBlockIndex = m / rowsPerBlock
    val remainderColBlockIndex = n / colsPerBlock
    val remainderRowBlockSize = (m % rowsPerBlock).toInt
    val remainderColBlockSize = (n % colsPerBlock).toInt
    val numRowBlocks = math.ceil(m.toDouble / rowsPerBlock).toInt
    val numColBlocks = math.ceil(n.toDouble / colsPerBlock).toInt

    val blocks = rows.rdd.flatMap { ir: IndexedVector =>
      val blockRow = ir.index / rowsPerBlock
      val rowInBlock = ir.index % rowsPerBlock

      ir.vector match {
        case SparseVector(size, indices, values) =>
          indices.zip(values).map { case (index, value) =>
            val blockColumn = index / colsPerBlock
            val columnInBlock = index % colsPerBlock
            ((blockRow.toInt, blockColumn.toInt), (rowInBlock.toInt, Array((value, columnInBlock))))
          }
        case DenseVector(values) =>
          values.grouped(colsPerBlock)
            .zipWithIndex
            .map { case (values, blockColumn) =>
              ((blockRow.toInt, blockColumn), (rowInBlock.toInt, values.zipWithIndex))
            }
      }
    }.groupByKey(GridPartitioner(numRowBlocks, numColBlocks, rows.rdd.getNumPartitions)).map {
      case ((blockRow, blockColumn), itr) =>
        val actualNumRows =
          if (blockRow == remainderRowBlockIndex) remainderRowBlockSize else rowsPerBlock
        val actualNumColumns =
          if (blockColumn == remainderColBlockIndex) remainderColBlockSize else colsPerBlock

        val arraySize = actualNumRows * actualNumColumns
        val matrixAsArray = new Array[Double](arraySize)
        var countForValues = 0
        itr.foreach { case (rowWithinBlock, valuesWithColumns) =>
          valuesWithColumns.foreach { case (value, columnWithinBlock) =>
            matrixAsArray.update(columnWithinBlock * actualNumRows + rowWithinBlock, value)
            countForValues += 1
          }
        }
        val denseMatrix = new DenseMatrix(actualNumRows, actualNumColumns, matrixAsArray)
        val finalMatrix = if (countForValues / arraySize.toDouble >= 0.1) {
          denseMatrix
        } else {
          denseMatrix.toSparse
        }

        ((blockRow, blockColumn), finalMatrix)
    }
    import rows.sparkSession.implicits._
    new BlockMatrix(blocks.toDS(), rowsPerBlock, colsPerBlock, m, n)
  }

  private[ml] def toBreeze(): BDM[Double] = {
    val m = numRows().toInt
    val n = numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    rows.collect().foreach { case IndexedVector(rowIndex, vector) =>
      val i = rowIndex.toInt
      vector.foreachActive { case (j, v) =>
        mat(i, j) = v
      }
    }
    mat
  }

  def toRowMatrix(): RowMatrix = {
    import rows.sparkSession.implicits._
    new RowMatrix(
      rows.mapPartitions(_.map((row) => RowVector(row.vector))))
  }

  def multiply(other: IndexedRowMatrix): DistributedMatrix = {
    multiply(other.toCoordinateMatrix)
  }

  def multiply(other: CoordinateMatrix): DistributedMatrix = {
    toCoordinateMatrix.multiply(other)
  }

  def toCoordinateMatrix(): CoordinateMatrix = {
    import rows.sparkSession.implicits._

    new CoordinateMatrix(
      rows.flatMap(row => row.vector.toArray
        .zipWithIndex
        .filter(_._1 > 0)
        .map(pair => MatrixEntry(row.index, pair._2.toLong, pair._1)))
    )
  }

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(entries: Dataset[IndexedVector]) = this(entries, 0L, 0)

  override def numCols(): Long = {
    if (nCols <= 0) {
      // Calling `first` will throw an exception if `rows` is empty.
      nCols = rows.first().vector.size.toLong
    }
    nCols
  }

  override def numRows(): Long = {
    if (nRows <= 0L) {
      nRows = rows.groupBy().max("index").collect()(0).getLong(0) + 1
    }
    nRows
  }
}
