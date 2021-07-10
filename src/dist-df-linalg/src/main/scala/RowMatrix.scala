// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset

case class RowVector(vector: Vector) {
  def size: Int = vector.size
}

class RowMatrix(
  val rows: Dataset[RowVector],
  private var nRows: Long,
  private var nCols: Int) extends DistributedMatrix {
  //  def toBreeze() = ???

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(rows: Dataset[RowVector]) = this(rows, 0L, 0)

  /** Gets or computes the number of columns. */
  override def numCols(): Long = {
    if (nCols <= 0) {
      try {
        // Calling `first` will throw an exception if `rows` is empty.
        nCols = rows.first().size
      } catch {
        case err: UnsupportedOperationException =>
          sys.error("Cannot determine the number of cols because it is not specified in the " +
            "constructor and the rows RDD is empty.")
      }
    }
    nCols.toLong
  }

  /** Gets or computes the number of rows. */
  override def numRows(): Long = {
    if (nRows <= 0L) {
      nRows = rows.count()
      if (nRows == 0L) {
        sys.error("Cannot determine the number of rows because it is not specified in the " +
          "constructor and the rows RDD is empty.")
      }
    }
    nRows
  }
}
