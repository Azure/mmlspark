// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import org.apache.spark.sql.Dataset

class IndexedColMatrix(val rows: Dataset[IndexedVector],
  private var nRows: Long,
  private var nCols: Long) extends DistributedMatrix {
  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(entries: Dataset[IndexedVector]) = this(entries, 0L, 0)

  def transpose(): IndexedRowMatrix = {
    new IndexedRowMatrix(rows, nCols, nRows)
  }

  /** Gets or computes the number of rows. */
  override def numRows(): Long = {
    if (nRows <= 0) {
      // Calling `first` will throw an exception if `rows` is empty.
      nRows = rows.first().vector.size.toLong
    }
    nRows
  }

  /** Gets or computes the number of columns. */
  override def numCols(): Long = {
    if (nCols <= 0L) {
      nCols = rows.groupBy().max("index").collect()(0).getLong(0) + 1
    }
    nCols
  }
}
