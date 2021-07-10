// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.ml.linalg.{Vectors => V}
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.Dataset

import scala.language.implicitConversions

/**
  * Represents a distributively stored matrix backed by one or more DataSets.
  */
trait DistributedMatrix extends Serializable {

  /** Gets or computes the number of rows. */
  def numRows(): Long

  /** Gets or computes the number of columns. */
  def numCols(): Long

}

/**
  * SparseMatrix holds the static methods used by the SparseMatrix class
  */
object DistributedMatrix {
  def multiply(matrix: Any, other: Any): DistributedMatrix = {
    (matrix, other) match {
      case (left: IndexedRowMatrix, right: IndexedRowMatrix) => multiply(left, right)
      case (left: IndexedRowMatrix, right: CoordinateMatrix) => multiply(left, right)
      case (left: CoordinateMatrix, right: IndexedRowMatrix) => multiply(left, right)
      case (left: CoordinateMatrix, right: CoordinateMatrix) => multiply(left, right)
    }
  }

  def sparse(entries: Dataset[MatrixEntry]): CoordinateMatrix = new CoordinateMatrix(entries)

  private[linalg] def multiply(leftMatrix: IndexedRowMatrix, rightMatrix: IndexedRowMatrix): DistributedMatrix = {
    multiply(leftMatrix.toCoordinateMatrix, rightMatrix.toCoordinateMatrix)
  }

  private[linalg] def multiply(leftMatrix: IndexedRowMatrix, rightMatrix: CoordinateMatrix): DistributedMatrix = {
    multiply(leftMatrix.toCoordinateMatrix, rightMatrix)
  }

  private[linalg] def multiply(leftMatrix: CoordinateMatrix, rightMatrix: IndexedRowMatrix): DistributedMatrix = {
    multiply(leftMatrix, rightMatrix.toCoordinateMatrix)
  }
}
