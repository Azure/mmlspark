// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import org.apache.spark.ml.param.{BooleanParam, Params}

/** Controls hashing parameters such us number of bits (numbits) and how to handle collisions.
  */
trait HasSumCollisions extends Params {
  val sumCollisions = new BooleanParam(this, "sumCollisions", "Sums collisions if true, otherwise removes them")
  setDefault(sumCollisions -> true)

  def getSumCollisions: Boolean = $(sumCollisions)

  def setSumCollisions(value: Boolean): this.type = set(sumCollisions, value)
}
