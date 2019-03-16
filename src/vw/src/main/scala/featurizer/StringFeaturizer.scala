// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.featurizer

import org.apache.spark.sql.Row
import org.vowpalwabbit.bare.VowpalWabbitMurmur

import scala.collection.mutable.ArrayBuffer

class StringFeaturizer(override val fieldIdx:Int, val columnName:String, val namespaceHash:Int)
  extends Featurizer(fieldIdx) {
    override def featurize(row: Row, indices:ArrayBuffer[Int], values:ArrayBuffer[Double]) = {
      indices += Featurizer.maxIndexMask & VowpalWabbitMurmur.hash(columnName + row.getString(fieldIdx), namespaceHash)
      values += 1.0

      ()
    }
}
