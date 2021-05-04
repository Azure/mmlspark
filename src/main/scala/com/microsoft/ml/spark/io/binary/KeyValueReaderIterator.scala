// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.binary

import java.io.Closeable
import org.apache.hadoop.mapreduce.RecordReader

// Based on:
// https://github.com/apache/spark/blob/master/sql/core/src/main/scala/
//   org/apache/spark/sql/execution/datasources/RecordReaderIterator.scala

/** An adaptor from a Hadoop [[RecordReader]] to an [[Iterator]] over the keys and values returned.
  *
  * This file is based on spark's RecordReaderIterator.
  */
private class KeyValueReaderIterator[K, V] (
  private[this] var rowReader: RecordReader[K, V]) extends Iterator[(K, V)] with Closeable {
  private[this] var havePair = false
  private[this] var finished = false

  override def hasNext: Boolean = {
    if (!finished && !havePair) {
      finished = !rowReader.nextKeyValue
      if (finished) {
        // Close and release the reader here; close() will also be called when the task
        // completes, but for tasks that read from many files, it helps to release the
        // resources early.
        close()
      }
      havePair = !finished
    }
    !finished
  }

  override def next(): (K, V) = {
    if (!hasNext) {
      throw new java.util.NoSuchElementException("End of stream")
    }
    havePair = false
    (rowReader.getCurrentKey, rowReader.getCurrentValue)
  }

  override def close(): Unit = {
    if (rowReader != null) { //scalastyle:ignore null
      try {
        rowReader.close()
      } finally {
        rowReader = null //scalastyle:ignore null
      }
    }
  }

}
