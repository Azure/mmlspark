// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.http

import java.util.concurrent.LinkedBlockingQueue
import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.ml.{ComplexParamsWritable, Transformer}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.concurrent.blocking

object PartitionConsolidator extends DefaultParamsReadable[PartitionConsolidator]

class Consolidator[T] {

  val buffer = new LinkedBlockingQueue[T]()
  var workingPartitions = 0

  def add(e: T): Boolean = {
    buffer.add(e)
  }

  def chosenPartition(): Boolean = this.synchronized {
    workingPartitions == 0
  }

  def addWorker(): Unit = {
    this.synchronized(workingPartitions += 1)
  }

  def removeWorker(): Unit = {
    this.synchronized(workingPartitions -= 1)
  }

  def getWorkingPartitions: Int = this.synchronized {
    val wp = workingPartitions
    wp
  }

  private def chosenIterator(it: Iterator[T], gracePeriod: Int = 1000) = {
    new Iterator[Option[T]] {

      private def hasNextHelper(recurse: Boolean): Boolean = {
        !buffer.isEmpty ||
          it.hasNext ||
          getWorkingPartitions > 1 || {
          if (recurse) {
            blocking {
              Thread.sleep(gracePeriod.toLong)
            }
            hasNextHelper(false)
          } else {
            removeWorker()
            false
          }
        }
      }

      override def hasNext: Boolean = {
        hasNextHelper(true)
      }

      override def next(): Option[T] = {
        if (!buffer.isEmpty) {
          Some(buffer.take())
        } else if (it.hasNext) {
          Some(it.next())
        } else {
          None
        }
      }
    }
  }

  private def regularIterator(it: Iterator[T]) = {
    new Iterator[Option[T]] {
      private var isDone = false

      override def hasNext: Boolean = !isDone

      override def next(): Option[T] = {
        assert(!isDone)
        it.foreach(add)
        isDone = true
        removeWorker()
        None
      }
    }
  }

  def registerAndReceive(it: Iterator[T]): Iterator[Option[T]] = {
    val chosen = chosenPartition()
    addWorker()
    if (chosen) {
      chosenIterator(it)
    } else {
      regularIterator(it)
    }
  }

}

class PartitionConsolidator(val uid: String)
  extends Transformer with HTTPParams with HasInputCol
    with HasOutputCol
    with ComplexParamsWritable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("PartitionConsolidator"))

  val consolidatorHolder: SharedSingleton[Consolidator[Row]] = SharedSingleton {
    new Consolidator[Row]()
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      dataset.toDF().mapPartitions { it =>
        if (it.hasNext) {
          consolidatorHolder.get.registerAndReceive(it).flatten
        } else {
          Iterator()
        }
      }(RowEncoder(dataset.schema))
    })
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema
}
