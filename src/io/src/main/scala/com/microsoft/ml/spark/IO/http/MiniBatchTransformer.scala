// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.contracts.Wrappable
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

trait HasMaxBatchSize extends Params {
  val maxBatchSize: Param[Int] = new IntParam(
    this, "maxBatchSize", "The max size of the buffer")

  /** @group getParam */
  def getMaxBatchSize: Int = $(maxBatchSize)

  /** @group setParam */
  def setMaxBatchSize(value: Int): this.type = set(maxBatchSize, value)
}

object MiniBatchTransformer extends DefaultParamsReadable[MiniBatchTransformer]

class MiniBatchTransformer(val uid: String)
    extends Transformer with HasMaxBatchSize with DefaultParamsWritable with Wrappable {

  def this() = this(Identifiable.randomUID("MiniBatchTransformer"))

  setDefault(maxBatchSize -> Integer.MAX_VALUE)

  def transpose(nestedSeq: Seq[Seq[Any]]): Seq[Seq[Any]] = {
    val innerLength = nestedSeq.head.length
    assert(nestedSeq.forall(_.lengthCompare(innerLength) == 0))
    (0 until innerLength).map(i => nestedSeq.map(inneSeq => inneSeq(i)))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.toDF().mapPartitions(it =>
      new BatchIterator(it, getMaxBatchSize).map(listOfRows =>
        Row.fromSeq(transpose(listOfRows.map(r => r.toSeq))))
    )(RowEncoder(transformSchema(dataset.schema)))
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.map(f => StructField(f.name, ArrayType(f.dataType))))
  }

}

object FlattenBatch extends DefaultParamsReadable[FlattenBatch]

class FlattenBatch(val uid: String)
    extends Transformer with Wrappable with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("FlattenBatch"))

  def transpose(nestedSeq: Seq[Seq[Any]]): Seq[Seq[Any]] = {
    val innerLength = nestedSeq.head.length
    assert(nestedSeq.forall(_.lengthCompare(innerLength) == 0))
    (0 until innerLength).map(i => nestedSeq.map(inneSeq => inneSeq(i)))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.toDF().mapPartitions(it =>
      it.flatMap { rowOfLists =>
        val transposed = transpose((0 until rowOfLists.length).map(rowOfLists.getSeq))
        transposed.map(Row.fromSeq)
      }
    )(RowEncoder(transformSchema(dataset.schema)))
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    assert(schema.fields.forall(sf => sf.dataType match {
      case _: ArrayType => true
      case _ => false
    }))
    StructType(schema.map(f => StructField(f.name, f.dataType.asInstanceOf[ArrayType].elementType)))
  }

}
