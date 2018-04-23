// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{ArrayType, IntegerType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalactic.Equality

class MiniBatchTransformerSuite extends TransformerFuzzing[MiniBatchTransformer] {

  def delayedIterator(n: Int, wait: Int = 5): Iterator[Int] = {
    (1 to n).toIterator.map { x =>
      Thread.sleep(x * wait.toLong)
      x
    }
  }

  import session.implicits._

  val n = 1000
  val df: DataFrame = sc.parallelize((1 to n).zip(List.fill(n)("foo"))).toDF("in1", "in2")

  test("basic functionality") {
    val delay = 5

    val slowDf = df.map { x => Thread.sleep(3 * delay.toLong); x }(RowEncoder(df.schema))

    val df2 = new MiniBatchTransformer().transform(slowDf)

    val df3 = df2.map { x => Thread.sleep(10 * delay.toLong); x }(RowEncoder(df2.schema))

    assert(df3.schema == new StructType()
             .add("in1", ArrayType(IntegerType))
             .add("in2", ArrayType(StringType)))

    val lengths = df3.collect().map(row => row.getSeq(0).length)
    assert(lengths.sum == n)
    assert(lengths.max > 2)

    val flattened = new FlattenBatch().transform(df3)

    flattened.show(truncate = false)
    df.show(truncate = false)

    assert(flattened.collect() === df.collect())
  }

  override def testObjects(): Seq[TestObject[MiniBatchTransformer]] = Seq(
    new TestObject(new MiniBatchTransformer(), df))

  override implicit lazy val dfEq: Equality[DataFrame] = new Equality[DataFrame]{
    def areEqual(a: DataFrame, bAny: Any): Boolean = bAny match {
      case ds: Dataset[_] =>
        val b = ds.toDF()
        baseDfEq.areEqual(new FlattenBatch().transform(a), new FlattenBatch().transform(b))
    }
  }

  override def reader: MLReadable[_] = MiniBatchTransformer

}

class FlattenBatchSuite extends TransformerFuzzing[FlattenBatch] {
  import session.implicits._
  val n = 1000
  val df: DataFrame = sc.parallelize((1 to n).zip(List.fill(n)("foo"))).toDF("in1", "in2")

  override def testObjects(): Seq[TestObject[FlattenBatch]] = Seq(
    new TestObject[FlattenBatch](new FlattenBatch(), new MiniBatchTransformer().transform(df))
  )

  override def reader: MLReadable[_] = FlattenBatch

}
