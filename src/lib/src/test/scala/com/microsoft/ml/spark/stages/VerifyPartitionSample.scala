// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class PartitionSampleSmokeTests extends TransformerFuzzing[PartitionSample] {

  import session.implicits._

  test("head 3") {
    val sampler = new PartitionSample().setMode("Head").setCount(3)
    val out = sampler.transform(makeDF)
    assert(out.count === 3)
    assert(makeDF.head === out.head)
  }

  test("random sample smoke") {
    val df = extendedDF(10)
    val sampler = new PartitionSample()
        .setMode("RandomSample")
        .setRandomSampleMode("Absolute")
        .setSeed(1)
        .setCount(10)
    val out = sampler.transform(df)
    assert(out.count < 16)
    assert(out.count > 5)

    val sampler2 = new PartitionSample()
        .setMode("RandomSample")
        .setRandomSampleMode("Percentage")
        .setSeed(1)
        .setPercent(0.5)
    val out2 = sampler2.transform(df)
    assert(out2.count < 100)
    assert(out2.count > 60)
  }

  def extendedDF(n: Int = 10): DataFrame = {
    (2 to n).map(_ => makeDF).foldLeft(makeDF)((a, b) => a.union(b))
  }

  lazy val makeDF: DataFrame = {
    Seq(( 1,  2),
        ( 3,  4),
        ( 5,  6),
        ( 7,  8),
        ( 9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 18),
        (19, 20),
        (21, 22),
        (23, 24),
        (25, 26),
        (27, 28),
        (29, 30),
        (31, 32))
      .toDF("Col1", "Col2")
  }

  override def testObjects(): Seq[TestObject[PartitionSample]] = Seq(
    new TestObject[PartitionSample](new PartitionSample().setMode("Head").setCount(3), makeDF))

  override def reader: MLReadable[_] = PartitionSample

}
