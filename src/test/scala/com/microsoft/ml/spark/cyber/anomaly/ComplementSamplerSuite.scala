// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cyber.anomaly

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, lit}

class ComplementSamplerSuite extends TransformerFuzzing[ComplementSampler] {

  import session.implicits._

  lazy val df: DataFrame = Seq(
    ("t1", 0, 0),
    ("t1", 1, 1),
    ("t1", 2, 2),
    ("t1", 3, 3),
    ("t1", 4, 4),
    ("t1", 5, 5),
    ("t2", 0, 0),
    ("t2", 1, 1),
    ("t2", 2, 2),
    ("t2", 3, 3)
  ).toDF("tenant", "user", "res").cache()

  def ca: ComplementSampler = new ComplementSampler()
    .setPartitionKey("tenant")
    .setInputCols(Array("user", "res"))

  def accessSet(df: DataFrame, tenant: String): Set[Seq[Int]] = {
    df
      .where(col("tenant") === lit(tenant))
      .collect().map(r => Seq(r.getInt(1), r.getInt(2)))
      .toSet
  }

  test("basic usage") {
    val results = ca.transform(df).cache()
    Seq("t1", "t2").map(tenant =>
      assert(accessSet(results, tenant).intersect(accessSet(df, tenant)).isEmpty)
    )
    assert(results.count == 10)
  }

  override def testObjects(): Seq[TestObject[ComplementSampler]] = Seq(
    new TestObject(ca, df)
  )

  override def reader: MLReadable[_] = ComplementSampler

}

