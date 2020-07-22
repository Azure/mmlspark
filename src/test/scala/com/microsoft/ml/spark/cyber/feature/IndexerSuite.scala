// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cyber.feature

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.functions.col

trait IndexerTestUtils extends TestBase {

  import session.implicits._

  lazy val df: DataFrame = Seq(
    ("1", "a", "A", 0, 0, 0),
    ("1", "b", "A", 1, 0, 1),
    ("1", "a", "B", 0, 1, 0),
    ("2", "aa", "AA", 1, 0, 3),
    ("2", "bb", "AA", 0, 0, 2),
    ("3", "b", "B", 0, 0, 4)
  ).toDF("tenant", "user", "res",
    "expected_uid", "expected_rid", "expected_uid_2").cache()

  def psi: PartitionedStringIndexer = new PartitionedStringIndexer()
    .setPartitionKey("tenant")
    .setInputCol("user")
    .setOutputCol("uid")
}

class PartitionedStringIndexerSuite extends EstimatorFuzzing[PartitionedStringIndexer] with IndexerTestUtils {

  test("basic usage") {
    val results = psi.fit(df).transform(df)
    assert(results.schema === df.schema.add("uid", LongType))
    assert(results.where(col("uid") =!= col("expected_uid")).count() === 0)
  }

  test("non overlapping") {
    val results = psi.setOverlappingIndicies(false).fit(df).transform(df)
    assert(results.where(col("uid") =!= col("expected_uid_2")).count() === 0)
  }

  test("inverse transform usage") {
    val model = psi.fit(df)
    val df2 = model.transform(df).withColumnRenamed("user", "expected_user")
    assert(model.inverseTransform(df2)
      .where(col("expected_user") =!= col("user")).count() === 0)
  }

  override def testObjects(): Seq[TestObject[PartitionedStringIndexer]] = Seq(
    new TestObject(psi, df)
  )

  override def reader: MLReadable[_] = PartitionedStringIndexer

  override def modelReader: MLReadable[_] = PartitionedStringIndexerModel

}

class PartitionedStringIndexerModelSuite extends TransformerFuzzing[PartitionedStringIndexerModel]
  with IndexerTestUtils {

  override def testObjects(): Seq[TestObject[PartitionedStringIndexerModel]] = Seq(
    new TestObject(psi.fit(df), df)
  )

  override def reader: MLReadable[_] = PartitionedStringIndexerModel

}

