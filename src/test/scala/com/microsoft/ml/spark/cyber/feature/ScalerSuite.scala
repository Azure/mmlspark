// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cyber.feature

import com.microsoft.ml.spark.core.test.base.{DataFrameEquality, TestBase}
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.scalatest.Assertion

trait ScalarTestUtilities extends TestBase with DataFrameEquality {

  import session.implicits._

  lazy val df: DataFrame = Seq(
    ("t1", "5", 500.0),
    ("t1", "6", 600.0),
    ("t2", "7", 700.0),
    ("t2", "8", 800.0),
    ("t3", "9", 900.0)
  ).toDF("tenent", "name", "score").cache()

  lazy val minMax: PartitionedMinMaxScaler = new PartitionedMinMaxScaler()
    .setPartitionKey("tenent")
    .setInputCol("score")
    .setOutputCol("scaled")

  lazy val standard: PartitionedStandardScaler = new PartitionedStandardScaler()
    .setPartitionKey("tenent")
    .setInputCol("score")
    .setOutputCol("scaled")

  override val epsilon: Double = .001

  def assertEqual(a: List[Double], b: List[Double]): Assertion = {
    assert(a.zip(b).forall(p => p._1 === p._2))
  }

}

class PartitionedMinMaxScalerSuite extends EstimatorFuzzing[PartitionedMinMaxScaler]
  with ScalarTestUtilities {

  test("basic usage") {
    val resultsDF = minMax.fit(df).transform(df)
    //resultsDF.printSchema()
    val results = resultsDF.select("scaled")
      .collect().map(_.getDouble(0)).toList
    assertEqual(results, List(0, 1, 0, 1, .5))
  }

  test("basic usage without partition key") {
    println(df.count()) // This is important!, if this is included and cross joins arent on this will fail
    val resultsDF = new PartitionedMinMaxScaler()
      .setInputCol("score")
      .setOutputCol("scaled")
      .fit(df)
      .transform(df)
    //resultsDF.printSchema()
    val results = resultsDF.select("scaled").collect().map(_.getDouble(0)).toList
    assertEqual(results, List(0.0, 0.25, 0.5, 0.75, 1.0))
  }

  override def testObjects(): Seq[TestObject[PartitionedMinMaxScaler]] = Seq(
    new TestObject(minMax, df)
  )

  override def reader: MLReadable[_] = PartitionedMinMaxScaler

  override def modelReader: MLReadable[_] = PartitionedMinMaxScalerModel
}

class PartitionedMinMaxModelSuite
  extends TransformerFuzzing[PartitionedMinMaxScalerModel]
    with ScalarTestUtilities {
  override def testObjects(): Seq[TestObject[PartitionedMinMaxScalerModel]] = Seq(
    new TestObject(minMax.fit(df), df)
  )

  override def reader: MLReadable[_] = PartitionedMinMaxScalerModel
}

class PartitionedStandardScalarSuite extends EstimatorFuzzing[PartitionedStandardScaler]
  with ScalarTestUtilities {

  test("basic usage") {
    val results = standard.fit(df).transform(df).select("scaled")
      .collect().map(_.getDouble(0)).toList
    assertEqual(results, List(-1.0, 1.0, -1.0, 1.0, 0.0))
  }

  override def testObjects(): Seq[TestObject[PartitionedStandardScaler]] = Seq(
    new TestObject(standard, df)
  )

  override def reader: MLReadable[_] = PartitionedStandardScaler

  override def modelReader: MLReadable[_] = PartitionedStandardScalerModel
}

class PartitionedStandardModelSuite
  extends TransformerFuzzing[PartitionedStandardScalerModel]
    with ScalarTestUtilities {
  override def testObjects(): Seq[TestObject[PartitionedStandardScalerModel]] = Seq(
    new TestObject(standard.fit(df), df)
  )

  override def reader: MLReadable[_] = PartitionedStandardScalerModel
}
