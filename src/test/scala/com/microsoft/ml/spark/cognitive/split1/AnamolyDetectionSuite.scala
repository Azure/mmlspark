// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split1

import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{col, collect_list, lit, struct}

trait AnomalyKey {
  lazy val anomalyKey: String = sys.env.getOrElse("ANOMALY_API_KEY", Secrets.AnomalyApiKey)
}

trait AnomalyDetectorSuiteBase extends TestBase with AnomalyKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("1972-01-01T00:00:00Z", 826.0),
    ("1972-02-01T00:00:00Z", 799.0),
    ("1972-03-01T00:00:00Z", 890.0),
    ("1972-04-01T00:00:00Z", 900.0),
    ("1972-05-01T00:00:00Z", 766.0),
    ("1972-06-01T00:00:00Z", 805.0),
    ("1972-07-01T00:00:00Z", 821.0),
    ("1972-08-01T00:00:00Z", 20000.0),
    ("1972-09-01T00:00:00Z", 883.0),
    ("1972-10-01T00:00:00Z", 898.0),
    ("1972-11-01T00:00:00Z", 957.0),
    ("1972-12-01T00:00:00Z", 924.0),
    ("1973-01-01T00:00:00Z", 881.0),
    ("1973-02-01T00:00:00Z", 837.0),
    ("1973-03-01T00:00:00Z", 90000.0)
  ).toDF("timestamp","value")
    .withColumn("group", lit(1))
    .withColumn("inputs", struct(col("timestamp"), col("value")))
    .groupBy(col("group"))
    .agg(collect_list(col("inputs")).alias("inputs"))

  lazy val df2: DataFrame = Seq(
    ("2000-01-24T08:46:00Z", 826.0),
    ("2000-01-24T08:47:00Z", 799.0),
    ("2000-01-24T08:48:00Z", 890.0),
    ("2000-01-24T08:49:00Z", 900.0),
    ("2000-01-24T08:50:00Z", 766.0),
    ("2000-01-24T08:51:00Z", 805.0),
    ("2000-01-24T08:52:00Z", 821.0),
    ("2000-01-24T08:53:00Z", 20000.0),
    ("2000-01-24T08:54:00Z", 883.0),
    ("2000-01-24T08:55:00Z", 898.0),
    ("2000-01-24T08:56:00Z", 957.0),
    ("2000-01-24T08:57:00Z", 924.0),
    ("2000-01-24T08:58:00Z", 881.0),
    ("2000-01-24T08:59:00Z", 837.0),
    ("2000-01-24T09:00:00Z", 90000.0)
  ).toDF("timestamp","value")
    .withColumn("group", lit(1))
    .withColumn("inputs", struct(col("timestamp"), col("value")))
    .groupBy(col("group"))
    .agg(collect_list(col("inputs")).alias("inputs"))

}

class DetectLastAnomalySuite extends TransformerFuzzing[DetectLastAnomaly] with AnomalyDetectorSuiteBase  {

  lazy val ad: DetectLastAnomaly = new DetectLastAnomaly()
    .setSubscriptionKey(anomalyKey)
    .setLocation("westus2")
    .setOutputCol("anomalies")
    .setSeriesCol("inputs")
    .setGranularity("monthly")
    .setErrorCol("errors")

  test("Basic Usage"){
    val fromRow = ADLastResponse.makeFromRowConverter
    val result = fromRow(ad.transform(df)
      .select("anomalies")
      .collect()
      .head.getStruct(0))
    assert(result.isAnomaly)
  }

  test("minutely Usage"){
    val fromRow = ADLastResponse.makeFromRowConverter
    val result = fromRow(ad.setGranularity("minutely").transform(df2)
      .select("anomalies")
      .collect()
      .head.getStruct(0))
    assert(result.isAnomaly)
  }

  override def testObjects(): Seq[TestObject[DetectLastAnomaly]] =
    Seq(new TestObject(ad, df))

  override def reader: MLReadable[_] = DetectLastAnomaly
}

class DetectAnomaliesSuite extends TransformerFuzzing[DetectAnomalies] with AnomalyDetectorSuiteBase  {

  lazy val ad: DetectAnomalies = new DetectAnomalies()
    .setSubscriptionKey(anomalyKey)
    .setLocation("westus2")
    .setOutputCol("anomalies")
    .setSeriesCol("inputs")
    .setGranularity("monthly")

  test("Basic Usage"){
    val fromRow = ADEntireResponse.makeFromRowConverter
    val result = fromRow(ad.transform(df)
      .select("anomalies")
      .collect()
      .head.getStruct(0))
    assert(result.isAnomaly.count({b => b}) == 2)
  }

  override def testObjects(): Seq[TestObject[DetectAnomalies]] =
    Seq(new TestObject(ad, df))

  override def reader: MLReadable[_] = DetectAnomalies
}

class SimpleDetectAnomaliesSuite extends TransformerFuzzing[SimpleDetectAnomalies]
  with AnomalyDetectorSuiteBase  {

  lazy val baseSeq = Seq(
    ("1972-01-01T00:00:00Z", 826.0),
    ("1972-02-01T00:00:00Z", 799.0),
    ("1972-03-01T00:00:00Z", 890.0),
    ("1972-04-01T00:00:00Z", 900.0),
    ("1972-05-01T00:00:00Z", 766.0),
    ("1972-06-01T00:00:00Z", 805.0),
    ("1972-07-01T00:00:00Z", 821.0),
    ("1972-08-01T00:00:00Z", 20000.0),
    ("1972-09-01T00:00:00Z", 883.0),
    ("1972-10-01T00:00:00Z", 898.0),
    ("1972-11-01T00:00:00Z", 957.0),
    ("1972-12-01T00:00:00Z", 924.0),
    ("1973-01-01T00:00:00Z", 881.0),
    ("1973-02-01T00:00:00Z", 837.0),
    ("1973-03-01T00:00:00Z", 9000.0)
  )

  import spark.implicits._

  lazy val sdf: DataFrame = baseSeq.map(p => (p._1,p._2,1.0))
    .++(baseSeq.map(p => (p._1,p._2,2.0)))
    .toDF("timestamp","value","group")

  lazy val sdf2: DataFrame = baseSeq.map(p => (p._1,p._2,1.0))
    .++(baseSeq.reverse.map(p => (p._1,p._2,2.0)))
    .toDF("timestamp","value","group")

  lazy val sdf3: DataFrame = baseSeq.map(p => (p._1,p._2,1.0))
    .++(baseSeq.reverse.take(2).map(p => (p._1,p._2,2.0)))
    .toDF("timestamp","value","group")

  lazy val sad: SimpleDetectAnomalies = new SimpleDetectAnomalies()
    .setSubscriptionKey(anomalyKey)
    .setLocation("westus2")
    .setOutputCol("anomalies")
    .setGroupbyCol("group")
    .setGranularity("monthly")

  test("Basic Usage"){
    val result = sad.transform(sdf)
      .collect().head.getAs[Row]("anomalies")
    assert(!result.getBoolean(0))
  }

  test("Reverse Reverse!"){
    sad.transform(sdf2)
      .show(truncate=false)
  }

  test("Error handling"){
    sad.transform(sdf3)
      .show(truncate=false)
  }

  //TODO Nulls, different cardinalities

  override def testObjects(): Seq[TestObject[SimpleDetectAnomalies]] =
    Seq(new TestObject(sad, sdf))

  override def reader: MLReadable[_] = SimpleDetectAnomalies
}
