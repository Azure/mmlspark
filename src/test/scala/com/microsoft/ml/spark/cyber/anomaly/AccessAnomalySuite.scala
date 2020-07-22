// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cyber.anomaly

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, col}
import org.apache.spark.sql.types.StringType

import scala.util.Random

trait AccessAnomalyTestUtils extends TestBase {
  import session.implicits._

  lazy val r = new Random(0)

  def sample(n: Int, low: Int, high: Int): Seq[Int] = {
    Seq.fill(n) {
      r.nextInt(high - low) + low
    }
  }

  lazy val users: Seq[Int] = sample(50, 0, 10) ++
    sample(50, 10, 20) ++
    sample(50, 20, 30)
  lazy val resources: Seq[Int] = sample(50, 0, 10) ++
    sample(50, 10, 20) ++
    sample(50, 20, 30)
  lazy val acceses: Seq[Int] = sample(150, 1, 10)
  lazy val tenant: Seq[Int] = sample(150, low = 0, high = 2)

  def zipDF(users: Seq[Int], resources: Seq[Int], acceses: Seq[Int], tenant: Seq[Int]): DataFrame = {
    Seq(users, resources, acceses, tenant)
      .transpose.map(l => (l(0), l(1), l(2), l(3)))
      .toDF("user", "res", "acc", "tenant")
      .withColumn("tenant", col("tenant").cast(StringType))
  }

  lazy val trainDF: DataFrame = zipDF(users, resources, acceses, tenant)

  lazy val usersAnomaly: Seq[Int] = users.take(10)
  lazy val resourcesAnomaly: Seq[Int] = resources.takeRight(10)
  lazy val accessesAnomaly: Seq[Int] = sample(10, 1, 10)
  lazy val tenantAnomaly: Seq[Int] = sample(10, low = 0, high = 2)

  lazy val testRegDF: DataFrame = trainDF
  lazy val testAnomalyDF: DataFrame = zipDF(usersAnomaly, resourcesAnomaly, accessesAnomaly, tenantAnomaly)

  def aa: AccessAnomaly = new AccessAnomaly()
    .setUserCol("user")
    .setResCol("res")
    .setAccessCol("acc")
    .setMaxIter(5)
    .setTenantCol("tenant")
    .setOutputCol("zscore")
    .setSeparateTenants(true)
}

class AccessAnomalySuite extends EstimatorFuzzing[AccessAnomaly] with AccessAnomalyTestUtils {
  override val epsilon = .1
  override val sortInDataframeEquality: Boolean = true

  test("basic usage") {
    val fitModel = aa.fit(trainDF)
    val avgRegScore = fitModel.transform(testRegDF)
      .groupBy().agg(avg("zscore")).collect().head.getDouble(0)
    val avgAnomalyScore = fitModel.transform(testAnomalyDF)
      .groupBy().agg(avg("zscore")).collect().head.getDouble(0)
    assert(avgAnomalyScore > .9)
    assert(avgRegScore < .5)
  }

  test("can work without acceses") {
    val fitModel = aa.fit(trainDF)
    val avgAnomalyScore = fitModel.transform(testAnomalyDF.drop("acc"))
      .groupBy().agg(avg("zscore")).collect().head.getDouble(0)
    assert(avgAnomalyScore > .9)
  }

  override def testObjects(): Seq[TestObject[AccessAnomaly]] = Seq(
    new TestObject(aa, trainDF, testAnomalyDF)
  )

  override def reader: MLReadable[_] = AccessAnomaly

  override def modelReader: MLReadable[_] = AccessAnomalyModel

}

class AccessAmonmalyModelSuite extends TransformerFuzzing[AccessAnomalyModel] with AccessAnomalyTestUtils {
  override val epsilon = .1
  override val sortInDataframeEquality: Boolean = true

  override def testObjects(): Seq[TestObject[AccessAnomalyModel]] = Seq(new TestObject(aa.fit(trainDF), testAnomalyDF))

  override def reader: MLReadable[_] = AccessAnomalyModel
}
