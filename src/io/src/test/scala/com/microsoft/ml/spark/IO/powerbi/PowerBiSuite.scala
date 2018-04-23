// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.powerbi

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.IO.binary.FileReaderUtils
import org.apache.spark.SparkException
import org.apache.spark.sql.functions.{current_timestamp, lit}

class PowerBiSuite extends TestBase with FileReaderUtils {

  lazy val url = sys.env("MML_POWERBI_URL")
  val df = session
    .createDataFrame(Seq(
      (Some(0), "a"),
      (Some(1), "b"),
      (Some(2), "c"),
      (Some(3), ""),
      (None, "bad_row")))
    .toDF("bar", "foo")
    .withColumn("baz", current_timestamp())
  val bigdf = (1 to 5).foldRight(df) { case (_, ldf) => ldf.union(df) }.repartition(2)

  test("write to powerBi", TestBase.BuildServer) {
    PowerBIWriter.write(df, url)
  }

  ignore("throw useful error message when given an improper dataset") {
    //TODO figure out why this does not throw errors on the build machine
    assertThrows[SparkException] {
      PowerBIWriter.write(df.withColumn("bad", lit("foo")), url)
    }
  }

  test("stream to powerBi", TestBase.BuildServer) {
    bigdf.write.parquet(tmpDir + "powerBI.parquet")
    val sdf = session.readStream.schema(df.schema).parquet(tmpDir + "powerBI.parquet")
    val q1 = PowerBIWriter.stream(sdf, url).start()
    q1.processAllAvailable()
  }

}
