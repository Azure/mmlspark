// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{StringType, StructType}

class SimpleHTTPTransformerSuite
    extends TransformerFuzzing[SimpleHTTPTransformer] with WithServer {

  import session.implicits._
  val df: DataFrame = sc.parallelize((1 to 10).map(Tuple1(_))).toDF("data")

  def simpleTransformer: SimpleHTTPTransformer =
    new SimpleHTTPTransformer()
      .setInputCol("data")
      .setOutputParser(new JSONOutputParser()
                         .setDataType(new StructType().add("blah", StringType)))
      .setUrl(url)
      .setOutputCol("results")

  test("HttpTransformerTest") {
    val results = simpleTransformer.transform(df).collect
    assert(results.length == 10)
    results.foreach(r =>
      assert(r.getStruct(1).getString(0) === "more blah"))
    assert(results(0).schema.fields.length == 2)
  }

  test("Concurrent HttpTransformerTest") {
    val results =
      new SimpleHTTPTransformer()
        .setInputCol("data")
        .setOutputParser(new JSONOutputParser()
                           .setDataType(new StructType().add("blah", StringType)))
        .setUrl(url)
        .setOutputCol("results")
        .setConcurrency(3)
        .transform(df)
        .collect
    assert(results.length  ==10)
    assert(results.forall(_.getStruct(1).getString(0) == "more blah"))
    assert(results(0).schema.fields.length == 2)
  }

  override def testObjects(): Seq[TestObject[SimpleHTTPTransformer]] =
    Seq(new TestObject(simpleTransformer, df))

  override def reader: MLReadable[_] = SimpleHTTPTransformer

}
