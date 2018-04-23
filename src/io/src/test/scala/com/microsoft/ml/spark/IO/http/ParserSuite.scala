// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.http.client.methods.HttpPost
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

trait ParserUtils extends WithServer {

  def sampleDf(spark: SparkSession): DataFrame = {
    val df = spark.createDataFrame((1 to 10).map(Tuple1(_)))
      .toDF("data")
    val df2 = new JSONInputParser().setInputCol("data")
      .setOutputCol("parsedInput").setUrl(url)
      .transform(df)
      .withColumn("unparsedOutput", udf({x: Int =>
        HTTPResponseData(
          Array(),
          EntityData(
            "{\"foo\": \"here\"}".getBytes, None, 0, None, false, false, false),
          StatusLineData(ProtocolVersionData("foo",1,1),200, "bar"),
          "en")
        }).apply(col("data"))
      )

    new JSONOutputParser()
      .setDataType(new StructType().add("foo", StringType))
      .setInputCol("unparsedOutput")
      .setOutputCol("parsedOutput")
      .transform(df2)
  }

  def makeTestObject[T <: Transformer](t: T, session: SparkSession): Seq[TestObject[T]] = {
    Seq(new TestObject(t, sampleDf(session)))
  }

}

class JsonInputParserSuite extends TransformerFuzzing[JSONInputParser] with ParserUtils {
  override def testObjects(): Seq[TestObject[JSONInputParser]] = makeTestObject(
    new JSONInputParser().setInputCol("data").setOutputCol("out")
      .setUrl(url), session)
  override def reader: MLReadable[_] = JSONInputParser
}

class JsonOutputParserSuite extends TransformerFuzzing[JSONOutputParser] with ParserUtils {
  override def testObjects(): Seq[TestObject[JSONOutputParser]] = makeTestObject(
    new JSONOutputParser().setInputCol("unparsedOutput").setOutputCol("out")
      .setDataType(new StructType().add("foo", StringType)), session)
  override def reader: MLReadable[_] = JSONOutputParser
}

class CustomInputParserSuite extends TransformerFuzzing[CustomInputParser] with ParserUtils {
  override def testObjects(): Seq[TestObject[CustomInputParser]] = makeTestObject(
    new CustomInputParser().setInputCol("data").setOutputCol("out")
      .setUDF({ x: Int => new HttpPost(s"http://$x") }), session)
  override def reader: MLReadable[_] = CustomInputParser
}

class CustomOutputParserSuite extends TransformerFuzzing[CustomOutputParser] with ParserUtils {
  override def testObjects(): Seq[TestObject[CustomOutputParser]] = makeTestObject(
    new CustomOutputParser().setInputCol("unparsedOutput").setOutputCol("out")
      .setUDF({ x: HTTPResponseData => x.locale }), session)
  override def reader: MLReadable[_] = CustomOutputParser
}
