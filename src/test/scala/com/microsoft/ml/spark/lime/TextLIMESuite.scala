// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lime

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import com.microsoft.ml.spark.stages.UDFTransformer
import com.microsoft.ml.spark.stages.udfs.get_value_udf
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.scalactic.Equality
import org.scalatest.Assertion

class TextLIMESuite extends TransformerFuzzing[TextLIME] {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("hi this is example 1", 1.0),
    ("hi this is cat 1", 0.0),
    ("hi this is example 1", 1.0),
    ("foo this is example 1", 1.0),
    ("hi this is example 1", 1.0),
    ("hi this is cat 1", 0.0),
    ("hi this is example 1", 1.0),
    ("hi this is example 1", 1.0),
    ("hi this is example 1", 1.0),
    ("hi bar is cat 1", 0.0),
    ("hi this is example 1", 1.0)
  ).toDF("text", "label")

  lazy val tok: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
  lazy val si: HashingTF = new HashingTF().setInputCol("tokens").setOutputCol("features")
  lazy val lr: LogisticRegression = new LogisticRegression()
    .setFeaturesCol("features").setLabelCol("label").setProbabilityCol("prob")
  lazy val selectClass: UDFTransformer = new UDFTransformer()
    .setInputCol("prob")
    .setOutputCol("prob")
    .setUDF(get_value_udf(1))

  lazy val textClassifier: Pipeline = new Pipeline().setStages(Array(tok, si, lr, selectClass))

  lazy val fitModel: PipelineModel = textClassifier.fit(df)

  lazy val textLime: TextLIME = new TextLIME()
    .setModel(fitModel)
    .setInputCol("text")
    .setPredictionCol("prob")
    .setOutputCol("interpretationWeights")
    .setTokenCol("interpretationTokens")
    .setNSamples(1000)

  lazy val interpretedDataFrame: DataFrame = textLime.transform(df)


  test(" lime text") {
    val results = interpretedDataFrame
      .select(
        UDFUtils.oldUdf({ vec: org.apache.spark.ml.linalg.Vector => vec(3) },
          DoubleType)(col("interpretationWeights")).alias("weight"),
        col("interpretationTokens").getItem(3).alias("word")
      )

    assert(results.collect().forall { r =>
      val weight = r.getAs[Double]("weight")
      r.getAs[String]("word") match {
        case "cat" => weight < 0
        case "example" => weight > .2
        case w => throw new IllegalArgumentException(s"unknown token $w")
      }
    })
  }

  override def testObjects(): Seq[TestObject[TextLIME]] = Seq(new TestObject(textLime, df))

  override def reader: MLReadable[_] = ImageLIME

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    super.assertDFEq(
      df1.select("interpretationTokens", "text", "label"),
      df2.select("interpretationTokens", "text", "label"))(eq)
  }

}
