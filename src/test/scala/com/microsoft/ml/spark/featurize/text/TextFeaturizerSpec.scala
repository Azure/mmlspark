// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.featurize.text

import com.microsoft.ml.spark.core.schema.DatasetExtensions._
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{NGram, Tokenizer}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class TextFeaturizerSpec extends EstimatorFuzzing[TextFeaturizer]{
  override val testFitting = true

  lazy val dfRaw: DataFrame = spark
    .createDataFrame(Seq((0, "Hi I"),
                         (1, "I wish for snow today"),
                         (2, "we Cant go to the park, because of the snow!"),
                         (3, "")))
    .toDF("label", "sentence")
  lazy val dfTok: DataFrame = new Tokenizer()
    .setInputCol("sentence")
    .setOutputCol("tokens")
    .transform(dfRaw)
  lazy val dfNgram: DataFrame =
    new NGram().setInputCol("tokens").setOutputCol("ngrams").transform(dfTok)

  test("operate on sentences,tokens,or ngrams") {
    val tfRaw = new TextFeaturizer()
      .setInputCol("sentence")
      .setOutputCol("features")
      .setNumFeatures(20)
    val tfTok = new TextFeaturizer()
      .setUseTokenizer(false)
      .setInputCol("tokens")
      .setOutputCol("features")
      .setNumFeatures(20)
    val tfNgram = new TextFeaturizer()
      .setUseTokenizer(false)
      .setUseNGram(false)
      .setInputCol("ngrams")
      .setOutputCol("features")
      .setNumFeatures(20)

    val dfRaw2 = tfRaw.fit(dfRaw).transform(dfRaw)
    val dfTok2 = tfTok.fit(dfTok).transform(dfTok)
    val dfNgram2 = tfNgram.fit(dfNgram).transform(dfNgram)

    val linesRaw = dfRaw2.getSVCol("features")
    val linesTok = dfTok2.getSVCol("features")
    val linesNgram = dfNgram2.getSVCol("features")

    assert(linesRaw.length == 4)
    assert(linesTok.length == 4)
    assert(linesNgram.length == 4)
    assert(linesRaw(0)(0) == 0.0)
    assert(linesTok(1)(9) == 0.0)
    assert(linesNgram(2)(7) == 1.8325814637483102)
    assert(linesNgram(3)(1) == 0.0)
  }

  test("throw errors if the schema is incorrect") {
    val tfRaw = new TextFeaturizer()
      .setUseTokenizer(true)
      .setInputCol("sentence")
      .setOutputCol("features")
      .setNumFeatures(20)
    val tfTok = new TextFeaturizer()
      .setUseTokenizer(false)
      .setInputCol("tokens")
      .setOutputCol("features")
      .setNumFeatures(20)
    assertSparkException[IllegalArgumentException](tfRaw.setInputCol("tokens"),           dfTok)
    assertSparkException[IllegalArgumentException](tfRaw.setInputCol("ngrams"),           dfNgram)
    assertSparkException[IllegalArgumentException](tfTok.setInputCol("sentence"),         dfRaw)
    assertSparkException[IllegalArgumentException](tfRaw.setInputCol("tokens_incorrect"), dfTok)
    assertSparkException[IllegalArgumentException](tfRaw.setOutputCol("tokens"),          dfTok)
  }

  override def testObjects(): Seq[TestObject[TextFeaturizer]] =
    List(new TestObject(new TextFeaturizer().setInputCol("sentence"), dfRaw))

  override def reader: MLReadable[_] = TextFeaturizer
  override def modelReader: MLReadable[_] = PipelineModel
}
