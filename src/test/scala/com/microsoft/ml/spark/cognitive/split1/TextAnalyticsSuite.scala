// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split1

import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import com.microsoft.ml.spark.stages.FixedMiniBatchTransformer
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row}

trait TextKey {
  lazy val textKey = sys.env.getOrElse("TEXT_API_KEY", Secrets.CognitiveApiKey)
}

class LanguageDetectorSuite extends TransformerFuzzing[LanguageDetectorV2] with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "Hello World",
    "Bonjour tout le monde",
    "La carretera estaba atascada. Había mucho tráfico el día de ayer.",
    ":) :( :D"
  ).toDF("text2")

  lazy val detector: LanguageDetectorV2 = new LanguageDetectorV2()
    .setSubscriptionKey(textKey)
    .setUrl("https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/languages")
    .setTextCol("text2")
    .setOutputCol("replies")

  test("Basic Usage") {
    val replies = detector.transform(df)
      .withColumn("lang", col("replies").getItem(0)
        .getItem("detectedLanguages").getItem(0)
        .getItem("name"))
      .select("lang")
      .collect().toList
    assert(replies(0).getString(0) == "English" && replies(2).getString(0) == "Spanish")
  }

  test("Is serializable ") {
    val replies = detector.transform(df.repartition(3))
      .withColumn("lang", col("replies").getItem(0)
        .getItem("detectedLanguages").getItem(0)
        .getItem("name"))
      .select("text2","lang")
      .sort("text2")
      .collect().toList
    assert(replies(2).getString(1) == "English" && replies(3).getString(1) == "Spanish")
  }

  test("Batch Usage") {
    val batchedDF = new FixedMiniBatchTransformer().setBatchSize(10).transform(df.coalesce(1))
    val tdf = detector.transform(batchedDF)
    val replies = tdf.collect().head.getAs[Seq[Row]]("replies")
    assert(replies.length == 4)
    val languages = replies.map(_.getAs[Seq[Row]]("detectedLanguages").head.getAs[String]("name")).toSet
    assert(languages("Spanish") && languages("English"))
  }

  override def testObjects(): Seq[TestObject[LanguageDetectorV2]] =
    Seq(new TestObject[LanguageDetectorV2](detector, df))

  override def reader: MLReadable[_] = LanguageDetectorV2
}

class LanguageDetectorV3Suite extends TransformerFuzzing[LanguageDetector] with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "Hello World",
    "Bonjour tout le monde",
    "La carretera estaba atascada. Había mucho tráfico el día de ayer.",
    ":) :( :D"
  ).toDF("text")

  lazy val detector: LanguageDetector = new LanguageDetector()
    .setSubscriptionKey(textKey)
    .setUrl("https://eastus.api.cognitive.microsoft.com/text/analytics/v3.0/languages")
    .setOutputCol("replies")

  test("Basic Usage") {
    val replies = detector.transform(df)
      .withColumn("lang", col("replies").getItem(0)
        .getItem("detectedLanguage")
        .getItem("name"))
      .select("lang")
      .collect().toList
    assert(replies(0).getString(0) == "English" && replies(2).getString(0) == "Spanish")
  }

  override def testObjects(): Seq[TestObject[LanguageDetector]] =
    Seq(new TestObject[LanguageDetector](detector, df))

  override def reader: MLReadable[_] = LanguageDetector
}

class EntityDetectorSuite extends TransformerFuzzing[EntityDetectorV2] with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("1", "Microsoft released Windows 10"),
    ("2", "In 1975, Bill Gates III and Paul Allen founded the company.")
  ).toDF("id", "text")

  lazy val detector: EntityDetectorV2 = new EntityDetectorV2()
    .setSubscriptionKey(textKey)
    .setUrl("https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/entities")
    .setLanguage("en")
    .setOutputCol("replies")

  test("Basic Usage") {
    val results = detector.transform(df)
      .withColumn("entities",
        col("replies").getItem(0).getItem("entities").getItem("name"))
      .select("id", "entities").collect().toList
    assert(results.head.getSeq[String](1).toSet
      .intersect(Set("Windows 10", "Windows 10 Mobile", "Microsoft")).size == 2)
  }

  override def testObjects(): Seq[TestObject[EntityDetectorV2]] =
    Seq(new TestObject[EntityDetectorV2](detector, df))

  override def reader: MLReadable[_] = EntityDetectorV2
}

class EntityDetectorSuiteV3 extends TransformerFuzzing[EntityDetector] with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("1", "Microsoft released Windows 10"),
    ("2", "In 1975, Bill Gates III and Paul Allen founded the company.")
  ).toDF("id", "text")

  lazy val detector: EntityDetector = new EntityDetector()
    .setSubscriptionKey(textKey)
    .setLocation("eastus")
    .setLanguage("en")
    .setOutputCol("replies")

  test("Basic Usage") {
    val results = detector.transform(df)
      .withColumn("entities",
        col("replies").getItem(0).getItem("entities").getItem("name"))
      .select("id", "entities").collect().toList
    println(results)
    assert(results.head.getSeq[String](1).toSet
      .intersect(Set("Windows 10", "Windows 10 Mobile", "Microsoft")).size == 2)
  }

  override def testObjects(): Seq[TestObject[EntityDetector]] =
    Seq(new TestObject[EntityDetector](detector, df))

  override def reader: MLReadable[_] = EntityDetector
}

trait TextSentimentBaseSuite extends TestBase with TextKey {
  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("en", "Hello world. This is some input text that I love."),
    ("fr", "Bonjour tout le monde"),
    ("es", "La carretera estaba atascada. Había mucho tráfico el día de ayer."),
    (null, "ich bin ein berliner"),
    (null, null),
    ("en", null)
  ).toDF("lang", "text")
}

class TextSentimentV3Suite extends TransformerFuzzing[TextSentiment] with TextSentimentBaseSuite {
  lazy val t: TextSentiment = new TextSentiment()
    .setSubscriptionKey(textKey)
    .setLocation("eastus")
    .setLanguageCol("lang")
    .setModelVersion("latest")
    .setShowStats(true)
    .setOutputCol("replies")

  test("Basic Usage") {
    val results = t.transform(df).select(
      col("replies").alias("scoredDocuments")
    ).collect().toList

    assert(List(4,5).forall(results(_).get(0) == null))
    assert(
      results(0).getSeq[Row](0).head.getString(0) == "positive" &&
      results(2).getSeq[Row](0).head.getString(0) == "negative")
  }

  test("batch usage"){
    val t = new TextSentiment()
      .setSubscriptionKey(textKey)
      .setLocation("eastus")
      .setTextCol("text")
      .setLanguage("en")
      .setOutputCol("score")
    val batchedDF = new FixedMiniBatchTransformer().setBatchSize(10).transform(df.coalesce(1))
    val tdf = t.transform(batchedDF)
    val replies = tdf.collect().head.getAs[Seq[Row]]("score")
    assert(replies.length == 6)
  }

  override def testObjects(): Seq[TestObject[TextSentiment]] =
    Seq(new TestObject[TextSentiment](t, df))

  override def reader: MLReadable[_] = TextSentiment
}

class TextSentimentSuite extends TransformerFuzzing[TextSentimentV2] with TextSentimentBaseSuite {

  lazy val t: TextSentimentV2 = new TextSentimentV2()
    .setSubscriptionKey(textKey)
    .setUrl("https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment")
    .setLanguageCol("lang")
    .setOutputCol("replies")

  test("Basic Usage") {
    val results = t.transform(df).withColumn("score",
      col("replies").getItem(0).getItem("score"))
      .select("score").collect().toList

    assert(List(4,5).forall(results(_).get(0) == null))
    assert(results(0).getFloat(0) > .5 && results(2).getFloat(0) < .5)
  }

  test("batch usage"){
    val t = new TextSentimentV2()
      .setSubscriptionKey(textKey)
      .setLocation("eastus")
      .setTextCol("text")
      .setLanguage("en")
      .setOutputCol("score")
    val batchedDF = new FixedMiniBatchTransformer().setBatchSize(10).transform(df.coalesce(1))
    val tdf = t.transform(batchedDF)
    val replies = tdf.collect().head.getAs[Seq[Row]]("score")
    assert(replies.length == 6)
  }

  override def testObjects(): Seq[TestObject[TextSentimentV2]] =
    Seq(new TestObject[TextSentimentV2](t, df))

  override def reader: MLReadable[_] = TextSentimentV2
}

class KeyPhraseExtractorSuite extends TransformerFuzzing[KeyPhraseExtractorV2] with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("en", "Hello world. This is some input text that I love."),
    ("fr", "Bonjour tout le monde"),
    ("es", "La carretera estaba atascada. Había mucho tráfico el día de ayer."),
    ("en", null)
  ).toDF("lang", "text")

  lazy val t: KeyPhraseExtractorV2 = new KeyPhraseExtractorV2()
    .setSubscriptionKey(textKey)
    .setUrl("https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/keyPhrases")
    .setLanguageCol("lang")
    .setOutputCol("replies")

  test("Basic Usage") {
    val results = t.transform(df).withColumn("phrases",
      col("replies").getItem(0).getItem("keyPhrases"))
      .select("phrases").collect().toList

    println(results)

    assert(results(0).getSeq[String](0).toSet === Set("world", "input text"))
    assert(results(2).getSeq[String](0).toSet === Set("carretera", "tráfico", "día"))
  }

  override def testObjects(): Seq[TestObject[KeyPhraseExtractorV2]] =
    Seq(new TestObject[KeyPhraseExtractorV2](t, df))

  override def reader: MLReadable[_] = KeyPhraseExtractorV2
}

class KeyPhraseExtractorV3Suite extends TransformerFuzzing[KeyPhraseExtractor] with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("en", "Hello world. This is some input text that I love."),
    ("fr", "Bonjour tout le monde"),
    ("es", "La carretera estaba atascada. Había mucho tráfico el día de ayer."),
    ("en", null)
  ).toDF("lang", "text")

  lazy val t: KeyPhraseExtractor = new KeyPhraseExtractor()
    .setSubscriptionKey(textKey)
    .setUrl("https://eastus.api.cognitive.microsoft.com/text/analytics/v3.0/keyPhrases")
    .setLanguageCol("lang")
    .setOutputCol("replies")

  test("Basic Usage") {
    val results = t.transform(df).withColumn("phrases",
      col("replies").getItem(0).getItem("keyPhrases"))
      .select("phrases").collect().toList

    println(results)

    assert(results(0).getSeq[String](0).toSet === Set("Hello world", "input text"))
    assert(results(2).getSeq[String](0).toSet === Set("mucho tráfico", "carretera", "ayer"))
  }

  override def testObjects(): Seq[TestObject[KeyPhraseExtractor]] =
    Seq(new TestObject[KeyPhraseExtractor](t, df))

  override def reader: MLReadable[_] = KeyPhraseExtractor
}

class NERSuite extends TransformerFuzzing[NERV2] with TextKey {
  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("1", "en", "Jeff bought three dozen eggs because there was a 50% discount."),
    ("2", "en", "In 1929, The Great Depression began. By 1933, the GDP in America fell by 25%."),
    ("3", "es", "Jeff fue a la cine la semana pasada")
  ).toDF("id", "language", "text")

  lazy val n: NERV2 = new NERV2()
    .setSubscriptionKey(textKey)
    .setLocation("eastus")
    .setLanguage("en")
    .setOutputCol("response")

  test("Basic Usage") {
    val results = n.transform(df)
    val matches = results.withColumn("match",
      col("response")
        .getItem(0)
        .getItem("entities")
        .getItem("matches")
        .getItem(0)
        .getItem(0))
      .select("match")

    val testRow = matches.collect().head(0).asInstanceOf[GenericRowWithSchema]

    assert(testRow.getAs[String]("text") === "Jeff")
    assert(testRow.getAs[Int]("offset") === 0)
    assert(testRow.getAs[Int]("length") === 4)

  }

  test("Usage in Spanish") {
    val results = n.transform(df)
    val matches = results.withColumn("match",
      col("response")
        .getItem(0)
        .getItem("entities")
        .getItem("matches")
        .getItem(0)
        .getItem(0))
      .select("match")

    val testRow = matches.collect()(2).asInstanceOf[GenericRowWithSchema]

    assert(testRow.getAs[String]("text") === "Jeff")
    assert(testRow.getAs[Int]("offset") === 3)
    assert(testRow.getAs[Int]("length") === 5)

  }

  override def testObjects(): Seq[TestObject[NERV2]] =
    Seq(new TestObject[NERV2](n, df))

  override def reader: MLReadable[_] = NERV2
}

class NERSuiteV3 extends TransformerFuzzing[NER] with TextKey {
  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("1", "en", "I had a wonderful trip to Seattle last week."),
    ("2", "en", "I visited Space Needle 2 times.")
  ).toDF("id", "language", "text")

  lazy val n: NER = new NER()
    .setSubscriptionKey(textKey)
    .setLocation("eastus")
    .setLanguage("en")
    .setOutputCol("response")

  test("Basic Usage") {
    val results = n.transform(df)
    val matches = results.withColumn("match",
      col("response")
        .getItem(0)
        .getItem("entities")
        .getItem(0))
      .select("match")

    val testRow = matches.collect().head(0).asInstanceOf[GenericRowWithSchema]

    assert(testRow.getAs[String]("text") === "trip")
    assert(testRow.getAs[Int]("offset") === 18)
    assert(testRow.getAs[Int]("length") === 4)
    assert(testRow.getAs[Double]("confidenceScore") > 0.7)
    assert(testRow.getAs[String]("category") === "Event")

  }

  override def testObjects(): Seq[TestObject[NER]] =
    Seq(new TestObject[NER](n, df))

  override def reader: MLReadable[_] = NER
}
