package com.microsoft.ml.spark.cognitive.split1

import com.azure.ai.textanalytics.models.TextAnalyticsRequestOptions
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.ml.param.DataFrameEquality
import org.apache.spark.sql.DataFrame

class DetectedLanguageSuitev4 extends TestBase with DataFrameEquality with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("us","Hello world. This is some input text that I love."),
    ("us","Glaciers are huge rivers of ice that ooze their way over land," +
      "powered by gravity and their own sheer weight."),
    ("us", "La carretera estaba atascada. Había mucho tráfico el día de ayer.")
  ).toDF("countryHint","text2")

  val options: Option[TextAnalyticsRequestOptions] = Some(new TextAnalyticsRequestOptions()
    .setIncludeStatistics(true))

  lazy val detector: TextAnalyticsLanguageDetection = new TextAnalyticsLanguageDetection(options)
    .setSubscriptionKey(textKey)
    .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
    .setInputCol("text2")
    .setLangCol("countryHint")
    .setOutputCol("Out")

  test("Language Detection - Basic Usage") {
    val replies = detector.transform(df)
      .select("name", "iso6391Name")
      .collect()
    assert(replies(0).getString(0) == "English" && replies(2).getString(0) == "Spanish" )
    assert(replies(0).getString(1) == "en" && replies(2).getString(1) == "es")
  }

  test("Language Detection - Print Schema") {
    detector.transform(df).printSchema()
  }
}

class TextSentimentSuiteV4 extends TestBase with DataFrameEquality with TextKey {
  import spark.implicits._

  val options: Option[TextAnalyticsRequestOptions] = Some(new TextAnalyticsRequestOptions()
    .setIncludeStatistics(true))

  lazy val df: DataFrame = Seq(
    ("en", "Hello world. This is some input text that I love."),
    ("en", "I am sad"),
    ("en", "I am feeling okay")
  ).toDF("lang", "text")

  lazy val detector: TextSentimentV4 = new TextSentimentV4(options)
    .setSubscriptionKey(textKey)
    .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
    .setInputCol("text")
    .setLangCol("lang")
    .setOutputCol("Out")

  test("Sentiment Analysis - Basic Usage") {
    val replies = detector.transform(df)
      .select("sentiment", "confidenceScores", "sentences", "warnings")
      .collect()
    assert(replies(0).getString(0) == "positive" && replies(1).getString(0) == "negative"
      && replies(2).getString(0) == "neutral")
  }
}

  class KeyPhraseExtractionSuiteV4 extends TestBase with DataFrameEquality with TextKey {
    import spark.implicits._

    val options: Option[TextAnalyticsRequestOptions] = Some(new TextAnalyticsRequestOptions()
      .setIncludeStatistics(true))

    lazy val df2: DataFrame = Seq(
      ("en","Hello world. This is some input text that I love."),
      ("en","Glaciers are huge rivers of ice that ooze their way over land," +
        "powered by gravity and their own sheer weight."),
      ("es", "La carretera estaba atascada. Había mucho tráfico el día de ayer.")
    ).toDF("langCol","text2")

    lazy val extractor: TextAnalyticsKeyphraseExtraction = new TextAnalyticsKeyphraseExtraction(options)
      .setSubscriptionKey(textKey)
      .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
      .setInputCol("text2")
      .setLangCol("langCol")
      .setOutputCol("Out")

    test("KPE - Basic Usage") {
      val replies = extractor.transform(df2)
        .select("keyPhrases")
        .collect()
      assert(replies(0).getSeq[String](0).toSet === Set("Hello world", "input text"))
      assert(replies(1).getSeq[String](0).toSet === Set("land", "sheer weight",
        "gravity", "way", "Glaciers", "ice", "huge rivers"))
      assert(replies(2).getSeq[String](0).toSet === Set("mucho tráfico", "carretera", "ayer"))
    }
  }
