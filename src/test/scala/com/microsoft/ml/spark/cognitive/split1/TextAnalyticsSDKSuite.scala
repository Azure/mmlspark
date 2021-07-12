package com.microsoft.ml.spark.cognitive.split1

import com.azure.ai.textanalytics.models.TextAnalyticsRequestOptions
import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.TransformerFuzzing
import com.microsoft.ml.spark.stages.FixedMiniBatchTransformer
import org.apache.spark.ml.param.DataFrameEquality
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col
import com.microsoft.ml.spark.stages.FixedMiniBatchTransformer



class DetectedLanguageSuitev4 extends TestBase with DataFrameEquality with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    Seq("Hello World"),
    Seq("Bonjour tout le monde"),
    Seq("La carretera estaba atascada. Había mucho tráfico el día de ayer."),
    Seq("世界您好"),
    Seq(":) :( :D"),
  ).toDF("text")

  val options: Option[TextAnalyticsRequestOptionsV4] = Some(new TextAnalyticsRequestOptionsV4("", true, false))

  lazy val detector: TextAnalyticsLanguageDetection = new TextAnalyticsLanguageDetection(options)
    .setSubscriptionKey(textKey)
    .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
    .setInputCol("text")
    .setOutputCol("output")

  test("Language Detection - Basic Usage") {
    val replies = detector.transform(df)
      .select("output")
      .collect()
    assert(replies(0).schema(0).name == "output")
    df.printSchema()
    df.show()
    replies.foreach { row =>
      row.toSeq.foreach{col => println(col) }
    }
  }

  test("Language Detection - Batch Usage") {
    val batchedDF = new FixedMiniBatchTransformer().setBatchSize(10).transform(df.coalesce(1))
    val tdf = detector.transform(batchedDF)
    val replies = tdf.collect().head.getAs[Seq[Row]]("replies")
    assert(replies.length == 4)
    val languages = replies.map(_.getAs[Seq[Row]]("detectedLanguages").head.getAs[String]("name")).toSet
    assert(languages("Spanish") && languages("English"))
  }
}
class TextSentimentSuiteV4 extends TestBase with DataFrameEquality with TextKey {

  import spark.implicits._

  val options: Option[TextAnalyticsRequestOptionsV4] = Some(new TextAnalyticsRequestOptionsV4("", true, false))

  lazy val df: DataFrame = Seq(
    Seq("Hello world. This is some input text that I love.", "this is something I hate"),
    Seq("I am sad"),
    Seq("I am feeling okay")
  ).toDF("id", "text")

  lazy val detector: TextSentimentV4 = new TextSentimentV4(options)
    .setSubscriptionKey(textKey)
    .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
    .setInputCol("text")
    .setOutputCol("output")

  test("Sentiment Analysis - Basic Usage") {
    val replies = detector.transform(df)
      .select("output")
      .collect()
    assert(replies(0).schema(0).name == "output")
    df.printSchema()
    df.show()
    replies.foreach { row =>
      row.toSeq.foreach { col => println(col) }
    }
  }

  lazy val batchedDF: DataFrame = Seq(
    Seq("I hate the rain.",
    "I love Vancouver.")
  ).toDF("text")

  lazy val detector2: TextSentimentV4 = new TextSentimentV4(options)
    .setSubscriptionKey(textKey)
    .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
    .setInputCol("text")
    .setOutputCol("output")

  test("func foo 2") {
    detector2.transform(batchedDF).printSchema()
    detector2.transform(batchedDF).show()
  }

  test("func Basic Usage2") {
    val replies = detector2.transform(batchedDF)
    .select(col("output").alias("scoredDocuments"))
      .collect().toList

    assert(
      replies(0).getSeq[Row](0).head.getString(0) == "negative" &&
        replies(1).getSeq[Row](0).head.getString(0) == "positive")
  }
}

  class KeyPhraseExtractionSuiteV4 extends TestBase with DataFrameEquality with TextKey {
    import spark.implicits._

    lazy val df2: DataFrame = Seq(
      Seq("Hello world. This is some input text that I love.", "I really enjoy coffee in the morning."),
      Seq("Glaciers are huge rivers of ice that ooze their way over land," +
        " powered by gravity and their own sheer weight."),
      Seq("Hello")
    ).toDF("text")

    val options: Option[TextAnalyticsRequestOptionsV4] = Some(new TextAnalyticsRequestOptionsV4("", true, false))

    lazy val extractor: TextAnalyticsKeyphraseExtraction = new TextAnalyticsKeyphraseExtraction(options)
      .setSubscriptionKey(textKey)
      .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
      .setInputCol("text")
      .setOutputCol("output")

    test("KPE - Basic Usage") {
      val replies = extractor.transform(df2)
        .select("output")
        .collect()
      assert(replies(0).schema(0).name == "output")
      df2.printSchema()
      df2.show()
      replies.foreach { row =>
        row.toSeq.foreach { col => println(col) }
      }
    }
  }

