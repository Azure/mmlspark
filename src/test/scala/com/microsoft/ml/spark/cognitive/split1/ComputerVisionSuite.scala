// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split1

import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.test.base.{Flaky, TestBase}
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.NamespaceInjections.pipelineModel
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.functions.typedLit
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalactic.Equality
import com.microsoft.ml.spark.core.spark.FluentAPI._

trait CognitiveKey {
  lazy val cognitiveKey: String = sys.env.getOrElse("COGNITIVE_API_KEY", Secrets.CognitiveApiKey)
}

trait OCRUtils extends TestBase {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/OCR/test1.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/OCR/test2.png",
    "https://mmlspark.blob.core.windows.net/datasets/OCR/test3.png"
  ).toDF("url")

  lazy val pdfDf: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/OCR/paper.pdf"
  ).toDF("url")

  lazy val bytesDF: DataFrame = BingImageSearch
    .downloadFromUrls("url", "imageBytes", 4, 10000)
    .transform(df)
    .select("imageBytes")

}

class OCRSuite extends TransformerFuzzing[OCR] with CognitiveKey with Flaky with OCRUtils {

  lazy val ocr: OCR = new OCR()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setDetectOrientation(true)
    .setOutputCol("ocr")

  lazy val bytesOCR: OCR = new OCR()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageBytesCol("imageBytes")
    .setDetectOrientation(true)
    .setOutputCol("bocr")

  test("Getters") {
    assert(ocr.getDetectOrientation)
    assert(ocr.getImageUrlCol === "url")
    assert(ocr.getSubscriptionKey == cognitiveKey)
    assert(bytesOCR.getImageBytesCol === "imageBytes")
  }

  test("Basic Usage with URL") {
    val model = pipelineModel(Array(
      ocr,
      OCR.flatten("ocr", "ocr")
    ))
    val results = model.transform(df).collect()
    assert(results(2).getString(2).startsWith("This is a lot of 12 point text"))
  }

  test("Basic Usage with Bytes") {
    val model = pipelineModel(Array(
      bytesOCR,
      OCR.flatten("bocr", "bocr")
    ))

    val results = model.transform(bytesDF).collect()
    assert(results(2).getString(2).startsWith("This is a lot of 12 point text"))
  }

  override def testObjects(): Seq[TestObject[OCR]] =
    Seq(new TestObject(ocr, df))

  override def reader: MLReadable[_] = OCR
}

class AnalyzeImageSuite extends TransformerFuzzing[AnalyzeImage] with CognitiveKey with Flaky {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("https://mmlspark.blob.core.windows.net/datasets/OCR/test1.jpg", "en"),
    ("https://mmlspark.blob.core.windows.net/datasets/OCR/test2.png", null), //scalastyle:ignore null
    ("https://mmlspark.blob.core.windows.net/datasets/OCR/test3.png", "en")
  ).toDF("url", "language")

  def baseAI: AnalyzeImage = new AnalyzeImage()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setOutputCol("features")

  lazy val ai: AnalyzeImage = baseAI
    .setImageUrlCol("url")
    .setLanguageCol("language")
    .setVisualFeatures(
      Seq("Categories", "Tags", "Description", "Faces", "ImageType", "Color", "Adult", "Objects", "Brands"))
    .setDetails(Seq("Celebrities", "Landmarks"))

  lazy val bytesDF: DataFrame = BingImageSearch
    .downloadFromUrls("url", "imageBytes", 4, 10000)
    .transform(df)
    .drop("url")

  lazy val bytesAI: AnalyzeImage = baseAI
    .setImageBytesCol("imageBytes")
    .setLanguageCol("language")
    .setVisualFeatures(
      Seq("Categories", "Tags", "Description", "Faces", "ImageType", "Color", "Adult", "Objects", "Brands")
    )
    .setDetails(Seq("Celebrities", "Landmarks"))

  test("full parametrization") {
    val row = (Seq("Categories"), "en", Seq("Celebrities"),
      "https://mmlspark.blob.core.windows.net/datasets/OCR/test1.jpg")
    val df = Seq(row).toDF()

    val staticAi = baseAI
      .setVisualFeatures(row._1)
      .setLanguage(row._2)
      .setDetails(row._3)
      .setImageUrl(row._4)

    val dynamicAi = baseAI
      .setVisualFeaturesCol("_1")
      .setLanguageCol("_2")
      .setDetailsCol("_3")
      .setImageUrlCol("_4")

    assert(dynamicAi.getVisualFeaturesCol == "_1")
    assert(dynamicAi.getLanguageCol == "_2")
    assert(dynamicAi.getDetailsCol == "_3")
    assert(dynamicAi.getImageUrlCol == "_4")
    assert(staticAi.getVisualFeatures == row._1)
    assert(staticAi.getLanguage == row._2)
    assert(staticAi.getDetails == row._3)
    assert(staticAi.getImageUrl == row._4)
    assert(staticAi.transform(df).collect().head.getAs[Row]("features") != null)
    assert(dynamicAi.transform(df).collect().head.getAs[Row]("features") != null)
  }

  test("Basic Usage with URL") {
    val fromRow = AIResponse.makeFromRowConverter
    val responses = ai.transform(df).select("features")
      .collect().toList.map(r =>
      fromRow(r.getStruct(0)))
    assert(responses.head.categories.get.head.name === "others_")
    assert(responses(1).categories.get.head.name === "text_sign")
  }

  test("Basic Usage with Bytes") {
    val fromRow = AIResponse.makeFromRowConverter
    val responses = bytesAI.transform(bytesDF).select("features")
      .collect().toList.map(r => fromRow(r.getStruct(0)))
    assert(responses.head.categories.get.head.name === "others_")
    assert(responses(1).categories.get.head.name === "text_sign")
  }

  test("Basic Usage with Bytes and null col") {
    val fromRow = AIResponse.makeFromRowConverter
    val responses = bytesAI.setImageUrlCol("url")
      .transform(bytesDF.withColumn("url", typedLit(null: String))) //scalastyle:ignore null
      .select("features")
      .collect().toList.map(r => fromRow(r.getStruct(0)))
    assert(responses.head.categories.get.head.name === "others_")
    assert(responses(1).categories.get.head.name === "text_sign")
  }

  override def testObjects(): Seq[TestObject[AnalyzeImage]] =
    Seq(new TestObject(ai, df))

  override def reader: MLReadable[_] = AnalyzeImage

  override implicit lazy val dfEq: Equality[DataFrame] = new Equality[DataFrame] {
    def areEqual(a: DataFrame, bAny: Any): Boolean = bAny match {
      case b: Dataset[_] =>
        baseDfEq.areEqual( //TODO remove flakiness fixing hack
          a.select("features.description.tags"),
          b.select("features.description.tags"))
    }
  }

}

class RecognizeTextSuite extends TransformerFuzzing[RecognizeText]
  with CognitiveKey with Flaky with OCRUtils {

  lazy val rt: RecognizeText = new RecognizeText()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setMode("Printed")
    .setOutputCol("ocr")
    .setConcurrency(5)

  lazy val bytesRT: RecognizeText = new RecognizeText()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageBytesCol("imageBytes")
    .setMode("Printed")
    .setOutputCol("ocr")
    .setConcurrency(5)

  test("Basic Usage with URL") {
    val results = df.mlTransform(rt, RecognizeText.flatten("ocr", "ocr"))
      .select("ocr")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "OPENS.ALL YOU HAVE TO DO IS WALK IN WHEN ONE DOOR CLOSES, ANOTHER CLOSED" ||
      headStr === "CLOSED WHEN ONE DOOR CLOSES, ANOTHER OPENS. ALL YOU HAVE TO DO IS WALK IN")
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF.mlTransform(bytesRT, RecognizeText.flatten("ocr", "ocr"))
      .select("ocr")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "OPENS.ALL YOU HAVE TO DO IS WALK IN WHEN ONE DOOR CLOSES, ANOTHER CLOSED" ||
      headStr === "CLOSED WHEN ONE DOOR CLOSES, ANOTHER OPENS. ALL YOU HAVE TO DO IS WALK IN")
  }

  override def testObjects(): Seq[TestObject[RecognizeText]] =
    Seq(new TestObject(rt, df))

  override def reader: MLReadable[_] = RecognizeText
}

class ReadSuite extends TransformerFuzzing[Read]
  with CognitiveKey with Flaky with OCRUtils {

  lazy val read: Read = new Read()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("ocr")
    .setConcurrency(5)

  lazy val bytesRead: Read = new Read()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageBytesCol("imageBytes")
    .setOutputCol("ocr")
    .setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("url", "ocr.analyzeResult.readResults")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage with URL") {
    val results = df.mlTransform(read, Read.flatten("ocr", "ocr"))
      .select("ocr")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "OPENS.ALL YOU HAVE TO DO IS WALK IN WHEN ONE DOOR CLOSES, ANOTHER CLOSED" ||
      headStr === "CLOSED WHEN ONE DOOR CLOSES, ANOTHER OPENS. ALL YOU HAVE TO DO IS WALK IN")
  }

  test("Basic Usage with pdf") {
    val results = pdfDf.mlTransform(read, Read.flatten("ocr", "ocr"))
      .select("ocr")
      .collect()
    val headStr = results.head.getString(0)
    val correctPrefix = "Full Tree Conditioned Tree Component Space " +
      "Efficiency Measured Data O(n × d) 380 MB Tree O((2n/l) × d)"

    assert(headStr.startsWith(correctPrefix))
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF.mlTransform(bytesRead, Read.flatten("ocr", "ocr"))
      .select("ocr")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "OPENS.ALL YOU HAVE TO DO IS WALK IN WHEN ONE DOOR CLOSES, ANOTHER CLOSED" ||
      headStr === "CLOSED WHEN ONE DOOR CLOSES, ANOTHER OPENS. ALL YOU HAVE TO DO IS WALK IN")
  }

  override def testObjects(): Seq[TestObject[Read]] =
    Seq(new TestObject(read, df))

  override def reader: MLReadable[_] = Read
}

class RecognizeDomainSpecificContentSuite extends TransformerFuzzing[RecognizeDomainSpecificContent]
  with CognitiveKey with Flaky {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test2.jpg"
  ).toDF("url")

  lazy val celeb: RecognizeDomainSpecificContent = new RecognizeDomainSpecificContent()
    .setSubscriptionKey(cognitiveKey)
    .setModel("celebrities")
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("celebs")

  lazy val bytesDF: DataFrame = BingImageSearch
    .downloadFromUrls("url", "imageBytes", 4, 10000)
    .transform(df)
    .select("imageBytes")

  lazy val bytesCeleb: RecognizeDomainSpecificContent = new RecognizeDomainSpecificContent()
    .setSubscriptionKey(cognitiveKey)
    .setModel("celebrities")
    .setLocation("eastus")
    .setImageBytesCol("imageBytes")
    .setOutputCol("celebs")

  test("Basic Usage with URL") {
    val model = pipelineModel(Array(
      celeb, RecognizeDomainSpecificContent.getMostProbableCeleb("celebs", "celebs")))
    val results = model.transform(df)
    assert(results.head().getString(2) === "Leonardo DiCaprio")
  }

  test("Basic Usage with Bytes") {
    val model = pipelineModel(Array(
      bytesCeleb, RecognizeDomainSpecificContent.getMostProbableCeleb("celebs", "celebs")))
    val results = model.transform(bytesDF)
    assert(results.head().getString(2) === "Leonardo DiCaprio")
  }

  override implicit lazy val dfEq: Equality[DataFrame] = new Equality[DataFrame] {
    def areEqual(a: DataFrame, bAny: Any): Boolean = bAny match {
      case b: Dataset[_] =>
        val t = RecognizeDomainSpecificContent.getMostProbableCeleb("celebs", "celebs")
        baseDfEq.areEqual(t.transform(a), t.transform(b))
    }
  }

  override def testObjects(): Seq[TestObject[RecognizeDomainSpecificContent]] =
    Seq(new TestObject(celeb, df))

  override def reader: MLReadable[_] = RecognizeDomainSpecificContent
}

class GenerateThumbnailsSuite extends TransformerFuzzing[GenerateThumbnails]
  with CognitiveKey with Flaky {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test1.jpg"
  ).toDF("url")

  lazy val t: GenerateThumbnails = new GenerateThumbnails()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setHeight(50).setWidth(50).setSmartCropping(true)
    .setImageUrlCol("url")
    .setOutputCol("thumbnails")

  lazy val bytesDF: DataFrame = BingImageSearch
    .downloadFromUrls("url", "imageBytes", 4, 10000)
    .transform(df)
    .select("imageBytes")

  lazy val bytesGT: GenerateThumbnails = new GenerateThumbnails()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setHeight(50).setWidth(50).setSmartCropping(true)
    .setImageBytesCol("imageBytes")
    .setOutputCol("thumbnails")

  test("Basic Usage with URL") {
    val results = t.transform(df)
    assert(results.head().getAs[Array[Byte]](2).length > 1000)
  }

  test("Basic Usage with Bytes") {
    val results = bytesGT.transform(bytesDF)
    assert(results.head().getAs[Array[Byte]](2).length > 1000)
  }

  override def testObjects(): Seq[TestObject[GenerateThumbnails]] =
    Seq(new TestObject(t, df))

  override def reader: MLReadable[_] = GenerateThumbnails
}

class TagImageSuite extends TransformerFuzzing[TagImage] with CognitiveKey with Flaky {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test1.jpg"
  ).toDF("url")

  lazy val t: TagImage = new TagImage()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("tags")

  lazy val bytesDF: DataFrame = BingImageSearch
    .downloadFromUrls("url", "imageBytes", 4, 10000)
    .transform(df)
    .select("imageBytes")

  lazy val bytesTI: TagImage = new TagImage()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageBytesCol("imageBytes")
    .setOutputCol("tags")

  test("Basic Usage with URL") {
    val results = t.transform(df)
    val tagResponse = results.head()
      .getAs[Row]("tags")
      .getSeq[Row](0)

    assert(tagResponse.map(_.getString(0)).toList.head === "person")
    assert(tagResponse.map(_.getDouble(1)).toList.head > .9)
  }

  test("Basic Usage with Bytes") {
    val results = bytesTI.transform(bytesDF)
    val tagResponse = results.head()
      .getAs[Row]("tags")
      .getSeq[Row](0)

    assert(tagResponse.map(_.getString(0)).toList.head === "person")
    assert(tagResponse.map(_.getDouble(1)).toList.head > .9)
  }

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    super.assertDFEq(df1.select("tags.tags"), df2.select("tags.tags"))(eq)
  }

  override def testObjects(): Seq[TestObject[TagImage]] =
    Seq(new TestObject(t, df))

  override def reader: MLReadable[_] = TagImage
}

class DescribeImageSuite extends TransformerFuzzing[DescribeImage]
  with CognitiveKey with Flaky {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test1.jpg"
  ).toDF("url")

  lazy val t: DescribeImage = new DescribeImage()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setMaxCandidates(3)
    .setImageUrlCol("url")
    .setOutputCol("descriptions")

  lazy val bytesDF: DataFrame = BingImageSearch
    .downloadFromUrls("url", "imageBytes", 4, 10000)
    .transform(df)
    .select("imageBytes")

  lazy val bytesDI: DescribeImage = new DescribeImage()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setMaxCandidates(3)
    .setImageBytesCol("imageBytes")
    .setOutputCol("descriptions")

  test("Basic Usage with URL") {
    val results = t.transform(df)
    val tags = results.select("descriptions").take(1).head
      .getStruct(0).getStruct(0).getSeq[String](0).toSet
    assert(tags("person") && tags("glasses"))
  }

  test("Basic Usage with Bytes") {
    val results = bytesDI.transform(bytesDF)
    val tags = results.select("descriptions").take(1).head
      .getStruct(0).getStruct(0).getSeq[String](0).toSet
    assert(tags("person") && tags("glasses"))
  }

  override def testObjects(): Seq[TestObject[DescribeImage]] =
    Seq(new TestObject(t, df))

  override def reader: MLReadable[_] = DescribeImage

}
