// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split1

import com.microsoft.ml.spark.FluentAPI._
import com.microsoft.ml.spark.cognitive.RESTHelpers.retry
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.env.StreamUtilities.using
import com.microsoft.ml.spark.core.test.base.{Flaky, TestBase}
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.commons.io.IOUtils
import org.apache.http.client.methods._
import org.apache.http.entity.StringEntity
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.scalactic.Equality
import spray.json._

import java.net.URI

object TrainCustomModelProtocol extends DefaultJsonProtocol {
  implicit val SourceFilterEnc: RootJsonFormat[SourceFilter] = jsonFormat2(SourceFilter)
  implicit val TrainCustomModelEnc: RootJsonFormat[TrainCustomModelSchema] = jsonFormat3(TrainCustomModelSchema)
}

import TrainCustomModelProtocol._

case class TrainCustomModelSchema(source: String, sourceFilter: SourceFilter, useLabelFile: Boolean)

case class SourceFilter(prefix: String, includeSubFolders: Boolean)

object FormRecognizerUtils extends CognitiveKey {

  import RESTHelpers._

  val PollingDelay = 1000

  def formSend(request: HttpRequestBase, path: String,
               params: Map[String, String] = Map()): String = {

    val paramString = if (params.isEmpty) {""} else {"?" + URLEncodingUtils.format(params)}
    request.setURI(new URI(path + paramString))

    retry(List(100, 500, 1000), { () =>
      request.addHeader("Ocp-Apim-Subscription-Key", cognitiveKey)
      request.addHeader("Content-Type", "application/json")
      using(Client.execute(request)) { response =>
        if (!response.getStatusLine.getStatusCode.toString.startsWith("2")) {
          val bodyOpt = request match {
            case er: HttpEntityEnclosingRequestBase => IOUtils.toString(er.getEntity.getContent, "UTF-8")
            case _ => ""
          }
          throw new RuntimeException(s"Failed: response: $response " + s"requestUrl: ${request.getURI}" +
              s"requestBody: $bodyOpt")
        }
        if (response.getStatusLine.getReasonPhrase == "No Content") {
          ""
        }
        else if (response.getStatusLine.getReasonPhrase == "Created") {
          response.getHeaders("Location").head.getValue
        }
        else {
          IOUtils.toString(response.getEntity.getContent, "UTF-8")
        }
      }.get
    })
  }

  def formDelete(path: String, params: Map[String, String] = Map()): String = {
    formSend(new HttpDelete(),
      "https://eastus.api.cognitive.microsoft.com/formrecognizer/v2.1/custom/models/" + path, params)
  }

  def formPost(path: String, body: TrainCustomModelSchema, params: Map[String, String] = Map())
                 (implicit format: JsonFormat[TrainCustomModelSchema]): String = {
    val post = new HttpPost()
    post.setEntity(new StringEntity(body.toJson.compactPrint))
    formSend(post, "https://eastus.api.cognitive.microsoft.com/formrecognizer/v2.1/custom/models" + path, params)
  }

  def formGet(path: String, params: Map[String, String] = Map()): String = {
    formSend(new HttpGet(), path, params)
  }
}

trait FormRecognizerUtils extends TestBase {

  import spark.implicits._

  def createTestDataframe(v: Seq[String], returnBytes: Boolean): DataFrame = {
    val df = v.toDF("source")
    if (returnBytes) {
      BingImageSearch
        .downloadFromUrls("source", "imageBytes", 4, 10000)
        .transform(df)
        .select("imageBytes")
    } else {
      df
    }
  }

  lazy val imageDf1: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/layout1.jpg"), returnBytes = false)

  lazy val bytesDF1: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/layout1.jpg"), returnBytes = true)

  lazy val imageDf2: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/receipt1.png"), returnBytes = false)

  lazy val bytesDF2: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/receipt1.png"), returnBytes = true)

  lazy val imageDf3: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/business_card.jpg"), returnBytes = false)

  lazy val bytesDF3: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/business_card.jpg"), returnBytes = true)

  lazy val imageDf4: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/invoice2.png"), returnBytes = false)

  lazy val bytesDF4: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/invoice2.png"), returnBytes = true)

  lazy val imageDf5: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/id1.jpg"), returnBytes = false)

  lazy val bytesDF5: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/id1.jpg"), returnBytes = true)

  lazy val pdfDf1: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/layout2.pdf"), returnBytes = false)

  lazy val pdfDf2: DataFrame = createTestDataframe(
    Seq("https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/invoice1.pdf",
    "https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/invoice3.pdf"), returnBytes = false)

  // TODO: renew the SAS after 2022-07-01 since it will expire
  lazy val trainingDataSAS: String = "https://mmlspark.blob.core.windows.net/datasets?sp=rl&st=2021" +
    "-06-30T04:29:50Z&se=2022-07-01T04:45:00Z&sv=2020-08-04&sr=c&sig=sdsOSpWptIoI3aSceGlGvQhjnOTJTAABghIajrOXJD8%3D"

  lazy val df: DataFrame = createTestDataframe(Seq(""), returnBytes = false)
}

class AnalyzeLayoutSuite extends TransformerFuzzing[AnalyzeLayout]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  lazy val analyzeLayout: AnalyzeLayout = new AnalyzeLayout()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageUrlCol("source").setOutputCol("layout").setConcurrency(5)

  lazy val bytesAnalyzeLayout: AnalyzeLayout = new AnalyzeLayout()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageBytesCol("imageBytes").setOutputCol("layout").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("source", "layout.analyzeResult.readResults")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage with URL") {
    val results = imageDf1.mlTransform(analyzeLayout,
      AnalyzeLayout.flattenReadResults("layout", "readlayout"),
      AnalyzeLayout.flattenPageResults("layout", "pageLayout"))
      .select("readlayout", "pageLayout")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr.startsWith("Purchase Order Hero Limited Purchase Order Company Phone: 555-348-6512 " +
      "Website: www.herolimited.com "))
    val pageHeadStr = results.head.getString(1)
    assert(pageHeadStr === "Details | Quantity | Unit Price | Total | Bindings | 20 | 1.00 | 20.00 | Covers Small" +
      " | 20 | 1.00 | 20.00 | Feather Bookmark | 20 | 5.00 | 100.00 | Copper Swirl Marker | 20 | 5.00 | " +
      "100.00\nSUBTOTAL | $140.00 | TAX | $4.00 |  |  | TOTAL | $144.00")
  }

  test("Basic Usage with pdf") {
    val results = pdfDf1.mlTransform(analyzeLayout,
      AnalyzeLayout.flattenReadResults("layout", "readlayout"),
      AnalyzeLayout.flattenPageResults("layout", "pageLayout"))
      .select("readlayout", "pageLayout")
      .collect()
    val headStr = results.head.getString(0)
    val correctPrefix = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q"
    assert(headStr.startsWith(correctPrefix))
    val pageHeadStr = results.head.getString(1)
    assert(pageHeadStr === "Title of each class | Trading Symbol | Name of exchange on which registered | " +
      "Common stock, $0.00000625 par value per share | MSFT | NASDAQ | 2.125% Notes due 2021 | MSFT | NASDAQ |" +
      " 3.125% Notes due 2028 | MSFT | NASDAQ | 2.625% Notes due 2033 | MSFT | NASDAQ")
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF1.mlTransform(bytesAnalyzeLayout,
      AnalyzeLayout.flattenReadResults("layout", "readlayout"),
      AnalyzeLayout.flattenPageResults("layout", "pageLayout"))
      .select("readlayout", "pageLayout")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr.startsWith("Purchase Order Hero Limited Purchase Order Company Phone: 555-348-6512" +
      " Website: www.herolimited.com "))
    val pageHeadStr = results.head.getString(1)
    assert(pageHeadStr === "Details | Quantity | Unit Price | Total | Bindings | 20 | 1.00 | 20.00 | Covers Small" +
      " | 20 | 1.00 | 20.00 | Feather Bookmark | 20 | 5.00 | 100.00 | Copper Swirl Marker | 20 | 5.00 | " +
      "100.00\nSUBTOTAL | $140.00 | TAX | $4.00 |  |  | TOTAL | $144.00")
  }

  override def testObjects(): Seq[TestObject[AnalyzeLayout]] =
    Seq(new TestObject(analyzeLayout, imageDf1))

  override def reader: MLReadable[_] = AnalyzeLayout
}

class AnalyzeReceiptsSuite extends TransformerFuzzing[AnalyzeReceipts]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  lazy val analyzeReceipts: AnalyzeReceipts = new AnalyzeReceipts()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageUrlCol("source").setOutputCol("receipts").setConcurrency(5)

  lazy val bytesAnalyzeReceipts: AnalyzeReceipts = new AnalyzeReceipts()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageBytesCol("imageBytes").setOutputCol("receipts").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("source", "receipts.analyzeResult.readResults")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage with URL") {
    val results = imageDf2.mlTransform(analyzeReceipts,
      AnalyzeReceipts.flattenReadResults("receipts", "readReceipts"),
      AnalyzeReceipts.flattenDocumentResults("receipts", "docReceipts"))
      .select("readReceipts", "docReceipts")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith(
      ("""{"Items":{"type":"array","valueArray":[{"type":"object",""" +
        """"valueObject":{"Name":{"type":"string","valueString":"Surface Pro 6","text":"Surface Pro 6""").stripMargin))
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF2.mlTransform(bytesAnalyzeReceipts,
      AnalyzeReceipts.flattenReadResults("receipts", "readReceipts"),
      AnalyzeReceipts.flattenDocumentResults("receipts", "docReceipts"))
      .select("readReceipts", "docReceipts")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith(
      ("""{"Items":{"type":"array","valueArray":[{"type":"object",""" +
        """"valueObject":{"Name":{"type":"string","valueString":"Surface Pro 6","text":"Surface Pro 6""").stripMargin))
  }

  override def testObjects(): Seq[TestObject[AnalyzeReceipts]] =
    Seq(new TestObject(analyzeReceipts, imageDf2))

  override def reader: MLReadable[_] = AnalyzeReceipts
}

class AnalyzeBusinessCardsSuite extends TransformerFuzzing[AnalyzeBusinessCards]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  lazy val analyzeBusinessCards: AnalyzeBusinessCards = new AnalyzeBusinessCards()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageUrlCol("source").setOutputCol("businessCards").setConcurrency(5)

  lazy val bytesAnalyzeBusinessCards: AnalyzeBusinessCards = new AnalyzeBusinessCards()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageBytesCol("imageBytes").setOutputCol("businessCards").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("source", "businessCards.analyzeResult.readResults")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage with URL") {
    val results = imageDf3.mlTransform(analyzeBusinessCards,
      AnalyzeBusinessCards.flattenReadResults("businessCards", "readBusinessCards"),
      AnalyzeBusinessCards.flattenDocumentResults("businessCards", "docBusinessCards"))
      .select("readBusinessCards", "docBusinessCards")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith((
      """{"Addresses":{"type":"array","valueArray":[{"type":""" +
      """"string","valueString":"2 Kingdom Street Paddington, London, W2 6BD""").stripMargin))
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF3.mlTransform(bytesAnalyzeBusinessCards,
      AnalyzeBusinessCards.flattenReadResults("businessCards", "readBusinessCards"),
      AnalyzeBusinessCards.flattenDocumentResults("businessCards", "docBusinessCards"))
      .select("readBusinessCards", "docBusinessCards")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith((
      """{"Addresses":{"type":"array","valueArray":[{"type":""" +
        """"string","valueString":"2 Kingdom Street Paddington, London, W2 6BD""").stripMargin))
  }

  override def testObjects(): Seq[TestObject[AnalyzeBusinessCards]] =
    Seq(new TestObject(analyzeBusinessCards, imageDf3))

  override def reader: MLReadable[_] = AnalyzeBusinessCards
}

class AnalyzeInvoicesSuite extends TransformerFuzzing[AnalyzeInvoices]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  lazy val analyzeInvoices: AnalyzeInvoices = new AnalyzeInvoices()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageUrlCol("source").setOutputCol("invoices").setConcurrency(5)

  lazy val bytesAnalyzeInvoices: AnalyzeInvoices = new AnalyzeInvoices()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageBytesCol("imageBytes").setOutputCol("invoices").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("source", "invoices.analyzeResult.readResults")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage with URL") {
    val results = imageDf4.mlTransform(analyzeInvoices,
      AnalyzeInvoices.flattenReadResults("invoices", "readInvoices"),
      AnalyzeInvoices.flattenDocumentResults("invoices", "docInvoices"))
      .select("readInvoices", "docInvoices")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith((
      """{"CustomerAddress":{"type":"string","valueString":"1020 Enterprise Way Sunnayvale, CA 87659","text":""" +
        """"1020 Enterprise Way Sunnayvale, CA 87659""").stripMargin))
  }

  test("Basic Usage with pdf") {
    val results = pdfDf2.mlTransform(analyzeInvoices,
      AnalyzeInvoices.flattenReadResults("invoices", "readInvoices"),
      AnalyzeInvoices.flattenDocumentResults("invoices", "docInvoices"))
      .select("readInvoices", "docInvoices")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith("""{"AmountDue":{"type":"number","valueNumber":610,"text":"$610.00"""))
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF4.mlTransform(bytesAnalyzeInvoices,
      AnalyzeInvoices.flattenReadResults("invoices", "readInvoices"),
      AnalyzeInvoices.flattenDocumentResults("invoices", "docInvoices"))
      .select("readInvoices", "docInvoices")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith((
      """{"CustomerAddress":{"type":"string","valueString":"1020 Enterprise Way Sunnayvale, CA 87659","text":""" +
        """"1020 Enterprise Way Sunnayvale, CA 87659""").stripMargin))
  }

  override def testObjects(): Seq[TestObject[AnalyzeInvoices]] =
    Seq(new TestObject(analyzeInvoices, imageDf4))

  override def reader: MLReadable[_] = AnalyzeInvoices
}

class AnalyzeIDDocumentsSuite extends TransformerFuzzing[AnalyzeIDDocuments]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  lazy val analyzeIDDocuments: AnalyzeIDDocuments = new AnalyzeIDDocuments()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageUrlCol("source").setOutputCol("ids").setConcurrency(5)

  lazy val bytesAnalyzeIDDocuments: AnalyzeIDDocuments = new AnalyzeIDDocuments()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setImageBytesCol("imageBytes").setOutputCol("ids").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("source", "ids.analyzeResult.readResults")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage with URL") {
    val results = imageDf5.mlTransform(analyzeIDDocuments,
      AnalyzeIDDocuments.flattenReadResults("ids", "readIds"),
      AnalyzeIDDocuments.flattenDocumentResults("ids", "docIds"))
      .select("readIds", "docIds")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith((
      """{"Address":{"type":"string","valueString":"123 STREET ADDRESS YOUR CITY WA 99999-1234","text":""" +
        """"123 STREET ADDRESS YOUR CITY WA 99999-1234""").stripMargin))
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF5.mlTransform(bytesAnalyzeIDDocuments,
      AnalyzeIDDocuments.flattenReadResults("ids", "readIds"),
      AnalyzeIDDocuments.flattenDocumentResults("ids", "docIds"))
      .select("readIds", "docIds")
      .collect()
    val headStr = results.head.getString(0)
    assert(headStr === "")
    val docHeadStr = results.head.getString(1)
    assert(docHeadStr.startsWith((
      """{"Address":{"type":"string","valueString":"123 STREET ADDRESS YOUR CITY WA 99999-1234","text":""" +
        """"123 STREET ADDRESS YOUR CITY WA 99999-1234""").stripMargin))
  }

  override def testObjects(): Seq[TestObject[AnalyzeIDDocuments]] =
    Seq(new TestObject(analyzeIDDocuments, imageDf5))

  override def reader: MLReadable[_] = AnalyzeIDDocuments
}

class ListCustomModelsSuite extends TransformerFuzzing[ListCustomModels]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  val getRequestUrl: String = FormRecognizerUtils.formPost("", TrainCustomModelSchema(
    trainingDataSAS, SourceFilter("CustomModelTrain", includeSubFolders = false), useLabelFile = false))

  val modelId: String = retry(List(10000, 20000, 30000), () => {
    val resp = FormRecognizerUtils.formGet(getRequestUrl)
    val modelInfo = resp.parseJson.asJsObject.fields.getOrElse("modelInfo", "")
    val status = modelInfo match {
      case x: JsObject => x.fields.getOrElse("status", "") match {
        case y: JsString => y.value
        case _ => throw new RuntimeException(s"No status found in response/modelInfo: $resp/$modelInfo")
      }
      case _ => throw new RuntimeException(s"No modelInfo found in response: $resp")
    }
    status match {
      case "ready" => modelInfo.asInstanceOf[JsObject].fields.getOrElse("modelId", "").asInstanceOf[JsString].value
      case "creating" => throw new RuntimeException("model creating ...")
      case s => throw new RuntimeException(s"Received unknown status code: $s")
    }
  })

  override def afterAll(): Unit = {
    if (modelId != "") {
      FormRecognizerUtils.formDelete(modelId)
    }
    super.afterAll()
  }

  lazy val listCustomModels: ListCustomModels = new ListCustomModels()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setOp("full").setOutputCol("models").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("models.summary.count")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("List model list details") {
    val results = df.mlTransform(listCustomModels,
      ListCustomModels.flattenModelList("models", "modelIds"))
      .select("modelIds")
      .collect()
    assert(results.head.getString(0) != "")
  }

  test("List model list summary") {
    val results = listCustomModels.setOp("summary").transform(df)
      .withColumn("modelCount", col("models").getField("summary").getField("count"))
      .select("modelCount")
      .collect()
    assert(results.head.getInt(0) >= 1)
  }

  override def testObjects(): Seq[TestObject[ListCustomModels]] =
    Seq(new TestObject(listCustomModels, df))

  override def reader: MLReadable[_] = ListCustomModels
}

class GetCustomModelSuite extends TransformerFuzzing[GetCustomModel]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  val getRequestUrl: String = FormRecognizerUtils.formPost("", TrainCustomModelSchema(
    trainingDataSAS, SourceFilter("CustomModelTrain", includeSubFolders = false), useLabelFile = false))

  val modelId: String = retry(List(10000, 20000, 30000), () => {
    val resp = FormRecognizerUtils.formGet(getRequestUrl)
    val modelInfo = resp.parseJson.asJsObject.fields.getOrElse("modelInfo", "")
    val status = modelInfo match {
      case x: JsObject => x.fields.getOrElse("status", "") match {
        case y: JsString => y.value
        case _ => throw new RuntimeException(s"No status found in response/modelInfo: $resp/$modelInfo")
      }
      case _ => throw new RuntimeException(s"No modelInfo found in response: $resp")
    }
    status match {
      case "ready" => modelInfo.asInstanceOf[JsObject].fields.getOrElse("modelId", "").asInstanceOf[JsString].value
      case "creating" => throw new RuntimeException("model creating ...")
      case s => throw new RuntimeException(s"Received unknown status code: $s")
    }
  })

  override def afterAll(): Unit = {
    if (modelId != "") {
      FormRecognizerUtils.formDelete(modelId)
    }
    super.afterAll()
  }

  lazy val getCustomModel: GetCustomModel = new GetCustomModel()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus")
    .setModelId(modelId).setIncludeKeys(true)
    .setOutputCol("model").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("model.trainResult.trainingDocuments")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Get model detail") {
    val results = getCustomModel.transform(df)
      .withColumn("keys", col("model").getField("keys"))
      .select("keys")
      .collect()
    assert(results.head.getString(0) ===
      ("""{"clusters":{"0":["BILL TO:","CUSTOMER ID:","CUSTOMER NAME:","DATE:","DESCRIPTION",""" +
        """"DUE DATE:","F.O.B. POINT","INVOICE:","P.O. NUMBER","QUANTITY","REMIT TO:","REQUISITIONER",""" +
        """"SALESPERSON","SERVICE ADDRESS:","SHIP TO:","SHIPPED VIA","TERMS","TOTAL","UNIT PRICE"]}}""").stripMargin)
  }

  override def testObjects(): Seq[TestObject[GetCustomModel]] =
    Seq(new TestObject(getCustomModel, df))

  override def reader: MLReadable[_] = GetCustomModel
}

class AnalyzeCustomModelSuite extends TransformerFuzzing[AnalyzeCustomModel]
  with CognitiveKey with Flaky with FormRecognizerUtils {

  val getRequestUrl: String = FormRecognizerUtils.formPost("", TrainCustomModelSchema(
    trainingDataSAS, SourceFilter("CustomModelTrain", includeSubFolders = false), useLabelFile = false))

  val modelId: String = retry(List(10000, 20000, 30000), () => {
    val resp = FormRecognizerUtils.formGet(getRequestUrl)
    val modelInfo = resp.parseJson.asJsObject.fields.getOrElse("modelInfo", "")
    val status = modelInfo match {
      case x: JsObject => x.fields.getOrElse("status", "") match {
        case y: JsString => y.value
        case _ => throw new RuntimeException(s"No status found in response/modelInfo: $resp/$modelInfo")
      }
      case _ => throw new RuntimeException(s"No modelInfo found in response: $resp")
    }
    status match {
      case "ready" => modelInfo.asInstanceOf[JsObject].fields.getOrElse("modelId", "").asInstanceOf[JsString].value
      case "creating" => throw new RuntimeException("model creating ...")
      case s => throw new RuntimeException(s"Received unknown status code: $s")
    }
  })

  override def afterAll(): Unit = {
    val listCustomModels: ListCustomModels = new ListCustomModels()
      .setSubscriptionKey(cognitiveKey).setLocation("eastus")
      .setOp("full").setOutputCol("models").setConcurrency(5)
    val results = listCustomModels.transform(df)
      .withColumn("modelIds", col("models").getField("modelList").getField("modelId"))
      .select("modelIds")
      .collect()
    val modelIds = results.flatMap(_.getAs[Seq[String]](0))
    modelIds.foreach(
      x => FormRecognizerUtils.formDelete(x)
    )
    super.afterAll()
  }

  lazy val analyzeCustomModel: AnalyzeCustomModel = new AnalyzeCustomModel()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus").setModelId(modelId)
    .setImageUrlCol("source").setOutputCol("form").setConcurrency(5)

  lazy val bytesAnalyzeCustomModel: AnalyzeCustomModel = new AnalyzeCustomModel()
    .setSubscriptionKey(cognitiveKey).setLocation("eastus").setModelId(modelId)
    .setImageBytesCol("imageBytes").setOutputCol("form").setConcurrency(5)

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = {
      df.select("source", "form.analyzeResult.readResults")
    }
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage with URL") {
    val results = imageDf4.mlTransform(analyzeCustomModel,
      AnalyzeCustomModel.flattenReadResults("form", "readForm"),
      AnalyzeCustomModel.flattenPageResults("form", "pageForm"),
      AnalyzeCustomModel.flattenDocumentResults("form", "docForm"))
      .select("readForm", "pageForm", "docForm")
      .collect()
    assert(results.head.getString(0) === "")
    assert(results.head.getString(1)
      .startsWith(("""KeyValuePairs: key: Invoice For: value: Microsoft 1020 Enterprise Way""")))
    assert(results.head.getString(2) === "")
  }

  test("Basic Usage with Bytes") {
    val results = bytesDF4.mlTransform(bytesAnalyzeCustomModel,
      AnalyzeCustomModel.flattenReadResults("form", "readForm"),
      AnalyzeCustomModel.flattenPageResults("form", "pageForm"),
      AnalyzeCustomModel.flattenDocumentResults("form", "docForm"))
      .select("readForm", "pageForm", "docForm")
      .collect()
    assert(results.head.getString(0) === "")
    assert(results.head.getString(1)
      .startsWith(("""KeyValuePairs: key: Invoice For: value: Microsoft 1020 Enterprise Way""")))
    assert(results.head.getString(2) === "")
  }

  override def testObjects(): Seq[TestObject[AnalyzeCustomModel]] =
    Seq(new TestObject(analyzeCustomModel, imageDf4))

  override def reader: MLReadable[_] = AnalyzeCustomModel
}
