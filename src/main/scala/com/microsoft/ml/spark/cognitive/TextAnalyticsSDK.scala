package com.microsoft.ml.spark.cognitive
import com.azure.ai.textanalytics.models.{AssessmentSentiment, DetectLanguageInput, DocumentSentiment, ExtractKeyPhraseResult, KeyPhrasesCollection, SentenceSentiment, SentimentConfidenceScores, TargetSentiment, TextAnalyticsRequestOptions, TextAnalyticsWarning, TextDocumentInput}
import com.azure.ai.textanalytics.implementation.models.SentenceOpinionSentiment
import com.azure.ai.textanalytics.{TextAnalyticsClient, TextAnalyticsClientBuilder}
import com.azure.core.credential.AzureKeyCredential
import com.microsoft.ml.spark.core.contracts.{HasConfidenceScoreCol, HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.schema.{DatasetExtensions, SparkBindings}
import com.microsoft.ml.spark.io.http.{HasErrorCol, SimpleHTTPTransformer}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.param.{Param, ParamMap, ServiceParam}
import org.apache.spark.ml.util.Identifiable._
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, NamespaceInjections, PipelineModel, Transformer}
import org.apache.spark.sql.functions.{array, col, struct}
import org.apache.spark.sql.types.{ArrayType, DataTypes, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import com.azure.ai.textanalytics.models
import com.azure.core.util.Context
import com.microsoft.ml.spark.stages.{DropColumns, Lambda, UDFTransformer}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import spray.json.DefaultJsonProtocol.{StringJsonFormat, seqFormat}

import java.net.URI
import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

trait HasTextCol extends HasServiceParams {
  val text = new ServiceParam[Seq[String]](this, "text", "the text in the request body", isRequired = true)

  def setTextCol(v: String): this.type = setVectorParam(text, v)

  def setText(v: Seq[String]): this.type = setScalarParam(text, v)

  def setText(v: String): this.type = setScalarParam(text, Seq(v))

  setDefault(text -> Right("text"))
}


trait HasLanguageCol extends HasServiceParams {
  val language = new ServiceParam[Seq[String]](this, "language",
    "the language code of the text (optional for some services)")

  def setLanguageCol(v: String): this.type = setVectorParam(language, v)

  def setLanguage(v: Seq[String]): this.type = setScalarParam(language, v)

  def setLanguage(v: String): this.type = setScalarParam(language, Seq(v))

  setDefault(language -> Left(Seq("en")))
}

abstract class TextAnalyticsSDKBase[T](val textAnalyticsOptions: Option[TextAnalyticsRequestOptionsV4] = None)
  extends Transformer
  with HasInputCol with HasErrorCol
  with HasEndpoint with HasSubscriptionKey
    with HasTextCol with HasLanguageCol
    with HasOutputCol
    with ComplexParamsWritable with BasicLogging {

  protected def outputSchema: StructType

  val responseTypeBinding: SparkBindings[TAResponseV4[T]]

  def invokeTextAnalytics(text: Seq[String]): TAResponseV4[T]

  protected lazy val textAnalyticsClient: TextAnalyticsClient =
    new TextAnalyticsClientBuilder()
      .credential(new AzureKeyCredential(getSubscriptionKey))
      .endpoint(getEndpoint)
      .buildClient()

  protected def transformTextRows(toRow: TAResponseV4[T] => Row)
                                 (rows: Iterator[Row]): Iterator[Row] = {
    rows.map { row =>
       val results = invokeTextAnalytics(getValue(row, text))
        Row.fromSeq(row.toSeq ++ Seq(toRow(results))) // Adding a new column
    }}

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val df = dataset.toDF
      val schema = dataset.schema
      val enc = RowEncoder(df.schema.add(getOutputCol, responseTypeBinding.schema))
      val toRow = responseTypeBinding.makeToRowConverter
      df.mapPartitions(transformTextRows(
        toRow,
      ))(enc)
    })
  }

  override def transformSchema(schema: StructType): StructType = {
    // Validate input schema
    val inputType = schema($(inputCol)).dataType
    require(inputType.equals(DataTypes.StringType), s"The input column must be of type String, but got $inputType")
    schema.add(getOutputCol, outputSchema)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}

object TextAnalyticsLanguageDetection extends ComplexParamsReadable[TextAnalyticsLanguageDetection]
class TextAnalyticsLanguageDetection(override val textAnalyticsOptions: Option[TextAnalyticsRequestOptionsV4] = None,
                                     override val uid: String = randomUID("TextAnalyticsLanguageDetection"))
  extends TextAnalyticsSDKBase[DetectedLanguageV4](textAnalyticsOptions)
  with HasConfidenceScoreCol {
  logClass()

  /**
   * Params for optional input column names.
   */
  // Country Hint
  final val countryHintCol: Param[String] = new Param[String](this, "countryHintCol", "country hint column name")

  final def getCountryHintCol: String = $(countryHintCol)

  final def setCountryHintCol(value: String): this.type = set(countryHintCol, value)
  setDefault(countryHintCol -> "CountryHint")

  override def outputSchema: StructType = DetectLanguageResponseV4.schema

  override val responseTypeBinding: SparkBindings[TAResponseV4[DetectedLanguageV4]] = DetectLanguageResponseV4

  override def invokeTextAnalytics(input: Seq[String]): TAResponseV4[DetectedLanguageV4] = {
    val document = input.toList.map(text => new DetectLanguageInput(scala.util.Random.nextInt(100).toString, text))
    val detectLanguageResultCollection = textAnalyticsClient.detectLanguageBatchWithResponse(
      document.asJava, null, Context.NONE).getValue
    val detectLanguageResult = detectLanguageResultCollection.asScala.head

    val languageResult = if (detectLanguageResult.isError) {
      None
    } else {
      Some(DetectedLanguageV4(
        detectLanguageResult.getPrimaryLanguage.getName,
        detectLanguageResult.getPrimaryLanguage.getIso6391Name,
        detectLanguageResult.getPrimaryLanguage.getConfidenceScore))
    }

    val error = if (detectLanguageResult.isError) {
      val error = detectLanguageResult.getError
      Some(TAErrorV4(error.getErrorCode.toString, error.getMessage, error.getTarget))
    } else {
      None
    }

    val stats = Option(detectLanguageResult.getStatistics) match {
      case Some(s) => Some(DocumentStatistics(s.getCharacterCount, s.getTransactionCount))
      case None => None
    }

    TAResponseV4[DetectedLanguageV4](
      languageResult,
      error,
      stats,
      Some(detectLanguageResultCollection.getModelVersion))
  }
}

object TextAnalyticsKeyphraseExtraction extends ComplexParamsReadable[TextAnalyticsKeyphraseExtraction]
class TextAnalyticsKeyphraseExtraction (override val textAnalyticsOptions: Option[TextAnalyticsRequestOptionsV4] = None,
                                     override val uid: String = randomUID("TextAnalyticsKeyphraseExtraction"))
  extends TextAnalyticsSDKBase[KeyphraseV4](textAnalyticsOptions) {
  logClass()

  override val responseTypeBinding: SparkBindings[TAResponseV4[KeyphraseV4]]
  = KeyPhraseResponseV4

  override def invokeTextAnalytics(input: Seq[String]): TAResponseV4[KeyphraseV4] = {
    val document = input.toList.map(text => new TextDocumentInput(scala.util.Random.nextInt(100).toString, text))
    val extractKeyPhrasesResultCollection = textAnalyticsClient.extractKeyPhrasesBatchWithResponse(
      document.asJava, null, Context.NONE).getValue

    val keyPhraseExtractionResult = extractKeyPhrasesResultCollection.asScala.head
    val keyPhraseDocument = keyPhraseExtractionResult.getKeyPhrases()

    val keyphraseResult = if (keyPhraseExtractionResult.isError) {
      None
    } else {
      Some(KeyphraseV4(
        keyPhraseDocument.asScala.toList,
        keyPhraseDocument.getWarnings.asScala.map(
          item => TAWarningV4(item.getWarningCode.toString,item.getMessage)
        ).toList))
    }

    val error = if (keyPhraseExtractionResult.isError) {
      val error = keyPhraseExtractionResult.getError
      Some(TAErrorV4(error.getErrorCode.toString, error.getMessage, error.getTarget))
    } else {
      None
    }

    val stats = Option(keyPhraseExtractionResult.getStatistics) match {
      case Some(s) => Some(DocumentStatistics(s.getCharacterCount, s.getTransactionCount))
      case None => None
    }

    TAResponseV4[KeyphraseV4](
      keyphraseResult,
      error,
      stats,
      Some(extractKeyPhrasesResultCollection.getModelVersion))
  }
  override def outputSchema: StructType = KeyPhraseResponseV4.schema
}

object TextSentimentV4 extends ComplexParamsReadable[TextSentimentV4]
class TextSentimentV4(override val textAnalyticsOptions: Option[TextAnalyticsRequestOptionsV4] = None,
                      override val uid: String = randomUID("TextSentimentV4"))
  extends TextAnalyticsSDKBase[SentimentScoredDocumentV4](textAnalyticsOptions)
    with HasConfidenceScoreCol {
  logClass()

  override val responseTypeBinding: SparkBindings[TAResponseV4[SentimentScoredDocumentV4]]
  = SentimentResponseV4
  override def invokeTextAnalytics(input: Seq[String]): TAResponseV4[SentimentScoredDocumentV4] = {
    val document = input.toList.map(text => new TextDocumentInput(scala.util.Random.nextInt(100).toString, text))

    val textSentimentResultCollection = textAnalyticsClient.analyzeSentimentBatchWithResponse(
     document.asJava, null, Context.NONE).getValue

    def getConfidenceScore(score: SentimentConfidenceScores): SentimentConfidenceScoreV4 = {
      SentimentConfidenceScoreV4(
        score.getNegative,
        score.getNeutral,
        score.getPositive)
    }

    def getTarget(target: TargetSentiment): TargetV4 = {
      TargetV4(
        target.getText,
        target.getSentiment.toString,
        getConfidenceScore(target.getConfidenceScores),
        target.getOffset,
        target.getLength)
    }

    def getAssessment(assess: AssessmentSentiment): AssessmentV4 = {
      AssessmentV4(
        assess.getText,
        assess.getSentiment.toString,
        getConfidenceScore(assess.getConfidenceScores),
        assess.isNegated,
        assess.getOffset,
        assess.getLength)
    }

    def getSentenceSentiment(sentencesent: SentenceSentiment): SentimentSentenceV4 = {
      SentimentSentenceV4(
        sentencesent.getText,
        sentencesent.getSentiment.toString,
        getConfidenceScore(sentencesent.getConfidenceScores),
        Option(sentencesent.getOpinions).map(sentmap =>
          sentmap.asScala.toList.map(op =>
            OpinionV4(getTarget(op.getTarget)
              , op.getAssessments.asScala.toList.map(assessment =>
                getAssessment(assessment))))),
        sentencesent.getOffset,
        sentencesent.getLength)
    }

    def getDocumentSentiment(doc: DocumentSentiment): SentimentScoredDocumentV4 = {
      SentimentScoredDocumentV4(
        doc.getSentiment.toString,
        getConfidenceScore(doc.getConfidenceScores),
        doc.getSentences.asScala.toList.map(sentenceSentiment =>
          getSentenceSentiment(sentenceSentiment)),
        doc.getWarnings.asScala.toList.map(warnings =>
          WarningsV4(warnings.getMessage, warnings.getWarningCode.toString)))
    }

    val textSentimentResult = textSentimentResultCollection.asScala.head
    val documentSentiment = textSentimentResult.getDocumentSentiment();

    val sentimentResult = if (textSentimentResult.isError){
      None
    } else {
      Some(getDocumentSentiment(documentSentiment))
    }

    val error = if (textSentimentResult.isError) {
      val error = textSentimentResult.getError
      Some(TAErrorV4(error.getErrorCode.toString, error.getMessage, error.getTarget))
    } else {
      None
    }

    val stats = Option(textSentimentResult.getStatistics).map(s =>
      DocumentStatistics(s.getCharacterCount, s.getTransactionCount))

    TAResponseV4[SentimentScoredDocumentV4](
      sentimentResult,
      error,
      stats,
      Some(textSentimentResultCollection.getModelVersion))
  }
  override def outputSchema: StructType = SentimentResponseV4.schema
}

  object SentimentResponseV4 extends SparkBindings[TAResponseV4[SentimentScoredDocumentV4]]

  case class SentimentConfidenceScoreV4(negative: Double, neutral: Double, positive: Double)

  case class SentimentScoredDocumentV4(sentiment: String,
                                       confidenceScores: SentimentConfidenceScoreV4,
                                       sentences: List[SentimentSentenceV4],
                                       warnings: List[WarningsV4])

  case class SentimentSentenceV4(text: String,
                                 sentiment: String,
                                 confidenceScores: SentimentConfidenceScoreV4,
                                 opinion: Option[List[OpinionV4]],
                                 offset: Int,
                                 length: Int)

  case class OpinionV4(target: TargetV4, assessment: List[AssessmentV4])

  case class TargetV4(text: String,
                      sentiment: String,
                      confidenceScores: SentimentConfidenceScoreV4,
                      offset: Int,
                      length: Int)

  case class AssessmentV4(text: String,
                          sentiment: String,
                          confidenceScores: SentimentConfidenceScoreV4,
                          isNegated: Boolean,
                          offset: Int,
                          length: Int)

  case class WarningsV4(text: String, warningCode: String)

  case class TextAnalyticsRequestOptionsV4(modelVersion: String,
                                           includeStatistics: Boolean,
                                           disableServiceLogs: Boolean)

