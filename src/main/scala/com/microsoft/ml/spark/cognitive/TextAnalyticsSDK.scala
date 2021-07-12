package com.microsoft.ml.spark.cognitive
import com.azure.ai.textanalytics.models.{AssessmentSentiment, DocumentSentiment, ExtractKeyPhraseResult,
  SentenceSentiment, SentimentConfidenceScores,TargetSentiment, TextAnalyticsRequestOptions, TextAnalyticsWarning}
import com.azure.ai.textanalytics.implementation.models.SentenceOpinionSentiment
import com.azure.ai.textanalytics.{TextAnalyticsClient, TextAnalyticsClientBuilder}
import com.azure.core.credential.AzureKeyCredential
import com.microsoft.ml.spark.core.contracts.{HasConfidenceScoreCol, HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.schema.SparkBindings
import com.microsoft.ml.spark.io.http.HasErrorCol
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.http.client.methods.HttpRequestBase
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.param.{Param, ParamMap, ServiceParam}
import org.apache.spark.ml.util.Identifiable._
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Transformer}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import com.azure.ai.textanalytics.models
import org.apache.spark.sql.Encoder

import java.net.URI
import scala.collection.JavaConverters._

abstract class TextAnalyticsSDKBase[T](val textAnalyticsOptions: Option[TextAnalyticsRequestOptions] = None)
  extends Transformer
  with HasInputCol with HasOutputCol with HasErrorCol
  with HasEndpoint with HasSubscriptionKey
  with ComplexParamsWritable with BasicLogging {

  protected val invokeTextAnalytics: String => TAResponseV4[T]

  protected def outputSchema: StructType

  protected lazy val textAnalyticsClient: TextAnalyticsClient =
    new TextAnalyticsClientBuilder()
      .credential(new AzureKeyCredential(getSubscriptionKey))
      .endpoint(getEndpoint)
      .buildClient()

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val invokeTextAnalyticsUdf = UDFUtils.oldUdf(invokeTextAnalytics, outputSchema)
      dataset.withColumn($(outputCol), invokeTextAnalyticsUdf(col(getInputCol)))
    })
  }

  override def transformSchema(schema: StructType): StructType = {
    // Validate input schema
    val inputType = schema(($(inputCol))).dataType
    require(inputType.equals(DataTypes.StringType), s"The input column must be of type String, but got $inputType")
    schema.add($(outputCol), outputSchema)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}

object TextAnalyticsLanguageDetection extends ComplexParamsReadable[TextAnalyticsLanguageDetection]

class TextAnalyticsLanguageDetection(override val textAnalyticsOptions: Option[TextAnalyticsRequestOptions] = None,
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

  override protected val invokeTextAnalytics: String => TAResponseV4[DetectedLanguageV4] = (text: String) =>
    {
      val detectLanguageResultCollection = textAnalyticsClient.detectLanguageBatch(
        Seq(text).asJava, null, textAnalyticsOptions.orNull)
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

class TextAnalyticsKeyphraseExtraction (override val textAnalyticsOptions: Option[TextAnalyticsRequestOptions] = None,
                                     override val uid: String = randomUID("TextAnalyticsKeyphraseExtraction"))
  extends TextAnalyticsSDKBase[KeyphraseV4](textAnalyticsOptions) {
  logClass()

  override def outputSchema: StructType = KeyPhraseResponseV4.schema

  override protected val invokeTextAnalytics: String => TAResponseV4[KeyphraseV4] = (text: String) =>
  {
    val ExtractKeyPhrasesResultCollection = textAnalyticsClient.extractKeyPhrasesBatch(
      Seq(text).asJava,"en", textAnalyticsOptions.orNull)
    val keyPhraseExtractionResult = ExtractKeyPhrasesResultCollection.asScala.head
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
      Some(ExtractKeyPhrasesResultCollection.getModelVersion))
  }
}
object TextSentimentV4 extends ComplexParamsReadable[TextSentimentV4]
class TextSentimentV4(override val textAnalyticsOptions: Option[TextAnalyticsRequestOptions] = None,
                      override val uid: String = randomUID("TextSentimentV4"))
  extends TextAnalyticsSDKBase[SentimentScoredDocumentV4](textAnalyticsOptions)
    with HasConfidenceScoreCol {
  logClass()

  override def outputSchema: StructType = SentimentResponseV4.schema

  override protected val invokeTextAnalytics: String => TAResponseV4[SentimentScoredDocumentV4] = (text: String) => {

    val textSentimentResultCollection = textAnalyticsClient.analyzeSentimentBatch(
      Seq(text).asJava, "en", textAnalyticsOptions.orNull)

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

    val sentimentResult = if (textSentimentResult.isError) {
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
}



