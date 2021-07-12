package com.microsoft.ml.spark.cognitive

import com.microsoft.ml.spark.core.schema.SparkBindings
import org.apache.spark.ml.ComplexParamsReadable


object DetectLanguageResponseV4 extends SparkBindings[TAResponseV4[DetectedLanguageV4]]

object KeyPhraseResponseV4 extends SparkBindings[TAResponseV4[KeyphraseV4]]

case class TAResponseV4[T](result: Option[T],
                           error: Option[TAErrorV4],
                           statistics: Option[DocumentStatistics],
                           modelVersion: Option[String])
case class DetectedLanguageV4(name: String, iso6391Name: String, confidenceScore: Double)

case class TAErrorV4(errorCode: String, errorMessage: String, target: String)

case class TAWarningV4 (warningCode: String, message: String)


case class KeyphraseV4(keyPhrases: List[String], warnings: List[TAWarningV4])
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

