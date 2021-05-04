// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split3

import java.io.{ByteArrayInputStream, File, FileInputStream}
import java.net.URI

import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.cognitive.split1.CognitiveKey
import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.commons.compress.utils.IOUtils
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.functions.{col, lit, to_json}
import org.apache.spark.sql.{DataFrame, Row}
import org.scalactic.Equality
import org.scalatest.Assertion

trait CustomSpeechKey {
  lazy val customSpeechKey: String = sys.env.getOrElse("CUSTOM_SPEECH_API_KEY", Secrets.CustomSpeechApiKey)
}

trait SpeechToTextSDKSuiteBase extends TestBase with CognitiveKey with CustomSpeechKey {

  import spark.implicits._

  val region = "eastus"
  lazy val resourcesDir = new File(new File(getClass.getResource("/").toURI).toString)
  val uri = new URI(s"https://$region.api.cognitive.microsoft.com/sts/v1.0/issuetoken")
  val language = "en-us"
  val profanity = "masked"
  val format = "simple"

  val streamUrl: String = "https://bitdash-a.akamaihd.net/content/MI201109210084_1/" +
    "m3u8s/f08e80da-bf1d-4e3d-8899-f0f6155f6efa.m3u8"

  val jaccardThreshold = 0.9

  lazy val audioPaths: Seq[File] = Seq("audio1.wav", "audio2.wav", "audio3.mp3",
    "dialogue.mp3", "mark.wav", "lily.wav")
    .map(new File(resourcesDir, _))

  lazy val audioBytes: Seq[Array[Byte]] = audioPaths.map(
    path => IOUtils.toByteArray(new FileInputStream(path))
  )

  lazy val Seq(bytes1, bytes2, bytes3, dialogueBytes, speaker1Bytes, speaker2Bytes) = audioBytes

  lazy val textPaths: Seq[File] = Seq("audio1.txt", "audio2.txt", "audio3.txt", "audio4.txt")
    .map(new File(resourcesDir, _))

  lazy val Seq(text1, text2, text3, text4) = textPaths.map(f =>
    StreamUtilities.usingSource(scala.io.Source.fromFile(f)) { source =>
      source.mkString
    }.get)

  lazy val Seq(audioDf1, audioDf2, audioDf3, dialogueDf) = audioBytes.take(4).map(bytes =>
    Seq(Tuple1(bytes)).toDF("audio")
  )

  lazy val dialogueDf2: DataFrame = Seq(
    (1, audioBytes(3)),
    (2, audioBytes(3))
  ).toDF("num", "audio")

  /** Simple similarity test using Jaccard index */
  def jaccardSimilarity(s1: String, s2: String): Double = {
    val a = s1.toLowerCase.sliding(2).toSet
    val b = s2.toLowerCase.sliding(2).toSet
    a.intersect(b).size.toDouble / (a | b).size.toDouble
  }

  def sdk: SpeechSDKBase

  def speechArrayToText(speechArray: Seq[SharedSpeechFields]): String = {
    speechArray.map(sr => sr.DisplayText.getOrElse("")).mkString(" ")
  }

  def speechTest(format: String, audioBytes: Array[Byte], expectedText: String): Assertion = {
    val resultArray = sdk.inputStreamToText(
      new ByteArrayInputStream(audioBytes),
      "wav",
      uri, cognitiveKey, profanity, language, format, None, Seq())
    val result = speechArrayToText(resultArray.toSeq)
    if (format == "simple") {
      resultArray.foreach { rp =>
        assert(rp.NBest.isEmpty)
      }
    } else {
      resultArray.foreach { rp =>
        assert(rp.NBest.get.nonEmpty)
      }
    }
    assert(jaccardSimilarity(expectedText, result) > .9)
  }

  def extractResults(df: DataFrame, streaming: Boolean): Seq[SpeechResponse] = {
    val toObj: Row => SpeechResponse = SpeechResponse.makeFromRowConverter
    val collectedResults = df.select("text").collect()
    if (streaming) {
      collectedResults.map(row => toObj(row.getAs[Row](0)))
    } else {
      collectedResults.flatMap(row => row.getSeq[Row](0).map(toObj))
    }
  }

  def dfTest(format: String,
             input: DataFrame,
             expectedText: String,
             verbose: Boolean = false,
             sdk: SpeechSDKBase = sdk,
             threshold: Double = jaccardThreshold): Assertion = {
    val resultSeq = extractResults(
      sdk.setFormat(format).transform(input),
      sdk.getStreamIntermediateResults)
    val result = speechArrayToText(resultSeq)

    if (verbose) {
      println(s"Expected: $expectedText")
      println(s"Actual: $result")
    }
    if (format == "simple") {
      resultSeq.foreach { rp =>
        assert(rp.NBest.isEmpty)
      }
    } else {
      resultSeq.foreach { rp =>
        assert(rp.NBest.get.nonEmpty)
      }
    }
    assert(jaccardSimilarity(expectedText, result) > threshold)
  }
}

class SpeechToTextSDKSuite extends TransformerFuzzing[SpeechToTextSDK] with SpeechToTextSDKSuiteBase {

  override val testFitting = false

  import spark.implicits._

  def sdk: SpeechToTextSDK = new SpeechToTextSDK()
    .setSubscriptionKey(cognitiveKey)
    .setLocation(region)
    .setOutputCol("text")
    .setAudioDataCol("audio")
    .setLanguage("en-US")
    .setProfanity("Masked")

  def customSdk: SpeechToTextSDK = sdk
    .setSubscriptionKey(customSpeechKey)
    .setEndpointId("395cdcf7-e7db-4083-aebe-868a7d80ca74")

  override lazy val dfEq: Equality[DataFrame] = new Equality[DataFrame] {
    override def areEqual(a: DataFrame, b: Any): Boolean = {
      jaccardSimilarity(
        speechArrayToText(extractResults(a, true)),
        speechArrayToText(extractResults(b.asInstanceOf[DataFrame], true))
      ) > jaccardThreshold
    }
  }

  test("Simple audioBytesToText 1") {
    speechTest("simple", bytes1, text1)
  }

  test("Detailed audioBytesToText 1") {
    speechTest("detailed", bytes1, text1)
  }

  ignore("Detailed audioBytesToText 2") {
    speechTest("detailed", bytes2, text2)
  }

  test("Simple audioBytesToText 2") {
    speechTest("simple", bytes2, text2)
  }

  test("Simple SDK Usage Audio 1") {
    dfTest("simple", audioDf1, text1)
  }

  test("Simple SDK Usage Audio 1 multi") {
    dfTest("simple", audioDf2, text2)
    dfTest("simple", audioDf1, text1)
    dfTest("simple", audioDf1, text1)
    dfTest("simple", audioDf1, text1)
  }

  test("Detailed SDK Usage Audio 1") {
    dfTest("detailed", audioDf1, text1)
  }

  test("Simple SDK Usage Audio 2") {
    dfTest("simple", audioDf2, text2)
  }

  test("Simple SDK Usage without streaming") {
    dfTest("simple", audioDf1, text1, sdk = sdk.setStreamIntermediateResults(false))
  }

  test("Custom SDK Usage") {
    dfTest("simple", audioDf1, text1, sdk = customSdk)
  }

  test("Detailed SDK Usage Audio 2") {
    dfTest("detailed", audioDf2, text2)
  }

  test("URI based access") {
    val uriDf = Seq(Tuple1(audioPaths(1).toURI.toString))
      .toDF("audio")
    dfTest("detailed", uriDf, text2)
  }

  test("URL based access") {
    tryWithRetries(Array(100, 500)) { () => //For handling flaky build machines
      val uriDf = Seq(Tuple1("https://mmlspark.blob.core.windows.net/datasets/Speech/audio2.wav"))
        .toDF("audio")
      dfTest("detailed", uriDf, text2)
    }
  }

  test("SAS URL based access") {
    val sasURL = "https://mmlspark.blob.core.windows.net/datasets/Speech/audio2.wav?sv=2019-12-12" +
      "&st=2021-01-25T16%3A40%3A13Z&se=2024-01-26T16%3A40%3A00Z" +
      "&sr=b&sp=r&sig=NpFm%2FJemAJOGIya1ykQ6f80YdvwpiAuJjnb2RVDtKro%3D"

    tryWithRetries(Array(100, 500)) { () => //For handling flaky build machines
      val uriDf = Seq(Tuple1(sasURL))
        .toDF("audio")
      dfTest("detailed", uriDf, text2, sdk = sdk)
    }
  }

  test("Detailed SDK with mp3 (Linux only)") {
    dfTest("detailed", audioDf3, text3, sdk = sdk.setFileType("mp3"), verbose = true, threshold = .6)
  }

  test("m3u8 based access") {
    val sdk2 = sdk.setExtraFfmpegArgs(Array("-t", "60"))
      .setLanguage("en-US")
    // 20 seconds of streaming
    tryWithRetries(Array(100, 500)) { () => //For handling flaky build machines
      val uriDf = Seq(Tuple1(streamUrl))
        .toDF("audio")
      dfTest(
        "detailed",
        uriDf, text4, sdk = sdk2, threshold = .6)
    }
  }

  test("m3u8 file writing") {
    val outputMp3 = new File(savePath, "output.mp3")
    val outputJson = new File(savePath, "output.json")

    try {
      val sdk2 = sdk.setExtraFfmpegArgs(Array("-t", "20"))
        .setRecordedFileNameCol("recordedFile")
        .setRecordAudioData(true)
        .setLanguage("en-US")

      // 20 seconds of streaming
      val uriDf = Seq(Tuple2(streamUrl, outputMp3.toString))
        .toDF("audio", "recordedFile")
      sdk2.transform(uriDf).write.mode("overwrite").json(outputJson.toString)

      assert(outputMp3.exists())
      assert(outputJson.exists())
    } finally {
      FileUtils.forceDelete(outputMp3)
      FileUtils.forceDelete(outputJson)
    }
  }

  test("API vs. SDK") {
    val stt = new SpeechToText()
      .setSubscriptionKey(cognitiveKey)
      .setLocation(region)
      .setOutputCol("text")
      .setAudioDataCol("audio")
      .setLanguage("en-US")
    val toObj: Row => SpeechResponse = SpeechResponse.makeFromRowConverter
    val apiResult = toObj(stt.setFormat("simple")
      .transform(audioDf2).select("text")
      .collect().head.getStruct(0)).DisplayText.getOrElse("")

    val sdkResult = speechArrayToText(sdk.setFormat(format)
      .transform(audioDf2)
      .select("text").collect()
      .map(row => toObj(row.getAs[Row](0)))
    )
    assert(jaccardSimilarity(apiResult, sdkResult) > jaccardThreshold)
  }

  override def testObjects(): Seq[TestObject[SpeechToTextSDK]] =
    Seq(new TestObject(sdk, audioDf2))

  override def reader: MLReadable[_] = SpeechToTextSDK
}

trait TranscriptionSecrets {
  lazy val conversationTranscriptionUrl: String = sys.env.getOrElse("CONVERSATION_TRANSCRIPTION_URL",
    Secrets.ConversationTranscriptionUrl)
  lazy val conversationTranscriptionKey: String = sys.env.getOrElse("CONVERSATION_TRANSCRIPTION_KEY",
    Secrets.ConversationTranscriptionKey)
}

class ConversationTranscriptionSuite extends TransformerFuzzing[ConversationTranscription]
  with SpeechToTextSDKSuiteBase with TranscriptionSecrets {

  override val testFitting = false

  import spark.implicits._

  override def sdk: ConversationTranscription = new ConversationTranscription()
    .setSubscriptionKey(conversationTranscriptionKey)
    .setUrl(conversationTranscriptionUrl)
    .setOutputCol("text")
    .setAudioDataCol("audio")
    .setLanguage("en-US")
    .setProfanity("Masked")

  override lazy val dfEq: Equality[DataFrame] = new Equality[DataFrame] {
    override def areEqual(a: DataFrame, b: Any): Boolean = {
      jaccardSimilarity(
        speechArrayToText(extractResults(a, true)),
        speechArrayToText(extractResults(b.asInstanceOf[DataFrame], true))
      ) > jaccardThreshold
    }
  }


  test("dialogue with participants") {
    val profile1 = SpeechAPI.getSpeakerProfile(audioPaths(4), conversationTranscriptionKey)
    val profile2 = SpeechAPI.getSpeakerProfile(audioPaths(5), conversationTranscriptionKey)
    val fromRow = TranscriptionResponse.makeFromRowConverter
    val speakers = sdk
      .setParticipants(Seq(
        ("user1", "en-US", profile1),
        ("user2", "en-US", profile2)
      ))
      .setFileType("mp3").transform(dialogueDf)
      .select(sdk.getOutputCol)
      .collect()
      .map(r => fromRow(r.getAs[Row]("text")).SpeakerId)
      .filterNot(sid => sid == "Unidentified")

    assert(speakers === Seq("user1", "user2", "user1", "user2"))
  }

  test("dialogue with participant col") {
    val profile1 = SpeechAPI.getSpeakerProfile(audioPaths(4), conversationTranscriptionKey)
    val profile2 = SpeechAPI.getSpeakerProfile(audioPaths(5), conversationTranscriptionKey)
    val participantDf = Seq(
      (1, Seq(TranscriptionParticipant("user1", "en-US", profile1),
        TranscriptionParticipant("user2", "en-US", profile2))),
      (2, null)
    ).toDF("num", "participants")
      .withColumn("participantsJson", to_json(col("participants")))

    val dialogueDf3 = dialogueDf2
      .join(participantDf, Seq("num"), joinType = "left")

    val fromRow = TranscriptionResponse.makeFromRowConverter
    val speakers = sdk
      .setParticipantsJsonCol("participantsJson")
      .setFileType("mp3")
      .transform(dialogueDf3)
      .select(sdk.getOutputCol)
      .collect()
      .map(r => fromRow(r.getAs[Row]("text")).SpeakerId)
      .filterNot(sid => sid == "Unidentified")

    assert(speakers === Seq("user1", "user2", "user1", "user2"))
  }

  ignore("dialogue without profiles") {
    val fromRow = TranscriptionResponse.makeFromRowConverter
    sdk
      .setFileType("mp3").transform(dialogueDf)
      .select(sdk.getOutputCol)
      .collect()
      .map(r => fromRow(r.getAs[Row]("text")))
      .foreach(println)
  }

  test("Simple SDK Usage Audio 1") {
    dfTest("simple", audioDf1, text1)
  }

  test("Simple SDK Usage Audio 2") {
    dfTest("simple", audioDf2, text2)
  }

  test("Simple SDK Usage without streaming") {
    dfTest("simple", audioDf1, text1, sdk = sdk.setStreamIntermediateResults(false))
  }

  test("URI based access") {
    val uriDf = Seq(Tuple1(audioPaths(1).toURI.toString))
      .toDF("audio")
    dfTest("simple", uriDf, text2)
  }

  test("URL based access") {
    tryWithRetries(Array(100, 500)) { () => //For handling flaky build machines
      val uriDf = Seq(Tuple1("https://mmlspark.blob.core.windows.net/datasets/Speech/audio2.wav"))
        .toDF("audio")
      dfTest("simple", uriDf, text2)
    }
  }

  test("SAS URL based access") {
    val sasURL = "https://mmlspark.blob.core.windows.net/datasets/Speech/audio2.wav" +
      "?sv=2019-12-12&st=2021-01-25T16%3A40%3A13Z&se=2024-01-26T16%3A40%3A00Z&sr=b&sp=r" +
      "&sig=NpFm%2FJemAJOGIya1ykQ6f80YdvwpiAuJjnb2RVDtKro%3D"

    tryWithRetries(Array(100, 500)) { () => //For handling flaky build machines
      val uriDf = Seq(Tuple1(sasURL))
        .toDF("audio")
      dfTest("simple", uriDf, text2, sdk = sdk)
    }
  }

  test("Detailed SDK with mp3 (Linux only)") {
    dfTest("simple", audioDf3, text3, sdk = sdk.setFileType("mp3"), verbose = true, threshold = .6)
  }

  test("m3u8 based access") {
    val sdk2 = sdk.setExtraFfmpegArgs(Array("-t", "60"))
      .setLanguage("en-US")
    // 20 seconds of streaming
    tryWithRetries(Array(100, 500)) { () => //For handling flaky build machines
      val uriDf = Seq(Tuple1(streamUrl))
        .toDF("audio")
      dfTest(
        "simple",
        uriDf, text4, sdk = sdk2, threshold = .6)
    }
  }

  test("m3u8 file writing") {
    val outputMp3 = new File(savePath, "output.mp3")
    val outputJson = new File(savePath, "output.json")

    try {
      val sdk2 = sdk.setExtraFfmpegArgs(Array("-t", "20"))
        .setRecordedFileNameCol("recordedFile")
        .setRecordAudioData(true)
        .setLanguage("en-US")

      // 20 seconds of streaming
      val uriDf = Seq(Tuple2(streamUrl, outputMp3.toString))
        .toDF("audio", "recordedFile")
      sdk2.transform(uriDf).write.mode("overwrite").json(outputJson.toString)

      assert(outputMp3.exists())
      assert(outputJson.exists())
    } finally {
      FileUtils.forceDelete(outputMp3)
      FileUtils.forceDelete(outputJson)
    }

  }

  override def testObjects(): Seq[TestObject[ConversationTranscription]] =
    Seq(new TestObject(sdk, audioDf2))

  override def reader: MLReadable[_] = ConversationTranscription
}

