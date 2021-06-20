// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.io.IOException

import sys.process._
import spray.json._
import DefaultJsonProtocol._

object Secrets {
  private val KvName = "mmlspark-keys"
  private val SubscriptionID = "e342c2c0-f844-4b18-9208-52c8c234c30e"

  protected def exec(command: String): String = {
    val os = sys.props("os.name").toLowerCase
    os match {
      case x if x contains "windows" => Seq("cmd", "/C") ++ Seq(command) !!
      case _ => command !!
    }
  }

  // Keep overhead of setting account down
  lazy val AccountString: String = {
    try {
      exec(s"az account set -s $SubscriptionID")
    } catch {
      case e: java.lang.RuntimeException =>
        println(s"Secret fetch error: ${e.toString}")
      case e: IOException =>
        println(s"Secret fetch error: ${e.toString}")
    }
    SubscriptionID
  }

  private def getSecret(secretName: String): String = {
    println(s"fetching secret: $secretName from $AccountString")
    val secretJson = exec(s"az keyvault secret show --vault-name $KvName --name $secretName")
    secretJson.parseJson.asJsObject().fields("value").convertTo[String]
  }

  lazy val CognitiveApiKey: String = getSecret("cognitive-api-key")
  lazy val CustomSpeechApiKey: String = getSecret("custom-speech-api-key")
  lazy val ConversationTranscriptionUrl: String = getSecret("conversation-transcription-url")
  lazy val ConversationTranscriptionKey: String = getSecret("conversation-transcription-key")

  lazy val AnomalyApiKey: String = getSecret("anomaly-api-key")
  lazy val AzureSearchKey: String = getSecret("azure-search-key")
  lazy val BingSearchKey: String = getSecret("bing-search-key")
  lazy val PowerbiURL: String = getSecret("powerbi-url")
  lazy val AdbToken: String = getSecret("adb-token")
  lazy val SynapseStorageKey: String = getSecret("mmlsparkbuildsynapse-key")
  lazy val SynapseSpnKey: String = getSecret("synapse-spn-key")

}
