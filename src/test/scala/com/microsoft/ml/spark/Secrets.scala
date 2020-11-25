// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import sys.process._
import spray.json._
import DefaultJsonProtocol._

object Secrets {
  private val KvName = "mmlspark-keys"
  private val SubscriptionID = "f9b96b36-1f5e-4021-8959-51527e26e6d3"

  protected def exec(command: String): String = {
    val os = sys.props("os.name").toLowerCase
    os match {
      case x if x contains "windows" => Seq("cmd", "/C") ++ Seq(command) !!
      case _ => command !!
    }
  }

  private def getSecret(secretName: String): String = {
    println(s"fetching secret: $secretName")
    exec(s"az account set -s $SubscriptionID")
    val secretJson = exec(s"az keyvault secret show --vault-name $KvName --name $secretName")
    secretJson.parseJson.asJsObject().fields("value").convertTo[String]
  }

  lazy val CognitiveApiKey: String = getSecret("cognitive-api-key")
  lazy val CustomSpeechApiKey: String = getSecret("custom-speech-api-key")
  lazy val AnomalyApiKey: String = getSecret("anomaly-api-key")
  lazy val AzureSearchKey: String = getSecret("azure-search-key")
  lazy val BingImageSearchKey: String = getSecret("bing-image-search-key")
  lazy val PowerbiURL: String = getSecret("powerbi-url")
  lazy val AdbToken: String = getSecret("adb-token")
  lazy val SynapseStorageKey: String = getSecret("synapse-storage-key")

}
