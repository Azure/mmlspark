// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.nbtest

import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.cognitive.RESTHelpers
import com.microsoft.ml.spark.core.env.FileUtilities
import org.apache.commons.io.IOUtils
import org.apache.http.client.entity.UrlEncodedFormEntity
import org.apache.http.client.methods.{HttpDelete, HttpGet, HttpPost}
import org.apache.http.entity.StringEntity
import org.apache.http.message.BasicNameValuePair
import org.json4s.JsonAST.JObject
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write
import org.json4s.{Formats, NoTypeHints}
import spray.json.DefaultJsonProtocol._
import spray.json._

import java.io.{File, InputStream}
import java.util
import scala.annotation.tailrec
import scala.concurrent.{TimeoutException, blocking}
import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

case class LivyBatch(id: Int,
                     state: String,
                     appId: Option[String],
                     appInfo: Option[JObject],
                     log: Seq[String])

case class LivyBatchJob(livyBatch: LivyBatch,
                        sparkPool: String,
                        livyUrl: String)

case class Application(state: String,
                       name: String,
                       livyId: String)

case class Applications(nJobs: Int,
                        sparkJobs: Seq[Application])

object SynapseUtilities {

  implicit val Fmts: Formats = Serialization.formats(NoTypeHints)
  lazy val Token: String = getSynapseToken

  val Folder = s"build_${BuildInfo.version}/scripts"
  val TimeoutInMillis: Int = 20 * 60 * 1000
  val StorageAccount: String = "mmlsparkbuildsynapse"
  val StorageContainer: String = "build"
  val TenantId: String = "72f988bf-86f1-41af-91ab-2d7cd011db47"
  val ClientId: String = "85dde348-dd2b-43e5-9f5a-22262af45332"

  def listPythonFiles(): Array[String] = {
    Option(
      FileUtilities
        .join(BuildInfo.baseDirectory, "notebooks", "samples")
        .getCanonicalFile
        .listFiles()
        .filter(_.getAbsolutePath.endsWith(".py"))
        .filter(_.getAbsolutePath.contains("-"))
        .filterNot(_.getAbsolutePath.contains("CyberML"))
        .filterNot(_.getAbsolutePath.contains("DeepLearning"))
        .filterNot(_.getAbsolutePath.contains("ConditionalKNN"))
        .filterNot(_.getAbsolutePath.contains("HyperParameterTuning"))
        .filterNot(_.getAbsolutePath.contains("Regressor.py"))
        .filterNot(_.getAbsolutePath.contains("Overview"))
        .filterNot(_.getAbsolutePath.contains("ModelInterpretation"))
        .map(file => file.getAbsolutePath))
      .get
      .sorted
  }

  def listPythonJobFiles(): Array[String] = {
    Option(
      FileUtilities
        .join(BuildInfo.baseDirectory, "notebooks", "samples")
        .getCanonicalFile
        .listFiles()
        .filter(_.getAbsolutePath.endsWith(".py"))
        .filterNot(_.getAbsolutePath.contains("-"))
        .filterNot(_.getAbsolutePath.contains(" "))
        .map(file => file.getAbsolutePath))
      .get
      .sorted
  }

  def listNoteBookFiles(): Array[String] = {
    Option(
      FileUtilities
        .join(BuildInfo.baseDirectory, "notebooks", "samples")
        .getCanonicalFile
        .listFiles()
        .filter(_.getAbsolutePath.endsWith(".ipynb"))
        .map(file => file.getAbsolutePath))
      .get
      .sorted
  }

  def postMortem(batch: LivyBatch, livyUrl: String): LivyBatch = {
    batch.log.foreach(println)
    write(batch)
    batch
  }

  def poll(id: Int, livyUrl: String, backOffs: List[Int] = List(100, 1000, 5000)): LivyBatch = {
    val getStatsRequest = new HttpGet(s"$livyUrl/$id")
    getStatsRequest.setHeader("Authorization", s"Bearer $Token")
    val statsResponse = RESTHelpers.safeSend(getStatsRequest, backoffs = backOffs, close = false)
    val batch = parse(IOUtils.toString(statsResponse.getEntity.getContent, "utf-8")).extract[LivyBatch]
    statsResponse.close()
    batch
  }

  def showSubmittingJobs(workspaceName: String, poolName: String): Applications = {
    val uri: String =
      "https://" +
        s"$workspaceName.dev.azuresynapse.net" +
        "/monitoring/workloadTypes/spark/applications" +
        "?api-version=2020-10-01-preview" +
        "&filter=(((state%20eq%20%27Queued%27)%20or%20(state%20eq%20%27Submitting%27))" +
        s"%20and%20(sparkPoolName%20eq%20%27$poolName%27))"
    val getRequest = new HttpGet(uri)
    getRequest.setHeader("Authorization", s"Bearer $Token")
    val jobsResponse = RESTHelpers.safeSend(getRequest, close = false)
    val activeJobs =
      parse(IOUtils.toString(jobsResponse.getEntity.getContent, "utf-8"))
        .extract[Applications]
    activeJobs
  }

  def checkPool(workspaceName: String, poolName: String): Boolean = {
    val nSubmittingJob = showSubmittingJobs(workspaceName, poolName).nJobs
    nSubmittingJob == 0
  }

  @tailrec
  def monitorPool(workspaceName: String, sparkPools: Array[String]): String = {

    val readyPools = sparkPools.filter(checkPool(workspaceName, _))

    if (readyPools.length > 0) {
      val readyPool = readyPools(0)
      println(s"Spark Pool: $readyPool is ready")
      readyPool
    }
    else {
      println(s"None spark pool is ready to submit job, waiting 10s")
      blocking {
        Thread.sleep(10000)
      }
      monitorPool(workspaceName, sparkPools)
    }
  }

  @tailrec
  def retry(id: Int, livyUrl: String, timeout: Int, startTime: Long): LivyBatch = {
    val batch = poll(id, livyUrl)
    println(s"batch state $id : ${batch.state}")
    if (batch.state == "success") {
      batch
    } else {
      if ((System.currentTimeMillis() - startTime) > timeout) {
        throw new TimeoutException(s"Job $id timed out.")
      }
      else if (batch.state == "dead") {
        postMortem(batch, livyUrl)
        throw new RuntimeException(s"Dead")
      }
      else if (batch.state == "error") {
        postMortem(batch, livyUrl)
        throw new RuntimeException(s"Error")
      }
      else {
        blocking {
          Thread.sleep(8000)
        }
        println(s"retrying id $id")
        retry(id, livyUrl, timeout, startTime)
      }
    }
  }

  def uploadAndSubmitNotebook(livyUrl: String, notebookPath: String): LivyBatch = {
    val convertedPyScript = new File(notebookPath)
    val abfssPath = uploadScript(convertedPyScript.getAbsolutePath, s"$Folder/${convertedPyScript.getName}")
    submitRun(livyUrl, abfssPath)
  }

  private def uploadScript(file: String, dest: String): String = {
    exec(s"az storage fs file upload " +
      s" -s $file -p $dest -f $StorageContainer " +
      s" --overwrite true " +
      s" --account-name $StorageAccount --account-key ${Secrets.SynapseStorageKey}")
    s"abfss://$StorageContainer@$StorageAccount.dfs.core.windows.net/$dest"
  }

  def exec(command: String): String = {
    val os = sys.props("os.name").toLowerCase
    os match {
      case x if x contains "windows" => Seq("cmd", "/C") ++ Seq(command) !!
      case _ => command !!
    }
  }

  def getSynapseToken: String = {
    val spnKey: String = Secrets.SynapseSpnKey

    val uri: String = s"https://login.microsoftonline.com/$TenantId/oauth2/token"

    val createRequest = new HttpPost(uri)
    createRequest.setHeader("Content-Type", "application/x-www-form-urlencoded")

    val bodyList: util.List[BasicNameValuePair] = new util.ArrayList[BasicNameValuePair]()
    bodyList.add(new BasicNameValuePair("grant_type", "client_credentials"))
    bodyList.add(new BasicNameValuePair("client_id", s"$ClientId"))
    bodyList.add(new BasicNameValuePair("client_secret", s"$spnKey"))
    bodyList.add(new BasicNameValuePair("resource", "https://dev.azuresynapse.net/"))

    createRequest.setEntity(new UrlEncodedFormEntity(bodyList, "UTF-8"))

    val response = RESTHelpers.safeSend(createRequest, close = false)
    val inputStream: InputStream = response.getEntity.getContent
    val pageContent: String = Source.fromInputStream(inputStream).mkString
    pageContent.parseJson.asJsObject().fields("access_token").convertTo[String]
  }

  def cancelRun(livyUrl: String, batchId: Int): Unit = {
    val createRequest = new HttpDelete(s"$livyUrl/$batchId")
    createRequest.setHeader("Authorization", s"Bearer $Token")
    val response = RESTHelpers.safeSend(createRequest, close = false)
    println(response.getEntity.getContent)
  }

  private def submitRun(livyUrl: String, path: String): LivyBatch = {
    // MMLSpark info
    val truncatedScalaVersion: String =
      BuildInfo.scalaVersion
        .split(".".toCharArray.head)
        .dropRight(1)
        .mkString(".")
    val deploymentBuild = s"com.microsoft.ml.spark:${BuildInfo.name}_$truncatedScalaVersion:${BuildInfo.version}"
    val repository = "https://mmlspark.azureedge.net/maven"
    val sparkPackages: Array[String] = Array(deploymentBuild)
    val jobName = path.split('/').last.replace(".py", "")
    val excludes: String =
      "org.scala-lang:scala-reflect," +
        "org.apache.spark:spark-tags_2.12," +
        "org.scalactic:scalactic_2.12," +
        "org.scalatest:scalatest_2.12"
    val packageVersion: String = sparkPackages.map(s => s.trim).mkString(",")
    val livyPayload: String =
      s"""
         |{
         | "file" : "$path",
         | "name" : "$jobName",
         | "driverMemory" : "28g",
         | "driverCores" : 4,
         | "executorMemory" : "28g",
         | "executorCores" : 4,
         | "numExecutors" : 2,
         | "conf" :
         |     {
         |         "spark.jars.packages" : "$packageVersion",
         |         "spark.jars.repositories" : "$repository",
         |         "spark.jars.excludes": "$excludes",
         |         "spark.driver.userClassPathFirst": "true",
         |         "spark.executor.userClassPathFirst": "true"
         |     }
         | }
      """.stripMargin

    val createRequest = new HttpPost(livyUrl)
    createRequest.setHeader("Content-Type", "application/json")
    createRequest.setHeader("Authorization", s"Bearer $Token")
    createRequest.setEntity(new StringEntity(livyPayload))
    val response = RESTHelpers.safeSend(createRequest, close=false)
    val content: String = IOUtils.toString(response.getEntity.getContent, "utf-8")
    val batch: LivyBatch = parse(content).extract[LivyBatch]
    val status: Int = response.getStatusLine.getStatusCode
    assert(status == 200)
    batch
  }
}
