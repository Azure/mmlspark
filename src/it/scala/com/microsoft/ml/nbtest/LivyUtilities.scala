package com.microsoft.ml.nbtest

import java.io.{File, FileInputStream}
import java.util.concurrent.TimeUnit

import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.build.BuildInfo
import org.apache.http.client.methods.{HttpDelete, HttpGet, HttpPost}
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write
import com.microsoft.ml.spark.cognitive.RESTHelpers
import com.microsoft.ml.spark.core.env.FileUtilities
import org.apache.commons.io.IOUtils
import org.apache.http.entity.StringEntity
import org.json4s.{Formats, NoTypeHints}
import org.json4s.JsonAST.JObject
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.spark_project.guava.io.BaseEncoding
import spray.json._
import DefaultJsonProtocol._
import sys.process._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future, TimeoutException, blocking}

case class LivyBatch(id: Int,
                     state: String,
                     appId: Option[String],
                     appInfo: Option[JObject],
                     log: Seq[String])

case class LivyLogs(id: Int,
                    from: Int,
                    total: Int,
                    log: Seq[String])
//noinspection ScalaStyle
object LivyUtilities {

  implicit val formats: Formats = Serialization.formats(NoTypeHints)

  val NotebookFiles: Array[String] = Option(
    FileUtilities.join(BuildInfo.baseDirectory, "notebooks", "samples").getCanonicalFile.listFiles().map(file => file.getAbsolutePath)
  ).get

  lazy val Token: String = getSynapseToken()

  val Folder = s"build_${BuildInfo.version}/scripts"
  val TimeoutInMillis: Int = 20 * 60 * 1000

  def poll (id: Int, livyUrl: String, backoffs: List[Int] = List(100, 1000, 5000)): LivyBatch = {
    val getStatsRequest = new HttpGet(s"$livyUrl/batches/$id")
    val statsResponse = RESTHelpers.safeSend(getStatsRequest, backoffs = backoffs, close=false)
    val batch = parse(IOUtils.toString(statsResponse.getEntity.getContent, "utf-8")).extract[LivyBatch]
    statsResponse.close()
    batch
  }

  def getLogs(id: Int, livyUrl: String, backoffs: List[Int] = List(100, 500, 1000)): Seq[String] = {
    val getLogsRequest = new HttpGet(s"$livyUrl/batches/$id/log?from=0&size=10000")
    val statsResponse = RESTHelpers.safeSend(getLogsRequest, backoffs=backoffs, close=false)
    val batchString = parse(IOUtils.toString(statsResponse.getEntity.getContent, "utf-8"))
    val batch = batchString.extract[LivyLogs]
    statsResponse.close()
    batch.log
  }

  def postMortem(batch: LivyBatch, livyUrl: String) : LivyBatch = {
    getLogs(batch.id, livyUrl).foreach(println)
    write(batch)
    batch
  }

  def retry(id: Int, livyUrl: String, timeout: Int, startTime: Long): LivyBatch = {
      val batch = poll(id, livyUrl)
      println(s"batch state $id : ${batch.state}")
      if (batch.state == "success") {
        batch
      }else{
        if((System.currentTimeMillis() - startTime) < timeout){
          throw new TimeoutException(s"Job $id timed out.")
        }
        else if (batch.state == "dead"){
          postMortem(batch, livyUrl)
          throw new RuntimeException(s"Dead")
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

  def uploadAndSubmitNotebook(livyUrl: String, notebookPath: String ): LivyBatch = {
    val convertedPyScript = convertNotebook(notebookPath)
    val abfssPath = uploadScript(convertedPyScript.getAbsolutePath, s"$Folder/${convertedPyScript.getName}")
    submitRun(livyUrl, abfssPath)
  }

  def cancelRun(livyUrl: String, batchId: Int) : Unit = {
    val createRequest = new HttpDelete(s"$livyUrl/batches/$batchId")
    val response = RESTHelpers.safeSend(createRequest, close = false)
    println(response.getEntity.getContent)
  }

  private def submitRun(livyUrl: String, path: String): LivyBatch =
  {
    // MMLSpark info
    val truncatedScalaVersion: String = BuildInfo.scalaVersion
      .split(".".toCharArray.head).dropRight(1).mkString(".")
    val deploymentBuild = s"com.microsoft.ml.spark:${BuildInfo.name}_$truncatedScalaVersion:${BuildInfo.version}"
    val repository = "https://mmlspark.azureedge.net/maven"

    val sparkPackages = Array(deploymentBuild)
    val livyPayload =
      s"""
         |{
         | "file" : "$path",
         | "conf" :
         |      {
         |        "spark.jars.packages" : "${sparkPackages.map(s => s.trim).mkString(",")}",
         |        "spark.jars.repositories" : s"$repository"
         |      }
         | }
      """.stripMargin

    val createRequest = new HttpPost(s"$livyUrl/batches")
    println(s"livy payload: $livyPayload")
    createRequest.setHeader("Content-Type", "application/json")
    createRequest.setHeader("Authorization", s"Bearer $Token")
    createRequest.setEntity(new StringEntity(livyPayload))
    val response = RESTHelpers.safeSend(createRequest, close = false)

    println(response.getEntity.getContent)
    val content = IOUtils.toString(response.getEntity.getContent, "utf-8")
    val batch = parse(content).extract[LivyBatch]
    val status = response.getStatusLine.getStatusCode
    assert(status == 201)
    batch
  }

  private def uploadScript(file: String, dest: String): String = {
    val fileName = new File(file).getName
    exec(s"az storage fs file upload -s $file -p $dest -f synapse --acount-name mmlsparkdemo3 --account-key ${Secrets.SynapseStorageKey}")
    s"abfss://synapse@mmlsparkdemo3.dfs.core.windows.net/$dest"
  }

  private def convertNotebook(notebookPath: String): File = {
    val os = sys.props("os.name").toLowerCase
    os match {
      case x if x contains "windows" => exec(s"conda activate mmlspark && jupyter nbconvert --to script $notebookPath")
      case _ => exec(s"conda init bash && conda activate mmlspark && jupyter nbconvert --to script $notebookPath")
    }

    new File(notebookPath.replace(".ipynb", ".py"))
  }

  private def getSynapseToken(): String = {
    val secretJson = exec(s"az account get-access-token --resource https://dev.azuresynapse.net")
    secretJson.parseJson.asJsObject().fields("accessToken").convertTo[String]
  }

  private def exec(command: String): String = {
    val os = sys.props("os.name").toLowerCase
    os match {
      case x if x contains "windows" => Seq("cmd", "/C") ++ Seq(command) !!
      case _ => command !!
    }
  }
}
