// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.split2

import java.io.File
import java.util.UUID
import java.util.concurrent.{Executors, TimeUnit, TimeoutException}

import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.test.base.{Flaky, TestBase}
import com.microsoft.ml.spark.io.IOImplicits._
import com.microsoft.ml.spark.io.http.HTTPSchema.string_to_response
import com.microsoft.ml.spark.io.http.SharedSingleton
import com.microsoft.ml.spark.io.split1.WithFreeUrl
import org.apache.commons.io.IOUtils
import org.apache.http.client.config.RequestConfig
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.{FileEntity, StringEntity}
import org.apache.http.impl.client.{BasicResponseHandler, CloseableHttpClient, HttpClientBuilder}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.execution.streaming.DistributedHTTPSourceProvider
import org.apache.spark.sql.functions.{col, length}
import org.apache.spark.sql.streaming.{DataStreamReader, DataStreamWriter, StreamingQuery}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

import scala.concurrent.duration.{Duration, FiniteDuration}
import scala.concurrent.{Await, ExecutionContext, ExecutionContextExecutor, Future}
import scala.util.parsing.json.JSONObject

trait HasHttpClient {
  lazy val requestTimeout = 600000

  lazy val requestConfig: RequestConfig = RequestConfig.custom()
    .setConnectTimeout(requestTimeout)
    .setConnectionRequestTimeout(requestTimeout)
    .setSocketTimeout(requestTimeout)
    .build()

  lazy val client: CloseableHttpClient = HttpClientBuilder
    .create().setDefaultRequestConfig(requestConfig).build()
}

trait HTTPTestUtils extends TestBase with WithFreeUrl with HasHttpClient {

  def waitForServer(server: StreamingQuery, maxTimeWaited: Int = 20000, checkEvery: Int = 100): Unit = {
    var waited = 0
    while (waited < maxTimeWaited) {
      if (!server.isActive) throw server.exception.get
      if (server.recentProgress.length > 1) return
      Thread.sleep(checkEvery.toLong)
      waited += checkEvery
    }
    throw new TimeoutException(s"Server Did not start within $maxTimeWaited ms")
  }

  def sendStringRequest(client: CloseableHttpClient,
                        url: String = url,
                        payload: String = "foo",
                        targetCode: Int = 200): (String, Double) = {
    val post = new HttpPost(url)
    val e = new StringEntity(payload)
    post.setEntity(e)
    //println("request sent")
    val t0 = System.nanoTime()
    val res = client.execute(post)
    val t1 = System.nanoTime()

    assert(targetCode === res.getStatusLine.getStatusCode)
    val out = if (targetCode == res.getStatusLine.getStatusCode && !targetCode.toString.startsWith("2")) {
      null
    } else {
      new BasicResponseHandler().handleResponse(res)
    }
    res.close()
    //println("request suceeded")
    (out, (t1 - t0).toDouble / 1e6)
  }

  def using(c: StreamingQuery)(t: => Unit): Unit = {
    try {
      t
    } finally {
      c.stop()
      Thread.sleep(1000)
    }
  }

  def sendStringRequestAsync(client: CloseableHttpClient, url: String = url): Future[(String, Double)] = {
    Future {
      sendStringRequest(client, url = url)
    }(ExecutionContext.global)
  }

  def sendJsonRequest(client: CloseableHttpClient, map: Map[String, Any], url: String): String = {
    val post = new HttpPost(url)
    val params = new StringEntity(JSONObject(map).toString())
    post.addHeader("content-type", "application/json")
    post.setEntity(params)
    val res = client.execute(post)
    val out = new BasicResponseHandler().handleResponse(res)
    res.close()
    out
  }

  def sendJsonRequest(client: CloseableHttpClient, payload: Int, url: String): (String, Long) = {
    val post = new HttpPost(url)
    val e = new StringEntity("{\"data\":" + s"$payload}")
    post.setEntity(e)
    val t0 = System.currentTimeMillis()
    val res = client.execute(post)
    val t1 = System.currentTimeMillis()
    val out = new BasicResponseHandler().handleResponse(res)
    res.close()
    (out, t1 - t0)
  }

  def sendJsonRequestAsync(client: CloseableHttpClient, map: Map[String, Any], url: String = url): Future[String] = {
    Future {
      sendJsonRequest(client, map, url = url)
    }(ExecutionContext.global)
  }

  def sendFileRequest(client: CloseableHttpClient, url: String = url): (String, Double) = {
    val post = new HttpPost(url)
    val e = new FileEntity(FileUtilities.join(
      BuildInfo.datasetDir, "Images", "Grocery", "testImages", "WIN_20160803_11_28_42_Pro.jpg"))
    post.setEntity(e)
    val t0 = System.nanoTime()
    val res = client.execute(post)
    val out = new BasicResponseHandler().handleResponse(res)
    res.close()
    val t1 = System.nanoTime()
    (out, (t1 - t0).toDouble / 1e6)
  }

  def mean(xs: List[Int]): Double = xs match {
    case Nil => 0.0
    case ys => ys.sum / ys.size.toDouble
  }

  def median(xs: List[Int]): Double = xs match {
    case Nil => 0.0
    case ys => ys.sorted.apply(ys.length / 2)
  }

  def stddev(xs: List[Int], avg: Double): Double = xs match {
    case Nil => 0.0
    case ys => math.sqrt(ys.foldLeft(0.0) {
      (a, e) => a + math.pow(e - avg, 2.0)
    } / xs.size)
  }

  lazy val requestDuration: FiniteDuration = Duration(10, TimeUnit.SECONDS)

  lazy implicit val ec: ExecutionContextExecutor = ExecutionContext.fromExecutor(Executors.newCachedThreadPool())

  def assertLatency(responsesWithLatencies: Seq[(String, Double)], cutoff: Double): Unit = {
    val latencies = responsesWithLatencies.drop(3).map(_._2.toInt).toList
    //responsesWithLatencies.foreach(r => println(r._1))
    val medianLatency = median(latencies)
    val meanLatency = mean(latencies)
    val stdLatency = stddev(latencies, meanLatency)
    println(s"Median Latency = $medianLatency")
    println(s"Latency = $meanLatency +/- $stdLatency")
    assert(medianLatency < cutoff)
    ()
  }

}

// TODO add tests for shuffles
class DistributedHTTPSuite extends TestBase with Flaky with HTTPTestUtils {
  // Logger.getRootLogger.setLevel(Level.WARN)
  // Logger.getLogger(classOf[DistributedHTTPSource]).setLevel(Level.INFO)
  // Logger.getLogger(classOf[JVMSharedServer]).setLevel(Level.INFO)

  def baseReaderDist: DataStreamReader = {
    spark.readStream.distributedServer
      .address(host, port, "foo")
      .option("maxPartitions", 3)
  }

  def baseReader: DataStreamReader = {
    spark.readStream.server
      .address(host, port, "foo")
      .option("maxPartitions", 3)
  }

  def baseWriterDist(df: DataFrame): DataStreamWriter[Row] = {
    df.writeStream
      .distributedServer
      .option("name", "foo")
      .queryName("foo")
      .option("checkpointLocation",
        new File(tmpDir.toFile, s"checkpoints-${UUID.randomUUID()}").toString)
  }

  def baseWriter(df: DataFrame): DataStreamWriter[Row] = {
    df.writeStream
      .server
      .option("name", "foo")
      .queryName("foo")
      .option("checkpointLocation",
        new File(tmpDir.toFile, s"checkpoints-${UUID.randomUUID()}").toString)
  }

  def createServer(): DataStreamWriter[Row] = {
    println(classOf[DistributedHTTPSourceProvider].getName)
    baseWriterDist(baseReaderDist
      .load()
      .withColumn("contentLength", col("request.entity.contentLength"))
      .withColumn("reply", string_to_response(col("contentLength").cast(StringType))))
  }

  test("standard client") {
    val server = createServer().start()
    using(server) {
      waitForServer(server)
      val responses = List(
        sendJsonRequest(client, Map("foo" -> 1, "bar" -> "here"), url),
        sendJsonRequest(client, Map("foo" -> 2, "bar" -> "heree"), url),
        sendJsonRequest(client, Map("foo" -> 3, "bar" -> "hereee"), url),
        sendJsonRequest(client, Map("foo" -> 4, "bar" -> "hereeee"), url)
      )
      val correctResponses = List(27, 28, 29, 30).map(_.toString)

      assert(responses === correctResponses)

      (1 to 20).map(i => sendJsonRequest(client, Map("foo" -> 1, "bar" -> "here"), url))
        .foreach(resp => assert(resp === "27"))
    }
  }

  test("test implicits") {
    val server = baseWriter(baseReader
      .load()
      .withColumn("contentLength", col("request.entity.contentLength"))
      .withColumn("reply", string_to_response(col("contentLength").cast(StringType))))
      .start()

    using(server) {
      waitForServer(server)
      val responses = List(
        sendJsonRequest(client, Map("foo" -> 1, "bar" -> "here"), url),
        sendJsonRequest(client, Map("foo" -> 2, "bar" -> "heree"), url),
        sendJsonRequest(client, Map("foo" -> 3, "bar" -> "hereee"), url),
        sendJsonRequest(client, Map("foo" -> 4, "bar" -> "hereeee"), url)
      )
      val correctResponses = List(27, 28, 29, 30).map(_.toString)

      assert(responses === correctResponses)

      (1 to 20).map(i => sendJsonRequest(client, Map("foo" -> 1, "bar" -> "here"), url))
        .foreach(resp => assert(resp === "27"))
    }

  }

  test("test implicits 2") {

    val server = baseWriter(baseReader
      .load()
      .parseRequest(apiName, BinaryType)
      .withColumn("length", length(col("bytes")))
      .makeReply("length"))
      .start()

    using(server) {
      waitForServer(server)
      val responsesWithLatencies = (1 to 10).map(i => sendFileRequest(client))

      val latencies = responsesWithLatencies.drop(3).map(_._2.toInt).toList
      val meanLatency = mean(latencies)
      val stdLatency = stddev(latencies, meanLatency)
      println(s"Latency = $meanLatency +/- $stdLatency")

      responsesWithLatencies.foreach(s => assert(s._1 === "{\"length\":279186}"))
    }
  }

  test("test implicits 2 distributed") {

    val server = baseWriterDist(baseReaderDist
      .load()
      .parseRequest(apiName, BinaryType)
      .withColumn("length", length(col("bytes")))
      .makeReply("length"))
      .start()

    using(server) {
      waitForServer(server)
      val responsesWithLatencies = (1 to 10).map(i =>
        sendFileRequest(client)
      )

      val latencies = responsesWithLatencies.drop(3).map(_._2.toInt).toList
      val meanLatency = mean(latencies)
      val stdLatency = stddev(latencies, meanLatency)
      println(s"Latency = $meanLatency +/- $stdLatency")

      responsesWithLatencies.foreach(s => assert(s._1 === "{\"length\":279186}"))
    }

  }

  test("python client") {
    val server = createServer().start()

    using(server) {
      waitForServer(server)

      val pythonClientCode =
        s"""import requests
           |import threading
           |import time
           |
         |exitFlag = 0
           |s = requests.Session()
           |
         |class myThread(threading.Thread):
           |    def __init__(self, threadID):
           |        threading.Thread.__init__(self)
           |        self.threadID = threadID
           |
         |    def run(self):
           |        print("Starting " + str(self.threadID))
           |        r = s.post("$url",
           |                          data={"number": 12524, "type": "issue", "action": "show"},
           |                          headers = {"content-type": "application/json"},
           |                          timeout=15)
           |
         |        assert r.status_code==200
           |        print("Exiting {} with code {}".format(self.threadID, r.status_code))
           |
         |
         |threads = []
           |for i in range(4):
           |    # Create new threads
           |    t = myThread(i)
           |    t.start()
           |    threads.append(t)
           |
         |""".stripMargin

      val pythonFile = new File(tmpDir.toFile, "pythonClient.py")
      FileUtilities.writeFile(pythonFile, pythonClientCode)

      val processes = (1 to 50).map(_ =>
        Runtime.getRuntime.exec(s"python ${pythonFile.getAbsolutePath}")
      )

      processes.foreach { p =>
        p.waitFor
        val error = IOUtils.toString(p.getErrorStream)
        assert(error === "")
      }
    }

  }

  test("async client") {
    val server = createServer().start()

    using(server) {
      waitForServer(server)

      val futures = List(
        sendJsonRequestAsync(client, Map("foo" -> 1, "bar" -> "here")),
        sendJsonRequestAsync(client, Map("foo" -> 2, "bar" -> "heree")),
        sendJsonRequestAsync(client, Map("foo" -> 3, "bar" -> "hereee")),
        sendJsonRequestAsync(client, Map("foo" -> 4, "bar" -> "hereeee"))
      )

      val responses = futures.map(f => Await.result(f, Duration.Inf))
      val correctResponses = List(27, 28, 29, 30).map(_.toString)

      assert(responses === correctResponses)

      (1 to 20).map(i => sendJsonRequestAsync(client, Map("foo" -> 1, "bar" -> "here")))
        .foreach { f =>
          val resp = Await.result(f, Duration(10, TimeUnit.SECONDS))
          assert(resp === "27")
        }
    }
  }

  test("State can be saved in a Shared singleton") {
    object Holder extends Serializable {

      class FooHolder {
        var state = 0

        def increment(): Unit = synchronized {
          state += 1
        }
      }

      val Foo: SharedSingleton[Holder.FooHolder] = SharedSingleton {
        new FooHolder
      }

      import spark.sqlContext.implicits._

      val DF: DataFrame = spark.sparkContext.parallelize(Seq(Tuple1("placeholder")))
        .toDF("plcaholder")
        .mapPartitions { _ =>
          Foo.get.increment()
          Iterator(Row(Foo.get.state))
        }(RowEncoder(new StructType().add("state", IntegerType))).cache()
      val States1: Array[Row] = DF.collect()

      val DF2: DataFrame = DF.mapPartitions { _ =>
        Iterator(Row(Foo.get.state))
      }(RowEncoder(new StructType().add("state", IntegerType)))
      val States2: Array[Row] = DF2.collect()
      assert(States2.forall(_.getInt(0) === States2.length))
    }
  }

}
