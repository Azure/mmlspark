// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import java.net.{InetAddress, InetSocketAddress}
import java.util.UUID
import java.util.concurrent.Executors
import javax.annotation.concurrent.GuardedBy

import com.microsoft.ml.spark.core.env.StreamUtilities.using
import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import org.apache.commons.io.IOUtils
import org.apache.spark.internal.Logging
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.execution.streaming._
import org.apache.spark.sql.functions.{col, struct, to_json}
import org.apache.spark.sql.sources.{DataSourceRegister, StreamSinkProvider, StreamSourceProvider}
import org.apache.spark.sql.streaming.OutputMode
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer
import scala.collection.{immutable, mutable}

object DistributedHTTPSource {

  val SCHEMA: StructType = {
    StructType(
      StructField("id", StringType) ::
        StructField("value", StringType) ::
        Nil)
  }

}

class MultiChannelMap[K, V](var nLists: Int) {

  @GuardedBy("this")
  private val map: mutable.HashMap[K, V] = new mutable.HashMap()

  @GuardedBy("this")
  private val lists: mutable.ListBuffer[mutable.ListBuffer[(K, V)]] =
    new mutable.ListBuffer() ++ (1 to nLists).map(_ => new mutable.ListBuffer[(K, V)]())

  @GuardedBy("this")
  private var getIndex = 0

  def updateNLists(n: Int): Unit = synchronized {
    getIndex = 0
    addIndex = 0
    val oldNLists = nLists
    nLists = n
    if (n > oldNLists) {
      lists.append(new mutable.ListBuffer[(K, V)]())
    } else if (n < nLists) {
      val listsToDisperse = lists.slice(0, oldNLists - n)
      lists.remove(0, oldNLists - n)
      listsToDisperse.flatten.foreach(p => addToNextList(p._1, p._2))
    }
  }

  private def getAndIncrementGetIndex(): Int = synchronized {
    val oldIndex = getIndex
    getIndex = (getIndex + 1) % nLists
    oldIndex
  }

  def get(k: K): V = synchronized(map(k))

  private def getAll(i: Int): mutable.ListBuffer[(K, V)] =
    synchronized(lists(i))

  def nextList(): ListBuffer[(K, V)] = synchronized {
    getAll(getAndIncrementGetIndex())
  }

  private def add(i: Int, k: K, v: V): Unit = synchronized {
    map.update(k: K, v: V)
    lists(i).append((k, v))
  }

  @GuardedBy("this")
  private var addIndex = 0

  private def getAndIncrementAddIndex(): Int = synchronized {
    val oldIndex = addIndex
    addIndex = (addIndex + 1) % nLists
    oldIndex
  }

  def addToNextList(k: K, v: V): Unit = synchronized {
    add(getAndIncrementAddIndex(), k, v)
  }

  def size: Int = synchronized(map.size)

}

class JVMSharedServer(name: String, host: String,
                      port: Int, maxAttempts: Int,
                      handleResponseErrors: Boolean) extends Logging {

  type Body = String
  type Exchange = HttpExchange
  type Request = (Body, Exchange)
  type ID = String

  @GuardedBy("this")
  private var requestsSeen = 0L
  @GuardedBy("this")
  private var requestsAccepted = 0L
  @GuardedBy("this")
  private var requestsAnswered = 0L
  @GuardedBy("this")
  private var currentBatch = 0L
  @GuardedBy("this")
  private var earliestBatch = 0L
  @GuardedBy("this")
  private var nPartitions = 0

  val serverIdentity: String = UUID.randomUUID().toString

  @GuardedBy("this")
  private val batchesToRequests: mutable.HashMap[Long, MultiChannelMap[ID, Request]] =
    new mutable.HashMap[Long, MultiChannelMap[ID, Request]]()

  def updateNPartitions(n: Int): Unit = synchronized {
    nPartitions = n
    batchesToRequests.valuesIterator.foreach(_.updateNLists(n))
  }

  def updateCurrentBatch(currentOffset: Long): Unit = synchronized {
    currentBatch = currentOffset
  }

  def incrementCurrentBatch(): Unit = synchronized {
    currentBatch = currentBatch + 1
  }

  def getRequests(start: Long, end: Long): immutable.IndexedSeq[(ID, Body)] = synchronized {
    // Increment the current batch so new requests are routed to the next batch
    if (end == currentBatch) incrementCurrentBatch()
    val requests = (start to end).flatMap { i =>
        val mcm = batchesToRequests.getOrElse(i, new MultiChannelMap[ID, Request](nPartitions))
        mcm.nextList().map(p => (p._1, p._2._1))
      }
    logDebug(s"getting ${requests.length} on $address batch $currentBatch")
    requests
  }

  def trimBatchesBefore(batch: Long): Unit = synchronized {
    val start = earliestBatch
    earliestBatch = batch
      (start to batch).foreach(batchesToRequests.remove)
    logDebug(s"trimming ${(start to batch).toList} on $address")
  }

  private def responseHelper(request: HttpExchange, code: Int, response: String): Unit = synchronized {
    val bytes = response.getBytes("UTF-8")
    request.synchronized {
      request.getResponseHeaders.add("Content-Type", "application/json")
      request.getResponseHeaders.add("Access-Control-Allow-Origin", "*")
      request.getResponseHeaders.add("Access-Control-Allow-Method", "GET, PUT, POST, DELETE, HEAD")
      request.getResponseHeaders.add("Access-Control-Allow-Headers", "*")
      request.sendResponseHeaders(code, 0)
      using(request.getResponseBody){os =>
        os.write(bytes)
        os.flush()
      }
      request.close()
    }
    requestsAnswered += 1
  }

  private def respond(request: HttpExchange, code: Int, response: String): Unit = synchronized {
    logDebug(s"responding $response batch: $currentBatch ip: $address")
    if (handleResponseErrors) {
      try {
        responseHelper(request, code, response)
      } catch {
        case e: Exception =>
          logError(s"ERROR on server $address: " + e.getMessage)
          logError("TRACEBACK: " + e.getStackTrace.mkString("\n"))
      }
    } else {
      responseHelper(request, code, response)
    }
    logDebug(s"responding $response batch: $currentBatch ip: $address")
    ()
  }

  def respond(batch: Long, uuid: String, code: Int, response: String): Unit = synchronized {
    val request = batchesToRequests(batch).get(uuid)._2
    respond(request, code, response)
  }

  private class RequestHandler extends HttpHandler {

    private def getBody(request: HttpExchange) = synchronized {
      using(request.getRequestBody) {
        IOUtils.toString(_, "UTF-8")
      }.get
    }

    override def handle(request: HttpExchange): Unit = synchronized {
      requestsSeen += 1
      if (request.getRequestMethod == "OPTIONS") {
        respond(request, 200, "")
      } else if (request.getRequestMethod != "POST") {
        respond(request, 405, "Only POSTs accepted")
      } else {
        val headers = request.getRequestHeaders
        if (headers.containsKey("Content-type")) {
          headers.get("Content-type").get(0) match {
            case "application/json" =>
              val body = getBody(request)
              val uuid = UUID.randomUUID().toString
              requestsAccepted += 1
              val cb = currentBatch.longValue()
              batchesToRequests.get(cb) match {
                case None =>
                  val mcm = new MultiChannelMap[ID, Request](nPartitions)
                  mcm.addToNextList(uuid, (body, request))
                  batchesToRequests.update(cb, mcm)
                case Some(mcm) => mcm.addToNextList(uuid, (body, request))
              }
              logDebug(s"handling $body batch: $currentBatch ip: $address")
              ()
            case _ =>
              respond(request, 400, "Content-type needs to be application/json")
          }
        }
      }
    }
  }

  private def tryCreateServer(host: String, startingPort: Int, triesLeft: Int): (HttpServer, Int) = {
    if (triesLeft == 0) {
      throw new java.net.BindException("Could not find open ports in the range," +
        " try increasing the number of ports to try")
    }
    try {
      val server = HttpServer.create(new InetSocketAddress(InetAddress.getByName(host), startingPort), 100)
      (server, startingPort)
    } catch {
      case _: java.net.BindException =>
        tryCreateServer(host, startingPort + 1, triesLeft - 1)
    }
  }

  @GuardedBy("this")
  private val (server, serverPort) = tryCreateServer(host, port, maxAttempts)
  server.createContext(s"/$name", new RequestHandler)
  server.setExecutor(Executors.newFixedThreadPool(100))
  server.start()

  val address: String = server.getAddress.getHostString + ":" + serverPort
  val machine: String = InetAddress.getLocalHost.toString

  def stop(): Unit = synchronized {
    server.stop(0)
    server.synchronized {
      server.notifyAll()
    }
  }

}

class DistributedHTTPSource(name: String,
                            host: String,
                            port: Int,
                            maxPortAttempts: Int,
                            maxPartitions: Option[Int],
                            handleResponseErrors: Boolean,
                            sqlContext: SQLContext)
    extends Source with Logging with Serializable {

  import sqlContext.implicits._

  private[spark] val server = SharedSingleton {
    new JVMSharedServer(name, host, port, maxPortAttempts, handleResponseErrors)
  }

  // Access point to run code on nodes through mapPartitions
  // TODO do this by hooking deeper into spark,
  // TODO allow for dynamic allocation
  private[spark] val serverInfoDF: DataFrame = {
    val enc = RowEncoder(new StructType()
      .add("machine", StringType)
      .add("ip", StringType)
      .add("id", StringType))
    val serverInfo = sqlContext.sparkContext
      .parallelize(Seq(Tuple1("placeholder")),
        maxPartitions.getOrElse(sqlContext.sparkContext.defaultParallelism))
      .toDF("placeholder")
      .mapPartitions { _ =>
        val s = server.get
        Iterator(Row(s.machine, s.address, s.serverIdentity))
      }(enc).cache()

    val serverToNumPartitions = serverInfo
      .select("id").groupBy("id").count().collect().map(r => (r.getString(0), r.getLong(1).toInt)).toMap
    val serverInfoConfigured = serverInfo.mapPartitions { it =>
      server.get.updateNPartitions(serverToNumPartitions(server.get.serverIdentity))
      it
    }(enc).cache()
    serverInfoConfigured.collect() // materialize to trigger setup

    logInfo("Got or Created services: "
      + serverInfoConfigured.collect().map {r =>
        s"machine: ${r.getString(0)}, address: ${r.getString(1)}, guid: ${r.getString(2)}"
      }.toList.mkString(", "))
    serverInfoConfigured
  }

  @GuardedBy("this")
  protected var currentOffset: LongOffset = new LongOffset(-1)

  @GuardedBy("this")
  protected var lastOffsetCommitted: LongOffset = new LongOffset(-1)

  /** Returns the schema of the data from this source */
  override def schema: StructType = DistributedHTTPSource.SCHEMA

  // Note we assume that this function is only called once during the polling of a new batch
  override def getOffset: Option[Offset] = synchronized {
    currentOffset += 1
    Some(currentOffset)
  }

  /** Returns the data that is between the offsets (`start`, `end`]. */
  override def getBatch(start: Option[Offset], end: Offset): DataFrame = synchronized {
    val startOrdinal =
      start.flatMap(LongOffset.convert).getOrElse(LongOffset(-1)).offset
    val endOrdinal = LongOffset.convert(end).getOrElse(LongOffset(-1)).offset

    serverInfoDF.mapPartitions { _ =>
      val s = server.get
      s.updateCurrentBatch(currentOffset.offset)
      s.getRequests(startOrdinal, endOrdinal)
        .map(Row.fromTuple).toIterator
    }(RowEncoder(DistributedHTTPSource.SCHEMA))
  }

  override def commit(end: Offset): Unit = synchronized {
    val newOffset = LongOffset.convert(end).getOrElse(
      sys.error(s"DistributedHTTPSource.commit() received an offset ($end) that did not " +
        s"originate with an instance of this class")
    )
    if (newOffset.offset < lastOffsetCommitted.offset) {
      sys.error(s"Offsets committed out of order: $lastOffsetCommitted followed by $end")
    }
    serverInfoDF.foreachPartition(_ =>
      server.get.trimBatchesBefore(newOffset.offset))
    lastOffsetCommitted = newOffset
  }

  /** Stop this source. */
  override def stop(): Unit = synchronized {
    serverInfoDF.foreachPartition(_ =>
      server.get.stop())
    ()
  }

  override def toString: String = s"DistributedHTTPSource[name: $name, host: $host, port: $port]"

}

class DistributedHTTPSourceProvider extends StreamSourceProvider with DataSourceRegister with Logging {

  /** Returns the name and schema of the source that can be used to continually read data. */
  override def sourceSchema(sqlContext: SQLContext,
                            schema: Option[StructType],
                            providerName: String,
                            parameters: Map[String, String]): (String, StructType) = {
    if (!parameters.contains("host")) {
      throw new NamespaceInjections.InjectedAnalysisException("Set a host to read from with option(\"host\", ...).")
    }
    if (!parameters.contains("port")) {
      throw new NamespaceInjections.InjectedAnalysisException("Set a port to read from with option(\"port\", ...).")
    }
    if (!parameters.contains("name")) {
      throw new NamespaceInjections.InjectedAnalysisException("Set a name of the API which is used for routing")
    }
    ("DistributedHTTP", DistributedHTTPSource.SCHEMA)
  }

  override def createSource(sqlContext: SQLContext,
                            metadataPath: String,
                            schema: Option[StructType],
                            providerName: String,
                            parameters: Map[String, String]): Source = {
    val host = parameters("host")
    val port = parameters("port").toInt
    val name = parameters("name")
    val maxPartitions = parameters.get("maxPartitions").map(_.toInt)
    val maxAttempts = parameters.getOrElse("maxPortAttempts", "10").toInt
    val handleResponseErrors = parameters.getOrElse("handleResponseErrors", "false").toBoolean
    val source = new DistributedHTTPSource(
      name, host, port, maxAttempts, maxPartitions, handleResponseErrors, sqlContext)
    DistributedHTTPSink.activeSinks(parameters("name")).linkWithSource(source)

    parameters.get("deployLoadBalancer") match {
      case Some(s) if s.toBoolean =>
        //AzureLoadBalancer.deployFromParameters(parameters)
        throw new NotImplementedError("Support for automatic deployment coming soon")
      case _ =>
    }
    source
  }

   /** String that represents the format that this data source provider uses. */
  override def shortName(): String = "DistributedHTTP"

}

class DistributedHTTPSink(val options: Map[String, String])
    extends Sink with Logging with Serializable {

  if (!options.contains("name")) {
    throw new NamespaceInjections.InjectedAnalysisException("Set a name of an API to reply to")
  }
  val name = options("name")

  DistributedHTTPSink.activeSinks.update(name, this)

  private var source: DistributedHTTPSource = _

  def getServerAddresses: List[String] = {
    source.serverInfoDF.collect().toList.map(_.getString(0)).distinct
  }

  private[spark] def linkWithSource(s: DistributedHTTPSource): Unit = {
    source = s
  }

  private def server: SharedSingleton[JVMSharedServer] = source.server

  override def addBatch(batchId: Long, data: DataFrame): Unit = synchronized {
    val valueCol = options.getOrElse("replyCol", "value")
    data.select(
      col(options.getOrElse("idCol", "id")),
      to_json(struct(valueCol)).alias(valueCol)
    ).foreach { row =>
      server.get.respond(batchId, row.getAs[String](0), 200, row.getString(1))
    }
  }

}

class DistributedHTTPSinkProvider extends StreamSinkProvider with DataSourceRegister {

  def createSink(sqlContext: SQLContext,
                 parameters: Map[String, String],
                 partitionColumns: Seq[String],
                 outputMode: OutputMode): Sink = {
    new DistributedHTTPSink(parameters)
  }

  def shortName(): String = "DistributedHTTP"

}

object DistributedHTTPSink {

  val activeSinks: mutable.Map[String, DistributedHTTPSink] = mutable.Map()

}
