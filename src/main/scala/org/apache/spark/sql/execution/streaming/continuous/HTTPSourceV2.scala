// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.sql.execution.streaming.continuous

import java.io.{BufferedReader, InputStreamReader}
import java.net.{InetAddress, InetSocketAddress, ServerSocket, URL}
import java.util
import java.util.concurrent.{Executors, LinkedBlockingQueue, TimeUnit}
import java.util.{Optional, UUID}

import com.jcraft.jsch.Session
import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.io.http._
import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import javax.annotation.concurrent.GuardedBy
import org.apache.commons.io.IOUtils
import org.apache.http.client.config.RequestConfig
import org.apache.http.client.methods.HttpPost
import org.apache.http.conn.util.InetAddressUtils
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClientBuilder
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability}
import org.apache.spark.sql.connector.read.streaming._
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.execution.streaming.HTTPServerUtils
import org.apache.spark.sql.internal.connector.SimpleTableProvider
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.{SparkContext, TaskContext}
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.parallel.mutable.{ParHashMap, ParHashSet}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.Try

private[streaming] case class ServiceInfo(name: String,
                                          host: String,
                                          port: Int,
                                          path: String,
                                          localIp: String,
                                          publicIp: Option[String])


class HTTPSourceTable(options: CaseInsensitiveStringMap)
  extends Table with SupportsRead with Logging {

  override def name(): String = {
    s"HTTPStream($options)"
  }

  override def schema(): StructType = HTTPSourceV2.Schema

  override def capabilities(): util.Set[TableCapability] = {
    Set(TableCapability.MICRO_BATCH_READ, TableCapability.CONTINUOUS_READ).asJava
  }

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder = () => new Scan {
    override def readSchema(): StructType = HTTPSourceV2.Schema

    override def toMicroBatchStream(checkpointLocation: String): MicroBatchStream = {
      logInfo("Creating Microbatch reader")
      new HTTPMicroBatchReader(continuous = false, options = options)
    }


    override def toContinuousStream(checkpointLocation: String): ContinuousStream =
      new HTTPContinuousReader(options = options)
  }
}

class HTTPSourceProviderV2 extends SimpleTableProvider with DataSourceRegister {

  override def getTable(options: CaseInsensitiveStringMap): Table = {
    new HTTPSourceTable(options: CaseInsensitiveStringMap)
  }

  override def shortName(): String = "HTTPv2"
}


object HTTPSourceProviderV2 {
  val VERSION = 2
}

private[streaming] object HTTPOffset {
  def getStartingOffset(numPartitions: Int): HTTPOffset = {
    HTTPOffset((0 until numPartitions).map(x => (x, 0L)).toMap)
  }

  def increment(offset: HTTPOffset): HTTPOffset = {
    HTTPOffset(offset.partitionToValue.mapValues(v => v + 1))
  }

}

private[streaming] case class HTTPOffset(partitionToValue: Map[Int, Long])
  extends Offset {
  implicit val defaultFormats: DefaultFormats = DefaultFormats
  override val json: String = Serialization.write(partitionToValue)
}

private[streaming] case class HTTPPartitionOffset(partition: Int, epoch: Long) extends PartitionOffset

object HTTPSourceV2 {
  val NumPartitions = "numPartitions"
  val Host = "host"
  val Port = "port"
  val Path = "path"
  val NAME = "name"
  val EpochLength = "epochLength" // in millis

  val IdSchema: StructType = new StructType()
    .add("originatingService", StringType)
    .add("requestId", StringType)
    .add("partitionId", IntegerType)

  val Schema: StructType = {
    new StructType().add("id", IdSchema).add("request", HTTPSchema.Request)
  }

}

private[streaming] object DriverServiceUtils {

  private def createServiceOnFreePort(path: String,
                                      host: String,
                                      handler: HttpHandler): HttpServer = {
    val port: Int = StreamUtilities.using(new ServerSocket(0))(_.getLocalPort).get
    val server = HttpServer.create(new InetSocketAddress(host, port), 100)
    server.setExecutor(Executors.newFixedThreadPool(100))
    server.createContext(s"/$path", handler)
    server.start()
    server
  }

  class DriverServiceHandler(name: String) extends HttpHandler {

    implicit val defaultFormats: DefaultFormats = DefaultFormats

    //scalastyle:off magic.number
    //scalastyle:off null
    override def handle(request: HttpExchange): Unit = {
      try {
        val info = Serialization.read[ServiceInfo](
          IOUtils.toString(request.getRequestBody))
        HTTPServerUtils.respond(request, HTTPResponseData(
          Array(), None,
          StatusLineData(null, 200, "Success"),
          "en")
        )
        HTTPSourceStateHolder.addServiceInfo(name, info)
      } finally {
        HTTPServerUtils.respond(request, HTTPResponseData(
          Array(), None,
          StatusLineData(null, 400, "Could not parse request to service info"),
          "en")
        )
      }
    }

    //scalastyle:on magic.number
    //scalastyle:on null
  }

  private def getHostToIP(hostname: String): String = {
    if (InetAddressUtils.isIPv4Address(hostname) || InetAddressUtils.isIPv6Address(hostname))
      hostname
    else
      InetAddress.getByName(hostname).getHostAddress
  }

  def getDriverHost: String = {
    val blockManager = SparkContext.getActive.get.env.blockManager
    blockManager.master.getMemoryStatus.toList.flatMap({ case (blockManagerId, _) =>
      if (blockManagerId.executorId == "driver") Some(getHostToIP(blockManagerId.host))
      else None
    }).head
  }

  def createDriverService(name: String): HttpServer = {
    createServiceOnFreePort(
      "driverService", "0.0.0.0", new DriverServiceHandler(name))
  }
}

private[streaming] case class WorkerServiceConfig(host: String,
                                                  port: Int,
                                                  path: String,
                                                  forwardingOptions: collection.Map[String, String],
                                                  driverServiceHost: String,
                                                  driverServicePort: Int,
                                                  epochLength: Long
                                                 )

private[streaming] class HTTPMicroBatchReader(continuous: Boolean, options: CaseInsensitiveStringMap)
  extends MicroBatchStream with Logging {
  implicit val defaultFormats: DefaultFormats = DefaultFormats

  val numPartitions: Int = options.getInt(HTTPSourceV2.NumPartitions, 2)
  val host: String = options.get(HTTPSourceV2.Host, "localhost")
  val port: Int = options.getInt(HTTPSourceV2.Port, 8888)
  val path: String = options.get(HTTPSourceV2.Path)
  val name: String = options.get(HTTPSourceV2.NAME)
  val epochLength: Long = options.getLong(HTTPSourceV2.EpochLength, 30000)

  val forwardingOptions: collection.Map[String, String] = options.asCaseSensitiveMap().asScala
    .filter { case (k, _) => k.startsWith("forwarding") }

  HTTPSourceStateHolder.initServiceInfo(name, path)

  private lazy val driverService: HttpServer =
    DriverServiceUtils.createDriverService(name)

  override def deserializeOffset(json: String): Offset = {
    HTTPOffset(Serialization.read[Map[Int, Long]](json))
  }

  protected var startOffset: HTTPOffset = initialOffset.asInstanceOf[HTTPOffset]
  protected var currentOffset: HTTPOffset = initialOffset.asInstanceOf[HTTPOffset]
  protected var endOffset: HTTPOffset = initialOffset.asInstanceOf[HTTPOffset]

  override def initialOffset: Offset = {
    HTTPOffset.getStartingOffset(numPartitions)
  }

  override def latestOffset: Offset = {
    if (endOffset == currentOffset) {
      endOffset = HTTPOffset(endOffset.partitionToValue.mapValues(i => i + 1))
    }
    endOffset
  }

  private def getPartitionOffsetMap(offset: Offset): Map[Int, Long] = {
    val partitionMap = offset match {
      case off: HTTPOffset => off.partitionToValue
      case off =>
        throw new IllegalArgumentException(
          s"invalid offset type ${off.getClass} for ContinuousHTTPSource")
    }
    if (partitionMap.keySet.size != numPartitions) {
      throw new IllegalArgumentException(
        s"The previous run contained ${partitionMap.keySet.size} partitions, but" +
          s" $numPartitions partitions are currently configured. The numPartitions option" +
          " cannot be changed.")
    }
    partitionMap
  }

  override def planInputPartitions(start: Offset, end: Offset): Array[InputPartition] = {
    assert(start != null,
      "start offset should already be set before create read tasks.")
    if (!continuous) {
      assert(end != null,
        "end offset should already be set before create read tasks.")
    }

    val startMap = getPartitionOffsetMap(start)
    val endMap = if (!continuous) Some(getPartitionOffsetMap(end)) else None
    currentOffset = endOffset

    val config = WorkerServiceConfig(host, port, path, forwardingOptions,
      DriverServiceUtils.getDriverHost, driverService.getAddress.getPort, epochLength)
    Range(0, numPartitions).map { i =>
      HTTPInputPartition(continuous, name, config, startMap(i), endMap.map(_ (i)), i)
        : InputPartition
    }.toArray
  }

  override def commit(end: Offset): Unit = {}

  override def stop(): Unit = {
    logDebug("Stopping 1")
    driverService.stop(0)
    HTTPSourceStateHolder.cleanUp(name)
  }

  override def createReaderFactory(): PartitionReaderFactory = {
    HTTPSourceReaderFactory
  }

}

object HTTPSourceReaderFactory extends ContinuousPartitionReaderFactory {
  override def createReader(partition: InputPartition): ContinuousPartitionReader[InternalRow] = {
    val p = partition.asInstanceOf[HTTPInputPartition]
    new HTTPInputPartitionReader(p.continuous,
      p.name: String,
      p.config: WorkerServiceConfig,
      p.startValue: Long,
      p.endValue: Option[Long],
      p.partitionIndex: Int)
  }
}

private[streaming] class HTTPContinuousReader(options: CaseInsensitiveStringMap)
  extends HTTPMicroBatchReader(continuous = true, options = options) with ContinuousStream {

  def initialOffset(start: Optional[Offset]): Unit =
    this.startOffset = start.orElse(HTTPOffset.getStartingOffset(numPartitions))
      .asInstanceOf[HTTPOffset]

  override def mergeOffsets(offsets: Array[PartitionOffset]): Offset = {
    assert(offsets.length == numPartitions)
    val tuples = offsets.map {
      case HTTPPartitionOffset(i, epoch) => (i, epoch)
    }
    HTTPOffset(Map(tuples: _*))
  }

  override def planInputPartitions(start: Offset): Array[InputPartition] =
    planInputPartitions(start, null)

  override def createContinuousReaderFactory(): ContinuousPartitionReaderFactory = {
    HTTPSourceReaderFactory
  }
}

private[streaming] case class HTTPInputPartition(continuous: Boolean,
                                                 name: String,
                                                 config: WorkerServiceConfig,
                                                 startValue: Long,
                                                 endValue: Option[Long],
                                                 partitionIndex: Int
                                                )
  extends InputPartition

object HTTPSourceStateHolder {

  private val ServiceInformation: mutable.Map[String, ParHashSet[ServiceInfo]] = mutable.Map()

  private[streaming] def initServiceInfo(name: String, path: String): Unit = {
    assert(HTTPSourceStateHolder.ServiceInformation.get(name).isEmpty,
      "Cannot make 2 services with the same name")
    HTTPSourceStateHolder.ServiceInformation.update(name, new ParHashSet[ServiceInfo]())
  }

  private[streaming] def addServiceInfo(name: String, info: ServiceInfo): Unit = {
    val infoSet = HTTPSourceStateHolder.ServiceInformation
      .getOrElse(info.path, new ParHashSet[ServiceInfo]())
    infoSet += info
    HTTPSourceStateHolder.ServiceInformation.update(info.name, infoSet)
  }

  private[streaming] def removeServiceInfo(name: String): Unit = {
    HTTPSourceStateHolder.ServiceInformation.remove(name)
    ()
  }

  @GuardedBy("this")
  private val Clients: mutable.Map[String, WorkerClient] = mutable.Map()

  private[streaming] def getOrCreateClient(name: String): WorkerClient = synchronized {
    HTTPSourceStateHolder.Clients.getOrElse(name, {
      val client = new WorkerClient
      HTTPSourceStateHolder.Clients.update(name, client)
      client
    })
  }

  private[streaming] def removeClient(name: String): Unit = synchronized {
    HTTPSourceStateHolder.Clients.get(name).foreach { c =>
      c.close()
      HTTPSourceStateHolder.Clients.remove(name)
    }
  }

  @GuardedBy("this")
  private val Servers: mutable.Map[String, WorkerServer] = mutable.Map()

  private[streaming] def getServer(name: String): WorkerServer = {
    HTTPSourceStateHolder.Servers(name)
  }

  private[streaming] def getOrCreateServer(name: String,
                                           epoch: Long,
                                           partitionId: Int,
                                           isContinuous: Boolean,
                                           client: WorkerClient,
                                           config: WorkerServiceConfig
                                          ): WorkerServer = synchronized {
    val s = HTTPSourceStateHolder.Servers.getOrElse(name, {
      val server = new WorkerServer(name, isContinuous, client, config)
      HTTPSourceStateHolder.Servers.update(name, server)
      server
    })
    s.registerPartition(epoch, partitionId)
    s
  }

  private[streaming] def removeServer(name: String): Unit = synchronized {
    HTTPSourceStateHolder.Servers.get(name).foreach { c =>
      c.close()
      HTTPSourceStateHolder.Servers.remove(name)
    }
  }

  private[streaming] implicit val defaultFormats: DefaultFormats = DefaultFormats // scalastyle:ignore field.name

  def serviceInfoJson(name: String): String = {
    try {
      Serialization.write(ServiceInformation(name).toArray)
    } catch {
      case e: Throwable =>
        throw e
    }
  }

  def serviceInfo(name: String): Array[ServiceInfo] = {
    ServiceInformation(name).toArray
  }

  def cleanUp(name: String): Unit = {
    removeServer(name)
    removeClient(name)
    removeServiceInfo(name)
  }

}

private[streaming] class CachedRequest(val e: HttpExchange, val id: String) {
  private var cached: Option[HTTPRequestData] = None

  var isCached = false

  def getRowRep: HTTPRequestData = {
    cached.getOrElse {
      val res = HTTPRequestData.fromHTTPExchange(e)
      cached = Some(res)
      isCached = true
      res
    }
  }

}

private[streaming] class WorkerClient extends AutoCloseable {

  private val internalClient = {
    val requestTimeout = 60000
    val requestConfig = RequestConfig.custom()
      .setConnectTimeout(requestTimeout)
      .setConnectionRequestTimeout(requestTimeout)
      .setSocketTimeout(requestTimeout)
      .build()
    HttpClientBuilder.create()
      .setDefaultRequestConfig(requestConfig)
      .build()
  }

  protected[streaming] def reportServerToDriver(driverAddress: String, serviceInfo: ServiceInfo): Unit = {
    implicit val defaultFormats: DefaultFormats = DefaultFormats
    val post = new HttpPost(driverAddress)
    val info = Serialization.write(serviceInfo)
    post.setEntity(new StringEntity(info))
    val resp = internalClient.execute(post)
    assert(resp.getStatusLine.getStatusCode == 200, resp)
    resp.close()
  }

  override def close(): Unit = {
    internalClient.close()
  }
}

private[streaming] class WorkerServer(val name: String,
                                      val isContinuous: Boolean,
                                      val client: WorkerClient,
                                      val config: WorkerServiceConfig)
  extends AutoCloseable with Logging {

  type PID = Int
  type RID = String
  type Epoch = Long

  @GuardedBy("this")
  private var epoch: Long = 0

  def registerPartition(localEpoch: Epoch, partitionId: PID): Unit = synchronized {
    if (registeredPartitions.get(partitionId).isEmpty) {
      logInfo(s"registering $partitionId localEpoch:$localEpoch globalEpoch:$epoch")
      registeredPartitions.update(partitionId, localEpoch)
    } else {
      logInfo(s"re-registering $partitionId localEpoch:$localEpoch globalEpoch:$epoch")
      val previousEpoch = registeredPartitions(partitionId)
      registeredPartitions.update(partitionId, localEpoch)
      //there has been a failed partition and we need to rehydrate the queue
      if (previousEpoch == localEpoch) {
        logWarning(s"Adding to crash list localEpoch:$localEpoch globalEpoch:$epoch partition:$partitionId")
        val recoveredQueue = new LinkedBlockingQueue[CachedRequest]()
        recoveredQueue.addAll(historyQueues.getOrElse(
          (localEpoch, partitionId), new ListBuffer[CachedRequest]()).asJava)
        recoveredPartitions.update((localEpoch, partitionId), recoveredQueue)
      }
    }
  }

  @GuardedBy("this")
  private val registeredPartitions = new mutable.HashMap[PID, Epoch]

  @GuardedBy("this")
  private val requestQueues = new mutable.HashMap[Epoch, LinkedBlockingQueue[CachedRequest]]()
  requestQueues.update(0, new LinkedBlockingQueue[CachedRequest]())

  @GuardedBy("this")
  private val historyQueues = new mutable.HashMap[(Epoch, PID), mutable.ListBuffer[CachedRequest]]

  private[streaming] val recoveredPartitions = new mutable.HashMap[(Epoch, PID), LinkedBlockingQueue[CachedRequest]]

  private class PublicHandler extends HttpHandler {
    override def handle(request: HttpExchange): Unit = {
      logDebug(s"handling epoch: $epoch")
      val uuid = UUID.randomUUID().toString
      val cReq = new CachedRequest(request, uuid)
      requestQueues(epoch).put(cReq)
    }
  }

  private class InternalHandler extends HttpHandler {
    override def handle(request: HttpExchange): Unit = {
      //TODO
      throw new NotImplementedError("Have not implemented shuffle routing")
    }
  }

  def replyTo(machineIP: String, id: String, data: HTTPResponseData): Unit = {
    if (machineIP == localIp) {
      routingTable.get(id)
        .orElse {
          logWarning(s"Could not find request $id");
          None
        }
        .foreach { request =>
          HTTPServerUtils.respond(request.e, data)
          request.e.close()
          routingTable.remove(id)
        }

    } else {
      //TODO
      throw new NotImplementedError("Have not implemented shuffle routing")
    }

  }

  def commit(epochId: Epoch, partition: PID): Unit = {
    if (!isContinuous && epochId >= 0) {
      historyQueues.remove((epochId, partition)).foreach(hq =>
        hq.foreach { cr => routingTable.remove(cr.id); () }
      )
      recoveredPartitions.remove((epochId, partition))
      if (!historyQueues.keys.map(_._1).toSet(epochId)) {
        requestQueues.remove(epochId)
        ()
      }
      logDebug(s"Server State: ${historyQueues.size}, ${recoveredPartitions.size}, ${routingTable.size}")
    }
  }

  def commit(rid: String): Unit = {
    routingTable.remove(rid)
    ()
  }

  private def tryCreateServer(host: String,
                              startingPort: Int,
                              triesLeft: Int,
                              increment: Boolean = true): (HttpServer, Int) = {
    if (triesLeft == 0) {
      throw new java.net.BindException("Could not find open ports in the range," +
        " try increasing the number of ports to try")
    }
    try {
      val server = HttpServer.create(new InetSocketAddress(InetAddress.getByName(host), startingPort), 100)
      (server, startingPort)
    } catch {
      case _: java.net.BindException =>
        val inc = if (increment) 1 else 0
        tryCreateServer(host, startingPort + inc, triesLeft - 1)
    }
  }

  private def getMachineLocalIp: String = {
    InetAddress.getLocalHost.getHostAddress
  }

  private def getMachinePublicIp: Option[String] = {
    Try(new BufferedReader(new InputStreamReader(
      new URL("https://api.ipify.org/?format=text").openStream()))
      .readLine()).toOption
  }

  private val requestDataToRow = HTTPRequestData.makeToInternalRowConverter
  private lazy val localIp = getMachineLocalIp

  @GuardedBy("this")
  private var epochStart = System.currentTimeMillis()

  private def getNextRequestHelper(localEpoch: Epoch,
                                   partitionIndex: PID,
                                   continuous: Boolean): Option[CachedRequest] = {
    if (!continuous && recoveredPartitions.get((localEpoch, partitionIndex)).isDefined) {
      return Option(recoveredPartitions((localEpoch, partitionIndex)).poll())
    }

    val queue = requestQueues.getOrElseUpdate(
      localEpoch, new LinkedBlockingQueue[CachedRequest]())
    val timeout: Option[Either[Long, Long]] = if (continuous) {
      None
    } else if (localEpoch < epoch || Option(queue.peek()).isDefined) {
      Some(Left(0L))
    } else {
      Some(Right(config.epochLength - (System.currentTimeMillis() - epochStart)))
    }

    val result = timeout
      .map {
        case Left(0L) => Option(queue.poll())
        case Right(t) =>
          Option(queue.poll(t, TimeUnit.MILLISECONDS)).orElse {
            synchronized {
              //If the queue times out then we move to the next epoch
              epoch += 1
              val lbq = new LinkedBlockingQueue[CachedRequest]()
              requestQueues.update(epoch, lbq)
              epochStart = System.currentTimeMillis()
              None
            }
          }
        case _ => throw new IllegalArgumentException("Should not hit this path")
      }
      .orElse(Some(Some(queue.take())))
      .flatten
    result
  }

  def getNextRequest(localEpoch: Epoch, partitionIndex: PID, continuous: Boolean): Option[InternalRow] = {
    getNextRequestHelper(localEpoch, partitionIndex, continuous)
      .map { request =>
        routingTable.put(request.id, request)
        if (TaskContext.get().attemptNumber() == 0) {
          // If the request has never been materialized add it to the cache, otherwise we are in a retry and
          // should not modify the history
          historyQueues.getOrElseUpdate((localEpoch, partitionIndex), new ListBuffer[CachedRequest]())
            .append(request)
        }
        InternalRow(
          InternalRow(UTF8String.fromString(getMachineLocalIp), UTF8String.fromString(request.id), partitionIndex),
          Try(requestDataToRow(request.getRowRep)).toOption.orNull
        )
      }
  }

  logInfo(s"starting server at ${config.host}:${config.port}")
  val (server, foundPort) = tryCreateServer(config.host, config.port, 3, increment = false)
  server.createContext(s"/${config.path}", new PublicHandler)
  server.setExecutor(null) //scalastyle:ignore null
  server.start()
  logInfo(s"successfully started server at ${config.host}:$foundPort")

  val reporting: Future[Unit] = Future {
    client.reportServerToDriver(
      s"http://${config.driverServiceHost}:${config.driverServicePort}/driverService",
      ServiceInfo(name, config.host, foundPort, config.path, getMachineLocalIp, getMachinePublicIp)
    )
    logInfo(s"successfully replied to driver ${config.host}:$foundPort")
  }

  var forwardingSession: Option[Session] = None
  if (config.forwardingOptions.getOrElse("forwarding.enabled", "false").toBoolean) {
    val (session, _) = PortForwarding.forwardPortToRemote(
      config.forwardingOptions.toMap
        .updated("forwarding.localport", foundPort.toString)
        .updated("forwarding.localhost", config.host)
    )
    forwardingSession = Some(session)
  }

  private val routingTable: ParHashMap[String, CachedRequest] = ParHashMap()

  override def close(): Unit = {
    logDebug("stopping 2 ")
    server.stop(0)
    forwardingSession.foreach(_.disconnect())
  }

}

private[streaming] class HTTPInputPartitionReader(continuous: Boolean,
                                                  name: String,
                                                  config: WorkerServiceConfig,
                                                  val startEpoch: Long,
                                                  val endEpoch: Option[Long],
                                                  val partitionIndex: Int)
  extends ContinuousPartitionReader[InternalRow] with Logging {

  val client: WorkerClient = HTTPSourceStateHolder.getOrCreateClient(name)
  val server: WorkerServer = HTTPSourceStateHolder.getOrCreateServer(
    name, startEpoch, partitionIndex, continuous, client, config)

  private val currentEpoch = startEpoch
  private var rowsSeen: Long = 0
  private var currentRow: InternalRow = _

  def next(): Boolean = {
    logDebug(s"calling next: pi: $partitionIndex epoch:$currentEpoch rowsSeen:$rowsSeen")
    val reqOpt = server.getNextRequest(currentEpoch, partitionIndex, continuous)
    reqOpt.map { req =>
      rowsSeen += 1
      currentRow = req
      logDebug(s"Returning true pi: $partitionIndex epoch:$currentEpoch")
      true
    }.getOrElse {
      logDebug(s"Returning false pi: $partitionIndex epoch:$currentEpoch")
      false
    }
  }

  def get: InternalRow = currentRow

  def close(): Unit = {}

  def getOffset: PartitionOffset =
    HTTPPartitionOffset(partitionIndex, currentEpoch)

}
