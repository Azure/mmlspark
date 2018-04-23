// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import java.net.{InetAddress, InetSocketAddress}
import javax.annotation.concurrent.GuardedBy

import com.microsoft.ml.spark.core.env.StreamUtilities.using
import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import org.apache.commons.io.IOUtils
import org.apache.spark.internal.Logging
import org.apache.spark.sql._
import org.apache.spark.sql.execution.streaming._
import org.apache.spark.sql.functions.{col, struct, to_json}
import org.apache.spark.sql.sources.{DataSourceRegister, StreamSinkProvider, StreamSourceProvider}
import org.apache.spark.sql.streaming.OutputMode
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object HTTPSource {

  val SCHEMA = StructType(
    StructField("id", StringType) ::
      StructField("value", StringType) :: Nil)

  def respond(request: HttpExchange, code: Int, response: String): Unit = {
    request.sendResponseHeaders(code, response.length.toLong)
    using(request.getResponseBody) {
      _.write(response.getBytes)
    }.get
  }

  var replyCallbacks: mutable.Map[String, Iterable[(Long, String)] => Unit] = mutable.Map()

}

/** A source that reads text lines through a TCP socket, designed only for tutorials and debugging.
  * This source will *not* work in production applications due to multiple reasons, including no
  * support for fault recovery and keeping all of the text read in memory forever.
  */
class HTTPSource(name: String, host: String, port: Int, sqlContext: SQLContext)
    extends Source with Logging {

  import sqlContext.implicits._

  class QueueHandler extends HttpHandler {

    private def getBody(request: HttpExchange) = {
      using(request.getRequestBody) {
        IOUtils.toString(_, "UTF-8")
      }.get
    }

    override def handle(request: HttpExchange): Unit = {
      if (request.getRequestMethod != "POST") {
        HTTPSource.respond(request, 405, "Only POSTs accepted")
      }
      val headers = request.getRequestHeaders
      if (headers.containsKey("Content-type") &&
        headers.get("Content-type").get(0) != "application/json") {
        HTTPSource.respond(request, 400, "Content-type needs to be application/json")
      }
      val body = getBody(request)
      HTTPSource.this.synchronized {
        currentOffset = currentOffset + 1
        requests.append((currentOffset.offset, body, request))
      }
    }
  }

  /** All batches from `lastCommittedOffset + 1` to `currentOffset`, inclusive.
    * Stored in a ListBuffer to facilitate removing committed batches.
    */
  @GuardedBy("this")
  protected val requests: ListBuffer[(Long, String, HttpExchange)] = ListBuffer()

  @GuardedBy("this")
  protected var currentOffset: LongOffset = new LongOffset(-1)

  @GuardedBy("this")
  protected var lastOffsetCommitted: LongOffset = new LongOffset(-1)

  @GuardedBy("this")
  private val server = HttpServer.create(new InetSocketAddress(InetAddress.getByName(host), port), 0)
  server.createContext(s"/$name", new QueueHandler)
  server.setExecutor(null)
  server.start()
  HTTPSource.replyCallbacks.update(name, reply)

  /** Returns the schema of the data from this source */
  override def schema: StructType = HTTPSource.SCHEMA

  override def getOffset: Option[Offset] = synchronized {
    if (currentOffset.offset == -1) None else Some(currentOffset)
  }

  /** Returns the data that is between the offsets (`start`, `end`]. */
  override def getBatch(start: Option[Offset], end: Offset): DataFrame = synchronized {
    val startOrdinal =
      start.flatMap(LongOffset.convert).getOrElse(LongOffset(-1)).offset.toInt + 1
    val endOrdinal = LongOffset.convert(end).getOrElse(LongOffset(-1)).offset.toInt + 1
    // Internal buffer only holds the batches after lastOffsetCommitted
    val rawList = synchronized {
      val sliceStart = startOrdinal - lastOffsetCommitted.offset.toInt - 1
      val sliceEnd = endOrdinal - lastOffsetCommitted.offset.toInt - 1
      requests.slice(sliceStart, sliceEnd).map(t => (t._1, t._2))
    }
    val rawBatch = sqlContext.createDataset(rawList)
    rawBatch.toDF("id", "value")
  }

  def reply(replies: Iterable[(Long, String)]): Unit = {
    replies.foreach { case (id, response) =>
      val request = requests((id - lastOffsetCommitted.offset).toInt - 1)
      HTTPSource.respond(request._3, 200, response)
    }
  }

  override def commit(end: Offset): Unit = synchronized {
    val newOffset = LongOffset.convert(end).getOrElse(
      sys.error(s"TextSocketStream.commit() received an offset ($end) that did not " +
                  s"originate with an instance of this class"))

    val offsetDiff = (newOffset.offset - lastOffsetCommitted.offset).toInt

    if (offsetDiff < 0) {
      sys.error(s"Offsets committed out of order: $lastOffsetCommitted followed by $end")
    }

    requests.trimStart(offsetDiff)
    lastOffsetCommitted = newOffset
  }

  /** Stop this source. */
  override def stop(): Unit = synchronized {
    server.stop(0)
    HTTPSource.replyCallbacks.remove(name)
    ()
  }

  override def toString: String = s"HTTPSource[name: $name, host: $host, port: $port]"

}

class HTTPSourceProvider extends StreamSourceProvider with DataSourceRegister with Logging {

  /** Returns the name and schema of the source that can be used to continually read data. */
  override def sourceSchema(sqlContext: SQLContext,
                            schema: Option[StructType],
                            providerName: String,
                            parameters: Map[String, String]): (String, StructType) = {
    logWarning("The socket source should not be used for production applications! " +
                 "It does not support recovery.")
    if (!parameters.contains("host")) {
      throw new NamespaceInjections.InjectedAnalysisException("Set a host to read from with option(\"host\", ...).")
    }
    if (!parameters.contains("port")) {
      throw new NamespaceInjections.InjectedAnalysisException("Set a port to read from with option(\"port\", ...).")
    }
    if (!parameters.contains("name")) {
      throw new NamespaceInjections.InjectedAnalysisException("Set a name of the API which is used for routing")
    }
    ("HTTP", HTTPSource.SCHEMA)
  }

  override def createSource(sqlContext: SQLContext,
                            metadataPath: String,
                            schema: Option[StructType],
                            providerName: String,
                            parameters: Map[String, String]): Source = {
    val host = parameters("host")
    val port = parameters("port").toInt
    val name = parameters("name")
    val source = new HTTPSource(name, host, port, sqlContext)
    source
  }

  /** String that represents the format that this data source provider uses. */
  override def shortName(): String = "HTTP"

}

class HTTPSink(val options: Map[String, String]) extends Sink with Logging {

  if (!options.contains("name")) {
    throw new NamespaceInjections.InjectedAnalysisException("Set a name of an API to reply to")
  }

  override def addBatch(batchId: Long, data: DataFrame): Unit = synchronized {
    val valueCol = options.getOrElse("replyCol", "value")
    val replies = data
      .select(col(options.getOrElse("idCol", "id")),
              to_json(struct(valueCol)).alias(valueCol))
      .collect().map { row => (row.getLong(0), row.getString(1))}
    HTTPSource.replyCallbacks(options("name"))(replies)
  }

}

class HTTPSinkProvider extends StreamSinkProvider with DataSourceRegister {

  def createSink(sqlContext: SQLContext,
                 parameters: Map[String, String],
                 partitionColumns: Seq[String],
                 outputMode: OutputMode): Sink = {
    new HTTPSink(parameters)
  }

  def shortName(): String = "HTTP"

}
