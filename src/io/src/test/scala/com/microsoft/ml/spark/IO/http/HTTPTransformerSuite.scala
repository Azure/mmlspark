// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import java.net.{InetSocketAddress, ServerSocket}
import java.util.concurrent.Executors

import com.microsoft.ml.spark.core.env.StreamUtilities.using
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalactic.Equality

object ServerUtils {
  private def respond(request: HttpExchange, code: Int, response: String): Unit = synchronized {
    val bytes = response.getBytes("UTF-8")
    request.synchronized {
      request.getResponseHeaders.add("Content-Type", "application/json")
      request.sendResponseHeaders(code, 0)
      using(request.getResponseBody) { os =>
        os.write(bytes)
        os.flush()
      }
      request.close()
    }
  }

  private class RequestHandler extends HttpHandler {
    override def handle(request: HttpExchange): Unit = synchronized {
      respond(request, 200, "{\"blah\": \"more blah\"}")
    }
  }

  def createServer(host: String, port:Int, apiName: String): HttpServer = {
    val server = HttpServer.create(new InetSocketAddress(host,port), 100)
    server.createContext(s"/$apiName", new RequestHandler)
    server.setExecutor(Executors.newFixedThreadPool(100))
    server.start()
    server
  }
}

trait WithFreeUrl {
  val host              = "localhost"
  val apiName           = "foo"
  //Note this port should be used immediately to avoid race conditions
  lazy val port: Int    =
    using(new ServerSocket(0))(_.getLocalPort).get
  lazy val url:String   = {
    s"http://$host:$port/$apiName"
  }
}

trait WithServer extends TestBase with WithFreeUrl {
  var server: Option[HttpServer] = None

  override def beforeAll(): Unit = {
    server = Some(ServerUtils.createServer(host, port, apiName))
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    server.get.stop(0)
    super.afterAll()
  }
}

class HTTPTransformerSuite extends TransformerFuzzing[HTTPTransformer]
    with WithServer with ParserUtils {

  override def testObjects(): Seq[TestObject[HTTPTransformer]] = makeTestObject(
      new HTTPTransformer().setInputCol("parsedInput").setOutputCol("out"), session)

  override def reader: MLReadable[_] = HTTPTransformer

  //TODO this is needed because columns with a timestamp are added
  override implicit lazy val dfEq: Equality[DataFrame] =  new Equality[DataFrame]{
    def areEqual(a: DataFrame, bAny: Any): Boolean = bAny match {
      case ds:Dataset[_] =>
        val b = ds.toDF()
        if (a.columns !== b.columns) {
          return false
        }
        val aSort = a.sort().collect()
        val bSort = b.sort().collect()
        if (aSort.length != bSort.length){
          return false
        }
        true
    }
  }

}
