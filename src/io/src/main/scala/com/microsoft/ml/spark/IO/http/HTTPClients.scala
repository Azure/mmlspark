// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import org.apache.http.client.methods.{CloseableHttpResponse, HttpRequestBase}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}

import scala.concurrent.ExecutionContext
import scala.concurrent.duration.Duration

trait Handler {

  def handle(client: CloseableHttpClient, request: HTTPRequestData): HTTPResponseData

  protected def convertAndClose(response: CloseableHttpResponse): HTTPResponseData = {
    val rData = new HTTPResponseData(response)
    response.close()
    rData
  }

}

private[ml] trait HTTPClient extends BaseClient
  with AutoCloseable with Handler {

  override protected type Client = CloseableHttpClient
  override type ResponseType = HTTPResponseData
  override type RequestType = HTTPRequestData

  protected val internalClient: Client = HttpClientBuilder.create().build()

  override def close(): Unit = {
    internalClient.close()
  }

  protected def sendRequestWithContext(request: RequestWithContext): ResponseWithContext = {
    ResponseWithContext(handle(internalClient, request.request), request.context)
  }

}

object AdvancedHTTPHandling extends Handler {

  val retryTimes = Seq(100L, 1000L, 2000L, 4000L)

  private def sendWithRetries(client: CloseableHttpClient,
                              request: HttpRequestBase,
                              retriesLeft: Seq[Long]): HTTPResponseData = {
    val response = client.execute(request)
    val code = response.getStatusLine.getStatusCode
    val suceeded = code match {
      case 200 => true
      case 429 =>
        Option(response.getFirstHeader("Retry-After"))
          .foreach{h =>
            println(s"waiting ${h.getValue}")
            Thread.sleep(h.getValue.toLong * 1000)}
        response.close()
        false
      case _ => false
    }
    if (suceeded || retriesLeft.isEmpty) {
      convertAndClose(response)
    } else {
      response.close()
      Thread.sleep(retriesLeft.head)
      sendWithRetries(client, request, retriesLeft.tail)
    }
  }

  def handle(client: CloseableHttpClient, request: HTTPRequestData): HTTPResponseData = {
    sendWithRetries(client, request.toHTTPCore, retryTimes)
  }
}

object BasicHTTPHandling extends Handler {

  def handle(client: CloseableHttpClient, request: HTTPRequestData): HTTPResponseData =
    convertAndClose(client.execute(request.toHTTPCore))

}

abstract class AsyncHTTPClient(override val concurrency: Int, override val timeout: Duration)
                              (override implicit val ec: ExecutionContext)
  extends AsyncClient(concurrency, timeout)(ec) with HTTPClient

abstract class SingleThreadedHTTPClient extends HTTPClient with SingleThreadedClient
