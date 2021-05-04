// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.http

import org.apache.commons.io.IOUtils
import org.apache.http.client.config.RequestConfig
import org.apache.http.client.methods.{CloseableHttpResponse, HttpPost, HttpRequestBase}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager
import org.apache.spark.injections.UDFUtils
import org.apache.spark.internal.{Logging => SparkLogging}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StringType

import scala.concurrent.{ExecutionContext, blocking}
import scala.concurrent.duration.Duration
import scala.util.Try

trait Handler {

  def handle(client: CloseableHttpClient, request: HTTPRequestData): HTTPResponseData

}

private[ml] trait HTTPClient extends BaseClient
  with AutoCloseable with Handler {

  override protected type Client = CloseableHttpClient
  override type ResponseType = HTTPResponseData
  override type RequestType = HTTPRequestData

  protected val requestTimeout: Int

  protected val requestConfig: RequestConfig = RequestConfig.custom()
    .setConnectTimeout(requestTimeout)
    .setConnectionRequestTimeout(requestTimeout)
    .setSocketTimeout(requestTimeout)
    .build()

  protected val connectionManager: PoolingHttpClientConnectionManager = {
    val cm = new PoolingHttpClientConnectionManager()
    cm.setDefaultMaxPerRoute(Int.MaxValue) // Spark will handle the threading to avoid going over limits
    cm.setMaxTotal(Int.MaxValue)
    cm
  }

  protected val internalClient: Client = HttpClientBuilder.create()
    .setConnectionManager(connectionManager)
    .setDefaultRequestConfig(requestConfig).build()

  override def close(): Unit = {
    internalClient.close()
  }

  protected def sendRequestWithContext(request: RequestWithContext): ResponseWithContext = {
    request.request.map(req =>
      ResponseWithContext(Some(handle(internalClient, req)), request.context)
    ).getOrElse(ResponseWithContext(None, request.context))
  }

}

object HandlingUtils extends SparkLogging {
  private[ml] def convertAndClose(response: CloseableHttpResponse): HTTPResponseData = {
    val rData = new HTTPResponseData(response)
    response.close()
    rData
  }

  type HandlerFunc = (CloseableHttpClient, HTTPRequestData) => HTTPResponseData

  private[ml] def sendWithRetries(client: CloseableHttpClient,
                                  request: HttpRequestBase,
                                  retriesLeft: Array[Int]): CloseableHttpResponse = {
    val response = client.execute(request)
    val code = response.getStatusLine.getStatusCode
    //scalastyle:off magic.number
    val succeeded = code match {
      case 200 => true
      case 201 => true
      case 202 => true
      case 429 =>
        Option(response.getFirstHeader("Retry-After"))
          .foreach { h =>
            logInfo(s"waiting ${h.getValue} on ${
              request match {
                case p: HttpPost => p.getURI + "   " +
                  Try(IOUtils.toString(p.getEntity.getContent, "UTF-8")).getOrElse("")
                case _ => request.getURI
              }
            }")
            Thread.sleep(h.getValue.toLong * 1000)
          }
        false
      case 400 =>
        true
      case _ =>
        logWarning(s"got error  $code: ${response.getStatusLine.getReasonPhrase} on ${
          request match {
            case p: HttpPost => p.getURI + "   " +
              Try(IOUtils.toString(p.getEntity.getContent, "UTF-8")).getOrElse("")
            case _ => request.getURI
          }
        }")
        false
    }
    //scalastyle:on magic.number
    if (succeeded || retriesLeft.isEmpty) {
      response
    } else {
      response.close()
      Thread.sleep(retriesLeft.head.toLong)
      if (code == 429) { // Do not count rate limiting in number of failures
        sendWithRetries(client, request, retriesLeft)
      } else {
        sendWithRetries(client, request, retriesLeft.tail)
      }
    }
  }

  def advanced(retryTimes: Int*)(client: CloseableHttpClient,
                                 request: HTTPRequestData): HTTPResponseData = {
    try {
      val req = request.toHTTPCore
      val message = req match {
        case r: HttpPost => Try(IOUtils.toString(r.getEntity.getContent, "UTF-8")).getOrElse("")
        case r => r.getURI
      }
      logInfo(s"sending $message")
      val start = System.currentTimeMillis()
      val resp = sendWithRetries(client, req, retryTimes.toArray)
      logInfo(s"finished sending (${System.currentTimeMillis() - start}ms) $message")
      val respData = convertAndClose(resp)
      req.releaseConnection()
      respData
    } catch {
      case e: java.net.SocketTimeoutException =>
        logWarning(s"Encountered Socket Timeout: ${e.getMessage}")
        null //scalastyle:ignore null
    }
  }

  def advancedUDF(retryTimes: Int*): UserDefinedFunction =
    UDFUtils.oldUdf(advanced(retryTimes: _*) _, StringType)

  def basic(client: CloseableHttpClient, request: HTTPRequestData): HTTPResponseData = {
    val req = request.toHTTPCore
    val data = convertAndClose(client.execute(req))
    req.releaseConnection()
    data
  }

  def basicUDF: UserDefinedFunction = UDFUtils.oldUdf(basic _, StringType)
}

class AsyncHTTPClient(val handler: HandlingUtils.HandlerFunc,
                      override val concurrency: Int,
                      override val timeout: Duration,
                      val requestTimeout: Int)
                     (override implicit val ec: ExecutionContext)
  extends AsyncClient(concurrency, timeout)(ec) with HTTPClient {
  override def handle(client: CloseableHttpClient,
                      request: HTTPRequestData): HTTPResponseData = blocking {
    handler(client, request)
  }
}

class SingleThreadedHTTPClient(val handler: HandlingUtils.HandlerFunc, val requestTimeout: Int)
  extends HTTPClient with SingleThreadedClient {
  override def handle(client: CloseableHttpClient,
                      request: HTTPRequestData): HTTPResponseData = blocking {
    handler(client, request)
  }
}
