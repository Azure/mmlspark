// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import java.net.{SocketException, URI}

import org.apache.commons.io.IOUtils
import org.apache.http._
import org.apache.http.client.methods._
import org.apache.http.entity.{ByteArrayEntity, StringEntity}
import org.apache.http.message.BasicHeader
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, struct, typedLit, udf}
import org.apache.spark.sql.types.{DataType, StringType}
import org.apache.spark.sql.{Column, Row}

case class HeaderData(name: String, value: String) {

  def this(h: Header) = {
    this(h.getName, h.getValue)
  }

  def toHTTPCore: Header = new BasicHeader(name, value)

  def toRow: Row = {
    Row(name, value)
  }

}

object HeaderData {

  def fromRow(r: Row): HeaderData = {
    HeaderData(r.getString(0), r.getString(1))
  }

}

case class EntityData(content: Array[Byte],
                      contentEncoding: Option[HeaderData],
                      contentLenth: Long,
                      contentType: Option[HeaderData],
                      isChunked: Boolean,
                      isRepeatable: Boolean,
                      isStreaming: Boolean) {

  def this(e: HttpEntity) = {
    this(
         try {
           IOUtils.toByteArray(e.getContent)
         } catch {
           case _: SocketException => Array() //TODO investigate why sockets fail sometimes
         },
         Option(e.getContentEncoding).map(new HeaderData(_)),
         e.getContentLength,
         Option(e.getContentType).map(new HeaderData(_)),
         e.isChunked,
         e.isRepeatable,
         e.isStreaming)
  }

  def toHttpCore: HttpEntity = {
    val e = new ByteArrayEntity(content)
    contentEncoding.foreach { ce => e.setContentEncoding(ce.toHTTPCore) }
    assert(e.getContentLength == contentLenth)
    contentType.foreach(h => e.setContentType(h.toHTTPCore))
    e.setChunked(isChunked)
    assert(e.isRepeatable == isRepeatable)
    assert(e.isStreaming == isStreaming)
    e
  }

  def toRow: Row = {
    Row(content, contentEncoding.map(_.toRow).orNull,
      contentLenth, contentType.map(_.toRow).orNull,
      isChunked, isRepeatable, isStreaming)
  }

}

object EntityData {

  def fromRow(r: Row): EntityData = {
    EntityData(
      r.getAs[Array[Byte]](0),
      if (r.isNullAt(1)) None else Some(HeaderData.fromRow(r.getStruct(1))),
      r.getLong(2),
      if (r.isNullAt(3)) None else Some(HeaderData.fromRow(r.getStruct(3))),
      r.getBoolean(4),
      r.getBoolean(5),
      r.getBoolean(6))
  }

}

case class StatusLineData(protocolVersion: ProtocolVersionData,
                          statusCode: Int,
                          reasonPhrase: String) {

  def this(s: StatusLine) = {
    this(new ProtocolVersionData(s.getProtocolVersion),
         s.getStatusCode,
         s.getReasonPhrase)
  }

  def toRow: Row = {
    Row(protocolVersion.toRow, statusCode, reasonPhrase)
  }

}

object StatusLineData {

  def fromRow(r: Row): StatusLineData = {
    StatusLineData(
      ProtocolVersionData.fromRow(r.getStruct(0)),
      r.getInt(1),
      r.getString(2))
  }

}

case class HTTPResponseData(headers: Array[HeaderData],
                            entity: EntityData,
                            statusLine: StatusLineData,
                            locale: String) {

  def this(response: CloseableHttpResponse) = {
    this(response.getAllHeaders.map(new HeaderData(_)),
         new EntityData(response.getEntity),
         new StatusLineData(response.getStatusLine),
         response.getLocale.toString)
  }

  def toRow: Row = {
    Row.apply(headers.map(_.toRow), entity.toRow, statusLine.toRow, locale)
  }

}

object HTTPResponseData {

  def fromRow(r: Row): HTTPResponseData = {
    HTTPResponseData(
      r.getSeq[Row](0).map(HeaderData.fromRow).toArray,
      EntityData.fromRow(r.getStruct(1)),
      StatusLineData.fromRow(r.getStruct(2)),
      r.getString(3))
  }

}

case class ProtocolVersionData(protocol: String, major: Int, minor: Int) {

  def this(v: ProtocolVersion) = {
    this(v.getProtocol, v.getMajor, v.getMinor)
  }

  def toHTTPCore: ProtocolVersion = {
    new ProtocolVersion(protocol, major, minor)
  }

  def toRow: Row = {
    Row(protocol, major, minor)
  }

}

object ProtocolVersionData {

  def fromRow(r: Row): ProtocolVersionData = {
    ProtocolVersionData(r.getString(0), r.getInt(1), r.getInt(2))
  }

}

case class RequestLineData(method: String,
                           uri: String,
                           protoclVersion: Option[ProtocolVersionData]) {

  def this(l: RequestLine) = {
    this(l.getMethod,
         l.getUri,
         Some(new ProtocolVersionData(l.getProtocolVersion)))
  }

}

object RequestLineData {

  def fromRow(row: Row): RequestLineData = {
    RequestLineData(
      row.getString(0),
      row.getString(1),
      if (row.isNullAt(2)) None else Some(ProtocolVersionData.fromRow(row.getStruct(2))))
  }

}

case class HTTPRequestData(requestLine: RequestLineData,
                           headers: Array[HeaderData],
                           entity: Option[EntityData]) {

  def this(r: HttpRequestBase) = {
    this(new RequestLineData(r.getRequestLine),
         r.getAllHeaders.map(new HeaderData(_)),
         r match {
           case re: HttpEntityEnclosingRequestBase => Option(re.getEntity).map(new EntityData(_))
           case _ => None
         })
  }

  def toHTTPCore: HttpRequestBase = {
    val request = requestLine.method.toUpperCase match {
      case "GET"     => new HttpGet()
      case "HEAD"    => new HttpHead()
      case "DELETE"  => new HttpDelete()
      case "OPTIONS" => new HttpOptions()
      case "TRACE"   => new HttpTrace()
      case "POST"    => new HttpPost()
      case "PUT"     => new HttpPut()
      case "PATCH"   => new HttpPatch()
    }
    request match {
      case re: HttpEntityEnclosingRequestBase =>
        entity.foreach(e => re.setEntity(e.toHttpCore))
      case _ if entity.isDefined =>
        throw new IllegalArgumentException(s"Entity is defined but method is ${requestLine.method}")
      case _ =>
    }
    request.setURI(new URI(requestLine.uri))
    requestLine.protoclVersion.foreach(pv =>
      request.setProtocolVersion(pv.toHTTPCore))
    request.setHeaders(headers.map(_.toHTTPCore))
    request
  }

}

object HTTPRequestData {

  def fromRow(row: Row): HTTPRequestData = {
    assert(row.schema == HTTPSchema.request)
    HTTPRequestData(
      RequestLineData.fromRow(row.getStruct(0)),
      row.getSeq[Row](1).map(HeaderData.fromRow).toArray,
      if (row.isNullAt(2)) None else Some(EntityData.fromRow(row.getStruct(2)))
    )}

}

object HTTPSchema {

  val response: DataType = ScalaReflection.schemaFor[HTTPResponseData].dataType
  val request: DataType = ScalaReflection.schemaFor[HTTPRequestData].dataType

  def to_http(urlCol: String, headersCol: String, methodCol: String, jsonEntityCol: String): Column = {
    to_http(col(urlCol), col(headersCol), col(methodCol), col(jsonEntityCol))
  }

  private def stringToEntity(s: String): EntityData = {
    new EntityData(new StringEntity(s))
  }

  private def entityToString(e: EntityData): Option[String] = {
    if (e.content.isEmpty) {
      None
    } else {
      Some(IOUtils.toString(e.content,
        e.contentEncoding.map(h => h.value).getOrElse("UTF-8")))
    }
  }

  val entityToStringUDF: UserDefinedFunction =
    udf({ x: Row => entityToString(EntityData.fromRow(x))},
        StringType)

  val stringToEntityUDF: UserDefinedFunction = udf({ x: String => stringToEntity(x) },
    ScalaReflection.schemaFor[EntityData].dataType
  )

  def to_http(urlCol: Column, headersCol: Column, methodCol: Column, jsonEntityCol: Column): Column = {
    val pvd: Option[ProtocolVersionData] = None
    struct(
      struct(
        methodCol.alias("method"),
        urlCol.alias("uri"),
        typedLit(pvd).alias("protocolVersion")).alias("requestLine"),
      headersCol.alias("headers"),
      stringToEntityUDF(jsonEntityCol).alias("entity")
    ).cast(request)
  }

}
