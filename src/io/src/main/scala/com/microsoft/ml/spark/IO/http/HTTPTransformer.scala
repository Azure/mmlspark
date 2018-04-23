// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, Wrappable}
import org.apache.http.client.methods.{CloseableHttpResponse, HttpRequestBase}
import org.apache.http.impl.client.CloseableHttpClient
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.concurrent.ExecutionContext
import scala.concurrent.duration.Duration

trait HTTPParams extends Wrappable {
  val concurrency: Param[Int] = new IntParam(
    this, "concurrency", "max number of concurrent calls")

  /** @group getParam */
  def getConcurrency: Int = $(concurrency)

  /** @group setParam */
  def setConcurrency(value: Int): this.type = set(concurrency, value)

  val concurrentTimeout: Param[Double] = new DoubleParam(
    this, "concurrentTimeout", "max number seconds to wait on futures if concurrency >= 1")

  /** @group getParam */
  def getConcurrentTimeout: Double = $(concurrentTimeout)

  /** @group setParam */
  def setConcurrentTimeout(value: Double): this.type = set(concurrentTimeout, value)

  val handlingStrategy: Param[String] = new Param[String](
    this, "handlingStrategy", "Which strategy to use when handling requests",
    { x: String => Set("basic", "advanced")(x.toLowerCase) })

  /** @group getParam */
  def getHandlingStrategy: String = $(handlingStrategy)

  /** @group setParam */
  def setHandlingStrategy(value: String): this.type = set(handlingStrategy, value.toLowerCase)

  setDefault(concurrency -> 1, concurrentTimeout -> 100, handlingStrategy -> "advanced")

}

trait HasURL extends Wrappable {

  val url: Param[String] = new Param[String](this, "url", "Url of the service")

  /** @group getParam */
  def getUrl: String = $(url)

  /** @group setParam */
  def setUrl(value: String): this.type = set(url, value)

}

object HTTPTransformer extends ComplexParamsReadable[HTTPTransformer]

class HTTPTransformer(val uid: String)
  extends Transformer with HTTPParams with HasInputCol
    with HasOutputCol
    with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("HTTPTransformer"))

  val clientHolder = SharedVariable {
    val strategy: (CloseableHttpClient, HTTPRequestData) => HTTPResponseData =
      getHandlingStrategy match {
        case "basic" => BasicHTTPHandling.handle
        case "advanced" => AdvancedHTTPHandling.handle
      }

    getConcurrency match {
      case 1 => new SingleThreadedHTTPClient {
        override def handle(client: Client, request: RequestType) = strategy(client, request)
      }
      case n if n > 1 =>
        val dur = Duration.fromNanos((getConcurrentTimeout * math.pow(10, 9)).toLong)
        val ec = ExecutionContext.global
        new AsyncHTTPClient(n, dur)(ec) {
          override def handle(client: Client, request: RequestType) = strategy(client, request)
        }
    }
  }

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from column selection
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val enc = RowEncoder(transformSchema(df.schema))
    val colIndex = df.schema.fieldNames.indexOf(getInputCol)
    df.mapPartitions { it =>
      if (!it.hasNext) {
        Iterator()
      }else{
        val c = clientHolder.get
        val responsesWithContext = c.sendRequestsWithContext(it.map(row =>
          c.RequestWithContext(
            HTTPRequestData.fromRow(row.getStruct(colIndex)),
            Some(row))))
        responsesWithContext.map(rwc =>
          Row.merge(rwc.context.get.asInstanceOf[Row], Row(rwc.response.toRow)))
      }
    }(enc)
  }

  def copy(extra: ParamMap): HTTPTransformer = defaultCopy(extra)

  def transformSchema(schema: StructType): StructType = {
    assert(schema(getInputCol).dataType == HTTPSchema.request)
    schema.add(getOutputCol, HTTPSchema.response)
  }

}
