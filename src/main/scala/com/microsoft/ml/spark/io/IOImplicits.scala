// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io

import com.microsoft.ml.spark.io.binary.BinaryFileFormat
import com.microsoft.ml.spark.io.http.{HTTPRequestData, HTTPSchema}
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.source.image.PatchedImageFileFormat
import org.apache.spark.sql._
import org.apache.spark.sql.execution.streaming._
import org.apache.spark.sql.execution.streaming.continuous._
import org.apache.spark.sql.streaming.{DataStreamReader, DataStreamWriter}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import scala.language.implicitConversions

case class DataStreamReaderExtensions(dsr: DataStreamReader) {

  def server: DataStreamReader = {
    dsr.format(classOf[HTTPSourceProvider].getName)
  }

  def distributedServer: DataStreamReader = {
    dsr.format(classOf[DistributedHTTPSourceProvider].getName)
  }

  def continuousServer: DataStreamReader = {
    dsr.format(classOf[HTTPSourceProviderV2].getName)
  }

  def address(host: String, port: Int, api: String): DataStreamReader = {
    dsr.option("host", host).option("port", port.toLong).option("path", api)
  }

  def image: DataStreamReader = {
    dsr.format(classOf[PatchedImageFileFormat].getName)
      .schema(ImageSchema.imageSchema)
  }

  def binary: DataStreamReader = {
    dsr.format(classOf[BinaryFileFormat].getName)
  }

}

case class DataStreamWriterExtensions[T](dsw: DataStreamWriter[T]) {

  def server: DataStreamWriter[T] = {
    dsw.format(classOf[HTTPSinkProvider].getName)
  }

  def distributedServer: DataStreamWriter[T] = {
    dsw.format(classOf[DistributedHTTPSinkProvider].getName)
  }

  def continuousServer: DataStreamWriter[T] = {
    dsw.format(classOf[HTTPSinkProviderV2].getName)
  }

  def replyTo(name: String): DataStreamWriter[T] = {
    dsw.option("name", name)
  }

  def image: DataStreamWriter[T] = {
    dsw.format(classOf[PatchedImageFileFormat].getName)
  }

  def binary: DataStreamWriter[T] = {
    dsw.format(classOf[BinaryFileFormat].getName)
  }

}

case class DataFrameReaderExtensions(dsr: DataFrameReader) {

  def image: DataFrameReader = {
    dsr.format(classOf[PatchedImageFileFormat].getName)
  }

  def binary: DataFrameReader = {
    dsr.format(classOf[BinaryFileFormat].getName)
  }

}

case class DataFrameWriterExtensions[T](dsw: DataFrameWriter[T]) {

  def image: DataFrameWriter[T] = {
    dsw.format(classOf[PatchedImageFileFormat].getName)
  }

  def binary: DataFrameWriter[T] = {
    dsw.format(classOf[BinaryFileFormat].getName)
  }

}

case class DataFrameExtensions(df: DataFrame) {

  private def jsonParsingError(schema: DataType)(body: String): String = {
    s"JSON Parsing error, expected schema:\n ${schema.simpleString}\n received:\n $body"
  }

  private def fullJsonParsingSuccess(a: Any): Boolean = {
    a match {
      case s: Seq[_] => s.forall(fullJsonParsingSuccess)
      case a: Row => a.toSeq.forall(fullJsonParsingSuccess)
      case null => false //scalastyle:ignore null
      case _ => true
    }
  }

  private def partialJsonParsingSuccess(a: Any): Boolean = {
    a match {
      case null => false //scalastyle:ignore null
      case _ => true
    }
  }

  /**
    *
    * @param apiName
    * @param schema
    * @param idCol
    * @param requestCol
    * @param parsingCheck "none": to accept all requests,
    *                     "full": to ensure all fields and sub-fields are non-null,
    *                     "partial": to ensure the root structure was parsed correctly
    * @return
    */
  def parseRequest(apiName: String,
                   schema: DataType,
                   idCol: String = "id",
                   requestCol: String = "request",
                   parsingCheck: String = "none"): DataFrame = {
    assert(df.schema(idCol).dataType == HTTPSourceV2.IdSchema &&
      df.schema(requestCol).dataType == HTTPRequestData.schema)
    schema match {
      case BinaryType =>
        df.select(col(idCol), col(requestCol).getItem("entity").getItem("content").alias("bytes"))
      case _ =>
        val parsedDf = df
          .withColumn("body", HTTPSchema.request_to_string(col(requestCol)))
          .withColumn("parsed", from_json(col("body"), schema))
        if (parsingCheck.toLowerCase == "none") {
          parsedDf.select(idCol, "parsed.*")
        } else {
          val successCol = parsingCheck.toLowerCase  match {
            case "full"  => UDFUtils.oldUdf({x: Any => !fullJsonParsingSuccess(x)}, BooleanType)(col("parsed"))
            case "partial" => UDFUtils.oldUdf({x: Any => !partialJsonParsingSuccess(x)}, BooleanType)(col("parsed"))
            case _ => throw new IllegalArgumentException(
              s"Need to use either full, partial, or none. Received $parsingCheck")
          }

          val df1 = parsedDf
            .withColumn("didReply",
              when(successCol,
                ServingUDFs.sendReplyUDF(
                  lit(apiName),
                  ServingUDFs.makeReplyUDF(
                    UDFUtils.oldUdf(jsonParsingError(schema) _, StringType)(col("body")),
                    StringType,
                    code = lit(400), //scalastyle:ignore magic.number
                    reason = lit("JSON Parsing Failure")),
                  col("id")
                )
              )
                .otherwise(lit(null))) //scalastyle:ignore null
            .filter(col("didReply").isNull)

          df1.withColumn("parsed", UDFUtils.oldUdf({ x: Row =>
            x
          }, df1.schema("parsed").dataType)(col("parsed")))
            .select(idCol, "parsed.*")

        }
    }
  }

  def makeReply(replyCol: String, name: String = "reply"): DataFrame = {
    df.withColumn(name, ServingUDFs.makeReplyUDF(col(replyCol), df.schema(replyCol).dataType))
  }

}

object IOImplicits {
  implicit def dsrToDsre(dsr: DataStreamReader): DataStreamReaderExtensions =
    DataStreamReaderExtensions(dsr)

  implicit def dsreToDsr(dsre: DataStreamReaderExtensions): DataStreamReader =
    dsre.dsr

  implicit def dswToDswe[T](dsw: DataStreamWriter[T]): DataStreamWriterExtensions[T] =
    DataStreamWriterExtensions(dsw)

  implicit def dsweToDsw[T](dswe: DataStreamWriterExtensions[T]): DataStreamWriter[T] =
    dswe.dsw

  implicit def dfToDfe[T](df: DataFrame): DataFrameExtensions =
    DataFrameExtensions(df)

  implicit def dfseToDf[T](dfe: DataFrameExtensions): DataFrame =
    dfe.df

  implicit def dfrToDfre(dsr: DataFrameReader): DataFrameReaderExtensions =
    DataFrameReaderExtensions(dsr)

  implicit def dfreToDfr(dsre: DataFrameReaderExtensions): DataFrameReader =
    dsre.dsr

  implicit def dfwToDfwe[T](dsw: DataFrameWriter[T]): DataFrameWriterExtensions[T] =
    DataFrameWriterExtensions(dsw)

  implicit def dfweToDfw[T](dswe: DataFrameWriterExtensions[T]): DataFrameWriter[T] =
    dswe.dsw

}
