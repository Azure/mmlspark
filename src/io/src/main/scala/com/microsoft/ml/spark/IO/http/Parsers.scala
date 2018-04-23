// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.schema.DatasetExtensions.{findUnusedColumnName => newCol}
import com.microsoft.ml.spark.core.serialize.params.{DataTypeParam, MapParam, UDFParam, UDPyFParam}
import org.apache.http.client.methods.HttpRequestBase
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.execution.python.UserDefinedPythonFunction
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DataType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import spray.json.DefaultJsonProtocol._

import scala.reflect.runtime.universe.TypeTag

abstract class HTTPInputParser extends Transformer with HasOutputCol with HasInputCol {
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema.add(getOutputCol, HTTPSchema.request)

}

object JSONInputParser extends ComplexParamsReadable[JSONInputParser]

class JSONInputParser(val uid: String) extends HTTPInputParser with HasURL with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("JSONInputParser"))

  val headers: MapParam[String, String] = new MapParam[String, String](
    this, "headers", "headers of the request")

  /** @group getParam */
  def getHeaders: Map[String, String] = $(headers)

  /** @group setParam */
  def setHeaders(value: Map[String, String]): this.type = set(headers, value)

  setDefault(headers -> Map[String, String]())

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val colsToAvoid = df.schema.fieldNames.toSet ++ Set(getOutputCol)
    val entityCol   = newCol("entity")(colsToAvoid)
    val urlCol      = newCol("url")(colsToAvoid)
    val headersCol  = newCol("headers")(colsToAvoid)
    val requestCol  = newCol("request")(colsToAvoid)
    val methodCol   = newCol("method")(colsToAvoid)

    val headers = getHeaders.toArray.map(x =>
        HeaderData(x._1, x._2)) ++ Array(HeaderData("Content-type", "application/json"))

    df.withColumn(entityCol, df.schema(getInputCol).dataType match {
      case _: StructType => to_json(col(getInputCol))
      case _: ArrayType  => to_json(col(getInputCol))
      case _             => to_json(struct(getInputCol))
    }).withColumn(urlCol, lit(getUrl))
      .withColumn(headersCol, typedLit(headers))
      .withColumn(methodCol, lit("POST"))
      .withColumn(requestCol,
                  HTTPSchema.to_http(urlCol, headersCol, methodCol, entityCol))
      .drop(entityCol, urlCol, headersCol, methodCol)
      .withColumnRenamed(requestCol, getOutputCol)
  }

}

object CustomInputParser extends ComplexParamsReadable[CustomInputParser]

class CustomInputParser(val uid: String) extends HTTPInputParser with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("CustomInputParser"))

  val udfScala = new UDFParam(
      this, "udfScala", "User Defined Function to be applied to the DF input col",
      { x: UserDefinedFunction => x.dataType == HTTPSchema.request })

  val udfPython = new UDPyFParam(
      this, "udfPython", "User Defined Python Function to be applied to the DF input col",
      { x: UserDefinedPythonFunction => x.dataType == HTTPSchema.request })

  val udfParams = Seq(udfScala, udfPython)

  /** @group getParam */
  def getUDF: UserDefinedFunction = $(udfScala)

  /** @group getParam */
  def getUDPyF: UserDefinedPythonFunction = $(udfPython)

  /** @group setParam */
  def setUDF(value: UserDefinedFunction): this.type = {
    udfParams.filter(isSet).foreach(clear)
    set(udfScala, value)
  }

  /** @group setParam */
  def setUDF(value: UserDefinedPythonFunction): this.type = {
    udfParams.filter(isSet).foreach(clear)
    set(udfPython, value)
  }

  def setUDF[T](f: T => HttpRequestBase): this.type = {
    setUDF(udf({ x: T => new HTTPRequestData(f(x)) }, HTTPSchema.request))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val parseInputExpression = {
      (get(udfScala), get(udfPython)) match {
        case (Some(f), None) => f(col(getInputCol))
        case (None, Some(f)) => f(col(getInputCol))
        case _ => throw new IllegalArgumentException("Need to set either parseInput or parseInputPy")
      }
    }
    dataset.toDF().withColumn(getOutputCol, parseInputExpression)
  }

}

abstract class HTTPOutputParser extends Transformer with HasInputCol with HasOutputCol {
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}

object JSONOutputParser extends ComplexParamsReadable[JSONOutputParser]

class JSONOutputParser(val uid: String) extends HTTPOutputParser with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("JSONOutputParser"))

  val dataType: Param[DataType] = new DataTypeParam(
    this, "dataType", "format to parse the column to")

  /** @group getParam */
  def getDataType: DataType = $(dataType)

  /** @group setParam */
  def setDataType(value: DataType): this.type = set(dataType, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val stringEntityCol = HTTPSchema.entityToStringUDF(col(getInputCol + ".entity"))
    dataset.toDF
      .withColumn(getOutputCol, from_json(stringEntityCol, getDataType))
  }

  override def transformSchema(schema: StructType): StructType = {
    assert(schema(getInputCol).dataType == HTTPSchema.response)
    schema.add(getOutputCol, getDataType)
  }

}

object CustomOutputParser extends ComplexParamsReadable[CustomOutputParser]

class CustomOutputParser(val uid: String) extends HTTPOutputParser with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("CustomOutputParser"))

  val udfScala = new UDFParam(this, "udfScala", "User Defined Function to be applied to the DF input col")

  val udfPython = new UDPyFParam(
      this, "udfPython", "User Defined Python Function to be applied to the DF input col")

  val udfParams = Seq(udfScala, udfPython)

  /** @group getParam */
  def getUDF: UserDefinedFunction = $(udfScala)

  /** @group getParam */
  def getUDPyF: UserDefinedPythonFunction = $(udfPython)

  /** @group setParam */
  def setUDF(value: UserDefinedFunction): this.type = {
    udfParams.filter(isSet).foreach(clear)
    set(udfScala, value)
  }

  /** @group setParam */
  def setUDF(value: UserDefinedPythonFunction): this.type = {
    udfParams.filter(isSet).foreach(clear)
    set(udfPython, value)
  }

  def setUDF[T: TypeTag](f: HTTPResponseData => T): this.type = {
    setUDF(udf({ x: Row => f(HTTPResponseData.fromRow(x)) }))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val parseOutputExpression = {
      (get(udfScala), get(udfPython)) match {
        case (Some(f), None) => f(col(getInputCol))
        case (None, Some(f)) => f(col(getInputCol))
        case _ => throw new IllegalArgumentException("Need to set either parseOutput or parseOutputPy")
      }
    }
    dataset.toDF().withColumn(getOutputCol, parseOutputExpression)
  }

  override def transformSchema(schema: StructType): StructType = {
    assert(schema(getInputCol).dataType == HTTPSchema.response)
    schema.add(getOutputCol, {
      (get(udfScala), get(udfPython)) match {
        case (Some(f), None) => f.dataType
        case (None, Some(f)) => f.dataType
        case _ => throw new IllegalArgumentException("Need to set either parseOutput or parseOutputPy")
      }
    })
  }

}
