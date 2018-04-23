// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import java.sql.Timestamp

import com.microsoft.ml.spark.core.contracts.MMLParams
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/** DataConversion object. */
object DataConversion extends DefaultParamsReadable[DataConversion]

/** Converts the specified list of columns to the specified type.
  * Returns a new DataFrame with the converted columns
  * @param uid The id of the module
  */
class DataConversion(override val uid: String) extends Transformer with MMLParams {
  def this() = this(Identifiable.randomUID("DataConversion"))

  /** Comma separated list of columns whose type will be converted
    * @group param
    */
  val col: Param[String] = StringParam(this, "col",
    "Comma separated list of columns whose type will be converted", "")

  /** @group getParam */
  final def getCol: String = $(col)

  /** @group setParam */
  def setCol(value: String): this.type = set(col, value)

  /** The result type
    * @group param
    */
  val convertTo: Param[String] = StringParam(this, "convertTo", "The result type", "")

  /** @group getParam */
  final def getConvertTo: String = $(convertTo)

  /** @group setParam */
  def setConvertTo(value: String): this.type = set(convertTo, value)

  /** Format for DateTime when making DateTime:String conversions.
    * The default is yyyy-MM-dd HH:mm:ss
    * @group param
    */
  val dateTimeFormat: Param[String] = StringParam(this, "dateTimeFormat",
    "Format for DateTime when making DateTime:String conversions", "yyyy-MM-dd HH:mm:ss")

  /** @group getParam */
  final def getDateTimeFormat: String = $(dateTimeFormat)

  /** @group setParam */
  def setDateTimeFormat(value: String): this.type = set(dateTimeFormat, value)

  /** Apply the <code>DataConversion</code> transform to the dataset
    * @param dataset The dataset to be transformed
    * @return The transformed dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    require($(col) != null, "No column name specified")
    require(dataset != null, "No dataset supplied")
    require(dataset.columns.length != 0, "Dataset with no columns cannot be converted")
    val colsList = $(col).split(",").map(_.trim)
    val errorList = verifyCols(dataset.toDF(), colsList)
    if (errorList.nonEmpty) {
      throw new NoSuchElementException
    }
    var df = dataset.toDF

    val res: DataFrame =  {
      for (convCol <- colsList) {
        df = $(convertTo) match {
          case "boolean" => numericTransform(df, BooleanType, convCol)
          case "byte" => numericTransform(df, ByteType, convCol)
          case "short" => numericTransform(df, ShortType, convCol)
          case "integer" => numericTransform(df, IntegerType, convCol)
          case "long" => numericTransform(df, LongType, convCol)
          case "float" => numericTransform(df, FloatType, convCol)
          case "double" => numericTransform(df, DoubleType, convCol)
          case "string" => numericTransform(df, StringType, convCol)
          case "toCategorical" =>
            val model = new ValueIndexer().setInputCol(convCol).setOutputCol(convCol).fit(df)
            model.transform(df)
          case "clearCategorical" =>
            new IndexToValue().setInputCol(convCol).setOutputCol(convCol).transform(df)
          case "date" => toDateConversion(df, convCol)
        }
      }
      df
    }
    res
  }

  /** Transforms the dataset
    * @param dataset The input dataset, to be transformed
    * @param paramMap ParamMap which contains parameter value to override the default value
    * @return the DataFrame that results from data conversion
    */
  override def transform(dataset: Dataset[_], paramMap: ParamMap): DataFrame = {
    setCol(paramMap.getOrElse(new Param("col", "col","Name of column whose type will be converted"), ""))
    setConvertTo(paramMap.getOrElse(new Param("convertTo", "convertTo","Result type"), ""))
    setDateTimeFormat(paramMap.getOrElse(new Param("dateTimeFormat", "dateTimeFormat", "Time string format"), ""))
    transform(dataset)
  }

  /** Transform the schema
    * @param schema
    * @return modified schema
    */
  def transformSchema(schema: StructType): StructType = {
    System.err.println("transformSchema not implemented yet")
    schema
  }

  /** Copy the class, with extra params
    * @param extra
    * @return
    */
  def copy(extra: ParamMap): DataConversion = defaultCopy(extra)

  /** Convert to a numeric type or a string. If the input type was a TimestampType,
    * tnen do a different conversion?
    */
  private def numericTransform(df: DataFrame, outType: DataType, columnName: String): DataFrame = {
    val inType = df.schema(columnName).dataType
    if (inType == StringType && outType == BooleanType) throw new Exception("String to Boolean not supported")
    val res = inType match {
      case TimestampType => fromDateConversion(df, outType, columnName)
      case _ => df.withColumn(columnName, df(columnName).cast(outType).as(columnName))
    }
    res
  }

  /** Convert a TimestampType to a StringType or a LongType, else error. */
  private def fromDateConversion(df: DataFrame, outType: DataType, columnName: String): DataFrame = {
    require(outType == StringType || outType == LongType, "Date only converts to string or long")
    val res = outType match {
      case LongType =>
        val getTime = udf((t:java.sql.Timestamp)=>t.getTime)
        df.withColumn(columnName, getTime(df(columnName)))
      case StringType =>
        val parseTimeString = udf((t:java.sql.Timestamp)=>{
          val f:java.text.SimpleDateFormat = new java.text.SimpleDateFormat($(dateTimeFormat));f.format(t)})
        df.withColumn(columnName, parseTimeString(df(columnName)))
    }
    res
  }

  private def toDateConversion(df: DataFrame, columnName: String): DataFrame = {
    val inType = df.schema(columnName).dataType
    require(inType == StringType || inType == LongType, "Can only convert string or long to Date")
    val res = inType match {
      case StringType =>
        val f = new java.text.SimpleDateFormat($(dateTimeFormat))
        val parseTimeFromString = udf((t:String)=>{new Timestamp(f.parse(t).getTime)})
        df.withColumn(columnName, parseTimeFromString(df(columnName)).cast("timestamp")).as(columnName)
      case LongType =>
        val longToTimestamp = udf((t:Long)=>{new java.sql.Timestamp(t)})
        df.withColumn(columnName, longToTimestamp(df(columnName)))
    }
    res
  }

  private def verifyCols(df: DataFrame, req: Array[String]): List[String] = {
    req.foldLeft(List[String]()) { (l, r) =>
      if (df.columns.contains(r)) l
      else {
        System.err.println(s"DataFrame does not contain specified column: $r")
        r :: l
      }
    }
  }

}
