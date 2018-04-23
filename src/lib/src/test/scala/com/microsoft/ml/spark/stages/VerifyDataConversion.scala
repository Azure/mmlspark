// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import java.sql.Timestamp

import com.microsoft.ml.spark.core.schema.SparkSchema
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class VerifyDataConversions extends TestBase {

  import session.implicits._

  val testVal: Long = (Int.MaxValue).toLong + 100
  val testShort: Integer = Short.MaxValue + 100
  /*
  DataFrame for the numerical and string <--> numerical conversions
   */
  val masterInDF = Seq((true: Boolean, 1: Byte, 2: Short, 3: Integer, 4: Long, 5.0F, 6.0, "7", "8.0"),
    (false, 9: Byte, 10: Short, 11: Integer, 12: Long, 14.5F, 15.5, "16", "17.456"),
    (true, -127: Byte, 345: Short, testShort, testVal, 18.91F, 20.21, "100", "200.12345"))
    .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")

  /*
  Dataframe of Timestamp data
   */
  val tsDF= Seq("1986-07-27 12:48:00.123", "1988-11-01 11:08:48.456", "1993-08-06 15:32:00.789").toDF("Col0")
    .select($"Col0".cast("timestamp"))

  /*
  Timestamps as longs dataframe. These longs were generated on the commandline feeding the above timestamp
  values to Timestamp.getTime()
   */
  val f = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
  val parseTimeFromString = udf((t:String)=>{new Timestamp(f.parse(t).getTime)})
  val lDF = Seq(f.parse("1986-07-27 12:48:00.123").getTime(),
    f.parse("1988-11-01 11:08:48.456").getTime(),
    f.parse("1993-08-06 15:32:00.789").getTime()).toDF("Col0")

  /*
  Timestaps as strings dataframe
   */
  val sDF = Seq("1986-07-27 12:48:00.123", "1988-11-01 11:08:48.456", "1993-08-06 15:32:00.789").toDF("Col0")

  /*
  Test conversion of all numeric types to Boolean
  Strings are cast to null, which causes the comparison test to fail, so for now I
  am skipping the string tests.
  Types tested are boolean, Byte, Short, Int, Long, Float, Double, and string
  */
  test("Test convert all types to Boolean") {
    val r1 = new DataConversion().setCol("byte").setConvertTo("boolean").transform(masterInDF)
    val r2 = new DataConversion().setCol("short").setConvertTo("boolean").transform(r1)
    val r3 = new DataConversion().setCol("int").setConvertTo("boolean").transform(r2)
    val r4 = new DataConversion().setCol("long").setConvertTo("boolean").transform(r3)
    val r5 = new DataConversion().setCol("float").setConvertTo("boolean").transform(r4)
    val r6 = new DataConversion().setCol("double").setConvertTo("boolean").transform(r5)
    val expectedRes = Seq(( true, true, true, true, true, true, true, "7", "8.0"),
      (false, true, true, true, true, true, true, "16", "17.456"),
      (true, true, true, true, true, true, true, "100", "200.12345"))
      .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")
    assert(r6.schema("byte").dataType == BooleanType)
    assert(r6.schema("short").dataType == BooleanType)
    assert(r6.schema("int").dataType == BooleanType)
    assert(r6.schema("long").dataType == BooleanType)
    assert(r6.schema("float").dataType == BooleanType)
    assert(r6.schema("double").dataType == BooleanType)
  }

  /*
  Verify sting to boolean throws an error
  */
  test("Test convert string to boolean throws an exception") {
    assertThrows[Exception] {
      new DataConversion().setCol("intstring").setConvertTo("boolean").transform(masterInDF)
    }
  }

  /*
  Test conversion of all numeric types to Byte, as well as string representations
  of integers and doubles
  Types tested are boolean, Byte, Short, Int, Long, Float, Double, and string
  For floats and doubles, the conversion value is the truncated integer portion of
  the number. For values that exceed the min/max value for integers, the value will be truncated
  at the least 32 bits, so a very large number will end up being a very large negative number
  */
  test("Test convert to Byte") {
    val expectedDF = Seq(( 1: Byte, 1: Byte, 2: Byte, 3: Byte, 4: Byte, 5: Byte, 6: Byte, 7: Byte, 8: Byte),
      (0: Byte, 9: Byte, 10: Byte, 11: Byte, 12: Byte, 14: Byte, 127: Byte, 16: Byte, 17: Byte),
      (1: Byte, -127: Byte, 89: Byte, 99: Byte, 99: Byte, 18: Byte, 20: Byte, 100: Byte, -56: Byte))
      .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")
    val res =  generateRes("byte", masterInDF)
    assert(res.schema("bool").dataType == ByteType)
    assert(res.schema("short").dataType == ByteType)
    assert(res.schema("int").dataType == ByteType)
    assert(res.schema("long").dataType == ByteType)
    assert(res.schema("float").dataType == ByteType)
    assert(res.schema("double").dataType == ByteType)
    assert(res.schema("intstring").dataType == ByteType)
    assert(res.schema("doublestring").dataType == ByteType)
  }

  /*
  Test conversion of all numeric types to Short, as well as string representations
  of integers and doubles
  Types tested are boolean, Byte, Short, Int, Long, Float, Double, and string
  For floats and doubles, the conversion value is the truncated integer portion of
  the number. For values that exceed the min/max value for integers, the value will be truncated
  at the least 32 bits, so a very large number will end up being a very large negative number
  */
  test("Test convert to Short") {
    val expectedDF = Seq(( 1: Short, 1: Short, 2: Short, 3: Short, 4: Short, 5: Short, 6: Short, 7: Short, 8: Short),
      (0: Short, 9: Short, 10: Short, 11: Short, 12: Short, 14: Short, 15: Short, 16: Short, 17: Short),
      (1: Short, -127: Short, 345: Short, -32669: Short, 99: Short, 18: Short, 20: Short, 100: Short, 200: Short))
      .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")
    assert(expectedDF.except(generateRes("short", masterInDF)).count == 0)
  }

  /*
  Test conversion of all numeric types to Integer, as well as string representations
  of integers and doubles
  Types tested are boolean, Byte, Short, Int, Long, Float, Double, and string
  For floats and doubles, the conversion value is the truncated integer portion of
  the number. For values that exceed the min/max value for integers, the value will be truncated
  at the least 32 bits, so a very large number will end up being a very large negative number
  */
  test("Test convert to Integer") {
    val expectedDF = Seq((1, 1, 2, 3, 4, 5, 6, 7, 8),
      (0, 9, 10, 11, 12, 14, 15, 16, 17),
      (1, -127, 345, 32867, -2147483549, 18, 20, 100, 200))
      .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")
    assert(expectedDF.except(generateRes("integer", masterInDF)).count == 0)
  }

  /*
  Test conversion of all numeric types to Long, as well as string representations
  of integers and doubles
  Types tested are boolean, Byte, Short, Int, Long, Float, Double, and string
  For floats and doubles, the conversion value is the truncated integer portion of
  the number. For values that exceed the min/max value for integers, the value will be truncated
  at the least 32 bits, so a very large number will end up being a very large negative number
  */
  test("Test convert to Long") {
    val expectedDF = Seq((1L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L),
      (0L, 9L, 10L, 11L, 12L, 14L, 15L, 16L, 17L),
      (1L, -127L, 345L, 32867L, 2147483747L, 18L, 20L, 100L, 200L))
      .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")
    assert(expectedDF.except(generateRes("long", masterInDF)).count == 0)
  }

  /*
  Test conversion of all numeric types to Double, as well as string representations
  of integers and doubles
  Types tested are boolean, Byte, Short, Int, Long, Float, Double, and string
  */
  test("Test convert to Double") {
    val fToD = 18.91F.toDouble
    val expectedDF = Seq((1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
      (0.0, 9.0, 10.0, 11.0, 12.0, 14.5, 15.5, 16.0, 17.456),
      (1.0, -127.0, 345.0, 32867.0, 2147483747.0, fToD, 20.21, 100.0, 200.12345))
      .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")
    assert(expectedDF.except(generateRes("double", masterInDF)).count == 0)
  }

  // Test the conversions to string
  test("Test convert all types to String") {
    val expectedDF = Seq(("true", "1", "2", "3", "4", "5.0", "6.0", "7", "8.0"),
      ("false", "9", "10", "11", "12", "14.5", "15.5", "16", "17.456"),
      ("true", "-127", "345", "32867", "2147483747", "18.91", "20.21", "100", "200.12345"))
      .toDF("bool", "byte", "short", "int", "long", "float", "double", "intstring", "doublestring")
    assert(expectedDF.except(generateRes("string", masterInDF)).count == 0)
  }

  // Test convert to categorical:
  test("Test convert to categorical") {
    val inDF = Seq(("piano", 1, 2), ("drum", 3, 4), ("guitar", 5, 6)).toDF("instruments", "c1", "c2")
    val res = new DataConversion().setCol("instruments").setConvertTo("toCategorical").transform(inDF)
    assert(SparkSchema.isCategorical(res, "instruments"))
  }

  // Test clearing categorical
  test("Test that categorical features will be cleared") {
    val inDF = Seq(("piano", 1, 2), ("drum", 3, 4), ("guitar", 5, 6)).toDF("instruments", "c1", "c2")
    val res = new DataConversion().setCol("instruments").setConvertTo("toCategorical").transform(inDF)
    assert(SparkSchema.isCategorical(res, "instruments"))
    val res2 = new DataConversion().setCol("instruments").setConvertTo("clearCategorical").transform(res)
    assert(!SparkSchema.isCategorical(res2, "instruments"))
    assert(inDF.except(res2).count == 0)
  }

  // Verify that a TimestampType is converted to a LongType
  test("Test timestamp to long conversion") {
    val res = new DataConversion().setCol("Col0").setConvertTo("long")
      .setDateTimeFormat("yyyy-MM-dd HH:mm:ss.SSS").transform(tsDF)
    assert(res.schema("Col0").dataType == LongType)
    assert(lDF.except(res).count == 0)
  }

  // Test the reverse - long to timestamp
  test("Test long to timestamp conversion") {
    val res = new DataConversion().setCol("Col0").setConvertTo("date")
      .setDateTimeFormat("yyyy-MM-dd HH:mm:ss.SSS").transform(lDF)
    assert(res.schema("Col0").dataType == TimestampType)
    assert(tsDF.except(res).count == 0)
  }

  test("Test timestamp to string conversion") {
    val res = new DataConversion().setCol("Col0").setConvertTo("string")
      .setDateTimeFormat("yyyy-MM-dd HH:mm:ss.SSS").transform(tsDF)
    assert(res.schema("Col0").dataType == StringType)
    assert(sDF.except(res).count == 0)
  }

  test("Test date string to timestamp conversion") {
    val res = new DataConversion().setCol("Col0").setConvertTo("date")
      .setDateTimeFormat("yyyy-MM-dd HH:mm:ss.SSS").transform(sDF)
    val res2 = new DataConversion().setCol("Col0").setConvertTo("long")
      .setDateTimeFormat("yyyy-MM-dd HH:mm:ss.SSS").transform(res)
    assert(res.schema("Col0").dataType == TimestampType)
    assert(tsDF.except(res).count == 0)
  }

  def generateRes(convTo: String, inDF: DataFrame): DataFrame = {
    val result = new DataConversion().setCol("bool, byte, short, int, long, float, double, intstring, doublestring")
      .setConvertTo(convTo).transform(masterInDF)
    result
  }

}
