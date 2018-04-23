// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.scalatest.Assertion

class UDFTransformerSuite extends TestBase with TransformerFuzzing[UDFTransformer] {
  val baseDF: DataFrame = makeBasicDF()
  val baseNullableDF: DataFrame = makeBasicNullableDF()
  val outCol = "out"
  val stringToIntegerUDF: UserDefinedFunction = udf((_: String) => 1)

  def udfTransformerTest(baseDF: DataFrame,
                         testUDF: UserDefinedFunction,
                         inputCol: String,
                         outputCol: String = outCol): Assertion = {
    val result = new UDFTransformer().setUDF(testUDF)
      .setInputCol(inputCol).setOutputCol(outputCol)
      .transform(baseDF)
    val expected = baseDF.withColumn(outputCol, testUDF(baseDF(inputCol)))
    assert(verifyResult(expected, result))
  }

  def udfMultiColTransformerTest(baseDF: DataFrame,
                         testUDF: UserDefinedFunction,
                         inputCols: Array[String],
                         outputCol: String = outCol): Assertion = {
    val result = new UDFTransformer().setUDF(testUDF)
      .setInputCols(inputCols).setOutputCol(outputCol)
      .transform(baseDF)
    val expected = baseDF.withColumn(outputCol, testUDF(inputCols.map(baseDF(_)): _*))
    assert(verifyResult(expected, result))
  }

  test("Apply udf to column in a data frame: String to Integer") {
    val inColName = "words"
    udfTransformerTest(baseDF, stringToIntegerUDF, inColName)
    udfTransformerTest(baseNullableDF, stringToIntegerUDF, inColName)
  }

  test("Apply udf to column in a data frame with existing output col") {
    val inColName = "words"
    val outColName = "numbers"
    udfTransformerTest(baseDF, stringToIntegerUDF, inColName, outColName)
    udfTransformerTest(baseNullableDF, stringToIntegerUDF, inColName, outColName)
  }

  test("Apply udf to column in a data frame: Integer to String") {
    val intToStringUDF = udf((_: Integer) => "test")
    val inColName = "numbers"
    udfTransformerTest(baseDF, intToStringUDF, inColName)
    udfTransformerTest(baseNullableDF, intToStringUDF, inColName)
  }

  test("Apply udf to column in a data frame: Long to String") {
    val longToStringUDF = udf((_: Long) => "test")
    val inColName = "longs"
    udfTransformerTest(baseDF, longToStringUDF, inColName)
    udfTransformerTest(baseNullableDF, longToStringUDF, inColName)
  }

  test("Apply udf to column in a data frame: Double to String") {
    val doubleToStringUDF = udf((_: Double) => "test")
    val inColName = "doubles"
    udfTransformerTest(baseDF, doubleToStringUDF, inColName)
    udfTransformerTest(baseNullableDF, doubleToStringUDF, inColName)
  }

  test("Apply udf to columns in a data frame: (Double, Long) -> String") {
    val doubleToStringUDF = udf((_: Double, _: Long) => "test")
    val inColNames = Array("doubles", "longs")
    udfMultiColTransformerTest(baseDF, doubleToStringUDF, inColNames)
    udfMultiColTransformerTest(baseNullableDF, doubleToStringUDF, inColNames)
  }

  test("Apply udf to column in a data frame: Boolean to String") {
    val booleanToStringUDF = udf((_: Boolean) => "test")
    val inColName = "booleans"
    udfTransformerTest(baseDF, booleanToStringUDF, inColName)
    udfTransformerTest(baseNullableDF, booleanToStringUDF, inColName)
  }

  test("Apply inputCols after inputCol error") {
    assertThrows[IllegalArgumentException] {
      val inColName = "doubles"
      val inColNames = Array("doubles", "longs")
      new UDFTransformer().setInputCol(inColName).setInputCols(inColNames)
    }
  }

  test("Apply inputCol after inputCols error") {
    assertThrows[IllegalArgumentException] {
      val inColName = "doubles"
      val inColNames = Array("doubles", "longs")
      new UDFTransformer().setInputCols(inColNames).setInputCol(inColName)
    }
  }

  def testObjects(): Seq[TestObject[UDFTransformer]] = {
    List(new TestObject(
      new UDFTransformer().setUDF(stringToIntegerUDF)
        .setInputCol("numbers").setOutputCol(outCol),
      baseDF))
  }

  override def reader: MLReadable[_] = UDFTransformer

}
