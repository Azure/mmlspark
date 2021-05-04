// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class UnicodeNormalizeSuite extends TestBase with TransformerFuzzing[UnicodeNormalize] {
  val inputCol = "words1"
  val outputCol = "norm1"

  //scalastyle:off null
  lazy val wordDF: DataFrame = spark.createDataFrame(Seq(
    ("Schön", 1),
    ("Scho\u0308n", 1),
    (null, 1)))
    .toDF(inputCol, "dummy")

  lazy val expectedResultComposed: DataFrame = spark.createDataFrame(Seq(
    ("Schön", 1, "schön"),
    ("Scho\u0308n", 1, "schön"),
    (null, 1, null)))
    .toDF(inputCol, "dummy", outputCol)

  lazy val expectedResultDecomposed: DataFrame = spark.createDataFrame(Seq(
    ("Schön", 1, "sch\u0308n"),
    ("Scho\u0308n", 1, "sch\u0308n"),
    (null, 1, null)))
    .toDF(inputCol, "dummy", outputCol)
  //scalastyle:on null

  private def testForm(form: String, expected: DataFrame) = {
    val unicodeNormalize = new UnicodeNormalize()
      .setForm(form)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
    val result = unicodeNormalize.transform(wordDF)
    assert(verifyResult(result, expected))
  }

  test("Check for NFC forms") { testForm("NFC", expectedResultComposed) }

  test("Check for NFKC forms") { testForm("NFKC", expectedResultComposed) }

  test("Check for NFD forms") { testForm("NFD", expectedResultDecomposed) }

  test("Check for NFKD forms") { testForm("NFKD", expectedResultDecomposed) }

  def testObjects(): Seq[TestObject[UnicodeNormalize]] = List(new TestObject(
    new UnicodeNormalize().setInputCol("words").setOutputCol("out"), makeBasicDF()))

  override def reader: MLReadable[_] = UnicodeNormalize

}
