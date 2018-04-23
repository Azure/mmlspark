// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable

class SelectColumnsSuite extends TestBase with TransformerFuzzing[SelectColumns] {

  import session.implicits._

  test("Select all columns in a data frame") {
    val input = makeBasicDF()
    val result = new SelectColumns()
      .setCols(input.columns)
      .transform(input)
    assert(verifyResult(input, result))
  }

  test("Test: Select two columns in a data frame") {
    val expected = Seq(
      ("guitars", "drums"),
      ("piano", "trumpet"),
      ("bass", "cymbals")
    ).toDF("words", "more")
    val result = new SelectColumns()
      .setCols(Array("words", "more"))
      .transform(makeBasicDF())
    assert(verifyResult(expected, result))
  }

  test("Test: Select columns with spaces") {
    val expected = Seq(
      ("guitars", "drums"),
      ("piano", "trumpet"),
      ("bass", "cymbals")
    ).toDF("words", "Scored Labels")
    val result = new SelectColumns()
      .setCols(Array("words", "Scored Labels"))
      .transform(makeBasicDF().withColumnRenamed("more", "Scored Labels"))
    assert(verifyResult(expected, result))
  }

  test("Test: Select one column from the data frame") {
    val expected = Seq(
      "guitars",
      "piano",
      "bass"
    ).toDF("words")
    val result = new SelectColumns()
      .setCols(Array("words"))
      .transform(makeBasicDF())
    assert(verifyResult(expected, result))
  }

  test("Invalid column specified") {
    try {
      new SelectColumns().setCol("four").transform(makeBasicDF())
      fail()
    } catch {
      case _: NoSuchElementException =>
    }
  }

  def testObjects(): Seq[TestObject[SelectColumns]] = List(new TestObject(
    new SelectColumns().setCol("numbers"), makeBasicDF()))

  override def reader: MLReadable[_] = SelectColumns

}
