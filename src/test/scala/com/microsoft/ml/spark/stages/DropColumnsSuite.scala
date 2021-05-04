// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.codegen.CodegenConfig
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{PyTestFuzzing, TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable

class DropColumnsSuite extends TestBase with TransformerFuzzing[DropColumns] {

  import spark.implicits._

  test("Drop no columns in a data frame") {
    val input = makeBasicDF()
    val result = new DropColumns()
      .setCols(Array())
      .transform(input)
    assert(verifyResult(input, result))
  }

  test("Drop all but two columns in a data frame") {
    val keep = Set("words", "more")
    val input = makeBasicDF()
    val expected = Seq(
      ("guitars", "drums"),
      ("piano", "trumpet"),
      ("bass", "cymbals")
    ).toDF("words", "more")
    val result = new DropColumns()
      .setCols(input.columns.filterNot(keep.contains))
      .transform(makeBasicDF())
    assert(verifyResult(expected, result))
  }

  test("Drop columns with spaces") {
    val result = new DropColumns()
      .setCols(Array("Scored Labels"))
      .transform(makeBasicDF().withColumnRenamed("more", "Scored Labels"))
    assert(verifyResult(makeBasicDF().drop("more"), result))
  }

  test("Invalid column specified") {
    try {
      new DropColumns().setCol("four").transform(makeBasicDF())
      fail()
    } catch {
      case _: NoSuchElementException =>
    }
  }

  def testObjects(): Seq[TestObject[DropColumns]] = List(new TestObject(
    new DropColumns().setCol("numbers"), makeBasicDF()))

  override def reader: MLReadable[_] = DropColumns

}
