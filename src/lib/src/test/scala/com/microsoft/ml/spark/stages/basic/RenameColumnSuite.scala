// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable

class RenameColumnSuite extends TestBase with TransformerFuzzing[RenameColumn] {

  test("Rename columns in a data frame") {
    val base = makeBasicDF()
    val result = new RenameColumn().setInputCol("words").setOutputCol("out").transform(base)
    val expected = base.withColumnRenamed("words", "out")
    assert(verifyResult(expected, result))
  }

  test("Rename columns with outputColumn as existing column") {
    val base = makeBasicDF()
    val result = new RenameColumn().setInputCol("words").setOutputCol("numbers").transform(base)
    val expected = base.withColumnRenamed("words", "numbers")
    assert(verifyResult(expected, result))
  }

  def testObjects(): Seq[TestObject[RenameColumn]] = List(new TestObject(
    new RenameColumn().setInputCol("numbers").setOutputCol("out"), makeBasicDF()))

  override def reader: MLReadable[_] = RenameColumn

}
