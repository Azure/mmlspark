// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable

class SummarizeDataSuite extends TransformerFuzzing[SummarizeData] {

  test("Smoke test for summarizing basic DF - schema transform") {

    val input = makeBasicDF()
    val summary = new SummarizeData()
    val result = summary.transformSchema(input.schema)
    assert(result.length > 10)
  }

  test("Smoke test for summary params") {
    val s = new SummarizeData()
    assert(s.params.length == 5)
    assert(s.params.map(s.isSet).toSeq == (1 to s.params.length).map(i => false))

    val sNoCounts = s.setCounts(false).setPercentiles(false)
    assert(sNoCounts.params.map(sNoCounts.isSet).toSeq === Seq(false, true, false, true, false))
  }

  test("Smoke test for summarizing basic DF") {
    val input = makeBasicDF()
    val summary = new SummarizeData()
    val result = summary.transform(input)
    assert(result.count === input.columns.length)
    assert(result.columns.length > 18)
  }

  test("Smoke test for summarizing missings DF") {
    val input = makeBasicNullableDF()
    val summary = new SummarizeData()
    val result = summary.transform(input)
    assert(result.count === input.columns.length)
    assert(result.columns.length > 18)
  }

  test("Smoke test for subset summarizing missings DF") {
    val input = makeBasicNullableDF()
    val summary = new SummarizeData().setPercentiles(false).setCounts(false)
    val result = summary.transform(input)
    assert(result.count === input.columns.length)
    assert(result.columns.length < 11)
  }

  override def testObjects(): Seq[TestObject[SummarizeData]] = Seq(new TestObject(
    new SummarizeData(),
    makeBasicDF()
  ))

  override def reader: MLReadable[_] = SummarizeData

}
