// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.text

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.util.MLReadable

import scala.collection.mutable

class MultiNGramSpec extends TransformerFuzzing[MultiNGram] {

  lazy val dfRaw = session
    .createDataFrame(Seq(
      (0, "Hi I"),
      (1, "I wish for snow today"),
      (2, "we Cant go to the park, because of the snow!"),
      (3, ""),
      (4, (1 to 10).map(_.toString).mkString(" "))
    ))
    .toDF("label", "sentence")
  lazy val dfTok = new Tokenizer()
    .setInputCol("sentence")
    .setOutputCol("tokens")
    .transform(dfRaw)

  lazy val t = new MultiNGram()
    .setLengths(Array(1, 3, 4)).setInputCol("tokens").setOutputCol("ngrams")
  lazy val dfNgram = t.transform(dfTok)

  test("operate on tokens ") {
    val grams = dfNgram.collect().last.getAs[mutable.WrappedArray[String]]("ngrams").toSet
    assert(grams("1 2 3 4"))
    assert(grams("4"))
    assert(grams("2 3 4"))
    assert(grams.size == 25)
  }

  override def testObjects(): Seq[TestObject[MultiNGram]] =
    List(new TestObject(t, dfTok))

  override def reader: MLReadable[_] = MultiNGram

}
