// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.ml

import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.ml.feature.{NGram, Tokenizer}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

class NGramSpec extends TestBase {

  def ngramDFToScalaList(dataFrame: DataFrame, outputCol: String = "ngrams"): Array[List[Any]] = {
    dataFrame.select(dataFrame(outputCol)).collect()
      .map(_.getAs[mutable.WrappedArray[Any]](0).toList)
  }

  test("operation on tokenized strings") {
    val wordDataFrame = session.createDataFrame(Seq((0, Array("Hi", "I", "can", "not", "foo")),
                                                    (1, Array("I")),
                                                    (2, Array("Logistic", "regression")),
                                                    (3, Array("Log", "f", "reg"))))
      .toDF("label", "words")

    val ngramDF = new NGram().setN(3)
      .setInputCol("words").setOutputCol("ngrams")
      .transform(wordDataFrame)
    val ngrams = ngramDFToScalaList(ngramDF)
    assert(ngrams(0) === Array("Hi I can", "I can not", "can not foo"))
    assert(ngrams(1) === Array())
    assert(ngrams(2) === Array())
    assert(ngrams(3) === Array("Log f reg"))
  }

  test("supporting several values for n") {
    val ns = 1 to 6
    val words = Array("Hi", "I", "can", "not", "foo", "bar", "foo", "afk")
    val wordDataFrame = session.createDataFrame(Seq((0, words))).toDF("label", "words")
    val nGramResults = ns.map { n =>
      ngramDFToScalaList(
        new NGram().setN(n)
          .setInputCol("words").setOutputCol("ngrams")
          .transform(wordDataFrame))
      }
    ns.foreach { n =>
      assert(nGramResults(n-1)(0).head === words.take(n).mkString(" "))
    }
  }

  test("handling empty strings gracefully") {
    val wordDataFrame = session.createDataFrame(Seq((0, "hey you no way"),
                                                    (1, "")))
      .toDF("label", "sentence")

    val tokenized = new Tokenizer().setInputCol("sentence").setOutputCol("tokens").transform(wordDataFrame)
    val ngrams = new NGram().setInputCol("tokens").setOutputCol("ngrams").transform(tokenized)
    assert(ngramDFToScalaList(ngrams)(1) === Nil)
  }

  test("raise an error when applied to a null array") {
    val tokenDataFrame = session.createDataFrame(Seq(
      (0, Some(Array("Hi", "I", "can", "not", "foo"))),
      (1, None))
    ).toDF("label", "tokens")
    assertSparkException[org.apache.spark.SparkException](new NGram().setInputCol("tokens"), tokenDataFrame)
  }

  test("raise an error when given strange values of n") {
    List(0, -1, -10).foreach { n =>
      intercept[IllegalArgumentException] { new NGram().setN(n) }
    }
  }

}
