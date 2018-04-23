// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.ml

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.schema.DatasetExtensions._
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector

class HashingTFSpec extends TestBase {

  test("operation on tokenized strings") {
    val wordDataFrame = session.createDataFrame(Seq(
      (0, Array("Hi", "I", "can", "not", "foo", "foo")),
      (1, Array("I")),
      (2, Array("Logistic", "regression")),
      (3, Array("Log", "f", "reg"))
    )).toDF("label", "words")

    val hashDF = new HashingTF().setInputCol("words").setOutputCol("hashedTF").transform(wordDataFrame)
    val lines = hashDF.getSVCol("hashedTF")

    val trueLines = List(
      new SparseVector(262144, Array(36073,51654,113890,139098,242088), Array(1.0,2.0,1.0,1.0,1.0)),
      new SparseVector(262144, Array(113890), Array(1.0)),
      new SparseVector(262144, Array(13671,142455), Array(1.0,1.0)),
      new SparseVector(262144, Array(24152,74466,122984), Array(1.0,1.0,1.0))
    )
    assert(lines === trueLines)
  }

  test("support several values for number of features") {
    val featureSizes = List(1, 5, 100, 100000)
    val words = Array("Hi", "I", "can", "not", "foo", "bar", "foo", "afk")
    val wordDataFrame = session.createDataFrame(Seq((0, words))).toDF("label", "words")

    val fsResults = featureSizes.map { n =>
          new HashingTF()
            .setNumFeatures(n)
            .setInputCol("words")
            .setOutputCol("hashedTF")
            .transform(wordDataFrame)
            .getSVCol("hashedTF")(0)
      }
    val trueResults = Array(
      new SparseVector(     1, Array(0), Array(8.0)),
      new SparseVector(     5, Array(0,2,3), Array(4.0,2.0,2.0)),
      new SparseVector(   100, Array(0,10,18,33,62,67,80), Array(1.0,2.0,1.0,1.0,1.0,1.0,1.0)),
      new SparseVector(100000, Array(5833,9467,16680,29018,68900,85762,97510), Array(1.0,1.0,1.0,1.0,1.0,1.0,2.0))
    )
    assert(fsResults === trueResults)
  }

  test("treat empty strings as another word") {
    val wordDataFrame = session.createDataFrame(Seq(
      (0, "hey you no way"),
      (1, "")))
      .toDF("label", "sentence")

    val tokenized = new Tokenizer().setInputCol("sentence").setOutputCol("tokens").transform(wordDataFrame)
    val hashDF = new HashingTF().setInputCol("tokens").setOutputCol("HashedTF").transform(tokenized)

    val lines = hashDF.getSVCol("hashedTF")
      assert(lines(1) === new SparseVector(262144, Array(249180), Array(1.0)))
  }

  test("raise an error when applied to a null array") {
    val tokenDataFrame = session.createDataFrame(Seq(
      (0, Some(Array("Hi", "I", "can", "not", "foo"))),
      (1, None))
    ).toDF("label", "tokens")
    assertSparkException[org.apache.spark.SparkException](new HashingTF().setInputCol("tokens"), tokenDataFrame)
  }

  test("raise an error when given strange values of n") {
    List(0, -1, -10).foreach { n =>
      intercept[IllegalArgumentException] { new HashingTF().setNumFeatures(n) }
    }
  }

}
