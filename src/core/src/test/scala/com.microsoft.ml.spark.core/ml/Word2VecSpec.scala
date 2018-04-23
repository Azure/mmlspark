// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.ml

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.schema.DatasetExtensions._
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame

class Word2VecSpec extends TestBase {

  def genTokenizedText(): DataFrame = {
    session.createDataFrame(Seq(
      (0, Array("I", "walked", "the", "dog", "down", "the", "street")),
      (1, Array("I", "walked", "with", "the", "dog")),
      (2, Array("I", "walked", "the", "pup"))
    )).toDF("label", "words")
  }

  def genW2V(): Word2Vec = new Word2Vec().setSeed(1234).setMinCount(0)

  test("operation on tokenized strings") {
    val df = genTokenizedText()

    val df2 = genW2V().setVectorSize(2)
      .setInputCol("words").setOutputCol("features").fit(df).transform(df)

    val lines = df2.getDVCol("features")
    assert(lines.forall(_.size == 2))
  }

  test("return vectors") {
    val df = genTokenizedText()
    val model = genW2V().setVectorSize(2)
      .setInputCol("words").setOutputCol("features").fit(df)
    val vectors = model.getVectors.getDVCol("vector")
    assert(vectors(0).size == 2)
  }

  test("return synonyms") {
    val df = genTokenizedText()
    val model = genW2V().setVectorSize(2)
      .setInputCol("words").setOutputCol("features").fit(df)
    val synonyms = model.findSynonyms("dog", 2).getColAs[String]("word")
    assert(synonyms.length === 2)
  }

  test("raise an error when applied to a null array") {
    val tokenDataFrame = session.createDataFrame(Seq(
      (0, Some(Array("Hi", "I", "can", "not", "foo"))),
      (1, None))
    ).toDF("label", "tokens")
    assertSparkException[org.apache.spark.SparkException](genW2V().setInputCol("tokens"), tokenDataFrame)
  }

  test("raise an error when given strange values of parameters") {
    def base(): Word2Vec = genW2V().setInputCol("words")
    def assertIllegalArgument[T](f: T => Any, args: T*): Unit =
      args.foreach { n => interceptWithoutLogging[IllegalArgumentException] { f(n) } }
    assertIllegalArgument[Int](base.setMinCount,             -1, -10)
    assertIllegalArgument[Int](base.setMaxIter,              -1, -10)
    assertIllegalArgument[Int](base.setVectorSize,        0, -1, -10)
    assertIllegalArgument[Int](base.setWindowSize,        0, -1, -10)
    assertIllegalArgument[Int](base.setMaxSentenceLength, 0, -1, -10)
    assertIllegalArgument[Int](base.setNumPartitions,     0, -1, -10)
    assertIllegalArgument[Double](base.setStepSize, 0.0, -1.0, -10.0)
  }

  test("return a vector of zeros when it encounters an OOV word") {
    val df = genTokenizedText()
    val model = genW2V().setVectorSize(2).setMinCount(1).setInputCol("words").setOutputCol("features").fit(df)
    val df2 = session.createDataFrame(Seq(
      (0, Array("ketchup")))).toDF("label", "words")
    val results = model.transform(df2)
    val lines = results.getDVCol("features")
    val trueLines = List(new DenseVector(Array(0.0, 0.0)))
    assert(lines === trueLines)
  }

  test("be able to set vector size") {
    val df = genTokenizedText()
    val vectorSizes = List(1, 10, 100)
    vectorSizes.foreach { n =>
      val results =
          genW2V().setVectorSize(n)
            .setInputCol("words").setOutputCol("features").fit(df).transform(df)
            .getDVCol("features")
        assert(results(0).size === n)
    }
  }

}
