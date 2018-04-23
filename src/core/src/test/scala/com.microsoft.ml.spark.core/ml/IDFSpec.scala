// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.ml

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.schema.DatasetExtensions._
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}

class IDFSpec extends TestBase {

  test("operation on hashingTF output") {
    val sentenceData = session.createDataFrame(Seq((0, "Hi I"),
                                                   (1, "I wish"),
                                                   (2, "we Cant")))
      .toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    val lines = rescaledData.getSVCol("features")
    val trueLines = List(
      new SparseVector(20, Array(0,  9), Array(0.6931471805599453, 0.28768207245178085)),
      new SparseVector(20, Array(9, 15), Array(0.28768207245178085, 0.6931471805599453)),
      new SparseVector(20, Array(6, 13), Array(0.6931471805599453, 0.6931471805599453))
    )
    assert(lines === trueLines)
  }

  test("operation on dense or sparse vectors") {
    val denseVects = Seq((0, new DenseVector(Array(1, 1, 0, 0, 0))),
                         (1, new DenseVector(Array(0, 1, 1, 0, 0))),
                         (2, new DenseVector(Array(0, 0, 0, 1, 1))))

    val denseVectDF = session.createDataFrame(denseVects).toDF("label", "features")
    val sparseVectDF = session.createDataFrame(denseVects.map(p => (p._1, p._2.toSparse))).toDF("label", "features")

    val rescaledDD =
      new IDF().setInputCol("features").setOutputCol("scaledFeatures").fit(denseVectDF).transform(denseVectDF)
    val rescaledDS =
      new IDF().setInputCol("features").setOutputCol("scaledFeatures").fit(denseVectDF).transform(sparseVectDF)
    val rescaledSD =
      new IDF().setInputCol("features").setOutputCol("scaledFeatures").fit(sparseVectDF).transform(denseVectDF)
    val rescaledSS =
      new IDF().setInputCol("features").setOutputCol("scaledFeatures").fit(sparseVectDF).transform(sparseVectDF)

    val resultsD = List(rescaledDD, rescaledSD).map(_.getDVCol("scaledFeatures"))
    val resultsS = List(rescaledDS, rescaledSS).map(_.getSVCol("scaledFeatures"))

    assert(resultsD.head === resultsD(1))
    assert(resultsS.head === resultsS(1))
    assert(resultsD.head.map(_.toSparse) === resultsS.head)
  }

  test("raise an error when applied to a null array") {
    val df = session.createDataFrame(Seq((0, Some(new DenseVector(Array(1, 1, 0, 0, 0)))),
                                         (1, Some(new DenseVector(Array(0, 1, 1, 0, 0)))),
                                         (2, None)))
      .toDF("id", "features")
    val df2 = new IDF().setInputCol("features")
    withoutLogging {
      intercept[org.apache.spark.SparkException] {
        new IDF().setInputCol("features").fit(df)
      }
    }
  }

  test("support setting minDocFrequency") {
    val df = session.createDataFrame(Seq((0, new DenseVector(Array(1, 1, 0, 0, 0))),
                                         (1, new DenseVector(Array(0, 1, 1, 0, 0))),
                                         (2, new DenseVector(Array(0, 0, 0, 1, 1)))))
      .toDF("id", "features")

    val df2 = new IDF().setMinDocFreq(2)
      .setInputCol("features").setOutputCol("rescaledFeatures")
      .fit(df).transform(df)
    val lines = df2.getDVCol("rescaledFeatures")
    val trueLines = List(new DenseVector(Array(0.0, 0.28768207245178085, 0.0, 0.0, 0.0)),
                         new DenseVector(Array(0.0, 0.28768207245178085, 0.0, 0.0, 0.0)),
                         new DenseVector(Array(0.0, 0.0, 0.0, 0.0, 0.0)))
    assert(lines === trueLines)
  }

  ignore("raise an error when given strange values of minDocumentFrequency") {
    val df = session.createDataFrame(Seq((0, new DenseVector(Array(1, 1, 0, 0, 0))),
                                         (1, new DenseVector(Array(0, 1, 1, 0, 0))),
                                         (2, new DenseVector(Array(0, 0, 0, 1, 1)))))
      .toDF("id", "features")
    // new IDF().setMinDocFreq(-1).setInputCol("features").fit(df).transform(df).show()
    List(-1, -10).foreach { n =>
      val estimator = new IDF().setMinDocFreq(n).setInputCol("features")
      assertSparkException[IllegalArgumentException](estimator, df)
    }
  }

}
