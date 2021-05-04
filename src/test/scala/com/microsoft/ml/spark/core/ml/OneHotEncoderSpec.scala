// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.ml

import com.microsoft.ml.spark.core.schema.DatasetExtensions._
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark._
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.DataFrame

class OneHotEncoderSpec extends TestBase {

  test("expand category indicies") {
    val df = spark.createDataFrame(Seq((0, 0.0),
                                         (1, 1.0),
                                         (2, 0.0),
                                         (3, 2.0),
                                         (4, 1.0),
                                         (5, 0.0)))
      .toDF("id", "categoryIndex")

    val encoded =
      new OneHotEncoder()
        .setInputCols(Array("categoryIndex")).setOutputCols(Array("categoryVec"))
        .fit(df).transform(df)
    val oneHotList = encoded.getSVCol("categoryVec")
    val trueList = List(new SparseVector(2, Array(0), Array(1.0)),
                        new SparseVector(2, Array(1), Array(1.0)),
                        new SparseVector(2, Array(0), Array(1.0)),
                        new SparseVector(2, Array(),  Array()),
                        new SparseVector(2, Array(1), Array(1.0)),
                        new SparseVector(2, Array(0), Array(1.0)))
    assert(oneHotList === trueList)
  }

  test("support interger indicies") {
    val df = spark.createDataFrame(Seq((0, 0),
                                         (1, 1),
                                         (2, 0),
                                         (3, 2),
                                         (4, 1),
                                         (5, 0)
                                     ))
      .toDF("id", "categoryIndex")

    val encoded = new OneHotEncoder()
      .setInputCols(Array("categoryIndex")).setOutputCols(Array("categoryVec"))
      .fit(df).transform(df)
    val oneHotList = encoded.getSVCol("categoryVec")
    val trueList = List(new SparseVector(2, Array(0), Array(1.0)),
                        new SparseVector(2, Array(1), Array(1.0)),
                        new SparseVector(2, Array(0), Array(1.0)),
                        new SparseVector(2, Array(),  Array()),
                        new SparseVector(2, Array(1), Array(1.0)),
                        new SparseVector(2, Array(0), Array(1.0)))
    assert(oneHotList === trueList)
  }

  test("support not dropping the last feature") {
    val df = spark.createDataFrame(Seq((0, 0.0),
                                         (1, 1.0),
                                         (2, 0.0),
                                         (3, 2.0),
                                         (4, 1.0),
                                         (5, 0.0)
                                     ))
      .toDF("id", "categoryIndex")

    val encoded = new OneHotEncoder().setDropLast(false)
      .setInputCols(Array("categoryIndex")).setOutputCols(Array("categoryVec"))
      .fit(df).transform(df)
    val oneHotList = encoded.getSVCol("categoryVec")
    val trueList = List(new SparseVector(3, Array(0), Array(1.0)),
                        new SparseVector(3, Array(1), Array(1.0)),
                        new SparseVector(3, Array(0), Array(1.0)),
                        new SparseVector(3, Array(2), Array(1.0)),
                        new SparseVector(3, Array(1), Array(1.0)),
                        new SparseVector(3, Array(0), Array(1.0)))
    assert(oneHotList === trueList)
  }

  private def testOHE(data: DataFrame): Unit = {
    assertSparkException[SparkException](
      new OneHotEncoder()
        .setInputCols(Array("categoryIndex")).setOutputCols(Array("encodedOutput")),
      data.toDF("id", "categoryIndex"))
  }

  test("raise an error when applied to a null array") {
    testOHE(spark.createDataFrame(Seq((0, Some(0.0)), (1, Some(1.0)), (2, None))))
  }
  test("raise an error when it receives a strange float") {
    testOHE(spark.createDataFrame(Seq((0, 0.0), (1, 1.0), (2, 0.4))))
    testOHE(spark.createDataFrame(Seq((0, 0.0), (1, 1.0), (2, -1.0))))
  }

}
