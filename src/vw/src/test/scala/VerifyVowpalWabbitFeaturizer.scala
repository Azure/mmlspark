// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import scala.reflect.runtime.universe.TypeTag
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.vowpalwabbit.bare.VowpalWabbitMurmur
import com.microsoft.ml.spark.featurizer.Featurizer

class VerifyVowpalWabbitFeaturizer extends TestBase {

  case class Sample1(val str: String, val seq: Seq[String])
  case class Input[T] (val in: T)

  test("Verify VowpalWabbit Featurizer can be run with seq and string") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("str", "seq"))
      .setOutputCol("features")
    val df1 = session.createDataFrame(Seq(Sample1("abc", Seq("foo", "bar"))))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[Vector](0)

    val featurizer2 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("seq", "str"))
      .setOutputCol("features")
    val df2 = session.createDataFrame(Seq(Sample1("abc", Seq("foo", "bar"))))

    val v2 = featurizer2.transform(df2).select(col("features")).collect.apply(0).getAs[Vector](0)

    assert(v1.equals(v2))
  }

  private def testNumeric[T: TypeTag](v: T) = {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = session.createDataFrame(Seq(Input[T](v)))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 1)
    assert(v1.indices(0) == VowpalWabbitMurmur.hash("in", VowpalWabbitMurmur.hash("features", 0)))
    assert(v1.values(0) == v)
  }

  test("Verify VowpalWabbit Featurizer can be run with numeric") {
    testNumeric[Int](5)
    testNumeric[Short](5)
    testNumeric[Byte](5)
    testNumeric[Long](5)
    testNumeric[Double](5.2)
    testNumeric[Float](5.2f)
  }

  test("Verify VowpalWabbit Featurizer can be run with string") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = session.createDataFrame(Seq(Input[String]("markus")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 1)
    assert(v1.indices(0) == VowpalWabbitMurmur.hash("inmarkus", VowpalWabbitMurmur.hash("features", 0)))
    assert(v1.values(0) == 1.0)
  }

  test("Verify VowpalWabbit Featurizer can be run with ArrayString") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = session.createDataFrame(Seq(Input[Array[String]](Array("markus", "marie"))))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 2)
    assert(v1.indices(0) == (Featurizer.maxIndexMask &
      VowpalWabbitMurmur.hash("inmarkus", VowpalWabbitMurmur.hash("features", 0))))
    assert(v1.indices(1) == (Featurizer.maxIndexMask &
      VowpalWabbitMurmur.hash("inmarie", VowpalWabbitMurmur.hash("features", 0))))
    assert(v1.values(0) == 1.0)
    assert(v1.values(1) == 1.0)
  }

  private def testMap[T: TypeTag](v1: T, v2: T) = {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in"))
      .setOutputCol("features")

    val df1 = session.createDataFrame(Seq(Input[Map[String, T]](Map[String, T]("k1" -> v1, "k2" -> v2))))

    val vec = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(vec.numNonzeros == 2)

    // note: order depends on the hashes
    assert(vec.indices(1) == (Featurizer.maxIndexMask &
      VowpalWabbitMurmur.hash("ink1", VowpalWabbitMurmur.hash("features", 0))))
    assert(vec.indices(0) == (Featurizer.maxIndexMask &
      VowpalWabbitMurmur.hash("ink2", VowpalWabbitMurmur.hash("features", 0))))
    assert(vec.values(1) == v1)
    assert(vec.values(0) == v2)
  }

  test("Verify VowpalWabbit Featurizer can be run with MapStringDouble") {
    testMap[Int](5, 4)
    testMap[Short](5, 4)
    testMap[Byte](5, 4)
    testMap[Long](5, 4)
    testMap[Double](5.2, 3.1)
    testMap[Float](5.2f, 3.1f)
  }

  test("Verify VowpalWabbit Featurizer can be run with StringSplitString") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setStringSplitInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = session.createDataFrame(Seq(Input[String]("markus  marie")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 2)
    assert(v1.indices(0) == (Featurizer.maxIndexMask &
      VowpalWabbitMurmur.hash("inmarkus", VowpalWabbitMurmur.hash("features", 0))))
    assert(v1.indices(1) == (Featurizer.maxIndexMask &
      VowpalWabbitMurmur.hash("inmarie", VowpalWabbitMurmur.hash("features", 0))))
    assert(v1.values(0) == 1.0)
    assert(v1.values(1) == 1.0)
  }

  test("Verify VowpalWabbit Featurizer can generate duplicates") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setStringSplitInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = session.createDataFrame(Seq(Input[String]("markus markus")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 1)
    assert(v1.indices(0) == (Featurizer.maxIndexMask &
      VowpalWabbitMurmur.hash("inmarkus", VowpalWabbitMurmur.hash("features", 0))))
    assert(v1.values(0) == 1.0)
  }
}