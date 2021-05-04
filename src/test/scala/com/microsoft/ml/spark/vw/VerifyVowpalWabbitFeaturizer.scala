// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.vowpalwabbit.spark.VowpalWabbitMurmur

import scala.reflect.runtime.universe.TypeTag

class VerifyVowpalWabbitFeaturizer extends TestBase with TransformerFuzzing[VowpalWabbitFeaturizer] {

  lazy val defaultMask: Int = (1 << 30) - 1

  case class Sample1(str: String, seq: Seq[String])

  case class Input[T](in: T)

  case class Input2[T, S](in1: T, in2: S)

  lazy val namespaceFeatures: Int = VowpalWabbitMurmur.hash("features", 0)

  test("Verify order preserving") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setStringSplitInputCols(Array("in"))
      .setPreserveOrderNumBits(2)
      .setNumBits(18)
      .setPrefixStringsWithColumnName(false)
      .setOutputCol("features")
    val df1 = spark.createDataFrame(Seq(Input[String]("marie markus fun")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 3)

    val bitMask = (1 << 18) - 1

    // the order is the same as in the string above
    assert((bitMask & v1.indices(0)) == (bitMask &
      VowpalWabbitMurmur.hash("marie", namespaceFeatures)))
    assert((bitMask & v1.indices(1)) == (bitMask &
      VowpalWabbitMurmur.hash("markus", namespaceFeatures)))
    assert((bitMask & v1.indices(2)) == (bitMask &
      VowpalWabbitMurmur.hash("fun", namespaceFeatures)))
    assert(v1.values(0) == 1.0)
    assert(v1.values(1) == 1.0)
    assert(v1.values(2) == 1.0)
  }

  test("Verify VowpalWabbit Featurizer can be run with seq and string") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("str", "seq"))
      .setOutputCol("features")
    val df1 = spark.createDataFrame(Seq(Sample1("abc", Seq("foo", "bar"))))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[Vector](0)

    val featurizer2 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("seq", "str"))
      .setOutputCol("features")
    val df2 = spark.createDataFrame(Seq(Sample1("abc", Seq("foo", "bar"))))

    val v2 = featurizer2.transform(df2).select(col("features")).collect.apply(0).getAs[Vector](0)

    assert(v1.equals(v2))
  }

  private def testNumeric[T: TypeTag](v: T) = {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = spark.createDataFrame(Seq(Input[T](v)))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 1)
    assert(v1.indices(0) == VowpalWabbitMurmur.hash("in", namespaceFeatures))
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
    val df1 = spark.createDataFrame(Seq(Input[String]("markus")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 1)
    assert(v1.indices(0) == VowpalWabbitMurmur.hash("inmarkus", namespaceFeatures))
    assert(v1.values(0) == 1.0)
  }

  test("Verify VowpalWabbit Featurizer can be run with ArrayString") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = spark.createDataFrame(Seq(Input[Array[String]](Array("markus", "marie"))))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 2)

    // Seems a little counterintuitive but the namespaceHash is formed form the output column name and not the
    // input column.
    assert(v1.indices(0) == (defaultMask &
      VowpalWabbitMurmur.hash("markus", namespaceFeatures)))
    assert(v1.indices(1) == (defaultMask &
      VowpalWabbitMurmur.hash("marie", namespaceFeatures)))
    assert(v1.values(0) == 1.0)
    assert(v1.values(1) == 1.0)
  }

  private def testMap[T: TypeTag](v1: T, v2: T) = {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in"))
      .setOutputCol("features")

    val df1 = spark.createDataFrame(Seq(Input[Map[String, T]](Map[String, T]("k1" -> v1, "k2" -> v2))))

    val vec = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(vec.numNonzeros == 2)

    // note: order depends on the hashes
    assert(vec.indices(0) == (defaultMask &
      VowpalWabbitMurmur.hash("ink1", namespaceFeatures)))
    assert(vec.indices(1) == (defaultMask &
      VowpalWabbitMurmur.hash("ink2", namespaceFeatures)))
    assert(vec.values(0) == v1)
    assert(vec.values(1) == v2)
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
    val df1 = spark.createDataFrame(Seq(Input[String]("markus  marie")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 2)
    assert(v1.indices(0) == (defaultMask &
      VowpalWabbitMurmur.hash("inmarkus", namespaceFeatures)))
    assert(v1.indices(1) == (defaultMask &
      VowpalWabbitMurmur.hash("inmarie", namespaceFeatures)))
    assert(v1.values(0) == 1.0)
    assert(v1.values(1) == 1.0)
  }

  test("Verify VowpalWabbit Featurizer can generate duplicates") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setStringSplitInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = spark.createDataFrame(Seq(Input[String]("markus markus markus")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 1)
    assert(v1.indices(0) == (defaultMask &
      VowpalWabbitMurmur.hash("inmarkus", namespaceFeatures)))
    assert(v1.values(0) == 3.0)
  }

  test("Verify VowpalWabbit Featurizer can generate duplicates and remove") {
    val featurizer1 = new VowpalWabbitFeaturizer()
      .setSumCollisions(false)
      .setStringSplitInputCols(Array("in"))
      .setOutputCol("features")
    val df1 = spark.createDataFrame(Seq(Input[String]("markus markus")))

    val v1 = featurizer1.transform(df1).select(col("features")).collect.apply(0).getAs[SparseVector](0)

    assert(v1.numNonzeros == 1)
    assert(v1.indices(0) == (defaultMask &
      VowpalWabbitMurmur.hash("inmarkus", namespaceFeatures)))
    assert(v1.values(0) == 1.0)
  }

  test("Verify VowpalWabbit Featurizer output VectorUDT schema type") {
    val newSchema = new VowpalWabbitFeaturizer()
      .setInputCols(Array("data"))
      .setOutputCol("features")
      .transformSchema(new StructType(Array(StructField("data", DataTypes.DoubleType, true))))

    assert(newSchema.fields(1).name == "features")
    assert(newSchema.fields(1).dataType.typeName == "vector")
  }

  private def verifyArrays[T](actual: Array[T], expected: Array[T])(implicit ord: Ordering[T]) = {
    assert(actual.length == expected.length)

    (actual.sorted zip expected.sorted).forall { case (x, y) => x == y }
  }

  test("Verify VowpalWabbit Featurizer can combine vectors") {
    val df1 = spark.createDataFrame(Seq(Input2[Vector, Vector](
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(8, Array(1, 4), Array(5.0, 8.0))
    )))

    val featurizer = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in1", "in2"))
      .setOutputCol("features")
      .setNumBits(18)

    val dfOut = featurizer.transform(df1)

    val output = dfOut.head.getAs[SparseVector]("features")

    assert(output.size == 262144)
    assert(output.numNonzeros == 4)

    verifyArrays(output.values, Array(1.0, 7.0, 3.0, 8.0))
  }

  test("Verify VowpalWabbit Featurizer can combine vectors and remask") {
    val df1 = spark.createDataFrame(Seq(Input2[Vector, Vector](
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(8, Array(1, 4), Array(5.0, 8.0))
    )))

    val featurizer = new VowpalWabbitFeaturizer()
      .setInputCols(Array("in1", "in2"))
      .setOutputCol("features")
      .setNumBits(2)

    val dfOut = featurizer.transform(df1)

    val output = dfOut.head.getAs[SparseVector]("features")

    assert(output.size == 4)
    assert(output.numNonzeros == 3)

    verifyArrays(output.values, Array(9.0, 7.0, 3.0))
  }

  test("Check tamil encoding") {
    //noinspection ScalaStyle
    val df = spark.createDataFrame(Seq(
      ("ஜெய்-அஞ்சலி காதல் பற்றி ராய் லட்சுமி!", "output")))
      .toDF("a", "normalized")

    val featurizer = new VowpalWabbitFeaturizer()
      .setStringSplitInputCols(Array("a"))
      .setOutputCol("features")

    val result = featurizer.transform(df)
    val vec = result.head().getAs[SparseVector](2)

    assert(vec.numNonzeros == 6)
    verifyArrays(vec.indices, Array(260933379, 376953061, 482756394, 597851995, 781002072, 950998778))
  }

  def testObjects(): Seq[TestObject[VowpalWabbitFeaturizer]] = List(new TestObject(
    new VowpalWabbitFeaturizer().setInputCols(Array("words")).setOutputCol("out"), makeBasicDF()))

  override def reader: MLReadable[_] = VowpalWabbitFeaturizer
}
