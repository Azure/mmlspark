// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.nn

import breeze.linalg.DenseVector
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.ml.linalg.{DenseVector => SDV}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit

import scala.collection.immutable

trait BallTreeTestBase extends TestBase {

  lazy val uniformData: immutable.IndexedSeq[DenseVector[Double]] = {
    for (x <- 1 to 10; y <- 1 to 10; z <- 1 to 10)
      yield DenseVector((x * 2).toDouble, (y * 2).toDouble, (z * 2).toDouble)
  }

  lazy val uniformLabels: IndexedSeq[Int] = randomClassLabels(uniformData, 5)

  lazy val largeUniformData: immutable.IndexedSeq[DenseVector[Double]] = {
    for (x <- 1 to 100; y <- 1 to 100; z <- 1 to 100)
      yield DenseVector((x * 2).toDouble, (y * 2).toDouble, (z * 2).toDouble)
  }

  def twoClassLabels(data: IndexedSeq[_]): IndexedSeq[Int] =
    IndexedSeq.fill(data.length / 2)(1) ++ IndexedSeq.fill(data.length - data.length / 2)(2)

  def twoClassStringLabels(data: IndexedSeq[_]): IndexedSeq[String] =
    twoClassLabels(data).map(_.toString)
  def randomClassLabels(data: IndexedSeq[_], nClasses: Int): IndexedSeq[Int] = {
    val r = scala.util.Random
    data.map(_ => r.nextInt(nClasses))
  }

  def assertEquivalent(r1: Seq[BestMatch],
                       r2: Seq[BestMatch]): Unit = {
    assert(r1.length == r2.length)
    r1.zip(r2).foreach { case (cm, gt) =>
      assert(cm.distance === gt.distance)
    }
  }

  def randomData(size: Int, dim: Int): IndexedSeq[DenseVector[Double]] = {
    scala.util.Random.setSeed(10)
    IndexedSeq.fill(size)(DenseVector.fill(dim)((scala.util.Random.nextDouble - 0.5) * 2))
  }

  lazy val randomData: IndexedSeq[DenseVector[Double]] = randomData(10000, 3)

  lazy val df: DataFrame = spark
    .createDataFrame(uniformData.zip(uniformLabels).map(p =>
      (new SDV(p._1.data), "foo", p._2)
    ))
    .toDF("features", "values", "labels")

  lazy val stringDF: DataFrame = spark
    .createDataFrame(uniformData.zip(uniformLabels).map(p =>
      (new SDV(p._1.data), "foo", "class1")
    ))
    .toDF("features", "values", "labels")

  lazy val testDF: DataFrame = spark
    .createDataFrame(uniformData.zip(uniformLabels).take(5).map(p =>
      (new SDV(p._1.data), "foo", p._2)
    ))
    .toDF("features", "values", "labels")
    .withColumn("conditioner", lit(Array(0, 1)))

  lazy val testStringDF: DataFrame = spark
    .createDataFrame(uniformData.zip(uniformLabels).take(5).map(p =>
      (new SDV(p._1.data), "foo", "class1")
    ))
    .toDF("features", "values", "labels")
    .withColumn("conditioner", lit(Array("class1")))
}

class BallTreeTest extends BallTreeTestBase {
  def naiveSearch(haystack: IndexedSeq[DenseVector[Double]],
                  needle: DenseVector[Double],
                  k: Int): Seq[BestMatch] = {
    haystack.zipWithIndex
      .map(p => (p._2, p._1 dot needle))
      .sorted(Ordering.by({ p: (Any, Double) => p._2 }).reverse)
      .take(k)
      .map { case (i, d) => BestMatch(i, d) }
  }

  def compareToNaive(haystack: IndexedSeq[DenseVector[Double]],
                     k: Int,
                     needle: DenseVector[Double]): Unit = {
    val tree = BallTree(haystack, haystack.indices)
    val btMatches = time(tree.findMaximumInnerProducts(needle, k))
    val groundTruth = time(naiveSearch(haystack, needle, k))
    assertEquivalent(btMatches, groundTruth)
  }

  test("search should be exact") {
    val needle = DenseVector(9.0, 10.0, 11.0)
    val ks = Seq(1, 5, 10, 100)
    for (k <- ks) {
      compareToNaive(largeUniformData, k, needle)
    }
  }

  test("Balltree with uniform data") {
    val keys = Seq(DenseVector(0.0, 0.0, 0.0),
      DenseVector(2.0, 2.0, 20.0),
      DenseVector(9.0, 10.0, 11.0))
    for (key <- keys) {
      compareToNaive(uniformData, 5, key)
    }
  }

  test("Balltree with random data should be correct") {
    compareToNaive(randomData, 5, DenseVector(randomData(3).toArray))
  }

  test("Balltree with random data and number of best matches should be correct") {
    val needle = DenseVector(randomData(3).toArray)
    compareToNaive(randomData, 5, needle)
  }

}
