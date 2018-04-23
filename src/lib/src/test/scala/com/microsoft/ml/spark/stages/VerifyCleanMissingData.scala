// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import java.lang.{Boolean => JBoolean, Double => JDouble, Integer => JInt}

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

/** Tests to validate the functionality of Clean Missing Data estimator. */
class VerifyCleanMissingData extends TestBase with EstimatorFuzzing[CleanMissingData] {

  override val epsilon: Double = 0.01
  import session.implicits._
  def createMockDataset: DataFrame = {
    Seq[(JInt, JInt, JDouble, JDouble, JInt)](
      (0, 2, 0.50, 0.60, 0),
      (1, 3, 0.40, null, null),
      (0, 4, 0.78, 0.99, 2),
      (1, 5, 0.12, 0.34, 3),
      (0, 1, 0.50, 0.60, 0),
      (null, null, null, null, null),
      (0, 3, 0.78, 0.99, 2),
      (1, 4, 0.12, 0.34, 3),
      (0, null, 0.50, 0.60, 0),
      (1, 2, 0.40, 0.50, null),
      (0, 3, null, 0.99, 2),
      (1, 4, 0.12, 0.34, 3))
      .toDF("col1", "col2", "col3", "col4", "col5")
  }

  def createStringMockDataset: DataFrame = {
    Seq[(JInt, String)](
      (0, "hello"),
      (1, "world"),
      (0, null),
      (1, "test111"),
      (0, "some words for test"),
      (null, "test2"),
      (0, null),
      (1, "another test"))
      .toDF("col1", "col2")
  }

  def createBooleanMockDataset: DataFrame = {
    Seq[(JInt, JBoolean)](
      (0, true),
      (1, false),
      (0, null),
      (1, true),
      (0, false),
      (null, true),
      (0, null),
      (1, false))
      .toDF("col1", "col2")
  }

  test("Test for cleaning missing data with mean") {
    val dataset = createMockDataset
    val cmd = new CleanMissingData()
      .setInputCols(dataset.columns)
      .setOutputCols(dataset.columns)
      .setCleaningMode(CleanMissingData.meanOpt)
    val cmdModel = cmd.fit(dataset)
    val result = cmdModel.transform(dataset)
    // Calculate mean of column values
    val numCols = dataset.columns.length
    val meanValues = Array.ofDim[Double](numCols)
    val counts = Array.ofDim[Double](numCols)
    val collected = dataset.collect()
    collected.foreach(row => {
      for (i <- 0 until numCols) {
        val rawValue = row.get(i)
        val rowValue =
          if (rawValue == null) 0
          else if (i == 2 || i == 3) {
            counts(i) += 1
            row.get(i).asInstanceOf[JDouble].doubleValue()
          } else {
            counts(i) += 1
            row.get(i).asInstanceOf[JInt].doubleValue()
          }
        meanValues(i) += rowValue
      }
    })
    for (i <- 0 until numCols) {
      meanValues(i) /= counts(i)
      if (i != 2 && i != 3) {
        meanValues(i) = meanValues(i).toInt.toDouble
      }
    }
    verifyReplacementValues(dataset, result, meanValues)
  }

  test("Test for cleaning missing data with median") {
    val dataset = createMockDataset
    val cmd = new CleanMissingData()
      .setInputCols(dataset.columns)
      .setOutputCols(dataset.columns)
      .setCleaningMode(CleanMissingData.medianOpt)
    val cmdModel = cmd.fit(dataset)
    val result = cmdModel.transform(dataset)
    val medianValues = Array[Double](0, 3, 0.4, 0.6, 2)
    verifyReplacementValues(dataset, result, medianValues)
  }

  test("Test for cleaning missing data with custom value") {
    val dataset = createMockDataset
    val customValue = -1.5
    val cmd = new CleanMissingData()
      .setInputCols(dataset.columns)
      .setOutputCols(dataset.columns)
      .setCleaningMode(CleanMissingData.customOpt)
      .setCustomValue(customValue.toString)
    val cmdModel = cmd.fit(dataset)
    val result = cmdModel.transform(dataset)
    val replacesValues = Array.fill[Double](dataset.columns.length)(customValue)
    val numCols = replacesValues.length
    for (i <- 0 until numCols) {
      if (i != 2 && i != 3) {
        replacesValues(i) = replacesValues(i).toInt.toDouble
      }
    }
    verifyReplacementValues(dataset, result, replacesValues)
  }

  test("Test for cleaning missing data with string custom value") {
    val dataset = createStringMockDataset
    val customValue = "myCustomValue"
    val cmd = new CleanMissingData()
      .setInputCols(Array("col2"))
      .setOutputCols(Array("col2"))
      .setCleaningMode(CleanMissingData.customOpt)
      .setCustomValue(customValue)
    val cmdModel = cmd.fit(dataset)
    val result = cmdModel.transform(dataset)
    val replacesValues = Array.fill[String](dataset.columns.length)(customValue)
    verifyReplacementValues[String](dataset, result, replacesValues, Array(1))
  }

  test("Test for cleaning missing data with boolean custom value") {
    val dataset = createBooleanMockDataset
    val customValue = true
    val cmd = new CleanMissingData()
      .setInputCols(Array("col2"))
      .setOutputCols(Array("col2"))
      .setCleaningMode(CleanMissingData.customOpt)
      .setCustomValue(customValue.toString)
    val cmdModel = cmd.fit(dataset)
    val result = cmdModel.transform(dataset)
    val replacesValues = Array.fill[Boolean](dataset.columns.length)(customValue)
    verifyReplacementValues[Boolean](dataset, result, replacesValues, Array(1))
  }

  private def verifyReplacementValues(expected: DataFrame, result: DataFrame, expectedValues: Array[Double]) = {
    val collectedExp = expected.collect()
    val collectedResult = result.collect()
    val numRows = result.count().toInt
    val numCols = result.columns.length
    for (j <- 0 until numRows) {
      for (i <- 0 until numCols) {
        val row = collectedExp(j)
        val (rowValue, actualValue) =
          if (i == 2 || i == 3) {
            (row.get(i).asInstanceOf[JDouble], collectedResult(j)(i).asInstanceOf[Double])
          } else {
            (row.get(i).asInstanceOf[JInt], collectedResult(j)(i).asInstanceOf[Int].toDouble)
          }
        if (rowValue == null) {
          val expectedValue = expectedValues(i)
          assert(expectedValue === actualValue,
            s"Values do not match, expected: $expectedValue, result: $actualValue")
        }
      }
    }
  }

  private def verifyReplacementValues[T](expected: DataFrame,
                                         result: DataFrame,
                                         expectedValues: Array[T],
                                         columns: Array[Int]) = {
    val collectedExp = expected.collect()
    val collectedResult = result.collect()
    val numRows = result.count().toInt
    val numCols = result.columns.length
    for (j <- 0 until numRows) {
      for (i <- 0 until numCols) {
        if (columns.contains(i)) {
          val row = collectedExp(j)
          val (rowValue, actualValue) = (row.get(i), collectedResult(j)(i))
          if (rowValue == null) {
            val expectedValue = expectedValues(i)
            assert(expectedValue == actualValue,
              s"Values do not match, expected: $expectedValue, result: $actualValue")
          }
        }
      }
    }
  }

  lazy val dataset: DataFrame = createMockDataset
  override def testObjects(): Seq[TestObject[CleanMissingData]] = List(new TestObject(
    new CleanMissingData().setInputCols(dataset.columns).setOutputCols(dataset.columns), dataset))

  override def reader: MLReadable[_] = CleanMissingData
  override def modelReader: MLReadable[_] = CleanMissingDataModel
}
