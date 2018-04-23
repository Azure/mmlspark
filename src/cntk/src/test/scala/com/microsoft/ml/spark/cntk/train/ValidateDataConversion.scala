// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk.train

import java.io.File
import java.net.URI

import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.DataFrame
import org.slf4j.LoggerFactory

class ValidateDataConversion extends TestBase with TestFileCleanup {

  override var cleanupPath: File = new File(new URI(dir))

  test("vector to text") {
    val testVectors = List(
      new DenseVector(Array(1.0, 0.0)),
      new SparseVector(     1, Array(0), Array(8.0)),
      new SparseVector(   100, Array(0,10,18,33,62,67,80), Array(1.0,2.0,1.0,1.0,1.0,1.0,1.0)),
      new SparseVector(100000, Array(5833,9467,16680,29018,68900,85762,97510), Array(1.0,1.0,1.0,1.0,1.0,1.0,2.0))
    )

    val expected = Seq(
      "0:1.0 1:0.0 ",
      "0:8.0 ",
      "0:1.0 10:2.0 18:1.0 33:1.0 62:1.0 67:1.0 80:1.0 ",
      "5833:1.0 9467:1.0 16680:1.0 29018:1.0 68900:1.0 85762:1.0 97510:2.0 ")

    val outputs = testVectors.map(vector => DataTransferUtils.convertVectorToText(vector, CNTKLearner.sparseForm))
    assert(outputs === expected)
  }

  test("validate empty vectors to dense text") {
    val testVectors = List(
      new DenseVector(Array(1.0, 0.0, 0.0, 4.0)),
      new DenseVector(Array(0.0, 0.0, 0.0, 0.0)),
      new SparseVector(4, Array(), Array()),
      new DenseVector(Array(2.0, 3.0, 4.0, 5.0))
    )

    val expected = Seq(
      "1.0 0.0 0.0 4.0 ",
      "0.0 0.0 0.0 0.0 ",
      "0.0 0.0 0.0 0.0 ",
      "2.0 3.0 4.0 5.0 ")

    val outputs = testVectors.map(vector => DataTransferUtils.convertVectorToText(vector, CNTKLearner.denseForm))
    assert(outputs === expected)
  }

  val mockLabelColumn = "Label"

  def createMockDataset: DataFrame = {
    session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, 0),
      (1, 3, 0.40, 0.50, 1),
      (0, 4, 0.78, 0.99, 2),
      (1, 5, 0.12, 0.34, 3),
      (0, 1, 0.50, 0.60, 0),
      (1, 3, 0.40, 0.50, 1),
      (0, 3, 0.78, 0.99, 2),
      (1, 4, 0.12, 0.34, 3),
      (0, 0, 0.50, 0.60, 0),
      (1, 2, 0.40, 0.50, 1),
      (0, 3, 0.78, 0.99, 2),
      (1, 4, 0.12, 0.34, 3)))
      .toDF(mockLabelColumn, "col1", "col2", "col3", "col4")
  }

  test("Checkpoint the data") {
    val data = createMockDataset

    val rData = DataTransferUtils.reduceAndAssemble(
        data, mockLabelColumn, "feats", 10)
    val cdata = DataTransferUtils.convertDatasetToCNTKTextFormat(
        rData, mockLabelColumn, "feats", CNTKLearner.denseForm, CNTKLearner.denseForm)

    val log = LoggerFactory.getLogger(this.getClass.getName.stripSuffix("$"))
    val transfer = new LocalWriter(log, s"$dir/smoke")
    val path = transfer.checkpointToText(cdata)

    val out = session.read.text(path)

    assert(verifyResult(cdata, out))
  }

  test("Verify vector labels") {
    val data = createMockDataset
    val rData1 = DataTransferUtils.reduceAndAssemble(data, mockLabelColumn, "feats", 10)
    val rData = DataTransferUtils.reduceAndAssemble(rData1, "feats", "labels", 10)
    val cdata = DataTransferUtils.convertDatasetToCNTKTextFormat(
        rData, "labels", "feats", CNTKLearner.denseForm, CNTKLearner.denseForm)

    val log = LoggerFactory.getLogger(this.getClass.getName.stripSuffix("$"))
    val transfer = new LocalWriter(log, s"$dir/vectorlabel")
    val path = transfer.checkpointToText(cdata)

    val out = session.read.text(path)

    assert(verifyResult(cdata, out))
  }

}
