// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import java.io.File

import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.image.UnrollImage
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql._
import com.microsoft.ml.spark.io.IOImplicits._
import org.scalatest.Assertion

trait CNTKTestUtils extends TestBase {

  val filesRoot: String = BuildInfo.datasetDir.toString
  val imagePath: String = FileUtilities.join(filesRoot, "Images", "CIFAR").toString
  val modelPath: String = FileUtilities.join(filesRoot, "CNTKModel", "ConvNet_CIFAR10.model").toString
  val inputCol  = "cntk_images"
  val outputCol = "out"
  val labelCol  = "labels"

  val featureVectorLength: Int = 3 * 32 * 32
  lazy val saveFile: String = new File(tmpDir.toFile, "spark-z.model").toString

  def testModelDF(spark: SparkSession): DataFrame = {
    import spark.implicits._
    spark.sparkContext.parallelize(Seq(
    Array(1.32165250, -2.1215112, 0.63150704, 0.77315974, -1.28163720,
      -0.20210080, -2.2839167, -2.08691480, 5.08418200, -1.33741090),
    Array(3.44079640, 1.4877119, -0.74059330, -0.34381202, -2.48724990,
      -2.62866950, -3.1693816, -3.14182600, 4.76314800, 0.68712880),
    Array(-1.88747900, -4.7685330, 0.15169683, 6.80547570, -0.38405967,
      3.41065170, 1.3302778, -0.87714905, -2.18046050, -4.16661830),
    Array(5.01010300, 3.9860306, -1.36795600, -0.89830830, -4.49545430,
      -4.19537070, -4.4045380, -5.81759450, 6.93805700, 1.49001510),
    Array(-4.70754600, -6.0414960, 1.20658250, 5.40738300, 1.07661690,
      4.71566440, 4.3834330, -1.57187440, -2.96569730, -5.43208270),
    Array(-1.23873880, -3.2042341, 2.54533000, 5.51954800, 2.89042470,
      0.12380804, 3.8639085, -4.79466800, -2.41463420, -5.17418430))).toDF
  }

  def testImages(spark: SparkSession): DataFrame = {
    val images = spark.read.image.load(imagePath)

    val unroll = new UnrollImage().setInputCol("image").setOutputCol(inputCol)

    unroll.transform(images).select(inputCol)
  }

  def makeFakeData(spark: SparkSession, rows: Int, size: Int, outputDouble: Boolean = false): DataFrame = {
    import spark.implicits._
    if (outputDouble) {
      List
        .fill(rows)(List.fill(size)(0.0).toArray)
        .zip(List.fill(rows)(0.0))
        .toDF(inputCol, labelCol)
    } else {
      List
        .fill(rows)(List.fill(size)(0.0.toFloat).toArray)
        .zip(List.fill(rows)(0.0))
        .toDF(inputCol, labelCol)
    }
  }

  protected def compareToTestModel(result: DataFrame): Assertion = {
    //TODO improve checks
    assert(result.columns.toSet == Set(inputCol, outputCol))
    assert(result.count() == testModelDF(result.sparkSession).count())
    val max = result
      .select(outputCol)
      .collect()
      .map(row => row.getAs[DenseVector](0).toArray.max)
      .max
    assert(max < 10 & max > -10)
  }

}
