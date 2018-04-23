// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.featurize

import java.io.File
import java.nio.file.Files
import java.sql.{Date, Timestamp}
import java.util.GregorianCalendar

import com.microsoft.ml.spark.core.schema.ImageSchema
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import com.microsoft.ml.spark.stages.ValueIndexer
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{StructField, StructType}

class VerifyAssembleFeatures extends TestBase with EstimatorFuzzing[AssembleFeatures] {
  def testObjects(): Seq[TestObject[AssembleFeatures]] = List(new TestObject(
    new AssembleFeatures().setColumnsToFeaturize(Array("numbers")), makeBasicDF()))

  val reader: MLReadable[_] = AssembleFeatures
  val modelReader: MLReadable[_] = AssembleFeaturesModel
}

class VerifyFeaturize extends TestBase with EstimatorFuzzing[Featurize] {

  val mockLabelColumn = "Label"
  val featuresColumn = "testColumn"
  val targetDirectory = new File("target")
  val resourceDir: File = new File(new File(getClass.getResource("/").toURI), "../../../src/test/resources")

  def getResource(name: String): File ={
    new File(resourceDir, name)
  }

  val benchmarkBasicDataTypesFile = "benchmarkBasicDataTypes.json"
  val historicDataTypesFile: File = getResource(benchmarkBasicDataTypesFile)
  val benchmarkBasicDataTypesTempFile: File = getTempFile(benchmarkBasicDataTypesFile)

  val benchmarkVectorsFile = "benchmarkVectors.json"
  val historicVectorsFile: File = getResource(benchmarkVectorsFile)
  val benchmarkVectorsTempFile: File = getTempFile(benchmarkVectorsFile)

  val benchmarkStringFile = "benchmarkString.json"
  val historicStringFile: File = getResource(benchmarkStringFile)
  val benchmarkStringTempFile: File = getTempFile(benchmarkStringFile)

  val benchmarkStringMissingsFile = "benchmarkStringMissing.json"
  val historicStringMissingsFile: File = getResource(benchmarkStringMissingsFile)
  val benchmarkStringMissingsTempFile: File = getTempFile(benchmarkStringMissingsFile)

  val benchmarkOneHotFile = "benchmarkOneHot.json"
  val historicOneHotFile: File = getResource(benchmarkOneHotFile)
  val benchmarkOneHotTempFile: File = getTempFile(benchmarkOneHotFile)

  val benchmarkNoOneHotFile = "benchmarkNoOneHot.json"
  val historicNoOneHotFile: File = getResource(benchmarkNoOneHotFile)
  val benchmarkNoOneHotTempFile: File = getTempFile(benchmarkNoOneHotFile)

  val benchmarkOneHotMissingsFile = "benchmarkOneHotMissings.json"
  val historicOneHotMissingsFile: File = getResource(benchmarkOneHotMissingsFile)
  val benchmarkOneHotMissingsTempFile: File = getTempFile(benchmarkOneHotMissingsFile)

  val benchmarkNoOneHotMissingsFile = "benchmarkNoOneHotMissings.json"
  val historicNoOneHotMissingsFile: File = getResource(benchmarkNoOneHotMissingsFile)
  val benchmarkNoOneHotMissingsTempFile: File = getTempFile(benchmarkNoOneHotMissingsFile)

  val benchmarkStringIndexOneHotFile = "benchmarkStringIndexOneHot.json"
  val historicStringIndexOneHotFile: File = getResource(benchmarkStringIndexOneHotFile)
  val benchmarkStringIndexOneHotTempFile: File = getTempFile(benchmarkStringIndexOneHotFile)

  val benchmarkDateFile = "benchmarkDate.json"
  val historicDateFile: File = getResource(benchmarkDateFile)
  val benchmarkDateTempFile: File = getTempFile(benchmarkDateFile)

  private def getTempFile(fileName: String): File = {
    new File(targetDirectory,
             s"${fileName}_${System.currentTimeMillis}_.json")
  }

  // int label with features of:
  // long, double, boolean, int, byte, float
  lazy val mockDataset = session.createDataFrame(Seq(
    (0, 2L, 0.50, true,  0, 0.toByte,    12F),
    (1, 3L, 0.40, false, 1, 100.toByte,  30F),
    (0, 4L, 0.78, true,  2, 50.toByte,   12F),
    (1, 5L, 0.12, false, 3, 0.toByte,    12F),
    (0, 1L, 0.50, true,  0, 0.toByte,    30F),
    (1, 3L, 0.40, false, 1, 10.toByte,   12F),
    (0, 3L, 0.78, false, 2, 0.toByte,    12F),
    (1, 4L, 0.12, false, 3, 0.toByte,    12F),
    (0, 0L, 0.50, true,  0, 0.toByte,    12F),
    (1, 2L, 0.40, false, 1, 127.toByte,  30F),
    (0, 3L, 0.78, true,  2, -128.toByte, 12F),
    (1, 4L, 0.12, false, 3, 0.toByte,    12F)))
    .toDF(mockLabelColumn, "col1", "col2", "col3", "col4", "col5", "col6")

  test("Featurizing on some basic data types") {
    val result: DataFrame =
      featurizeAndVerifyResult(mockDataset,
                               benchmarkBasicDataTypesTempFile.toString,
                               historicDataTypesFile)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[DenseVector](featuresColumn).values.length == 6)
  }

  test("Featurizing with vector columns, sparse and dense") {
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, Vectors.sparse(3, Seq((0, 1.0), (2, 2.0))), 0.50, 0.60, 0, Vectors.dense(1.0, 0.1, -1.5)),
      (1, Vectors.dense(1.5, 0.2, -1.2), 0.40, 0.50, 1, Vectors.dense(1.5, 0.2, -1.2)),
      (1, Vectors.sparse(3, Seq((0, 1.0), (2, 2.0))), 0.12, 0.34, 3, Vectors.sparse(3, Seq((0, 1.0), (2, 2.0)))),
      (0, Vectors.dense(1.1, 0.5, -1.024), 0.50, 0.60, 0, Vectors.dense(1.0, 0.4, -1.23)),
      (1, Vectors.dense(1.1, 0.5, -1.056), 0.40, 0.50, 1, Vectors.dense(1.1, 0.5, -1.024)),
      (0, Vectors.dense(Double.NaN, 0.2, -1.23), 0.78, 0.99, 2, Vectors.dense(1.0, 0.1, -1.22)),
      (1, Vectors.dense(1.0, 0.4, -1.23), 0.12, 0.34, 3, Vectors.dense(Double.NaN, 0.2, -1.23))))
      .toDF(mockLabelColumn, "col1", "col2", "col3", "col4", "col5")

    val result: DataFrame = featurizeAndVerifyResult(dataset,
      benchmarkVectorsTempFile.toString,
      historicVectorsFile)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[DenseVector](featuresColumn).values.length == 9)
  }

  test("Featurizing with text columns - using hashing with count based feature selection") {
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, "pokemon are everywhere"),
      (1, 3, 0.40, 0.50, "they are in the woods"),
      (0, 4, 0.78, 0.99, "they are in the water"),
      (1, 5, 0.12, 0.34, "they are in the fields"),
      (0, 3, 0.78, 0.99, "pokemon - gotta catch em all")))
      .toDF(mockLabelColumn, "col1", "col2", "col3", "col4")

    val result: DataFrame = featurizeAndVerifyResult(dataset,
      benchmarkStringTempFile.toString,
      historicStringFile)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[SparseVector](featuresColumn).size == 11)
  }

  test("Featurizing with date and timestamp columns") {
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, new Date(new GregorianCalendar(2017, 6, 7).getTimeInMillis), new Timestamp(1000)),
      (1, 3, 0.40, 0.50, new Date(new GregorianCalendar(2017, 6, 8).getTimeInMillis), new Timestamp(2000)),
      (0, 4, 0.78, 0.99, new Date(new GregorianCalendar(2017, 6, 6).getTimeInMillis), new Timestamp(3000)),
      (1, 5, 0.12, 0.34, new Date(new GregorianCalendar(2016, 6, 5).getTimeInMillis), new Timestamp(4000)),
      (0, 3, 0.78, 0.99, new Date(new GregorianCalendar(2010, 6, 9).getTimeInMillis), new Timestamp(5000))))
      .toDF(mockLabelColumn, "col1", "col2", "col3", "date", "timestamp")

    val result: DataFrame = featurizeAndVerifyResult(dataset,
      benchmarkDateTempFile.toString,
      historicDateFile)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[DenseVector](featuresColumn).size == 16)
  }

  test("Featurizing with image columns") {
    val imageDFSchema = StructType(Array(StructField("image", ImageSchema.columnSchema, true)))
    val path1: String = "file:/home/ilya/lib/datasets/Images/CIFAR/00000.png"
    val path2: String = "file:/home/ilya/lib/datasets/Images/CIFAR/00001.png"
    val height = 32
    val width = 32
    val imgType = 16
    val colorBands = 3
    // Size is height * width * colorBands
    val imageSize = height * width * colorBands
    // Expected is image size with width and height
    val expectedSize = imageSize + 2
    val rowRDD: RDD[Row] = sc.parallelize(Seq[Row](
      Row(Row(path1, height, width, imgType, Array.fill[Byte](imageSize)(1))),
      Row(Row(path2, height, width, imgType, Array.fill[Byte](imageSize)(1)))
    ))
    val dataset = session.createDataFrame(rowRDD, imageDFSchema)
    val result: DataFrame = featurize(dataset, includeFeaturesColumns = false)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[DenseVector](featuresColumn).size == expectedSize)
  }

  test("Verify featurizing text data produces proper tokenized output") {
    val wordCountCol = "wordCount"
    val wordLengthCol = "wordLength"
    val textCol = "textCol"
    val mockAmazonData = session.createDataFrame(Seq(
      (1, 221, 4.42, "Ok~ but I think the Keirsey Temperment Test is more accurate - and cheaper.  This book has its " +
        "good points. If anything, it helps you put into words what you want  from a supervisor, but it is not very " +
        "accurate. The online test does not account for a difference between when 2 of their options are both " +
        "exactly like you, or if they don't describe you at all. This messes up the results, and it did not " +
        "describe me very well. I am not just in denial. I have taken a lot of personality type tests, like " +
        "the Keirsey Temperment sorter and have a pretty good idea of my strengths. So, although this book is " +
        "pretty good in making you understand the importance of incouraging your strengths, it still " +
        "leaves you wondering about how you fit in to their terminology.  As for using this book as a manager " +
        "to better serve your employees, I'v seen it done and it does not necessarily work because the strengths " +
        "spit out for people were not wholly accurate. The company I work for has done this, and most of the " +
        "people who were shifted around to better serve their strengths (according to this book) are very " +
        "unhappy in their new positions.  Your money can be spent better elsewhere. I say its only worth about $10"),
      (0, 138, 4.49, "I had a bad feeling about this!  And I was right!  I was intrigued by the title, which " +
        "supposedly links Jedi wisdom to Christianity.  Well, after 60 pages or so, I have got the feeling that the " +
        "Staub is trying to wrap Jedi in Christian cloth and failing at that. The author speaks of the difficulty in " +
        "leading a Christian life.  But, I say that any religious life (be it Christian, Islam or otherwise) is hard " +
        "because it turns the back on the norm or the conventional.   I am convinced that Yoda is a Zen master; " +
        "the Force is derived from Tao, not God as interpreted by the orthodox religion(I am purposefully leaving " +
        "out Christian Mysticism, which is another beast altogether.). A better book on the subject of theology " +
        "in Star wars is \"The Dharma of Star Wars.\""),
      (0, 43, 4.98, "Poorly written  I tried reading this book but found it so turgid and poorly written that I " +
        "put it down in frustration.  It reads like a translation from another language by an academic bureacrat. " +
        "The theme is interesting, the execution poor.  Cannot recommend")))
      .toDF(mockLabelColumn, wordCountCol, wordLengthCol, textCol)

    val featModel = new Featurize()
      .setFeatureColumns(Map { featuresColumn -> Array(wordCountCol, wordLengthCol, textCol) })
      .setNumberOfFeatures(100000).fit(mockAmazonData)
    val nonzeroValuesThreshold = 30
    featModel.transform(mockAmazonData).collect().foreach(
      row => assert(row.getAs[SparseVector](featuresColumn).indices.length > nonzeroValuesThreshold,
        "Strings improperly tokenized")
    )
  }

  test("Featurizing with text columns that have missing values - using hashing with count based feature selection") {
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, 2, 0.50, "pokemon are everywhere"),
      (1, 3, 0.40, null),
      (0, 4, 0.78, "they are in the water"),
      (1, 5, 0.12, "they are in the fields"),
      (0, 3, 0.78, null)))
      .toDF(mockLabelColumn, "col1", "col2", "col3")

    val result: DataFrame = featurizeAndVerifyResult(dataset,
      benchmarkStringMissingsTempFile.toString,
      historicStringMissingsFile)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[DenseVector](featuresColumn).size == 8)
  }

  test("Featurizing with categorical columns - using one hot encoding") {
    val cat = "Cat"
    val dog = "Dog"
    val bird = "Bird"
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, 2, 0.50, 0.60, dog, cat),
      (1, 3, 0.40, 0.50, cat, dog),
      (0, 4, 0.78, 0.99, dog, bird),
      (1, 5, 0.12, 0.34, cat, dog),
      (0, 3, 0.78, 0.99, dog, bird),
      (1, 4, 0.12, 0.34, bird, dog)))
      .toDF(mockLabelColumn, "col1", "col2", "col3", "col4", "col5")

    val model1 = new ValueIndexer().setInputCol("col4").setOutputCol("col4").fit(dataset)
    val model2 = new ValueIndexer().setInputCol("col5").setOutputCol("col5").fit(dataset)
    val catDataset = model1.transform(model2.transform(dataset))

    val result: DataFrame = featurizeAndVerifyResult(catDataset,
      benchmarkOneHotTempFile.toString,
      historicOneHotFile,
      oneHotEncode = true)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[DenseVector](featuresColumn).size == 7)

    // Verify without one-hot encoding we get expected data
    val resultNoOneHot: DataFrame = featurizeAndVerifyResult(catDataset,
      benchmarkNoOneHotTempFile.toString,
      historicNoOneHotFile)
    // Verify that features column has the correct number of slots
    assert(resultNoOneHot.first().getAs[DenseVector](featuresColumn).size == 5)

    // Verify get equivalent results if we use string indexer for making categoricals
    val tmp4col = "col4tmp"
    val tmp5col = "col5tmp"
    val strind1 = new StringIndexer().setInputCol("col4").setOutputCol(tmp4col)
    val strind2 = new StringIndexer().setInputCol("col5").setOutputCol(tmp5col)
    val fit1 = strind1.fit(dataset)
    val catResult1 = fit1.transform(dataset)
    val fit2 = strind2.fit(catResult1)
    val catResult2 = fit2.transform(catResult1)
      .drop("col4", "col5")
      .withColumnRenamed(tmp4col, "col4")
      .withColumnRenamed(tmp5col, "col5")

    val resultStringIndexer: DataFrame = featurizeAndVerifyResult(catResult2,
      benchmarkStringIndexOneHotTempFile.toString,
      historicStringIndexOneHotFile,
      oneHotEncode = true)
    // Verify that features column has the correct number of slots
    assert(resultStringIndexer.first().getAs[DenseVector](featuresColumn).size == 7)
  }

  // This test currently fails on ValueIndexer, where we should handle missing values (unlike spark,
  // which fails with a null reference exception)
  ignore("Featurizing with categorical columns that have missings - using one hot encoding") {
    val cat = "Cat"
    val dog = "Dog"
    val bird = "Bird"
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, cat),
      (1, null),
      (0, bird),
      (1, null),
      (0, bird),
      (1, dog)))
      .toDF(mockLabelColumn, "col1")

    val model1 = new ValueIndexer().setInputCol("col1").setOutputCol("col1").fit(dataset)
    val catDataset = model1.transform(dataset)

    val result: DataFrame = featurizeAndVerifyResult(catDataset,
      benchmarkOneHotMissingsTempFile.toString,
      historicOneHotMissingsFile,
      oneHotEncode = true)
    // Verify that features column has the correct number of slots
    assert(result.first().getAs[DenseVector](featuresColumn).size == 4)

    // Verify without one-hot encoding we get expected data
    val resultNoOneHot: DataFrame = featurizeAndVerifyResult(catDataset,
      benchmarkNoOneHotMissingsTempFile.toString,
      historicNoOneHotMissingsFile)
    // Verify that features column has the correct number of slots
    assert(resultNoOneHot.first().getAs[DenseVector](featuresColumn).size == 4)
  }

  private def featurize(dataset: DataFrame,
                        oneHotEncode: Boolean = false,
                        includeFeaturesColumns: Boolean = true): DataFrame = {
    val featureColumns = dataset.columns.filter(_ != mockLabelColumn)
    val feat = new Featurize()
      .setNumberOfFeatures(10)
      .setFeatureColumns(Map(featuresColumn -> featureColumns))
      .setOneHotEncodeCategoricals(oneHotEncode)
      .setAllowImages(true)
    val featModel = feat.fit(dataset)
    val result = featModel.transform(dataset)
    if (includeFeaturesColumns) result else result.select(featuresColumn)
  }

  private def featurizeAndVerifyResult(dataset: DataFrame,
                               tempFile: String,
                               historicFile: File,
                               oneHotEncode: Boolean = false,
                               includeFeaturesColumns: Boolean = true): DataFrame = {
    val result = featurize(dataset, oneHotEncode, includeFeaturesColumns)
    // Write out file so it is easy to compare the results
    result.repartition(1).write.json(tempFile)
    if (!Files.exists(historicFile.toPath)) {
      // Store result in file for future
      val directory = historicFile.toString.replace(".json", "")
      result.repartition(1).write.json(directory)
      val directoryFile = new File(directory)
      val jsonFile = directoryFile.listFiles().filter(file => file.toString.endsWith(".json"))(0)
      jsonFile.renameTo(historicFile)
      FileUtils.forceDelete(directoryFile)
    }
    val expResult = session.read.json(historicFile.toString)
    // Verify the results are the same
    verifyResult(expResult, result)
    result
  }

  override def testObjects: List[TestObject[Featurize]] = List(new TestObject(
    new Featurize().setFeatureColumns(Map(featuresColumn -> mockDataset.columns)), mockDataset))

  override def reader: MLReadable[_] = Featurize
  override def modelReader: MLReadable[_] = PipelineModel
}
