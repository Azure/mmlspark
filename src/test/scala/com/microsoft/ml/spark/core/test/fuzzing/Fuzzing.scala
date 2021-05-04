// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.fuzzing

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files

import com.microsoft.ml.spark.codegen.CodegenConfig
import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.commons.io.FileUtils
import org.apache.spark.ml._
import org.apache.spark.ml.param.{DataFrameEquality, ExternalPythonWrappableParam, ParamPair}
import org.apache.spark.ml.util.{MLReadable, MLWritable}
import org.apache.spark.sql.DataFrame
import com.microsoft.ml.spark.codegen.GenerationUtils._

/**
  * Class for holding test information, call by name to avoid uneccesary computations in test generations
  *
  * @param stage         Pipeline stage for testing
  * @param fitDFArg      Dataframe to fit
  * @param transDFArg    Dataframe to transform
  * @param validateDFArg Optional dataframe to validate against
  * @tparam S The type of the stage
  */
class TestObject[S <: PipelineStage](val stage: S,
                                     fitDFArg: => DataFrame,
                                     transDFArg: => DataFrame,
                                     validateDFArg: => Option[DataFrame]) {
  lazy val fitDF: DataFrame = fitDFArg
  lazy val transDF: DataFrame = transDFArg
  lazy val validateDF: Option[DataFrame] = validateDFArg

  def this(stage: S, df: => DataFrame) = {
    this(stage, df, df, None)
  }

  def this(stage: S, fitDF: => DataFrame, transDF: => DataFrame) = {
    this(stage, fitDF, transDF, None)
  }

}

trait PyTestFuzzing[S <: PipelineStage] extends TestBase with DataFrameEquality {

  def pyTestObjects(): Seq[TestObject[S]]

  val testClassName: String = this.getClass.getName.split(".".toCharArray).last

  val testDataDir: File = FileUtilities.join(
    CodegenConfig.TestDataDir, this.getClass.getName.split(".".toCharArray).last)

  def saveDataset(df: DataFrame, name: String): Unit = {
    df.write.mode("overwrite").parquet(new File(testDataDir, s"$name.parquet").toString)
  }

  def saveModel(model: S, name: String): Unit = {
    model match {
      case writable: MLWritable =>
        writable.write.overwrite().save(new File(testDataDir, s"$name.model").toString)
      case _ =>
        throw new IllegalArgumentException(s"${model.getClass.getName} is not writable")
    }

  }

  val testFitting = false

  def saveTestData(): Unit = {
    testDataDir.mkdirs()
    pyTestObjects().zipWithIndex.foreach { case (to, i) =>
      saveModel(to.stage, s"model-$i")
      if (testFitting) {
        saveDataset(to.fitDF, s"fit-$i")
        saveDataset(to.transDF, s"trans-$i")
        to.validateDF.foreach(saveDataset(_, s"val-$i"))
      }
    }
  }

  def pyTestInstantiateModel(stage: S, num: Int): String = {
    val fullParamMap = stage.extractParamMap().toSeq
    val partialParamMap = stage.extractParamMap().toSeq.filter(pp => stage.get(pp.param).isDefined)
    val stageName = stage.getClass.getName.split(".".toCharArray).last

    def instantiateModel(paramMap: Seq[ParamPair[_]]) = {
      val externalLoadlingLines = paramMap.flatMap { pp =>
        pp.param match {
          case ep: ExternalPythonWrappableParam[_] =>
            Some(ep.pyLoadLine(num))
          case _ => None
        }
      }.mkString("\n")
      s"""
         |$externalLoadlingLines
         |
         |model = $stageName(
         |${indent(paramMap.map(pyRenderParam(_)).mkString(",\n"), 1)}
         |)
         |
         |""".stripMargin
    }

    try {
      instantiateModel(fullParamMap)
    } catch {
      case _: NotImplementedError =>
        println(s"could not generate full test for ${stageName}, resorting to partial test")
        instantiateModel(partialParamMap)
    }
  }


  def makePyTests(testObject: TestObject[S], num: Int): String = {
    val stage = testObject.stage
    val stageName = stage.getClass.getName.split(".".toCharArray).last
    val fittingTest = stage match {
      case _: Estimator[_] if testFitting =>
        s"""
           |fdf = spark.read.parquet(join(test_data_dir, "fit-$num.parquet"))
           |tdf = spark.read.parquet(join(test_data_dir, "trans-$num.parquet"))
           |model.fit(fdf).transform(tdf).show()
           |""".stripMargin
      case _: Transformer if testFitting =>
        s"""
           |tdf = spark.read.parquet(join(test_data_dir, "trans-$num.parquet"))
           |model.transform(tdf).show()
           |""".stripMargin
      case _ => ""
    }

    s"""
       |def test_${stageName}_constructor_$num(self):
       |${indent(pyTestInstantiateModel(stage, num), 1)}
       |
       |    self.assert_correspondence(model, "py-constructor-model-$num.model", $num)
       |
       |${indent(fittingTest, 1)}
       |
       |""".stripMargin
  }


  def makePyTestFile(): Unit = {
    spark
    saveTestData()
    val generatedTests = pyTestObjects().zipWithIndex.map { case (to, i) => makePyTests(to, i) }
    val stage = pyTestObjects().head.stage
    val stageName = stage.getClass.getName.split(".".toCharArray).last
    val importPath = stage.getClass.getName.split(".".toCharArray).dropRight(1)
    val importPathString = importPath.mkString(".").replaceAllLiterally("com.microsoft.ml.spark", "mmlspark")
    val testClass =
      s"""import unittest
         |from mmlsparktest.spark import *
         |from $importPathString import $stageName
         |from os.path import join
         |import json
         |
         |test_data_dir = "${testDataDir.toString.replaceAllLiterally("\\", "\\\\")}"
         |
         |
         |class $testClassName(unittest.TestCase):
         |    def assert_correspondence(self, model, name, num):
         |        model.write().overwrite().save(join(test_data_dir, name))
         |        sc._jvm.com.microsoft.ml.spark.core.utils.ModelEquality.assertEqual(
         |            "${stage.getClass.getName}",
         |            str(join(test_data_dir, name)),
         |            str(join(test_data_dir, "model-{}.model".format(num)))
         |        )
         |
         |${indent(generatedTests.mkString("\n\n"), 1)}
         |
         |if __name__ == "__main__":
         |    result = unittest.main()
         |
         |""".stripMargin

    val testFolders = importPath.mkString(".")
      .replaceAllLiterally("com.microsoft.ml.spark", "mmlsparktest").split(".".toCharArray)
    val testDir = FileUtilities.join((Seq(CodegenConfig.PyTestDir.toString) ++ testFolders.toSeq): _*)
    testDir.mkdirs()
    Files.write(
      FileUtilities.join(testDir, "test_" + camelToSnake(testClassName) + ".py").toPath,
      testClass.getBytes(StandardCharsets.UTF_8))
  }

}

trait ExperimentFuzzing[S <: PipelineStage] extends TestBase with DataFrameEquality {

  def experimentTestObjects(): Seq[TestObject[S]]

  def runExperiment(s: S, fittingDF: DataFrame, transformingDF: DataFrame): DataFrame = {
    s match {
      case t: Transformer =>
        t.transform(transformingDF)
      case e: Estimator[_] =>
        e.fit(fittingDF).transform(transformingDF)
      case _ => throw new MatchError(s"$s is not a Transformer or Estimator")
    }
  }

  def testExperiments(): Unit = {
    experimentTestObjects().foreach { req =>
      val res = runExperiment(req.stage, req.fitDF, req.transDF)
      req.validateDF match {
        case Some(vdf) => assertDFEq(res, vdf)
        case None => ()
      }
    }
  }

  test("Experiment Fuzzing") {
    testExperiments()
  }

}

trait SerializationFuzzing[S <: PipelineStage with MLWritable] extends TestBase with DataFrameEquality {
  def serializationTestObjects(): Seq[TestObject[S]]

  def reader: MLReadable[_]

  def modelReader: MLReadable[_]

  val useShm: Boolean = sys.env.getOrElse("MMLSPARK_TEST_SHM", "false").toBoolean

  lazy val savePath: String = {
    if (useShm) {
      val f = new File(s"/dev/shm/SavedModels-${System.currentTimeMillis()}")
      f.mkdir()
      f.toString
    } else {
      Files.createTempDirectory("SavedModels-").toString
    }
  }

  val ignoreEstimators: Boolean = false

  private def testSerializationHelper(path: String,
                                      stage: PipelineStage with MLWritable,
                                      reader: MLReadable[_],
                                      fitDF: DataFrame, transDF: DataFrame): Unit = {
    try {
      stage.write.overwrite().save(path)
      assert(new File(path).exists())
      val loadedStage = reader.load(path)
      (stage, loadedStage) match {
        case (e1: Estimator[_], e2: Estimator[_]) =>
          val df1 = e1.fit(fitDF).transform(transDF)
          val df2 = e2.fit(fitDF).transform(transDF)
          assertDFEq(df1, df2)
        case (t1: Transformer, t2: Transformer) =>
          val df1 = t1.transform(transDF)
          val df2 = t2.transform(transDF)
          assertDFEq(df1, df2)
        case _ => throw new IllegalArgumentException(s"$stage and $loadedStage do not have proper types")
      }
      ()
    } finally {
      if (new File(path).exists()) FileUtils.forceDelete(new File(path))
    }
  }

  def testSerialization(): Unit = {
    try {
      serializationTestObjects().foreach { req =>
        val fitStage = req.stage match {
          case stage: Estimator[_] =>
            if (!ignoreEstimators) {
              testSerializationHelper(savePath + "/stage", stage, reader, req.fitDF, req.transDF)
            }
            stage.fit(req.fitDF).asInstanceOf[PipelineStage with MLWritable]
          case stage: Transformer => stage
          case s => throw new IllegalArgumentException(s"$s does not have correct type")
        }
        testSerializationHelper(savePath + "/fitStage", fitStage, modelReader, req.transDF, req.transDF)

        val pipe = new Pipeline().setStages(Array(req.stage.asInstanceOf[PipelineStage]))
        if (!ignoreEstimators) {
          testSerializationHelper(savePath + "/pipe", pipe, Pipeline, req.fitDF, req.transDF)
        }
        val fitPipe = pipe.fit(req.fitDF)
        testSerializationHelper(savePath + "/fitPipe", fitPipe, PipelineModel, req.transDF, req.transDF)
      }
    } finally {
      if (new File(savePath).exists) FileUtils.forceDelete(new File(savePath))
    }
  }

  test("Serialization Fuzzing") {
    testSerialization()
  }

}

trait Fuzzing[S <: PipelineStage with MLWritable] extends SerializationFuzzing[S]
  with ExperimentFuzzing[S] with PyTestFuzzing[S] {

  def testObjects(): Seq[TestObject[S]]

  def pyTestObjects(): Seq[TestObject[S]] = testObjects()

  def serializationTestObjects(): Seq[TestObject[S]] = testObjects()

  def experimentTestObjects(): Seq[TestObject[S]] = testObjects()

}

trait TransformerFuzzing[S <: Transformer with MLWritable] extends Fuzzing[S] {

  override val ignoreEstimators: Boolean = true

  override def modelReader: MLReadable[_] = reader

}

trait EstimatorFuzzing[S <: Estimator[_] with MLWritable] extends Fuzzing[S]
