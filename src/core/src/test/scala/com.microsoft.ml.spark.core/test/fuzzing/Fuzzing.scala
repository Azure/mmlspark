// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.fuzzing

import java.io.{FileNotFoundException, FileOutputStream}
import java.nio.file.{Files, Path, Paths}

import com.microsoft.ml.spark.core.env.FileUtilities.File
import com.microsoft.ml.spark.core.serialize.PythonWrappableParam
import com.microsoft.ml.spark.core.test.base.DataFrameEquality
import org.apache.commons.io.{FileUtils, IOUtils}
import org.apache.spark.ml._
import org.apache.spark.ml.param.{BooleanParam, ParamPair}
import org.apache.spark.ml.util.{MLReadable, MLWritable}
import org.apache.spark.sql.DataFrame

import scala.sys.process.Process

case class TestObject[S <: PipelineStage](stage: S,
                                          fitDF: DataFrame,
                                          transDF: DataFrame,
                                          validateDF: Option[DataFrame]) {
  def this(stage: S, df: DataFrame) = {
    this(stage, df, df, None)
  }

  def this(stage: S, fitDF: DataFrame, transDF: DataFrame) = {
    this(stage, fitDF, transDF, None)
  }

}

trait PyTestFuzzing[S <: PipelineStage] extends DataFrameEquality {

  def pyTestObjects(): Seq[TestObject[S]]

  protected val targetDir: Path = Paths.get(getClass.getClassLoader.getResource("").toURI).getParent.getParent

  protected lazy val scalaClass: Class[_ <: S] = pyTestObjects().head.stage.getClass
  protected lazy val pyClassName: String = scalaClass.getSimpleName
  protected lazy val pyModule: String = scalaClass.getName
    .replace("com.microsoft.ml.spark", "mmlspark")
    .split("\\.").dropRight(1).mkString(".")

  protected lazy val pySavedDatasetFolder: File = {
    val dir = new File(new File(targetDir.toFile, "test-pyData"), pyClassName)
    if (!dir.exists()) dir.mkdirs()
    dir
  }

  private def makePyDir(location: File) = {
    if (!location.exists()) location.mkdirs()
    val init = new File(location,"__init__.py")
    if (!init.exists()) init.createNewFile()
  }

  protected lazy val pyTestFolder: File = {
    val baseDir = new File(targetDir.toFile, "test-pyTests")
    makePyDir(baseDir)
    val dir = new File(baseDir, pyClassName)
    makePyDir(dir)
    dir
  }

  def savePyDatasets(): Unit = {
    pyTestObjects().zipWithIndex.foreach { case (to, i) =>
      val dir = new File(pySavedDatasetFolder, i.toString)
      if (dir.exists()) FileUtils.forceDelete(dir)
      dir.mkdirs()

      to.fitDF.write.parquet(new File(dir, "fit.parquet").toString)
      to.transDF.write.parquet(new File(dir, "trans.parquet").toString)
      to.validateDF.foreach(df => df.write.parquet(new File(dir, "val.parquet").toString))
    }
  }

  protected def pyParamSetter[T](p: ParamPair[T]): String = {
    val pythonValue = p.param match {
      case pyParam: PythonWrappableParam[T] => pyParam.pythonValueEncoding(p.value)
      case bp: BooleanParam => if (p.value.asInstanceOf[Boolean]) "True" else "False"
      case param => param.jsonEncode(p.value)
    }
    s".set${p.param.name.capitalize}($pythonValue)"
  }

  protected def pyInstantiateStage(s: S): String = {
    val setters = s.extractParamMap().toSeq.map(pp => pyParamSetter(pp))
    s"${s.getClass.getSimpleName}() \\\n" + setters.map(indent(_)).mkString(" \\\n")
  }

  protected def indent(code: String, tabs: Int = 1): String = {
    val padding = "    " * tabs
    code.split("\n").map(s => padding + s).mkString("\n")
  }

  protected def pyFuzzingFile(): String = {
    val testDefs = pyTestObjects().zipWithIndex.map { case (req, i) =>
      val name = i.toString
      pyFuzzingTest(
        name,
        req.stage,
        new File(new File(pySavedDatasetFolder, name), "fit.parquet"),
        new File(new File(pySavedDatasetFolder, name), "trans.parquet"),
        req.validateDF.map(_ => new File(new File(pySavedDatasetFolder, name), "val.parquet"))
      )
    }
    val projectDir = new File(new File(getClass.getResource("/").toURI), "../../../")
    val showVersionScript = new File(projectDir, "../../tools/runme/show-version")
    val mmlVersion     = sys.env.getOrElse("MML_VERSION", Process(showVersionScript.toString).!!.trim)
    val ivySettingsFile = new File(projectDir,"../ivysettings.xml").getAbsolutePath
    val scalaVersion = sys.env("SCALA_VERSION")
    val mmlPackage = s"com.microsoft.ml.spark:mmlspark_$scalaVersion:$mmlVersion"

    s"""|import unittest
        |from pyspark.sql import SparkSession
        |
        |spark = SparkSession.builder \\
        |    .master("local[*]") \\
        |    .appName("${pyClassName}Test") \\
        |    .config("spark.jars.ivySettings", "$ivySettingsFile") \\
        |    .config("spark.jars.packages", "$mmlPackage") \\
        |    .getOrCreate()
        |
        |from $pyModule import $pyClassName
        |
        |class ${pyClassName}FuzzingTest(unittest.TestCase):
        |${indent(testDefs.mkString("\n"))}
        |
        |
        |import os, xmlrunner
        |if __name__ == "__main__":
        |    result = unittest.main(testRunner=xmlrunner.XMLTestRunner(
        |        output=os.getenv("TEST_RESULTS","TestResults")),
        |        failfast=False, buffer=False, catchbreak=False)
        |""".stripMargin
  }

  protected def pyFuzzingTest(name: String,
                    stage: S,
                    fitPath: File,
                    transPath: File,
                    valPath: Option[File]): String = {
    val fittingCode = stage match {
      case _: Transformer =>
        s"""|transDf = spark.read.parquet("$transPath")
            |results = model.transform(transDf)""".stripMargin
      case _: Estimator[_] =>
        s"""|fitDf = spark.read.parquet("$fitPath")
            |transDf = spark.read.parquet("$transPath")
            |results = model.fit(fitDf).transform(transDf)""".stripMargin
    }
    val instantiator = pyInstantiateStage(stage)
    val body =
      s"""|model = $instantiator
          |$fittingCode
          |results.collect()""".stripMargin

    s"""|def testFuzzing$name(self):
        |${indent(body)}""".stripMargin
  }

  protected def pyTests: Seq[String] = {
    Seq(pyFuzzingFile())
  }

  def writePyTests(): Unit = {
    pyTests.zipWithIndex.foreach { case (test, i) =>
      IOUtils.write(
        test,
        new FileOutputStream(new File(pyTestFolder, s"test_$i.py")))
    }
  }

  test("Generate python tests"){
    savePyDatasets()
    writePyTests()
  }

}

trait ExperimentFuzzing[S <: PipelineStage] extends DataFrameEquality {

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
        case Some(vdf) => assert(res === vdf)
        case None => ()
      }
    }
  }

  test("Experiment Fuzzing") {
    testExperiments()
  }

}

trait SerializationFuzzing[S <: PipelineStage with MLWritable] extends DataFrameEquality {
  def serializationTestObjects(): Seq[TestObject[S]]

  def reader: MLReadable[_]

  def modelReader: MLReadable[_]

  val savePath: String = Files.createTempDirectory("SavedModels-").toString

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
          assert(e1.fit(fitDF).transform(transDF) === e2.fit(fitDF).transform(transDF))
        case (t1: Transformer, t2: Transformer) =>
          assert(t1.transform(transDF) === t2.transform(transDF))
        case _ => throw new IllegalArgumentException(s"$stage and $loadedStage do not have proper types")
      }
      ()
    } finally {
      try {
        FileUtils.forceDelete(new File(path))
      } catch {
        case _: FileNotFoundException =>
      }
      ()
    }
  }

  def testSerialization(): Unit = {
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
  }

  test("Serialization Fuzzing") {
    testSerialization()
  }

}

trait Fuzzing[S <: PipelineStage with MLWritable] extends PyTestFuzzing[S]
  with SerializationFuzzing[S] with ExperimentFuzzing[S] {

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
