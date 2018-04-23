// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.fuzzing

import com.microsoft.ml.spark.core.contracts.{HasFeaturesCol, HasInputCol, HasLabelCol, HasOutputCol}
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.utils.JarLoadingUtils
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{MLReadable, MLWritable}

import scala.language.existentials

/** Tests to validate fuzzing of modules. */
class FuzzingTest extends TestBase {

  // Needed because the session in TB is lazy
  session

  // Use this for more detailed output from the Jar Loader
  val debug = false

  // use this to quickly see all the results for all failing modules
  // Note that this could make the tests pass when they should be failing
  val disableFailure = false

  test("Verify stage fitting and transforming") {
    val exemptions: Set[String] = Set(
      "org.apache.spark.ml.feature.FastVectorAssembler",
      "com.microsoft.ml.spark.stages.ValueIndexerModel",
      "com.microsoft.ml.spark.cntk.train.CNTKLearner",
      "com.microsoft.ml.spark.stages.hyper.TuneHyperparameters",
      "com.microsoft.ml.spark.stages.ComputePerInstanceStatistics",
      "com.microsoft.ml.spark.stages.DataConversion"
    )
    val applicableStages = pipelineStages.filter(t => !exemptions(t.getClass.getName))
    val applicableClasses = applicableStages.map(_.getClass.asInstanceOf[Class[_]]).toSet
    val classToFuzzer: Map[Class[_], ExperimentFuzzing[_ <: PipelineStage]] =
      experimentFuzzers.map(f =>
        (f.experimentTestObjects().head.stage.getClass, f)).toMap
    val classesWithFuzzers = classToFuzzer.keys
    val classesWithoutFuzzers = applicableClasses.diff(classesWithFuzzers.toSet)
    assertOrLog(classesWithoutFuzzers.isEmpty,
                "These classes do not have Experiment fuzzers, \n" +
                  "(try extending Estimator/Transformer Fuzzing): \n" +
                  classesWithoutFuzzers.mkString("\n"))
  }

  test("Verify all stages can be serialized") {
    val exemptions: Set[String] = Set(
      "org.apache.spark.ml.feature.FastVectorAssembler",
      "com.microsoft.ml.spark.stages.ValueIndexerModel",
      "com.microsoft.ml.spark.cntk.train.CNTKLearner",
      "com.microsoft.ml.spark.stages.ComputePerInstanceStatistics",
      "com.microsoft.ml.spark.stages.DataConversion",
      "com.microsoft.ml.spark.stages.hyper.TuneHyperparameters"
    )
    val applicableStages = pipelineStages.filter(t => !exemptions(t.getClass.getName))
    val applicableClasses = applicableStages.map(_.getClass.asInstanceOf[Class[_]]).toSet
    val classToFuzzer: Map[Class[_], SerializationFuzzing[_ <: PipelineStage with MLWritable]] =
      serializationFuzzers.map(f => (f.serializationTestObjects().head.stage.getClass, f)).toMap
    val classesWithFuzzers = classToFuzzer.keys
    val classesWithoutFuzzers = applicableClasses.diff(classesWithFuzzers.toSet)
    assertOrLog(classesWithoutFuzzers.isEmpty,
      "These classes do not have Serialization fuzzers,\n" +
        "(try extending Estimator/Transformer Fuzzing):\n  " +
        classesWithoutFuzzers.mkString("\n  "))
  }

  ignore("Verify all stages can be tested in python") {
    val exemptions: Set[String] = Set(
      "org.apache.spark.ml.feature.FastVectorAssembler",
      "com.microsoft.ml.spark.stages.ValueIndexerModel",
      "com.microsoft.ml.spark.cntk.train.CNTKLearner",
      "com.microsoft.ml.spark.stages.ComputePerInstanceStatistics",
      "com.microsoft.ml.spark.stages.DataConversion",
      "com.microsoft.ml.spark.stages.hyper.TuneHyperparameters",
      "com.microsoft.ml.spark.lightgbm.LightGBMClassifier",
      "com.microsoft.ml.spark.lightgbm.LightGBMRegressor"
    )
    val applicableStages = pipelineStages.filter(t => !exemptions(t.getClass.getName))
    val applicableClasses = applicableStages.map(_.getClass.asInstanceOf[Class[_]]).toSet
    val classToFuzzer: Map[Class[_], PyTestFuzzing[_ <: PipelineStage]] =
      pytestFuzzers.map(f => (f.pyTestObjects().head.stage.getClass, f)).toMap
    val classesWithFuzzers = classToFuzzer.keys
    val classesWithoutFuzzers = applicableClasses.diff(classesWithFuzzers.toSet)
    assertOrLog(classesWithoutFuzzers.isEmpty, classesWithoutFuzzers.mkString("\n"))

    applicableClasses.foreach { clazz =>
      //classToFuzzer(clazz).saveDatasets()
      //classToFuzzer(clazz).getPyTests()
      // TODO implement logic for creating and running pytests
      // TODO maybe move to codegen

      println(s"$clazz round trip succeeded")
    }
  }

  // TODO verify that model UIDs match the class names, perhaps use a Trait

  test("Verify all pipeline stages don't have exotic characters") {
    val badChars = List(",", "\"", "'", ".")
    pipelineStages.foreach { pipelineStage =>
      pipelineStage.params.foreach { param =>
        assertOrLog(!param.name.contains(badChars), param.name)
        assertOrLog(!param.doc.contains("\""), param.doc)
      }
    }
  }

  test("Verify all pipeline stage values match their param names") {
    val exemptions: Set[String] = Set[String](
      "com.microsoft.ml.spark.stages.basic.UDFTransformer") // needs to hide setters from model
    pipelineStages.foreach { pipelineStage =>
      if (!exemptions(pipelineStage.getClass.getName)) {
        val paramFields =
          pipelineStage.getClass.getDeclaredFields
            .filter(f => classOf[Param[Any]].isAssignableFrom(f.getType))
        val paramNames = paramFields.map { f =>
          f.setAccessible(true)
          val p = f.get(pipelineStage)
          p.asInstanceOf[Param[Any]].name
        }
        val paramFieldNames = paramFields.map(_.getName)
        assertOrLog(paramNames === paramFieldNames, pipelineStage.getClass.getName)
      }
    }
  }

  test("Verify correct use of mixins") {
    val triggers = Map(
      "inputCol" -> classOf[HasInputCol],
      "inputColumn" -> classOf[HasInputCol],
      "outputCol" -> classOf[HasOutputCol],
      "outputColumn" -> classOf[HasOutputCol],
      "labelCol" -> classOf[HasLabelCol],
      "labelColumn" -> classOf[HasLabelCol],
      "featuresCol" -> classOf[HasFeaturesCol],
      "featuresColumn" -> classOf[HasFeaturesCol]
    )

    val exemptions = Set[String](
      "org.apache.spark.ml.feature.FastVectorAssembler", // In Spark namespace
      "com.microsoft.ml.spark.lightgbm.LightGBMClassifier", // HasFeaturesCol is part of spark's base class
      "com.microsoft.ml.spark.lightgbm.LightGBMRegressor", // HasFeaturesCol is part of spark's base class
      "com.microsoft.ml.spark.stages.text.TextFeaturizer" // needs to hide setters from model
    )
    pipelineStages.foreach { stage =>
      if (!exemptions(stage.getClass.getName)) {
        stage.params.foreach { param =>
          triggers.get(param.name) match {
            case Some(clazz) =>
              assertOrLog(clazz.isAssignableFrom(stage.getClass),
                stage.getClass.getName + " needs to extend " + clazz.getName)
            case None =>
          }
        }
      }
    }
  }

  private def assertOrLog(condition: Boolean, hint: String = "",
                          disableFailure: Boolean = disableFailure): Unit = {
    if (disableFailure && !condition) println(hint)
    else                              assert(condition, hint)
    ()
  }

  // set the context loader to pick up on the jars
  Thread.currentThread().setContextClassLoader(JarLoadingUtils.classLoader)

  private lazy val transformers: List[Transformer] = JarLoadingUtils.loadClass[Transformer](debug = debug)

  private lazy val estimators: List[Estimator[_]] = JarLoadingUtils.loadClass[Estimator[_]](debug = debug)

  private lazy val readers: List[MLReadable[_]] = JarLoadingUtils.loadObject[MLReadable[_]](debug = debug)

  private lazy val pipelineStages: List[PipelineStage] = JarLoadingUtils.loadClass[PipelineStage](debug = debug)

  private lazy val experimentFuzzers: List[ExperimentFuzzing[_ <: PipelineStage]] =
    JarLoadingUtils.loadTestClass[ExperimentFuzzing[_ <: PipelineStage]](debug = debug)

  private lazy val serializationFuzzers: List[SerializationFuzzing[_ <: PipelineStage with MLWritable]] =
    JarLoadingUtils.loadTestClass[SerializationFuzzing[_ <: PipelineStage with MLWritable]](debug = debug)

  private lazy val pytestFuzzers: List[PyTestFuzzing[_ <: PipelineStage]] =
    JarLoadingUtils.loadTestClass[PyTestFuzzing[_ <: PipelineStage]](debug = debug)

}
