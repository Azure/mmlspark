// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import com.microsoft.ml.spark.codegen.Config._
import com.microsoft.ml.spark.core.env.FileUtilities._
import org.apache.commons.lang3.StringUtils
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.{Estimator, PipelineStage, Transformer}

/** :: DeveloperApi ::
  * Abstraction for PySpark wrapper generators.
  */
abstract class PySparkWrapperTest(entryPoint: PipelineStage,
                                  entryPointName: String,
                                  entryPointQualifiedName: String,
                                  packageName: String) extends WritableWrapper {

  // general classes are imported from the mmlspark directy;
  // internal classes have to be imported from their packages
  private def importClass(entryPointName:String):String = {
    val packageString = {
      if (packageName.startsWith("com.microsoft.ml.spark")){
        packageName
          .drop("com.microsoft.ml.spark".length)
      } else {
        ""
      }
    }

    if (entryPointName startsWith internalPrefix) s"from mmlspark$packageString.$entryPointName import $entryPointName"
    else s"from mmlspark$packageString import $entryPointName"
  }

  protected def classTemplate(classParams: String, paramGettersAndSetters: String) =
    s"""|import unittest
        |import pandas as pd
        |import numpy as np
        |import pyspark.ml, pyspark.ml.feature
        |from pyspark import SparkContext
        |from pyspark.sql import SQLContext
        |from pyspark.ml.classification import LogisticRegression
        |from pyspark.ml.regression import LinearRegression
        |${importClass(entryPointName)}
        |from pyspark.ml.feature import Tokenizer
        |from mmlspark.stages.featurize import TrainClassifier
        |from mmlspark.stages import ValueIndexer
        |
        |sc = SparkContext()
        |
        |class ${entryPointName}Test(unittest.TestCase):
        |
        |    def test_${entryPointName}AllDefaults(self):
        |        my$entryPointName = $entryPointName()
        |        my$entryPointName.setParams($classParams)
        |        self.assertNotEqual(my$entryPointName, None)
        |
        |$paramGettersAndSetters
        |
        |""".stripMargin

  protected val unittestString =
    s"""|
        |import os, xmlrunner
        |if __name__ == "__main__":
        |    result = unittest.main(testRunner=xmlrunner.XMLTestRunner(output=os.getenv("TEST_RESULTS","TestResults")),
        |                           failfast=False, buffer=False, catchbreak=False)
        |""".stripMargin

  protected def setAndGetTemplate(paramName: String, value: String) =
    s"""|    def test_set$paramName(self):
        |        my$entryPointName = $entryPointName()
        |        val = $value
        |        my$entryPointName.set$paramName(val)
        |        retVal = my$entryPointName.get$paramName()
        |        self.assertEqual(val, retVal)
        |""".stripMargin

  protected def tryFitSetupTemplate(entryPointName: String) =
    s"""|    def test_$entryPointName(self):
        |        dog = "dog"
        |        cat = "cat"
        |        bird = "bird"
        |        tmp1  = {
        |            "col1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        |            "col2": [2, 3, 4, 5, 1, 3, 3, 4, 0, 2, 3, 4],
        |            "col3": [0.50, 0.40, 0.78, 0.12, 0.50, 0.40, 0.78, 0.12, 0.50, 0.40, 0.78, 0.12],
        |            "col4": [0.60, 0.50, 0.99, 0.34, 0.60, 0.50, 0.99, 0.34, 0.60, 0.50, 0.99, 0.34],
        |            "col5": [dog, cat, dog, cat, dog, bird, dog, cat, dog, bird, dog, cat],
        |            "col6": [cat, dog, bird, dog, bird, dog, cat, dog, cat, dog, bird, dog],
        |            "image": [cat, dog, bird, dog, bird, dog, cat, dog, cat, dog, bird, dog]
        |        }
        |        sqlC = SQLContext(sc)
        |        pddf = pd.DataFrame(tmp1)
        |        pddf["col1"] = pddf["col1"].astype(np.float64)
        |        pddf["col2"] = pddf["col2"].astype(np.int32)
        |        data = sqlC.createDataFrame(pddf)
        |""".stripMargin

  protected def tryTransformTemplate(entryPointName: String, param: String) =
      s"""|        my$entryPointName = $entryPointName($param)
          |        prediction = my$entryPointName.transform(data)
          |        self.assertNotEqual(prediction, None)
          |""".stripMargin

  protected def tryFitTemplate(entryPointName: String, model: String) =
      s"""|        my$entryPointName = $entryPointName(model=$model, labelCol="col1", numFeatures=5)
          |        model = my$entryPointName.fit(data)
          |        self.assertNotEqual(model, None)""".stripMargin

  protected def tryMultiColumnFitTemplate(entryPointName: String, model: String) =
      s"""|        my$entryPointName = $entryPointName(baseStage=$model, inputCols=["col1"], outputCols=["out"])
          |        model = my$entryPointName.fit(data)
          |        self.assertNotEqual(model, None)""".stripMargin
  private def evaluateSetupTemplate(entryPointName: String) =
    s"""|    def test_$entryPointName(self):
        |        data = {
        |            "labelColumn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        |            "col1": [2, 3, 4, 5, 1, 3, 3, 4, 0, 2, 3, 4],
        |            "col2": [0.50, 0.40, 0.78, 0.12, 0.50, 0.40, 0.78, 0.12, 0.50, 0.40, 0.78, 0.12],
        |            "col3": [0.60, 0.50, 0.99, 0.34, 0.60, 0.50, 0.99, 0.34, 0.60, 0.50, 0.99, 0.34],
        |            "col4": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        |        }
        |        sqlC = SQLContext(sc)
        |        pddf = pd.DataFrame(data)
        |        data = sqlC.createDataFrame(pddf)
        |        model = TrainClassifier(model=LogisticRegression(), labelCol="labelColumn",
        |                                numFeatures=256).fit(data)
        |""".stripMargin

  protected def computeStatisticsTemplate(entryPointName: String) =
    s"""|${evaluateSetupTemplate(entryPointName)}
        |        scoredData = model.transform(data)
        |        scoredData.limit(10).toPandas()
        |        evaluatedData = $entryPointName().transform(scoredData)
        |        self.assertNotEqual(evaluatedData, None)
        |""".stripMargin

  protected def valueIndexerModelTemplate(entryPointName: String) =
    s"""|${tryFitSetupTemplate(entryPointName)}
        |        valueData = $entryPointName(inputCol="col5", outputCol="catOutput",
        |                                    dataType="string", levels=["dog", "cat", "bird"]).transform(data)
        |        self.assertNotEqual(valueData, None)
        |""".stripMargin

  protected def indexToValueTemplate(entryPointName: String) =
    s"""|${tryFitSetupTemplate(entryPointName)}
        |        indexModel = ValueIndexer(inputCol="col5", outputCol="catOutput").fit(data)
        |        indexedData = indexModel.transform(data)
        |        valueData = $entryPointName(inputCol="catOutput", outputCol="origDomain").transform(indexedData)
        |        self.assertNotEqual(valueData, None)
        |""".stripMargin

  protected def evaluateTemplate(entryPointName: String) =
    s"""|${evaluateSetupTemplate(entryPointName)}
        |        model = TrainClassifier(model=LogisticRegression(), labelCol="labelColumn",
        |                                numFeatures=256).fit(data)
        |        evaluateModels = FindBestModel(models=[model, model]).fit(data)
        |        bestModel = evaluateModels.transform(data)
        |        self.assertNotEqual(bestModel, None)
        |""".stripMargin

  // These params are need custom handling. For now, just skip them so we have tests that pass.
  private lazy val skippedParams =  Set[String]("models", "model", "cntkModel", "stage")
  protected def isSkippedParam(paramName: String): Boolean = skippedParams.contains(paramName)
  protected def isModel(paramName: String): Boolean = paramName.toLowerCase() == "model"
  protected def isBaseTransformer(paramName: String): Boolean = paramName.toLowerCase() == "basetransformer"
  protected def tryFitString(entryPointName: String): String =
    if (entryPointName.contains("Regressor") && !entryPointName.contains("LightGBM"))
      tryFitTemplate(entryPointName, "LinearRegression(solver=\"l-bfgs\")")
    else if (entryPointName.contains("Classifier") && !entryPointName.contains("LightGBM"))
      tryFitTemplate(entryPointName, "LogisticRegression()")
    else if (entryPointName.contains("MultiColumnAdapter"))
      tryMultiColumnFitTemplate(entryPointName, "ValueIndexer()")
    else ""
  protected def computeStatisticsString(entryPointName: String): String = computeStatisticsTemplate(entryPointName)
  protected def evaluateString(entryPointName: String): String          = evaluateTemplate(entryPointName)
  protected def indexToValueString(entryPointName: String): String      = indexToValueTemplate(entryPointName)
  protected def valueIndexerModelString(entryPointName: String): String = valueIndexerModelTemplate(entryPointName)
  protected def tryTransformString(entryPointName: String): String = {
    val param: String =
      entryPointName match {
        case "_CNTKModel" | "MultiTokenizer" | "NltTokenizeTransform" | "TextTransform"
           | "TextNormalizerTransform" | "WordTokenizeTransform" => "inputCol=\"col5\""
        case "DataConversion"      => "col=\"col1\", convertTo=\"double\""
        case "DropColumns"         => "cols=[\"col1\"]"
        case "EnsembleByKey"       => "keys=[\"col1\"], cols=[\"col3\"]"
        case "FastVectorAssembler" => "inputCols=\"col1\""
        case "IndexToValue"        => "inputCol=\"catOutput\""
        case "MultiNGram"          => "inputColumns=np.array([ \"col5\", \"col6\" ])"
        case "RenameColumn"        => "inputCol=\"col5\", outputCol=\"catOutput1\""
        case "Repartition"         => "n=2"
        case "SelectColumns"       => "cols=[\"col1\"]"
        case "TextPreprocessor"    => "inputCol=\"col5\", outputCol=\"catOutput1\", normFunc=\"identity\""
        case "ValueIndexerModel"   => "inputCol=\"col5\", outputCol=\"catOutput\", " +
          "dataType=\"string\", levels=[\"dog\", \"cat\", \"bird\"]"
        case "WriteBlob"           => "blobPath=\"file:///tmp/" + java.util.UUID.randomUUID + ".tsv\""
        case _ => ""
      }
    tryTransformTemplate(entryPointName, param)
  }

  protected def getPythonizedDefault(paramDefault: String, paramType: String,
                                     defaultStringIsParsable: Boolean): String =
    paramType match {
      case "BooleanParam" =>
        StringUtils.capitalize(paramDefault)
      case "DoubleParam" | "FloatParam" | "IntParam" | "LongParam" =>
        paramDefault
      case x if x == "Param" || defaultStringIsParsable =>
        "\"" + paramDefault + "\""
      case _ =>
        "None"
    }

  protected def getParamDefault(param: Param[_]): (String, String) = {
    if (!entryPoint.hasDefault(param)) ("None", null)
    else {
      val paramParent: String = param.parent
      val paramDefault = entryPoint.getDefault(param).get.toString
      if (paramDefault.toLowerCase.contains(paramParent.toLowerCase))
        ("None",
         paramDefault.substring(paramDefault.lastIndexOf(paramParent) + paramParent.length))
      else {
        val defaultStringIsParsable: Boolean =
          try {
            entryPoint.getParam(param.name).w(paramDefault)
            true
          } catch {
            case e: Exception => false
          }
        (getPythonizedDefault(paramDefault, param.getClass.getSimpleName, defaultStringIsParsable),
         null)
      }
    }
  }

  protected def getPysparkWrapperTestBase: String = {
    // Iterate over the params to build strings
    val paramGettersAndSettersString =
      entryPoint.params.filter { param => !isSkippedParam(param.name)
      }.map { param =>
        val value = if (isModel(param.name)) "LogisticRegression()"
                    else if (isBaseTransformer(param.name)) "Tokenizer()"
                    else getParamDefault(param)._1
        setAndGetTemplate(StringUtils.capitalize(param.name), value)
      }.mkString("\n")
    val classParamsString =
      entryPoint.params.map(param => param.name + "=" + getParamDefault(param)._1).mkString(", ")
    classTemplate(classParamsString, paramGettersAndSettersString)
  }

  def pysparkWrapperTestBuilder(): String = {
    copyrightLines + getPysparkWrapperTestBase
  }

  def writeWrapperToFile(dir: File): Unit = {
    writeFile(new File(dir, entryPointName + "_tests.py"), pysparkWrapperTestBuilder())
  }

}

class PySparkTransformerWrapperTest(entryPoint: Transformer,
                                    entryPointName: String,
                                    entryPointQualifiedName: String,
                                    packageName: String)
  extends PySparkWrapperTest(entryPoint,
    entryPointName,
    entryPointQualifiedName,
    packageName) {

  // The transformer tests for FastVectorAssembler ... UnrollImage are disabled for the moment.
  override def pysparkWrapperTestBuilder(): String = {
    val transformTest =
      entryPointName match {
        case "ComputeModelStatistics" => computeStatisticsString(entryPointName)
        case "ComputePerInstanceStatistics" => computeStatisticsString(entryPointName)
        case "IndexToValue" => indexToValueString(entryPointName)
        case "ValueIndexerModel" => valueIndexerModelString(entryPointName)
        case "_CNTKModel" | "FastVectorAssembler" | "_UDFTransformer"
           | "MultiNGram" | "ImageFeaturizer" | "_ImageFeaturizer"
           | "_ImageTransformer" | "UnrollImage" | "HashTransform" | "Timer"
           | "StopWordsRemoverTransform"  | "ImageSetAugmenter" | "PowerBITransformer"
           | "CustomInputParser" | "CustomOutputParser" | "FlattenBatch"
           | "HTTPTransformer" | "JSONOutputParser" | "JSONInputParser"
           | "SimpleHTTPTransformer"
           => ""
        case _ =>
          tryFitSetupTemplate(entryPointName) + tryTransformString(entryPointName)
      }
    super.pysparkWrapperTestBuilder + transformTest + unittestString
  }

}

class PySparkEstimatorWrapperTest(entryPoint: Estimator[_],
                                  entryPointName: String,
                                  entryPointQualifiedName: String,
                                  companionModelName: String,
                                  companionModelQualifiedName: String,
                                  packageName:String)
    extends PySparkWrapperTest(entryPoint, entryPointName, entryPointQualifiedName, packageName) {

  private val modelName = entryPointName + "Model"

  override def pysparkWrapperTestBuilder(): String = {
    val testString =
      if (entryPointName == "FindBestModel")
        evaluateString(entryPointName)
      else
        tryFitSetupTemplate(entryPointName) + tryFitString(entryPointName)
    super.pysparkWrapperTestBuilder + testString + unittestString
  }

}
