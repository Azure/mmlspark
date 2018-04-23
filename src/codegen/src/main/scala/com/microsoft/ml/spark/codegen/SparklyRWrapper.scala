// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import com.microsoft.ml.spark.codegen.Config._
import com.microsoft.ml.spark.core.env.FileUtilities._
import org.apache.commons.lang3.StringUtils
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.{Estimator, PipelineStage, Transformer}

import scala.collection.mutable.ListBuffer

/** :: DeveloperApi ::
  * Abstraction for SparklyR wrapper generators.
  */
abstract class SparklyRWrapper(entryPoint: PipelineStage,
                               entryPointName: String,
                               entryPointQualifiedName: String) extends WritableWrapper {

  protected def functionTemplate(docString: String,
                                 classParamsString: String,
                                 setParams: String,
                                 modelStr: String,
                                 moduleAcc: String): String = {
    s"""|
        |$docString
        |ml_$entryPointName <- function(x$classParamsString)
        |{
        |  df <- spark_dataframe(x)
        |  sc <- spark_connection(df)
        |  env <- new.env(parent = emptyenv())
        |
        |  env$$model <- \"$entryPointQualifiedName\"
        |  mod <- invoke_new(sc, env$$model)
        |
        |  mod_parameterized <- mod %>%
        |$setParams
        |$modelStr
        |  transformed <- invoke($moduleAcc, \"transform\", df)
        |
        |  sdf_register(transformed)
        |}""".stripMargin
  }

  protected def header(simpleClassName: String): String = WrapperClassDoc.GenerateWrapperClassDoc(simpleClassName)
  protected def classDocTemplate(simpleClassName: String) = s"""${header(simpleClassName)}"""
  val modelStr: String
  val moduleAcc: String
  val psType: String
  val additionalParams: String

  protected def getRDefault(paramDefault: String, paramType: String,
                            defaultStringIsParsable: Boolean): String =
    paramType match {
      case "BooleanParam" =>
        StringUtils.upperCase(paramDefault)
      case "DoubleParam" | "FloatParam" | "IntParam" | "LongParam" =>
        paramDefault
      case x if x == "Param" || defaultStringIsParsable =>
        "\"" + paramDefault.replace("\\", "\\\\") + "\""
      case _ =>
        "NULL"
    }

  protected def getParamDefault(param: Param[_]): String = {
    if (!entryPoint.hasDefault(param)) "NULL"
    else {
      val paramDefault: String = entryPoint.getDefault(param).get.toString
      if (paramDefault.toLowerCase.contains(param.parent.toLowerCase)) "NULL"
      else getRDefault(paramDefault,
                       param.getClass.getSimpleName,
                       try {
                         entryPoint.getParam(param.name).w(paramDefault)
                         true
                       } catch {
                         case _: Exception => false
                       })
    }
  }

  protected def getParamConversion(paramType: String, paramName: String): String = {
    paramType match {
      case "BooleanParam"               => s"as.logical($paramName)"
      case "DoubleParam" | "FloatParam" => s"as.double($paramName)"
      case "StringArrayParam"           => s"as.array($paramName)"
      case "IntParam" | "LongParam"     => s"as.integer($paramName)"
      case "MapArrayParam" | "Param" | "StringParam" => paramName
      case _ => paramName
    }
  }

  private def paramDocTemplate(param: Param[_]): String = {
    s"""@param ${param.name} ${param.doc}"""
  }

  private def invokeParamStrTemplate(pname: String, param: Param[_]): String = {
    val convertedParam = getParamConversion(param.getClass.getSimpleName, pname)
    s"""${scopeDepth}invoke(\"set${StringUtils.capitalize(pname)}\", $convertedParam)""".stripMargin
  }

  protected def getSparklyRWrapperBase: String = {
    // Construct relevant strings
    val paramsAndDefaults = ListBuffer[String]()
    val setParamsList     = ListBuffer[String]()
    val paramDocList      = ListBuffer[String]()

    // Iterate over the params to build strings
    val allParams: Array[Param[_]] = entryPoint.params
    for (param <- allParams) {
      val pname = param.name
      setParamsList += invokeParamStrTemplate(pname, param)
      paramsAndDefaults += pname + "=" + getParamDefault(param)
      paramDocList += paramDocTemplate(param)
    }

    val funcParamsString = (if (paramsAndDefaults.isEmpty) ""
                            else paramsAndDefaults.mkString(", ", ", ", "")) +
                           additionalParams
    val setParams = setParamsList.mkString(" %>%\n")
    val simpleClassName = entryPoint.getClass.getSimpleName
    val classDocString  = classDocTemplate(simpleClassName).replace("\n", s"\n#' ${scopeDepth}")
    val paramDocString  = paramDocList.mkString("\n#' ")

    val docString =
      s"""|#' Spark ML -- $simpleClassName
          |#'
          |#' $classDocString
          |#' $paramDocString
          |#' @export""".stripMargin

    functionTemplate(docString, funcParamsString, setParams, modelStr, moduleAcc) + "\n"

  }

  def sparklyRWrapperBuilder(): String = {
    getSparklyRWrapperBase
  }

  def writeWrapperToFile(dir: File): Unit = {
    writeFile(sparklyRNamespacePath, s"export(ml_$entryPointName)\n",
              StandardOpenOption.APPEND)
    writeFile(new File(dir, s"$entryPointName.R"),
              copyrightLines + sparklyRWrapperBuilder())
  }

}

class SparklyRTransformerWrapper(entryPoint: Transformer,
                                 entryPointName: String,
                                 entryPointQualifiedName: String)
  extends SparklyRWrapper(entryPoint, entryPointName, entryPointQualifiedName) {

  override val modelStr         = ""
  override val moduleAcc        = "mod_parameterized"
  override val psType           = "Transformer"
  override val additionalParams = ""

}

class SparklyREstimatorWrapper(entryPoint: Estimator[_],
                               entryPointName: String,
                               entryPointQualifiedName: String,
                               companionModelName: String,
                               companionModelQualifiedName: String)
  extends SparklyRWrapper(entryPoint, entryPointName, entryPointQualifiedName) {

  override val modelStr: String =
  s"""|  mod_model_raw <- mod_parameterized %>%
      |    invoke(\"fit\", df)
      |
      |  mod_model <- sparklyr:::new_ml_model(mod_parameterized, mod_model_raw, mod_model_raw)
      |
      |  if (only.model)
      |    return(mod_model)
      |""".stripMargin
  override val moduleAcc = "mod_model$model"
  override val psType = "Estimator"
  override val additionalParams = ", only.model=FALSE"

}
