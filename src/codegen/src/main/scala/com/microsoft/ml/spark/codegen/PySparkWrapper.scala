// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import com.microsoft.ml.spark.codegen.Config._
import com.microsoft.ml.spark.core.env.FileUtilities.{File, _}
import org.apache.commons.lang3.StringUtils
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.{Estimator, PipelineStage, Transformer}

import scala.collection.mutable.ListBuffer

/** :: DeveloperApi ::
  * Abstraction for PySpark wrapper generators.
  */
abstract class PySparkWrapper(entryPoint: PipelineStage,
                              entryPointName: String,
                              entryPointQualifiedName: String)
    extends WritableWrapper {

  private val additionalImports = Map(
    ("complexTypes",
      s"from ${pyDir.getName}.TypeConversionUtils import generateTypeConverter, complexTypeConverter"),
    ("utils", s"from ${pyDir.getName}.Utils import *")
  )

  // Note: in the get/set with kwargs, there is an if/else that is due to the fact that since 2.1.1,
  //   kwargs is an instance attribute.  Once support for 2.1.0 is dropped, the else part of the
  //   if/else can be removed
  protected def classTemplate(importsString: String, inheritanceString: String,
                              classParamsString: String,
                              paramDefinitionsAndDefaultsString: String,
                              paramGettersAndSettersString: String,
                              classDocString: String, paramDocString: String,
                              classParamDocString: String): String = {
    s"""|$copyrightLines
        |
        |import sys
        |if sys.version >= '3':
        |    basestring = str
        |
        |from pyspark.ml.param.shared import *
        |from pyspark import keyword_only
        |from pyspark.ml.util import JavaMLReadable, JavaMLWritable
        |from pyspark.ml.wrapper import JavaTransformer, JavaEstimator, JavaModel
        |from pyspark.ml.common import inherit_doc
        |$importsString
        |
        |@inherit_doc
        |class $entryPointName($inheritanceString):
        |    "\""
        |$classDocString
        |
        |$classParamDocString
        |    "\""
        |
        |    @keyword_only
        |    def __init__(self, $classParamsString):
        |        super($entryPointName, self).__init__()
        |        self._java_obj = self._new_java_obj(\"$entryPointQualifiedName\")
        |$paramDefinitionsAndDefaultsString
        |        if hasattr(self, \"_input_kwargs\"):
        |            kwargs = self._input_kwargs
        |        else:
        |            kwargs = self.__init__._input_kwargs
        |        self.setParams(**kwargs)
        |
        |    @keyword_only
        |    def setParams(self, $classParamsString):
        |        "\""
        |        Set the (keyword only) parameters
        |
        |        Args:
        |
        |$paramDocString
        |        "\""
        |        if hasattr(self, \"_input_kwargs\"):
        |            kwargs = self._input_kwargs
        |        else:
        |            kwargs = self.__init__._input_kwargs
        |        return self._set(**kwargs)
        |
        |$paramGettersAndSettersString
        |""".stripMargin
  }

  // Complex parameters need type converters
  protected def defineComplexParamsTemplate(pname: String, explanation: String, other: String) =
    s"""        self.$pname = Param(self, \"$pname\", \"$explanation\", $other)"""
  protected def setTemplate(capName: String, pname: String, explanation: String): String = {
    s"""|    def set$capName(self, value):
        |        "\""
        |
        |        Args:
        |
        |$explanation
        |
        |        "\""
        |        self._set($pname=value)
        |        return self
        |
        |""".stripMargin
  }

  protected def getTemplate(capName: String, pname: String, docType: String, paramDef: String): String = {
    val res = paramDef.split(":", 2).map(_.trim)
    s"""|    def get$capName(self):
        |        "\""
        |
        |        Returns:
        |
        |            $docType: ${res(1)}
        |        "\""
        |        return self.getOrDefault(self.$pname)
        |
        |""".stripMargin
  }

  protected def getComplexTemplate(capName: String, pname: String, docType: String, paramDef: String): String = {
    val res = paramDef.split(":", 2).map(_.trim)
    s"""|    def get$capName(self):
        |        "\""
        |
        |        Returns:
        |
        |            $docType: ${res(1)}
        |        "\""
        |        return self._cache.get(\"$pname\", None)
        |
        |""".stripMargin
  }

  protected def saveLoadTemplate(entryPointQualifiedName: String, entryPointName: String): String = {
    s"""|    @classmethod
        |    def read(cls):
        |        "\"" Returns an MLReader instance for this class. "\""
        |        return JavaMMLReader(cls)
        |
        |    @staticmethod
        |    def getJavaPackage():
        |        "\"" Returns package name String. "\""
        |        return \"${entryPointQualifiedName}\"
        |
        |    @staticmethod
        |    def _from_java(java_stage):
        |        module_name=${entryPointName}.__module__
        |        module_name=module_name.rsplit(".", 1)[0] + ".${
      if (entryPointName.startsWith("_")) entryPointName.tail else entryPointName}"
        |        return from_java(java_stage, module_name)
        |""".stripMargin
  }

  protected val header = WrapperClassDoc.GenerateWrapperClassDoc(entryPointName)
  protected def classDocTemplate(entryPointName: String) = s"""$header"""

  protected def paramDocTemplate(explanation: String, docType: String, indent: String): String = {
    val res = explanation.split(":", 2).map(_.trim)
    s"""$indent${res(0)} ($docType): ${res(1)}"""
  }

  val psType: String
  private lazy val objectBaseClass: String = "Java" + psType
  private lazy val autoInheritedClasses =
    Seq("ComplexParamsMixin", "JavaMLReadable", "JavaMLWritable", objectBaseClass)
  // Complex types are not easily recognized by Py4j. They need special processing.
  private lazy val complexTypes =  Set[String](
    "TransformerParam",
    "TransformerArrayParam",
    "EstimatorParam",
    "PipelineStageParam",
    "EstimatorArrayParam")
  protected def isComplexType(paramType: String): Boolean = complexTypes.contains(paramType)

  protected def getParamExplanation(param: Param[_]): String = {
    // the result might have the instance UID: it shouldn't be in the generated code
    val expl = entryPoint.explainParam(param).replace(entryPoint.uid, "[self.uid]")
    // if no defaults, Spark's explainParam() will end with "(undefined)", remove it
    " *\\(undefined\\)$".r.replaceAllIn(expl, "")
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

  protected def getPythonizedDataType(paramType: String): String =
    paramType match {
      case "BooleanParam"     => "bool"
      case "IntParam"         => "int"
      case "LongParam"        => "long"
      case "FloatParam"       => "float"
      case "DoubleParam"      => "double"
      case "StringParam"      => "str"
      case "Param"            => "str"
      case "StringArrayParam" => "list"
      case "ByteArrayParam"   => "list"
      case "MapArrayParam"    => "dict"
      case _                  => "object"
    }

  protected def getParamDefault(param: Param[_]): (String, String) = {
    var paramDefault:   String = null
    var pyParamDefault: String = "None"
    var autogenSuffix:  String = null
    var defaultStringIsParsable: Boolean = true

    if (entryPoint.hasDefault(param)) {
      val paramParent: String = param.parent
      paramDefault = entryPoint.getDefault(param).get.toString
      if (paramDefault.toLowerCase.contains(paramParent.toLowerCase))
        autogenSuffix = paramDefault.substring(paramDefault.lastIndexOf(paramParent)
                                               + paramParent.length)
      else {
        try {
          entryPoint.getParam(param.name).w(paramDefault)
        } catch {
          case e: Exception =>
            defaultStringIsParsable = false
        }
        pyParamDefault = getPythonizedDefault(paramDefault,
          param.getClass.getSimpleName, defaultStringIsParsable)
      }
    }
    (pyParamDefault, autogenSuffix)
  }

  protected def getPysparkWrapperBase: String = {
    // Construct relevant strings
    val imports = ListBuffer[String](additionalImports("utils"))
    val inheritedClasses = ListBuffer[String]()
    inheritedClasses ++= autoInheritedClasses
    val paramsAndDefaults           = ListBuffer[String]()
    val paramDefinitionsAndDefaults = ListBuffer[String]()
    val paramGettersAndSetters      = ListBuffer[String]()
    val paramDocList                = ListBuffer[String]()
    val classParamDocList           = ListBuffer[String]()

    // Iterate over the params to build strings
    val allParams: Array[Param[_]] = entryPoint.params
    // Check for complex types
    if (allParams.exists(p => isComplexType(p.getClass.getSimpleName))) {
      // Add special imports
      imports += additionalImports("complexTypes")
      // Add cache
      paramDefinitionsAndDefaults += scopeDepth * 2 + "self._cache = {}"
    }
    for (param <- allParams) {
      val pname = param.name
      val docType = getPythonizedDataType(param.getClass.getSimpleName)
      paramGettersAndSetters +=
        setTemplate(StringUtils.capitalize(pname), pname,
          paramDocTemplate(getParamExplanation(param), docType, scopeDepth * 3))
      if (isComplexType(param.getClass.getSimpleName)) {
        paramDefinitionsAndDefaults +=
          defineComplexParamsTemplate(pname, getParamExplanation(param),
            s"""generateTypeConverter("$pname", self._cache, complexTypeConverter)""")
        paramGettersAndSetters +=
          getComplexTemplate(StringUtils.capitalize(pname), pname, docType, getParamExplanation(param))
        paramDocList +=
          paramDocTemplate(getParamExplanation(param), docType, scopeDepth * 3)
        classParamDocList +=
          paramDocTemplate(getParamExplanation(param), docType, scopeDepth * 2)
      } else {
        paramDefinitionsAndDefaults +=
          s"""${scopeDepth * 2}self.$pname = Param(self, \"$pname\", \"${getParamExplanation(param)}\")"""
        paramGettersAndSetters +=
          getTemplate(StringUtils.capitalize(pname), pname, docType, getParamExplanation(param))
        paramDocList +=
          paramDocTemplate(getParamExplanation(param), docType, scopeDepth * 3)
        classParamDocList +=
          paramDocTemplate(getParamExplanation(param), docType, scopeDepth * 2)
      }

      val (pyParamDefault, autogenSuffix) = getParamDefault(param)
      paramsAndDefaults += pname + "=" + pyParamDefault

      if (pyParamDefault != "None")
        paramDefinitionsAndDefaults += s"""${scopeDepth * 2}self._setDefault($pname=$pyParamDefault)"""
      else if (autogenSuffix != null)
        paramDefinitionsAndDefaults += s"""${scopeDepth * 2}self._setDefault($pname=self.uid + \"$autogenSuffix\")"""

    }

    // Build strings
    val importsString     = imports.mkString("\n")
    val inheritanceString = inheritedClasses.mkString(", ")
    val classParamsString = paramsAndDefaults.mkString(", ")
    val paramDefinitionsAndDefaultsString = paramDefinitionsAndDefaults.mkString("\n")
    val paramGettersAndSettersString = paramGettersAndSetters.mkString("\n")
    val classDocString = classDocTemplate(entryPointName)
    val paramDocString = paramDocList.mkString("\n")
    //val classParamDocString = classParamDocList.mkString("\n")
    val classParamDocString = {
      if (classParamDocList.isEmpty) ""
      else scopeDepth + "Args:\n\n" + classParamDocList.mkString("\n")
    }

    classTemplate(importsString, inheritanceString,
                    classParamsString,
                    paramDefinitionsAndDefaultsString, paramGettersAndSettersString,
                    classDocString, paramDocString, classParamDocString) + "\n" +
           saveLoadTemplate(entryPointQualifiedName, entryPointName)
  }

  def pysparkWrapperBuilder(): String = {
    getPysparkWrapperBase
  }

  def writeWrapperToFile(dir: File): Unit = {
    writeFile(new File(dir, entryPointName + ".py"), pysparkWrapperBuilder())
  }

}

class PySparkTransformerWrapper(entryPoint: Transformer,
                                entryPointName: String,
                                entryPointQualifiedName: String)
    extends PySparkWrapper(entryPoint,
                           entryPointName,
                           entryPointQualifiedName) {
  override val psType = "Transformer"
}

class PySparkEstimatorWrapper(entryPoint: Estimator[_],
                              entryPointName: String,
                              entryPointQualifiedName: String,
                              companionModelName: String,
                              companionModelQualifiedName: String)
  extends PySparkWrapper(entryPoint,
                         entryPointName,
                         entryPointQualifiedName) {

  private val createModelStringTemplate =
    s"""|    def _create_model(self, java_model):
        |        return $companionModelName(java_model)
        |
        |""".stripMargin

  private def modelClassString(className: String, superClass: String): String = {
    s"""|class ${className}(ComplexParamsMixin, JavaModel, JavaMLWritable, JavaMLReadable):
        |    "\""
        |    Model fitted by :class:`${superClass}`.
        |
        |    This class is left empty on purpose.
        |    All necessary methods are exposed through inheritance.
        |    "\""
        |""".stripMargin
  }

  override def pysparkWrapperBuilder(): String = {
    Seq(super.pysparkWrapperBuilder,
        createModelStringTemplate,
        modelClassString(companionModelName, entryPointName),
        saveLoadTemplate(companionModelQualifiedName, companionModelName),
        "").mkString("\n")
  }

  override val psType = "Estimator"

}
