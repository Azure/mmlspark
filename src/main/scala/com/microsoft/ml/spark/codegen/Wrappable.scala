// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import java.lang.reflect.ParameterizedType
import java.nio.charset.StandardCharsets
import java.nio.file.Files

import com.microsoft.ml.spark.core.env.FileUtilities
import com.microsoft.ml.spark.core.serialize.ComplexParam
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model, Transformer}

import scala.collection.Iterator.iterate
import scala.reflect.ClassTag

case class ParamInfo[T <: Param[_]: ClassTag](pyType: String,
                                              pyTypeConverter: Option[String],
                                              rTypeConverter: Option[String]) {

  def this(pyType: String, typeConverterArg: String, rTypeConverterArg: String) = {
    this(pyType, Some(typeConverterArg), Some(rTypeConverterArg))
  }

  def this(pyType: String) = {
    this(pyType, None, None)
  }

}

object DefaultParamInfo {
  val BooleanInfo = new ParamInfo[BooleanParam]("bool", "TypeConverters.toBoolean", "as.logical")
  val IntInfo = new ParamInfo[IntParam]("int", "TypeConverters.toInt", "as.integer")
  val LongInfo = new ParamInfo[LongParam]("long", None, Some("as.integer"))
  val FloatInfo = new ParamInfo[FloatParam]("float", "TypeConverters.toFloat", "as.double")
  val DoubleInfo = new ParamInfo[DoubleParam]("float", "TypeConverters.toFloat", "as.double")
  val StringInfo = new ParamInfo[Param[String]]("str", Some("TypeConverters.toString"), None)
  val StringArrayInfo = new ParamInfo[StringArrayParam]("list", "TypeConverters.toListString", "as.array")
  val DoubleArrayInfo = new ParamInfo[DoubleArrayParam]("list", "TypeConverters.toListFloat", "as.array")
  val IntArrayInfo = new ParamInfo[IntArrayParam]("list", "TypeConverters.toListInt", "as.array")
  val ByteArrayInfo = new ParamInfo[ByteArrayParam]("list")
  val MapArrayInfo = new ParamInfo[MapArrayParam]("dict")
  val MapInfo = new ParamInfo[MapParam[_, _]]("dict")
  val UnknownInfo = new ParamInfo[Param[_]]("object")

  //noinspection ScalaStyle
  def getParamInfo(dataType: Param[_]): ParamInfo[_] = {
    dataType match {
      case _: BooleanParam => BooleanInfo
      case _: IntParam => IntInfo
      case _: LongParam => LongInfo
      case _: FloatParam => FloatInfo
      case _: DoubleParam => DoubleInfo
      case _: StringArrayParam => StringArrayInfo
      case _: DoubleArrayParam => DoubleArrayInfo
      case _: IntArrayParam => IntArrayInfo
      case _: ByteArrayParam => ByteArrayInfo
      case _: MapArrayParam => MapArrayInfo
      case _: MapParam[_, _] => MapInfo
      //case _: Param[String] => StringInfo //TODO fix erasure issues
      case _ => UnknownInfo
    }
  }
}


trait BaseWrappable extends Params {
  protected lazy val copyrightLines: String =
    s"""|# Copyright (C) Microsoft Corporation. All rights reserved.
        |# Licensed under the MIT License. See LICENSE in project root for information.
        |""".stripMargin

  protected lazy val classNameHelper: String = this.getClass.getName.split(".".toCharArray).last

  protected def companionModelClassName: String = {
    val superClass = iterate[Class[_]](this.getClass)(_.getSuperclass)
      .find(c => Set("Estimator", "ProbabilisticClassifier", "Predictor", "BaseRegressor", "Ranker")(
        c.getSuperclass.getSimpleName))
      .get
    val typeArgs = this.getClass.getGenericSuperclass.asInstanceOf[ParameterizedType].getActualTypeArguments
    val modelTypeArg = superClass.getSuperclass.getSimpleName match {
      case "Estimator" =>
        typeArgs.head
      case model if Set("ProbabilisticClassifier", "BaseRegressor", "Predictor", "Ranker")(model) =>
        typeArgs.last
    }
    modelTypeArg.getTypeName
  }

}

trait PythonWrappable extends BaseWrappable {

  import GenerationUtils._
  import DefaultParamInfo._

  def pyAdditionalMethods: String = {
    ""
  }

  protected lazy val pyInternalWrapper = false

  protected lazy val pyClassName: String = {
    if (pyInternalWrapper) {
      "_" + classNameHelper
    } else {
      "" + classNameHelper
    }
  }

  protected lazy val pyObjectBaseClass: String = {
    this match {
      case _: Estimator[_] => "JavaEstimator"
      case _: Model[_] => "JavaModel"
      case _: Transformer => "JavaTransformer"
      case _: Evaluator => "JavaEvaluator"
    }
  }

  protected lazy val pyInheritedClasses =
    Seq("ComplexParamsMixin", "JavaMLReadable", "JavaMLWritable", pyObjectBaseClass)

  // TODO add default values
  protected lazy val pyClassDoc: String = {
    val argLines = this.params.map { p =>
      s"""${p.name} (${getParamInfo(p).pyType}): ${p.doc}"""
    }.mkString("\n")
    s"""|"\""
        |Args:
        |${indent(argLines, 1)}
        |"\""
        |""".stripMargin
  }

  protected lazy val pyParamsDefinitions: String = {
    this.params.map { p =>
      val typeConverterString = getParamInfo(p).pyTypeConverter.map(", typeConverter=" + _).getOrElse("")
      s"""|${p.name} = Param(Params._dummy(), "${p.name}", "${p.doc}"$typeConverterString)
          |""".stripMargin
    }.mkString("\n")
  }

  protected def pyParamArg[T](p: Param[T]): String = {
    (p, this.getDefault(p)) match {
      case (_: ServiceParam[_], _) =>
        s"${p.name}=None,\n${p.name}Col=None"
      case (_: ComplexParam[_], _) | (_, None) =>
        s"${p.name}=None"
      case (_, Some(v)) =>
        s"""${p.name}=${PythonWrappableParam.pyDefaultRender(v, p)}"""
    }
  }

  protected def pyParamDefault[T](p: Param[T]): Option[String] = {
    (p, this.getDefault(p)) match {
      case (_: ServiceParam[_], _) =>
        None
      case (_: ComplexParam[_], _) | (_, None) =>
        None
      case (_, Some(v)) =>
        Some(s"""self._setDefault(${pyParamArg(p)})""")
    }
  }

  protected def pyParamsArgs: String =
    this.params.map(pyParamArg(_)).mkString(",\n")

  protected def pyParamsDefaults: String =
    this.params.flatMap(pyParamDefault(_)).mkString("\n")

  protected def pyParamSetter(p: Param[_]): String = {
    val capName = p.name.capitalize
    val docString =
      s"""|"\""
          |Args:
          |    ${p.name}: ${p.doc}
          |"\""
          |""".stripMargin
    p match {
      case _: ServiceParam[_] =>
        s"""|def set$capName(self, value):
            |${indent(docString, 1)}
            |    if isinstance(value, list):
            |        value = SparkContext._active_spark_context._jvm.org.apache.spark.ml.param.ServiceParam.toSeq(value)
            |    self._java_obj = self._java_obj.set$capName(value)
            |    return self
            |
            |def set${capName}Col(self, value):
            |${indent(docString, 1)}
            |    self._java_obj = self._java_obj.set${capName}Col(value)
            |    return self
            |""".stripMargin
      case _ =>
        s"""|def set${p.name.capitalize}(self, value):
            |${indent(docString, 1)}
            |    self._set(${p.name}=value)
            |    return self
            |""".stripMargin
    }
  }

  protected def pyParamsSetters: String =
    this.params.map(pyParamSetter).mkString("\n")

  protected def pyExtraEstimatorMethods: String = {
    this match {
      case _: Estimator[_] =>
        s"""|def _create_model(self, java_model):
            |    try:
            |        model = ${companionModelClassName.split(".".toCharArray).last}(java_obj=java_model)
            |        model._transfer_params_from_java()
            |    except TypeError:
            |        model = ${companionModelClassName.split(".".toCharArray).last}._from_java(java_model)
            |    return model
            |
            |def _fit(self, dataset):
            |    java_model = self._fit_java(dataset)
            |    return self._create_model(java_model)
            |""".stripMargin
      case _ =>
        ""
    }
  }

  protected def pyExtraEstimatorImports: String = {
    this match {
      case _: Estimator[_] =>
        val companionModelImport = companionModelClassName
          .replaceAllLiterally("com.microsoft.ml.spark", "mmlspark")
          .replaceAllLiterally("org.apache.spark", "pyspark")
          .split(".".toCharArray)
        val path = if (companionModelImport.head == "pyspark") {
          companionModelImport.dropRight(1).mkString(".")
        } else {
          companionModelImport.mkString(".")
        }
        val modelName = companionModelImport.last
        s"from $path import $modelName"
      case _ =>
        ""
    }
  }

  protected def pyParamGetter(p: Param[_]): String = {
    val capName = p.name.capitalize
    val docString =
      s"""|"\""
          |Returns:
          |    ${p.name}: ${p.doc}
          |"\""
          |""".stripMargin
    p match {
      case _: DataFrameParam =>
        s"""|
            |def get$capName(self):
            |${indent(docString, 1)}
            |    ctx = SparkContext._active_spark_context
            |    sql_ctx = SQLContext.getOrCreate(ctx)
            |    return DataFrame(self._java_obj.get$capName(), sql_ctx)
            |""".stripMargin
      case _: TransformerParam | _: EstimatorParam | _: PipelineStageParam =>
        s"""|
            |def get$capName(self):
            |${indent(docString, 1)}
            |    return JavaParams._from_java(self._java_obj.get$capName())
            |""".stripMargin
      case _ =>
        s"""|
            |def get$capName(self):
            |${indent(docString, 1)}
            |    return self.getOrDefault(self.${p.name})
            |""".stripMargin
    }
  }

  protected def pyParamsGetters: String =
    this.params.map(pyParamGetter).mkString("\n")

  def pyInitFunc(): String = {
    s"""
       |@keyword_only
       |def __init__(
       |    self,
       |    java_obj=None,
       |${indent(pyParamsArgs, 1)}
       |    ):
       |    super($pyClassName, self).__init__()
       |    if java_obj is None:
       |        self._java_obj = self._new_java_obj("${this.getClass.getName}", self.uid)
       |    else:
       |        self._java_obj = java_obj
       |${indent(pyParamsDefaults, 1)}
       |    if hasattr(self, \"_input_kwargs\"):
       |        kwargs = self._input_kwargs
       |    else:
       |        kwargs = self.__init__._input_kwargs
       |
       |    if java_obj is None:
       |        for k,v in kwargs.items():
       |            if v is not None:
       |                getattr(self, "set" + k[0].upper() + k[1:])(v)
       |""".stripMargin

  }

  //noinspection ScalaStyle
  protected def pythonClass(): String = {
    s"""|$copyrightLines
        |
        |import sys
        |if sys.version >= '3':
        |    basestring = str
        |
        |from pyspark import SparkContext, SQLContext
        |from pyspark.sql import DataFrame
        |from pyspark.ml.param.shared import *
        |from pyspark import keyword_only
        |from pyspark.ml.util import JavaMLReadable, JavaMLWritable
        |from mmlspark.core.serialize.java_params_patch import *
        |from pyspark.ml.wrapper import JavaTransformer, JavaEstimator, JavaModel
        |from pyspark.ml.evaluation import JavaEvaluator
        |from pyspark.ml.common import inherit_doc
        |from mmlspark.core.schema.Utils import *
        |from pyspark.ml.param import TypeConverters
        |from mmlspark.core.schema.TypeConversionUtils import generateTypeConverter, complexTypeConverter
        |$pyExtraEstimatorImports
        |
        |@inherit_doc
        |class $pyClassName(${pyInheritedClasses.mkString(", ")}):
        |${indent(pyClassDoc, 1)}
        |
        |${indent(pyParamsDefinitions, 1)}
        |
        |${indent(pyInitFunc(), 1)}
        |
        |    @keyword_only
        |    def setParams(
        |        self,
        |${indent(pyParamsArgs, 2)}
        |        ):
        |        "\""
        |        Set the (keyword only) parameters
        |        "\""
        |        if hasattr(self, \"_input_kwargs\"):
        |            kwargs = self._input_kwargs
        |        else:
        |            kwargs = self.__init__._input_kwargs
        |        return self._set(**kwargs)
        |
        |    @classmethod
        |    def read(cls):
        |        "\"" Returns an MLReader instance for this class. "\""
        |        return JavaMMLReader(cls)
        |
        |    @staticmethod
        |    def getJavaPackage():
        |        "\"" Returns package name String. "\""
        |        return "${this.getClass.getName}"
        |
        |    @staticmethod
        |    def _from_java(java_stage):
        |        module_name=$pyClassName.__module__
        |        module_name=module_name.rsplit(".", 1)[0] + ".$classNameHelper"
        |        return from_java(java_stage, module_name)
        |
        |${indent(pyParamsSetters, 1)}
        |
        |${indent(pyParamsGetters, 1)}
        |
        |${indent(pyExtraEstimatorMethods, 1)}
        |
        |${indent(pyAdditionalMethods, 1)}
        """.stripMargin
  }

  def makePyFile(): Unit = {
    val importPath = this.getClass.getName.split(".".toCharArray).dropRight(1)
    val srcFolders = importPath.mkString(".")
      .replaceAllLiterally("com.microsoft.ml.spark", "mmlspark").split(".".toCharArray)
    val srcDir = FileUtilities.join((Seq(CodegenConfig.PySrcDir.toString) ++ srcFolders.toSeq): _*)
    srcDir.mkdirs()
    Files.write(
      FileUtilities.join(srcDir, pyClassName + ".py").toPath,
      pythonClass().getBytes(StandardCharsets.UTF_8))
  }

}

trait RWrappable extends BaseWrappable {

  import GenerationUtils._

  protected lazy val rInternalWrapper = false

  protected lazy val rFuncName: String = {
    if (rInternalWrapper) {
      "_ml_" + camelToSnake(classNameHelper)
    } else {
      "ml_" + camelToSnake(classNameHelper)
    }
  }

  protected def rParamsArgs: String =
    this.params.map(rParamArg(_) + ",\n").mkString("")

  protected def rParamArg[T](p: Param[T]): String = {
    (p, this.getDefault(p)) match {
      case (_: ServiceParam[_], _) =>
        s"${p.name}=NULL,\n${p.name}Col=NULL"
      case (_: ComplexParam[_], _) | (_, None) =>
        s"${p.name}=NULL"
      case (_, Some(v)) =>
        s"""${p.name}=${RWrappableParam.rDefaultRender(v, p)}"""
    }
  }

  protected def rDocString: String = {
    val paramDocLines = this.params.map(p =>
      s"#' @param ${p.name} ${p.doc}"
    ).mkString("\n")
    s"""
       |#' $classNameHelper
       |#'
       |${paramDocLines}
       |#' @export
       |""".stripMargin
  }

  protected def rSetterLines: String = {
    this.params.map { p =>
      val value = DefaultParamInfo.getParamInfo(p)
        .rTypeConverter.map(tc => s"$tc(${p.name})").getOrElse(p.name)
      p match {
        case p: ServiceParam[_] =>
          s"""invoke("set${p.name.capitalize}Col", ${p.name}Col) %>%\ninvoke("set${p.name.capitalize}", $value)"""
        case p =>
          s"""invoke("set${p.name.capitalize}", $value)"""
      }
    }.mkString(" %>%\n")
  }

  protected def rExtraInitLines: String = {
    this match {
      case _: Estimator[_] =>
        "unfit.model=FALSE,\nonly.model=FALSE,\n"
      case _ =>
        "only.model=FALSE,\n"
    }
  }

  protected def rExtraBodyLines: String = {
    this match {
      case _: Estimator[_] =>
        s"""
           |if (unfit.model)
           |    return(mod_parameterized)
           |transformer <- mod_parameterized %>%
           |    invoke("fit", df)
           |scala_transformer_class <- "${companionModelClassName}"
           |""".stripMargin
      case _ =>
        s"""
           |transformer <- mod_parameterized
           |scala_transformer_class <- scala_class
           |""".stripMargin
    }
  }

  protected def rClass(): String = {
    s"""
       |$copyrightLines
       |
       |$rDocString
       |$rFuncName <- function(
       |    x,
       |${indent(rParamsArgs, 1)}
       |${indent(rExtraInitLines, 1)}
       |    uid=random_string("${rFuncName}"),
       |    ...)
       |{
       |    if (unfit.model) {
       |        sc <- x
       |    } else {
       |        df <- spark_dataframe(x)
       |        sc <- spark_connection(df)
       |    }
       |    scala_class <- "${this.getClass.getName}"
       |    mod <- invoke_new(sc, scala_class, uid = uid)
       |    mod_parameterized <- mod %>%
       |${indent(rSetterLines, 2)}
       |${indent(rExtraBodyLines, 1)}
       |    if (only.model)
       |        return(sparklyr:::new_ml_transformer(transformer, class=scala_transformer_class))
       |    transformed <- invoke(transformer, "transform", df)
       |    sdf_register(transformed)
       |}
       |""".stripMargin

  }

  def makeRFile(): Unit = {
    CodegenConfig.RSrcDir.mkdirs()
    Files.write(
      FileUtilities.join(CodegenConfig.RSrcDir, rFuncName + ".R").toPath,
      rClass().getBytes(StandardCharsets.UTF_8))
  }

}

trait Wrappable extends PythonWrappable with RWrappable

