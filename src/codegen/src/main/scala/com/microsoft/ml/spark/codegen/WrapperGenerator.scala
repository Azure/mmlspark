// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import java.lang.reflect.{ParameterizedType, Type}
import java.util.jar._

import com.microsoft.ml.spark.codegen.Config._
import com.microsoft.ml.spark.core.env.FileUtilities.{File, writeFile}
import com.microsoft.ml.spark.core.env.InternalWrapper
import com.microsoft.ml.spark.core.env.StreamUtilities._
import org.apache.spark.ml.{Estimator, Transformer}

import scala.collection.Iterator.iterate
import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.internal.util.ScalaClassLoader.URLClassLoader
import scala.reflect.runtime.universe._

abstract class WrapperGenerator {

  def wrapperName(myClass: Class[_]): String

  def generateEstimatorWrapper(entryPoint: Estimator[_],
                               entryPointName: String,
                               entryPointQualifiedName: String,
                               companionModelName: String,
                               companionModelQualifiedName: String): WritableWrapper

  def generateEstimatorTestWrapper(entryPoint: Estimator[_],
                                   entryPointName: String,
                                   entryPointQualifiedName: String,
                                   companionModelName: String,
                                   companionModelQualifiedName: String,
                                   packageName: String): Option[WritableWrapper]

  def generateTransformerWrapper(entryPoint: Transformer,
                                 entryPointName: String,
                                 entryPointQualifiedName: String): WritableWrapper

  def generateTransformerTestWrapper(entryPoint: Transformer,
                                     entryPointName: String,
                                     entryPointQualifiedName: String,
                                     packageName: String): Option[WritableWrapper]

  def wrapperDir: File

  def wrapperTestDir: File

  def writeWrappersToFile(myClass: Class[_], qualifiedClassName: String): Unit = {
    try {
      val classInstance = myClass.newInstance()

      val (wrapper: WritableWrapper, wrapperTests: Option[WritableWrapper]) =
        classInstance match {
          case t: Transformer =>
            val className = wrapperName(myClass)
            (generateTransformerWrapper(t, className, qualifiedClassName),
             generateTransformerTestWrapper(
               t, className, qualifiedClassName, t.getClass.getPackage.getName))
          case e: Estimator[_] =>
            val sc = iterate[Class[_]](myClass)(_.getSuperclass)
                     .find(c => Seq("Estimator", "Predictor").contains(c.getSuperclass.getSimpleName))
                     .get
            val typeArgs = sc.getGenericSuperclass.asInstanceOf[ParameterizedType]
              .getActualTypeArguments
            val getModelFromGenericType = (modelType: Type) => {
              val modelClass = modelType.getTypeName.split("<").head
              (modelClass.split("\\.").last, modelClass)
            }
            val (modelClass, modelQualifiedClass) = sc.getSuperclass.getSimpleName match {
              case "Estimator" => getModelFromGenericType(typeArgs.head)
              case "Predictor" => getModelFromGenericType(typeArgs(2))
            }

            val className = wrapperName(myClass)
            (generateEstimatorWrapper(e, className, qualifiedClassName, modelClass, modelQualifiedClass),
             generateEstimatorTestWrapper(
               e, className, qualifiedClassName, modelClass, modelQualifiedClass, e.getClass.getPackage.getName))
          case _ => return
        }
      val packageFolders =
        if (myClass.getPackage.getName.startsWith("com.microsoft.ml.spark")){
          myClass.getPackage.getName
            .replace("com.microsoft.ml.spark.","")
            .replace(".","/")
        }else{
          ""
        }

      val destDir= new File(wrapperDir, packageFolders)
      if (!destDir.exists()) destDir.mkdirs()
      wrapper.writeWrapperToFile(destDir)
      if (wrapperTests.isDefined) wrapperTests.get.writeWrapperToFile(wrapperTestDir)
      if (debugMode) println(s"Generated wrapper for class ${myClass.getSimpleName}")
    } catch {
      // Classes without default constructor
      case ie: InstantiationException =>
        if (debugMode) println(s"Could not generate wrapper for class ${myClass.getSimpleName}: $ie")
      // Classes with "private" modifiers on constructors
      case iae: IllegalAccessException =>
        if (debugMode) println(s"Could not generate wrapper for class ${myClass.getSimpleName}: $iae")
      // Classes that require runtime library loading
      case ule: UnsatisfiedLinkError =>
        if (debugMode) println(s"Could not generate wrapper for class ${myClass.getSimpleName}: $ule")
      case e: Exception =>
        println(s"Could not generate wrapper for class ${myClass.getSimpleName}: ${e.printStackTrace}")
    }
  }

  def getWrappersFromJarFile(jarFilePath: String, cl2: URLClassLoader): Unit = {
    val cld = new URLClassLoader(Array(new File(jarFilePath).toURI.toURL), cl2)
    val jfd = new JarFile(jarFilePath)

    using(Seq(cld, jfd)) { s =>
      val cl = s(0).asInstanceOf[URLClassLoader]
      val jarFile = s(1).asInstanceOf[JarFile]
      val _ = jarFile.entries.asScala
        .filter(e => e.getName.endsWith(".class"))
        .map(e => e.getName.replace("/", ".").stripSuffix(".class"))
        .filter(q => { val clazz = cl.loadClass(q)
                       try {
                         clazz.getEnclosingClass == null
                       } catch {
                         case _: java.lang.NoClassDefFoundError => false
                       }})
        .toList
        .sorted
        .foreach(q => writeWrappersToFile(cl.loadClass(q), q))
    }.get

  }

  def generateWrappers(): Unit = {
    val jarFiles = outputDir.listFiles.filter(_.getName.endsWith(".jar")).sortBy(_.getName)
    val jarUrls = jarFiles.map(_.toURI.toURL)
    using(Seq(new URLClassLoader(jarUrls, this.getClass.getClassLoader))) { s =>
      jarFiles.foreach(f => getWrappersFromJarFile(f.getAbsolutePath, s(0)))
    }.get
  }

}

object PySparkWrapperGenerator {
  def apply(): Unit = {
    new PySparkWrapperGenerator().generateWrappers()
  }
}

class PySparkWrapperGenerator extends WrapperGenerator {
  override def wrapperDir: File = pyDir
  override def wrapperTestDir: File = pyTestDir

  // check if the class is annotated with InternalWrapper
  private[spark] def needsInternalWrapper(myClass: Class[_]):Boolean = {
    val typ: ClassSymbol = runtimeMirror(myClass.getClassLoader).classSymbol(myClass)
    typ.annotations.exists(a => a.tree.tpe =:= typeOf[InternalWrapper])
  }

  def wrapperName(myClass: Class[_]):String = {
    val prefix = if (needsInternalWrapper(myClass)) internalPrefix else ""
    prefix + myClass.getSimpleName
  }

  def generateEstimatorWrapper(entryPoint: Estimator[_],
                               entryPointName: String,
                               entryPointQualifiedName: String,
                               companionModelName: String,
                               companionModelQualifiedName: String): WritableWrapper = {
    new PySparkEstimatorWrapper(entryPoint,
                                entryPointName,
                                entryPointQualifiedName,
                                companionModelName,
                                companionModelQualifiedName)
  }

  def generateEstimatorTestWrapper(entryPoint: Estimator[_],
                                   entryPointName: String,
                                   entryPointQualifiedName: String,
                                   companionModelName: String,
                                   companionModelQualifiedName: String,
                                   packageName: String): Option[WritableWrapper] = {
    Some(new PySparkEstimatorWrapperTest(entryPoint,
                                         entryPointName,
                                         entryPointQualifiedName,
                                         companionModelName,
                                         companionModelQualifiedName,
                                         packageName))
  }

  def generateTransformerWrapper(entryPoint: Transformer,
                                 entryPointName: String,
                                 entryPointQualifiedName: String): WritableWrapper = {
    new PySparkTransformerWrapper(entryPoint, entryPointName, entryPointQualifiedName)
  }

  def generateTransformerTestWrapper(entryPoint: Transformer,
                                     entryPointName: String,
                                     entryPointQualifiedName: String,
                                     packageName: String): Option[WritableWrapper] = {
    Some(new PySparkTransformerWrapperTest(entryPoint, entryPointName, entryPointQualifiedName, packageName))
  }
}

object SparklyRWrapperGenerator {
  def apply(): Unit = {
    new SparklyRWrapperGenerator().generateWrappers()
  }
}

class SparklyRWrapperGenerator extends WrapperGenerator {
  override def wrapperDir: File = rSrcDir
  override def wrapperTestDir: File = rTestDir

  // description file; need to encode version as decimal
  val today = new java.text.SimpleDateFormat("yyyy-MM-dd")
                .format(new java.util.Date())
  val ver0 = "\\.dev|\\+".r.replaceAllIn(mmlVer, "-")
  val ver  = "\\.g([0-9a-f]+)".r.replaceAllIn(ver0, m =>
      "." + scala.math.BigInt(m.group(1), 16).toString)
  val actualVer = if (ver == mmlVer) "" else s"\nMMLSparkVersion: $mmlVer"
  writeFile(new File(rDir, "DESCRIPTION"),
            s"""|Package: mmlspark
                |Title: Access to MMLSpark via R
                |Description: Provides an interface to MMLSpark.
                |Version: $ver$actualVer
                |Date: $today
                |Author: Microsoft Corporation
                |Maintainer: MMLSpark Team <mmlspark-support@microsoft.com>
                |URL: https://github.com/Azure/mmlspark
                |BugReports: https://github.com/Azure/mmlspark/issues
                |Depends:
                |    R (>= 2.12.0)
                |Imports:
                |    sparklyr
                |License: MIT
                |""".stripMargin)

  // generate a new namespace file, import sparklyr
  writeFile(sparklyRNamespacePath,
            s"""|$copyrightLines
                |import(sparklyr)
                |
                |export(sdf_transform)
                |""".stripMargin)

  def wrapperName(myClass: Class[_]): String =
    myClass.getSimpleName.foldLeft((true, ""))((base, c) => {
      val ignoreCaps = base._1
      val partialStr = base._2
      if (!c.isUpper)      (false, partialStr + c)
      else if (ignoreCaps) (true,  partialStr + c.toLower)
      else                 (true,  partialStr + "_" + c.toLower)
    })._2

  def generateEstimatorWrapper(entryPoint: Estimator[_],
                               entryPointName: String,
                               entryPointQualifiedName: String,
                               companionModelName: String,
                               companionModelQualifiedName: String): WritableWrapper = {
    new SparklyREstimatorWrapper(entryPoint,
                                 entryPointName,
                                 entryPointQualifiedName,
                                 companionModelName,
                                 companionModelQualifiedName)
  }

  def generateEstimatorTestWrapper(entryPoint: Estimator[_],
                                   entryPointName: String,
                                   entryPointQualifiedName: String,
                                   companionModelName: String,
                                   companionModelQualifiedName: String,
                                   packageName: String): Option[WritableWrapper] = {
    None
  }

  def generateTransformerWrapper(entryPoint: Transformer,
                                 entryPointName: String,
                                 entryPointQualifiedName: String): WritableWrapper = {
    new  SparklyRTransformerWrapper(entryPoint, entryPointName, entryPointQualifiedName)
  }

  def generateTransformerTestWrapper(entryPoint: Transformer,
                                     entryPointName: String,
                                     entryPointQualifiedName: String,
                                     packageName: String): Option[WritableWrapper] = {
    None
  }

}
