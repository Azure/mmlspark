// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.pytest

import java.io.{InputStream, ObjectInputStream, ObjectStreamClass}
import java.net.{URL, URLClassLoader}

import com.microsoft.ml.spark.core.env.FileUtilities._
import com.microsoft.ml.spark.core.test.fuzzing.PyTestFuzzing
import org.apache.commons.io.FileUtils
import org.scalatest.exceptions.TestFailedException

import scala.reflect.{ClassTag, _}

/** Contains logic for loading classes. */
object LocalTestLoadingUtils {

  val resourcesDirectory = new File(getClass.getResource("/").toURI)

  private val testFileLocs = allFiles(resourcesDirectory, file => file.getName.endsWith(".class"))
    .map(file => file.getCanonicalPath.replace(resourcesDirectory.getCanonicalPath + "/", ""))

  private def addURL(url: URL): Unit = {
    val classLoader = ClassLoader.getSystemClassLoader.asInstanceOf[URLClassLoader]
    val clazz = classOf[URLClassLoader]
    val method = clazz.getDeclaredMethod("addURL", classOf[URL])
    method.setAccessible(true)
    method.invoke(classLoader, url)
    ()
  }

  val classLoader: ClassLoader = {
    testFileLocs.map(file =>
      new File(file).toURI.toURL
    ).foreach(addURL)
    ClassLoader.getSystemClassLoader
  }


  lazy val loadedTestClasses: List[Class[_]] = {
    val classNames = testFileLocs.map(je => je.stripSuffix(".class").replace("/", "."))
    classNames.map { name =>
      classLoader.loadClass(name): Class[_]
    }.toList
  }

  private def catchInstantiationErrors[T](clazz: Class[_], func: Function[Class[_], T], debug: Boolean): Option[T] = {
    def log(message: String) = {
      if (debug) println(message)
    }

    try {
      Some(func(clazz))
    } catch {
      // Classes without default constructor
      case ie: InstantiationException =>
        log(s"Could not generate wrapper without default constructor for " +
          s"class ${clazz.getSimpleName}: $ie")
        None
      // Classes with "private" modifiers on constructors
      case iae: IllegalAccessException =>
        log(s"Could not generate wrapper due to private modifiers or constructors for " +
          s"class ${clazz.getSimpleName}: $iae")
        None
      case ncd: NoClassDefFoundError =>
        log(s"Could not generate wrapper because no class definition found for class " +
          s"${clazz.getSimpleName}: $ncd")
        None
      case ule: UnsatisfiedLinkError =>
        log(s"Could not generate wrapper due to link error from: " +
          s"${clazz.getSimpleName}: $ule")
        None
      case e: TestFailedException =>
        log(s"Could not generate wrapper due to TestFailedException")
        None
      case e: Exception =>
        log(s"Could not generate wrapper for class ${clazz.getSimpleName}: ${e.printStackTrace()}")
        None
    }
  }

  def loadTest[T: ClassTag](instantiate: Class[_] => Any,
                            classFilter: Class[_] => Boolean,
                            debug: Boolean): List[T] = {
    loadedTestClasses.filter(lc => classTag[T].runtimeClass.isAssignableFrom(lc) && classFilter(lc)).flatMap { lc =>
      catchInstantiationErrors(lc, instantiate, debug)
    }.asInstanceOf[List[T]]
  }

  def loadTestClass[T: ClassTag](filter: Class[_]=> Boolean,
                                 debug: Boolean): List[T] =
    loadTest[T](lc => lc.newInstance(), filter, debug)

}

class ContextObjectInputStream(input: InputStream) extends ObjectInputStream(input) {
  protected override def resolveClass(desc: ObjectStreamClass): Class[_] = {
    try {
      Class.forName(desc.getName, false, Thread.currentThread().getContextClassLoader)
    } catch {
      case _: ClassNotFoundException => super.resolveClass(desc)
    }
  }
}

abstract class PyTestGeneratorBase {

  def exceptions: Set[Class[_]] = Set()

  def applicableTests: List[PyTestFuzzing[_]] = LocalTestLoadingUtils
    .loadTestClass[PyTestFuzzing[_]]({tc => !exceptions(tc)}, false)

  def userProvidedPyTestDir =
    new File(new File(getClass.getResource("/").toURI), "../../../src/test/python")

  def targetPyTestDir =
    new File(new File(getClass.getResource("/").toURI), "../../test-pyTests")

  def targetPyDataDir =
    new File(new File(getClass.getResource("/").toURI), "../../test-pyData")

  def main(args: Array[String]): Unit = {
    if (targetPyTestDir.exists())
      FileUtils.deleteDirectory(targetPyTestDir)
    if (targetPyDataDir.exists())
      FileUtils.deleteDirectory(targetPyDataDir)
    if (userProvidedPyTestDir.exists())
      FileUtils.copyDirectory(userProvidedPyTestDir, targetPyTestDir)
    applicableTests.foreach { tc =>
      println(s"saving datasets for $tc")
      tc.savePyDatasets()
      println(s"writing pytests for $tc")
      tc.writePyTests()
    }
  }

}