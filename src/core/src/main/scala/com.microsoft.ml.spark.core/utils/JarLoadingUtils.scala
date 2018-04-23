// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.utils

import java.io.{InputStream, ObjectInputStream, ObjectStreamClass}
import java.net.{URL, URLClassLoader}
import java.util.jar.JarFile

import com.microsoft.ml.spark.core.env.FileUtilities._
import org.scalatest.exceptions.TestFailedException

import scala.reflect.ClassTag
import scala.reflect._
import collection.JavaConverters._

/** Contains logic for loading classes. */
object JarLoadingUtils {

  private val jarRelPath = "target/scala-" + sys.env("SCALA_VERSION")
  private val testRelPath = "test-classes"
  private val projectRoots = "project/project-roots.txt"

  private val outputDirs = {
    val thisFile = new File(getClass.getProtectionDomain.getCodeSource.getLocation.getPath)
    val levelsToSrc = 4
    val topDir = (1 to levelsToSrc).foldLeft(thisFile) {case (f, i) => f.getParentFile}
    val rootsFile = new File(topDir, projectRoots)
    val roots = readFile(rootsFile, _.getLines.toList)
    roots.map { root => new File(new File(topDir, root), jarRelPath)}
  }

  private val testOutputDirs = {
    outputDirs.flatMap(dir => {
      val filePath = new File(dir, testRelPath)
      if (filePath.exists()) {
        Some(filePath)
      } else {
        None
      }
    })
  }

  private val jarFileLocs = outputDirs.flatMap(dir =>
    allFiles(dir, file => file.getName.endsWith(".jar")))

  private val testFileLocs = testOutputDirs.flatMap(dir =>
    allFiles(dir, file => file.getName.endsWith(".class"))
      .map(file => file.getCanonicalPath.replace(dir.getCanonicalPath + "/", "")))

  private val jarURLs = jarFileLocs.map(_.toURI.toURL)

  private def addURL(url: URL): Unit = {
    val classLoader = ClassLoader.getSystemClassLoader.asInstanceOf[URLClassLoader]
    val clazz = classOf[URLClassLoader]
    val method = clazz.getDeclaredMethod("addURL", classOf[URL])
    method.setAccessible(true)
    method.invoke(classLoader, url)
    ()
  }

  val classLoader: ClassLoader = {
    jarURLs.union(testOutputDirs
      .map(file =>
        new File(file.getCanonicalPath).toURI.toURL)
    ).toArray.foreach(addURL)
    ClassLoader.getSystemClassLoader
  }

  private lazy val loadedClasses: List[Class[_]] = {
    val jarFiles = jarFileLocs.map(jf => new JarFile(jf.getAbsolutePath))
    try {
      val classNames = jarFiles.flatMap(_.entries().asScala)
        .filter(je => je.getName.endsWith(".class"))
        .map(je => je.getName.replace("/", ".").stripSuffix(".class"))
      classNames.map(name => classLoader.loadClass(name))
    } finally {
      jarFiles.foreach(jf => jf.close())
    }
  }

  lazy val loadedTestClasses: List[Class[_]] = {
    val classNames = testFileLocs.map(je => je.stripSuffix(".class").replace("/", "."))
    classNames.map { name =>
      classLoader.loadClass(name): Class[_]
    }
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

  def load[T: ClassTag](instantiate: Class[_] => Any, debug: Boolean): List[T] = {
    loadedClasses.filter(lc => classTag[T].runtimeClass.isAssignableFrom(lc)).flatMap { lc =>
      catchInstantiationErrors(lc, instantiate, debug)
    }.asInstanceOf[List[T]]
  }

  def loadClass[T: ClassTag](debug: Boolean): List[T] = load[T](lc => lc.newInstance(), debug)

  def loadTest[T: ClassTag](instantiate: Class[_] => Any, debug: Boolean): List[T] = {
    loadedTestClasses.filter(lc => classTag[T].runtimeClass.isAssignableFrom(lc)).flatMap { lc =>
      catchInstantiationErrors(lc, instantiate, debug)
    }.asInstanceOf[List[T]]
  }

  def loadTestClass[T: ClassTag](debug: Boolean): List[T] = loadTest[T](lc => lc.newInstance(), debug)

  def loadObject[T: ClassTag](debug: Boolean): List[T] = load[T](
    lc =>{
      val cons = lc.getDeclaredConstructors()(0)
      cons.setAccessible(true)
      cons.newInstance()}
    ,
    debug)

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
