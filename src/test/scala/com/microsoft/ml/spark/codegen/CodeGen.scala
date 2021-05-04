// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import java.io.File
import com.microsoft.ml.spark.build.BuildInfo
import com.microsoft.ml.spark.codegen.CodegenConfig._
import com.microsoft.ml.spark.core.env.FileUtilities._
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.PyTestFuzzing
import com.microsoft.ml.spark.core.utils.JarLoadingUtils.instantiateServices
import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils._

object CodeGenUtils {
  def clean(dir: File): Unit = if (dir.exists()) FileUtils.forceDelete(dir)

  def toDir(f: File): File = new File(f, File.separator)
}

object CodeGen {

  import CodeGenUtils._

  def generatePythonClasses(): Unit = {
    instantiateServices[PythonWrappable].foreach { w =>
      w.makePyFile()
    }
  }

  def generateRClasses(): Unit = {
    instantiateServices[RWrappable].foreach { w =>
      w.makeRFile()
    }
  }

  private def makeInitFiles(packageFolder: String = ""): Unit = {
    val dir = new File(new File(PySrcDir, "mmlspark"), packageFolder)
    val packageString = if (packageFolder != "") packageFolder.replace("/", ".") else ""
    val importStrings =
      dir.listFiles.filter(_.isFile).sorted
        .map(_.getName)
        .filter(name => name.endsWith(".py") && !name.startsWith("_") && !name.startsWith("test"))
        .map(name => s"from mmlspark$packageString.${getBaseName(name)} import *\n").mkString("")
    writeFile(new File(dir, "__init__.py"), packageHelp(importStrings))
    dir.listFiles().filter(_.isDirectory).foreach(f =>
      makeInitFiles(packageFolder + "/" + f.getName)
    )
  }

  //noinspection ScalaStyle
  def generateRPackageData(): Unit = {
    // description file; need to encode version as decimal
    val today = new java.text.SimpleDateFormat("yyyy-MM-dd")
      .format(new java.util.Date())

    RSrcDir.mkdirs()
    writeFile(new File(RSrcDir.getParentFile, "DESCRIPTION"),
      s"""|Package: mmlspark
          |Title: Access to MMLSpark via R
          |Description: Provides an interface to MMLSpark.
          |Version: ${BuildInfo.rVersion}
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
          |Suggests:
          |    testthat (>= 3.0.0)
          |Config/testthat/edition: 3
          |""".stripMargin)

    writeFile(new File(RSrcDir, "package_register.R"),
      s"""|#' @import sparklyr
          |spark_dependencies <- function(spark_version, scala_version, ...) {
          |    spark_dependency(
          |        jars = c(),
          |        packages = c(
          |            sprintf("com.microsoft.ml.spark:mmlspark_%s:${BuildInfo.version}", scala_version)
          |        ),
          |        repositories = c("https://mmlspark.azureedge.net/maven")
          |    )
          |}
          |
          |#' @import sparklyr
          |.onLoad <- function(libname, pkgname) {
          |    sparklyr::register_extension(pkgname)
          |}
          |""".stripMargin)

    writeFile(new File(RSrcDir.getParentFile, "mmlspark.Rproj"),
      """
        |Version: 1.0
        |
        |RestoreWorkspace: Default
        |SaveWorkspace: Default
        |AlwaysSaveHistory: Default
        |
        |EnableCodeIndexing: Yes
        |UseSpacesForTab: Yes
        |NumSpacesForTab: 4
        |Encoding: UTF-8
        |
        |RnwWeave: Sweave
        |LaTeX: pdfLaTeX
        |
        |BuildType: Package
        |PackageUseDevtools: Yes
        |PackageInstallArgs: --no-multiarch --with-keep.source
        |
        |""".stripMargin)

  }

  def rGen(): Unit = {
    clean(RSrcRoot)
    generateRPackageData()
    generateRClasses()
    FileUtils.copyDirectoryToDirectory(toDir(RSrcOverrideDir), toDir(RSrcDir))
    FileUtils.copyDirectoryToDirectory(toDir(RTestOverrideDir), toDir(RTestDir))
  }

  def pyGen(): Unit = {
    clean(PySrcDir)
    generatePythonClasses()
    TestBase.stopSparkSession()
    FileUtils.copyDirectoryToDirectory(toDir(PySrcOverrideDir), toDir(PySrcDir))
    makeInitFiles()
  }

  def main(args: Array[String]): Unit = {
    clean(PackageDir)
    rGen()
    pyGen()
  }

}

object TestGen {

  import CodeGenUtils._

  def generatePythonTests(): Unit = {
    instantiateServices[PyTestFuzzing[_]].foreach { ltc =>
      try {
        ltc.makePyTestFile()
      } catch {
        case _: NotImplementedError =>
          println(s"ERROR: Could not generate test for ${ltc.testClassName} because of Complex Parameters")
      }
    }
  }

  private def makeInitFiles(packageFolder: String = ""): Unit = {
    val dir = new File(new File(PyTestDir, "mmlsparktest"), packageFolder)
    writeFile(new File(dir, "__init__.py"), "")
    dir.listFiles().filter(_.isDirectory).foreach(f =>
      makeInitFiles(packageFolder + "/" + f.getName)
    )
  }

  def main(args: Array[String]): Unit = {
    clean(TestDataDir)
    clean(PyTestDir)
    generatePythonTests()
    TestBase.stopSparkSession()
    FileUtils.copyDirectoryToDirectory(toDir(PyTestOverrideDir), toDir(PyTestDir))
    makeInitFiles()
  }
}
