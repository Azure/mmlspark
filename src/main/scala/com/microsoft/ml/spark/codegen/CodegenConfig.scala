// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import java.io.File

import com.microsoft.ml.spark.build.BuildInfo

object CodegenConfig {
  val DebugMode: Boolean = sys.env.getOrElse("DEBUGMODE", "").trim.toLowerCase == "true"

  val TopDir: File = BuildInfo.baseDirectory
  val Version: String = BuildInfo.version
  val PackageName: String = BuildInfo.name
  val TargetDir = new File(TopDir, s"target/scala-${BuildInfo.scalaVersion.slice(0,4)}")
  val ScalaSrcDir = "src/main/scala"

  val GeneratedDir = new File(TargetDir, "generated")
  val PackageDir = new File(GeneratedDir, "package")
  val SrcDir = new File(GeneratedDir, "src")
  val TestDir = new File(GeneratedDir, "test")
  val DocDir = new File(GeneratedDir, "doc")
  val TestDataDir = new File(GeneratedDir, "test-data")

  //Python Codegen Constant
  val PySrcDir = new File(SrcDir, "python")
  val PyPackageDir = new File(PackageDir, "python")
  val PyTestDir = new File(TestDir, "python")
  val PySrcOverrideDir = new File(TopDir, "src/main/python")
  val PyTestOverrideDir = new File(TopDir, "src/test/python")

  //R Codegen Constants
  val RSrcRoot = new File(SrcDir, "R")
  val RSrcDir = new File(RSrcRoot, "mmlspark/R")
  val RPackageDir = new File(PackageDir, "R")
  val RTestDir = new File(RSrcRoot, "mmlspark/tests")

  val RTestOverrideDir = new File(TopDir, "src/test/R")
  val RSrcOverrideDir = new File(TopDir, "src/main/R")

  //val rPackageFile = new File(rPackageDir, s"mmlspark-$mmlVer.zip")

  val InternalPrefix = "_"
  val ScopeDepth: String = " " * 4

  val CopyrightLines: String =
    s"""|# Copyright (C) Microsoft Corporation. All rights reserved.
        |# Licensed under the MIT License. See LICENSE in project root for information.
        |""".stripMargin

  // The __init__.py file
  def packageHelp(importString: String): String = {
    s"""|$CopyrightLines
        |
        |"\""
        |MicrosoftML is a library of Python classes to interface with the
        |Microsoft scala APIs to utilize Apache Spark to create distibuted
        |machine learning models.
        |
        |MicrosoftML simplifies training and scoring classifiers and
        |regressors, as well as facilitating the creation of models using the
        |CNTK library, images, and text.
        |"\""
        |
        |__version__ = "${BuildInfo.pythonizedVersion}"
        |__spark_package_version__ = "${BuildInfo.version}"
        |
        |$importString
        |""".stripMargin
  }
}
