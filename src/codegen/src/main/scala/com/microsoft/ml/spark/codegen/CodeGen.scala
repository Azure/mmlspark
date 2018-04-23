// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import java.util.regex.Pattern

import com.microsoft.ml.spark.codegen.Config._
import com.microsoft.ml.spark.codegen.DocGen._
import com.microsoft.ml.spark.codegen.WrapperClassDoc._
import com.microsoft.ml.spark.core.env.FileUtilities.{File, _}
import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils._

import scala.util.matching.Regex

object CodeGen {

  def copyAllFiles(fromDir: File, rx: Regex, toDir: File): Unit = {
    if (!fromDir.isDirectory) { println(s"'$fromDir' is not a directory"); return }
    allFiles(fromDir, if (rx == null) null else f => rx.findFirstIn(f.getName).isDefined)
      .foreach{x => copyFile(x, toDir, overwrite=true)}
  }
  def copyAllFiles(fromDir: File, extension: String, toDir: File): Unit =
    copyAllFiles(fromDir,
                 if (extension == null || extension == "") null
                 else (Pattern.quote("." + extension) + "$").r,
                 toDir)

  def copyPythonDirFromRoots(fromDir: File, roots: List[String], relPath: String, toDir: File): Unit = {
    roots.foreach { root =>
      val dir = new File(new File(fromDir, root), relPath)
      if (dir.exists && dir.isDirectory) {
        FileUtils.copyDirectory(dir, toDir)
      }
    }
  }

  def copyAllFilesFromRoots(fromDir: File, roots: List[String], relPath: String,
                            extension: String, toDir: File): Unit = {
    roots.foreach { root =>
      val dir = new File(new File(fromDir, root), relPath)
      if (dir.exists && dir.isDirectory) {
        copyAllFiles(dir, extension, toDir)
      }
    }
  }
  def copyAllFilesFromRoots(fromDir: File, roots: List[String], relPath: String,
                            rx: Regex, toDir: File): Unit = {
    roots.foreach { root =>
      val dir = new File(new File(fromDir, root), relPath)
      if (dir.exists && dir.isDirectory) copyAllFiles(dir, rx, toDir)
    }
  }

  def generateArtifacts(): Unit = {
    println(s"""|Running code generation with config:
                |  topDir:    $topDir
                |  srcDir:    $srcDir
                |  outputDir: $outputDir
                |  pyDir:     $pyDir
                |  pyTestDir: $pyTestDir
                |  pyDocDir:  $pyDocDir
                |  rDir:      $rDir""".stripMargin)
    val roots = // note: excludes the toplevel project
      if (!rootsFile.exists) sys.error(s"Could not find roots file at $rootsFile")
      else readFile(rootsFile, _.getLines.toList).filter(_ != ".")
    println("Creating temp folders")
    pyDir.mkdirs
    pyTestDir.mkdirs
    pyDocDir.mkdirs
    tmpDocDir.mkdirs
    rSrcDir.mkdirs
    println("Copying jar files to output directory")
    copyAllFilesFromRoots(srcDir, roots, jarRelPath,
                          (Pattern.quote("-" + mmlVer + ".jar") + "$").r,
                          outputDir)
    println("Copying source python files")
    copyPythonDirFromRoots(srcDir, roots, pyRelPath, pyDir)
    println("Copying source R files")
    copyAllFilesFromRoots(srcDir, roots, rRelPath, "R", rSrcDir)
    println("Copying python class doc files")
    copyAllFilesFromRoots(srcDir, roots, txtRelPath, "txt", tmpDocDir)
    println("Generating python APIs")
    PySparkWrapperGenerator()
    println("Generating R APIs")
    SparklyRWrapperGenerator()
    println("Generating .rst files for the Python APIs documentation")
    genRstFiles()
    // build init file
    makeInitFiles()
    // package python+r zip files
    zipFolder(pyDir, pyZipFile)
    zipFolder(rDir,  rZipFile)
    FileUtils.forceDelete(rDir)
    // leave the python source files, so they will be included in the super-jar
    // FileUtils.forceDelete(pyDir)
    // delete the text files with the Python Class descriptions - truly temporary
    FileUtils.forceDelete(tmpDocDir)
  }

  def makeInitFiles(packageFolder: String = ""): Unit = {
    val dir = new File(pyDir, packageFolder)
    val packageString = if (packageFolder != "") packageFolder.replace("/",".") else ""
    val importStrings =
      allTopLevelFiles(dir, f => "^[a-zA-Z]\\w*[.]py$".r.findFirstIn(f.getName).isDefined)
        .map(f => s"from mmlspark$packageString.${getBaseName(f.getName)} import *\n").mkString("")
    writeFile(new File(dir, "__init__.py"), packageHelp(importStrings))
    dir.listFiles().filter(_.isDirectory).foreach(f =>
      makeInitFiles(packageFolder +"/" + f.getName)
    )
  }

  def main(args: Array[String]): Unit = {
    org.apache.log4j.BasicConfigurator.configure(new org.apache.log4j.varia.NullAppender())
    generateArtifacts()
  }

}
