// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.env

import java.io.File
import java.nio.file.{Files, StandardCopyOption}

import scala.io.{BufferedSource, Source}

object FileUtilities {

  def join(folders: String*): File = {
    folders.tail.foldLeft(new File(folders.head)) { case (f, s) => new File(f, s) }
  }

  def join(base: File, folders: String*): File = {
    folders.foldLeft(base) { case (f, s) => new File(f, s) }
  }

  // Same for StandardOpenOption
  type StandardOpenOption = java.nio.file.StandardOpenOption
  object StandardOpenOption {
    import java.nio.file.{StandardOpenOption => S}
    val APPEND = S.APPEND
    val CREATE = S.CREATE
  }

  def allFiles(dir: File, pred: File => Boolean = null): Array[File] = {
    def loop(dir: File): Array[File] = {
      val (dirs, files) = dir.listFiles.sorted.partition(_.isDirectory)
      (if (pred == null) files else files.filter(pred)) ++ dirs.flatMap(loop)
    }
    loop(dir)
  }

  // readFile takes a file name or a File, and function to extract a value from
  // BufferedSource which defaults to _.mkString; performs the read, closes the
  // source, and returns the result
  def readFile[T](file: File, read: BufferedSource => T): T = {
    val i = Source.fromFile(file)
    try read(i) finally i.close
  }
  def readFile(file: File): String = readFile(file, _.mkString)

  def writeFile(file: File, stuff: Any, flags: StandardOpenOption*): Unit = {
    Files.write(file.toPath, stuff.toString.getBytes(), flags: _*)
    ()
  }

  def copyFile(from: File, toDir: File, overwrite: Boolean = false): Unit = {
    Files.copy(from.toPath, new File(toDir, from.getName).toPath,
               (if (overwrite) Seq(StandardCopyOption.REPLACE_EXISTING)
                else Seq()): _*)
    ()
  }

  // Perhaps this should move into a more specific place, not a generic file utils thing
  def zipFolder(dir: File, out: File): Unit = {
    import java.io.{BufferedInputStream, FileInputStream, FileOutputStream}
    import java.util.zip.{ZipEntry, ZipOutputStream}
    val bufferSize = 2 * 1024
    val data = new Array[Byte](bufferSize)
    val zip = new ZipOutputStream(new FileOutputStream(out))
    val prefixLen = dir.getParentFile.toString.length + 1
    allFiles(dir).foreach { file =>
      zip.putNextEntry(new ZipEntry(file.toString.substring(prefixLen).replace(java.io.File.separator, "/")))
      val in = new BufferedInputStream(new FileInputStream(file), bufferSize)
      var b = 0
      while (b >= 0) { zip.write(data, 0, b); b = in.read(data, 0, bufferSize) }
      in.close()
      zip.closeEntry()
    }
    zip.close()
  }

}
