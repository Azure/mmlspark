// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.binary

import com.microsoft.ml.spark.core.schema.BinaryFileSchema
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.spark.binary.BinaryFileFormat
import org.apache.spark.sql.{DataFrame, SparkSession}

object BinaryFileReader {

  private def recursePath(fileSystem: FileSystem,
                          path: Path,
                          pathFilter:FileStatus => Boolean,
                          visitedSymlinks: Set[Path]): Array[Path] ={
    val filteredPaths = fileSystem.listStatus(path).filter(pathFilter)
    val filteredDirs = filteredPaths.filter(fs => fs.isDirectory & !visitedSymlinks(fs.getPath))
    val symlinksFound = visitedSymlinks ++ filteredDirs.filter(_.isSymlink).map(_.getPath)
    filteredPaths.map(_.getPath) ++ filteredDirs.map(_.getPath)
      .flatMap(p => recursePath(fileSystem, p, pathFilter, symlinksFound))
  }

  def recursePath(fileSystem: FileSystem, path: Path, pathFilter:FileStatus => Boolean): Array[Path] ={
    recursePath(fileSystem, path, pathFilter, Set())
  }

  /** Read the directory of binary files from the local or remote source
    *
    * @param path       Path to the directory
    * @param recursive  Recursive search flag
    * @return           DataFrame with a single column of "binaryFiles", see "columnSchema" for details
    */
  def read(path: String, recursive: Boolean, spark: SparkSession,
           sampleRatio: Double = 1, inspectZip: Boolean = true, seed: Long = 0L): DataFrame = {
    val p = new Path(path)
    val globs = if (recursive) {
      recursePath(p.getFileSystem(spark.sparkContext.hadoopConfiguration), p, {fs => fs.isDirectory})
        .map(g => g) ++ Array(p)
    } else {
      Array(p)
    }
    spark.read.format(classOf[BinaryFileFormat].getName)
      .option("subsample", sampleRatio)
      .option("seed", seed)
      .option("inspectZip",inspectZip).load(globs.map(g => g.toString):_*)
  }

  /** Read the directory of binary files from the local or remote source
    *
    * @param path       Path to the directory
    * @return           DataFrame with a single column of "binaryFiles", see "columnSchema" for details
    */
  def stream(path: String, spark: SparkSession,
           sampleRatio: Double = 1, inspectZip: Boolean = true, seed: Long = 0L): DataFrame = {
    val p = new Path(path)
    spark.readStream.format(classOf[BinaryFileFormat].getName)
      .option("subsample", sampleRatio)
      .option("seed", seed)
      .option("inspectZip",inspectZip).schema(BinaryFileSchema.schema).load(p.toString)
  }

}
