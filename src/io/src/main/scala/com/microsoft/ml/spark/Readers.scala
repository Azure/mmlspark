// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import com.microsoft.ml.spark.IO.binary.BinaryFileReader
import com.microsoft.ml.spark.IO.image.ImageReader
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.language.implicitConversions

/** Implicit conversion allows sparkSession.readImages(...) syntax
  * Example:
  *     import com.microsoft.ml.spark.Readers.implicits._
  *     sparkSession.readImages(path, recursive = false)
  */
object Readers {

  object implicits {

    class Session(sparkSession: SparkSession) {

      /** @param path         Path to the files directory
        * @param recursive    Recursive path search flag
        * @param sampleRatio  Fraction of the files loaded
        * @param inspectZip   Whether zip files are treated as directories
        * @return Dataframe with a single column "value" of binary files, see BinaryFileSchema for details
        */
      def readBinaryFiles(path: String, recursive: Boolean,
                          sampleRatio: Double = 1, inspectZip: Boolean = true, seed: Long = 0L): DataFrame =
        BinaryFileReader.read(path, recursive, sparkSession, sampleRatio, inspectZip, seed)

      /** Read the directory of images from the local or remote source
        *
        * @param path         Path to the image directory
        * @param recursive    Recursive path search flag
        * @param sampleRatio  Fraction of the files loaded
        * @param inspectZip   Whether zip files are treated as directories
        * @return Dataframe with a single column "image" of images, see ImageSchema for details
        */
      def readImages(path: String, recursive: Boolean,
                     sampleRatio: Double = 1, inspectZip: Boolean = true, seed: Long = 0L): DataFrame =
        ImageReader.read(path, recursive, sparkSession, sampleRatio, inspectZip, seed)
    }

    implicit def ImplicitSession(sparkSession: SparkSession):Session = new Session(sparkSession)

  }
}
