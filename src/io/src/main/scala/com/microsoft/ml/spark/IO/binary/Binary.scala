// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.binary

import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.language.implicitConversions

/** Implicit conversion allows sparkSession.readImages(...) syntax
  * Example:
  *     import com.microsoft.ml.spark.Readers.implicits._
  *     sparkSession.readImages(path, recursive = false)
  */
object Binary {

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
    }

    implicit def ImplicitSession(sparkSession: SparkSession):Session = new Session(sparkSession)

  }

}
