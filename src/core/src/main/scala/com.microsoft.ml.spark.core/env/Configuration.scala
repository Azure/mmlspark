// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.env

import java.util.Properties
import scala.sys.process._

import org.apache.spark._
import org.apache.spark.sql.SparkSession

// For development convenience - not hard to reimplement the pieces used here
import com.typesafe.config.{Config, ConfigFactory}

// This is meant to provide a uniform means of configuring the
// SDK and extension packages in a Spark-compatible form while
// also allowing for env vars as we may use via the CLI
abstract class Configuration(config: Config) {
  private val namespace = "mmlspark"

  protected def subspace: String

  def root: String = combine(namespace, subspace)

  private def combine(names: String*): String = names.mkString(".")
}

class MMLConfig(config: Config) extends Configuration(config) {
  override val subspace = "sdk"
}

object MMLConfig {
  // Use spark model of one config/JVM
  private lazy val baseConfig = new MMLConfig(ConfigFactory.load())
  def get(): MMLConfig = baseConfig

  private def combine(names: String*): String = names.mkString(".")
}

// Move to CNTK subpackage
class CNTKConfig(config: Config) extends MMLConfig(config) {
  override val subspace = "cntk"
  // Danil brings up a good point - device configuration is confusing
  // we need to not only say number of devices but also which ones in the
  // GPU list to use (1080 gaming + Titan DL for example)
}

// Move to TLC subpackage
class TLCConfig(config: Config) extends MMLConfig(config) {
  override val subspace = "tlc"
}
