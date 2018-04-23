// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.env

import java.nio.file.Paths

import org.slf4j.{Logger, LoggerFactory}

object EnvironmentUtils {

  // We should use Apache Commons Lang instead
  def IsWindows: Boolean = System.getProperty("os.name").toLowerCase().indexOf("win") >= 0

  // Make this overrideable so people have control over the granularity
  private def getEnvInfo(log: Logger): Option[String] = {
    log.info(s"Computing GPU count on ${if (IsWindows) "Windows" else "Linux"}")
    val nvsmicmd = if (IsWindows) {
      // Unlikely nvidia is on the path
      val nvsmi = Paths.get(
        System.getenv("ProgramFiles"),
        "NVIDIA Corporation",
        "NVSMI",
        "nvidia-smi.exe").toAbsolutePath.toString
      "\"" + nvsmi + "\""
    } else {
      "nvidia-smi"
    }
    // Probably a more Scala-idiomatic way to do this
    try {
      Some(ProcessUtils.getProcessOutput(log, s"$nvsmicmd -L"))
    } catch {
      // Use the logging API to do this properly
      case e: Exception => {
        log.warn(s"Couldn't query Nvidia SMI for GPU info: $e")
        None
      }
    }
  }

  lazy val GPUCount: Option[Int] = {
    val log = LoggerFactory.getLogger(this.getClass.getName.stripSuffix("$"))
    val envInfo = getEnvInfo(log)
    if (envInfo.isEmpty) None else {
      // Commons Lang has isNotBlank
      val gpucnt = envInfo.get.split("\n").count(!_.trim.isEmpty)
      log.info(s"$gpucnt GPUs detected")
      Some(gpucnt)
    }
  }

  def getEnv(s:String): String = {
    System.getenv(s) match {
      case null =>
        throw new IllegalArgumentException(s"Environment variable $s not found")
      case v => v
    }
  }

}
