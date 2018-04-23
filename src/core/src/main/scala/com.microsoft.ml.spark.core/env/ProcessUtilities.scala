// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.env

import scala.sys.process._
import org.slf4j.Logger

object ProcessUtils {

  // These are only here until we create a more robust
  // stream-redirected utility
  def getProcessOutput(log: Logger, cmd: ProcessBuilder): String = {
    log.info(s"Capturing external process:\n $cmd")
    val ret = cmd.!!
    log.info(s"$ret...done!")
    ret
  }

  def runProcess(log: Logger, cmd: ProcessBuilder): Int = {
    log.info(s"Executing external process:\n $cmd")
    val ret = cmd .!
    log.info(s"$ret...done!")
    ret
  }

}
