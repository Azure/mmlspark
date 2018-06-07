// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.slf4j.LoggerFactory

class ValidateConfiguration extends TestBase {

  test("Basic BrainScript config E2E") {
    val relativeOutRoot = "out"
    val remappedInPath = "in.txt"
    val dataFormat = "text"

    val config = new BrainScriptBuilder()
      .setOutputRoot(relativeOutRoot)
      .setInputFile(
        new InputData(
          dataFormat,
          remappedInPath,
          Map("features" -> InputShape(10000, "sparse"),
            "labels" -> InputShape(1, "dense"))))

    val log = LoggerFactory.getLogger(this.getClass.getName.stripSuffix("$"))
    val cb = new CNTKCommandBuilder(log, false)
      .appendOverrideConfig(config.toOverrideConfig)
    println(cb.configs)
  }

}
