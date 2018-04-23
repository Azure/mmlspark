// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable

class CheckpointDataSuite extends TransformerFuzzing[CheckpointData] {

  test("Smoke test for Spark session version") {
    assert(session.sparkContext.version ==
             sys.env.getOrElse("SPARK_VERSION",
                               sys.error("Missing $SPARK_VER environment variable")))
  }

  test("Cache DF") {
    val input = makeBasicDF()
    input.createOrReplaceTempView("cachingDFView")

    val checkpointer = new CheckpointData().setDiskIncluded(false).setRemoveCheckpoint(false)
    checkpointer.transform(input)

    assert(input.sqlContext.isCached("cachingDFView"))
  }

  test("Remove Cache on DF") {
    assert(session.sqlContext.isCached("cachingDFView"))
    val input = session.table("cachingDFView")

    val checkpointer = new CheckpointData().setDiskIncluded(false).setRemoveCheckpoint(true)
    checkpointer.transform(input)

    assert(!input.sqlContext.isCached("cachingDFView"))
  }

  override def testObjects(): Seq[TestObject[CheckpointData]] = Seq(new TestObject(
    new CheckpointData().setDiskIncluded(false).setRemoveCheckpoint(false),
    makeBasicDF()
  ))

  override def reader: MLReadable[_] = CheckpointData
}
