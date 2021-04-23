// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.nbtest

import com.microsoft.ml.spark.core.test.base.TestBase

import java.util.concurrent.TimeUnit
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.language.existentials

/** Tests to validate fuzzing of modules. */
class SynapseTests extends TestBase {

  test("convert") {
    SynapseUtilities.convertNotebook()
  }

  test("SynapsePROD") {
    val workspaceName = "wenqxsynapseppe"
    val poolName = "spark3pool"
    val livyUrl = "https://" +
      workspaceName +
      ".dev.azuresynapse-dogfood.net/livyApi/versions/2019-11-01-preview/sparkPools/" +
      poolName +
      "/batches"

    val livyBatches = SynapseUtilities.listPythonFiles()
      .filterNot(_.contains(" "))
      .filterNot(_.contains("-"))
      .map(f => {
        val livyBatch: LivyBatch = SynapseUtilities.uploadAndSubmitNotebook(livyUrl, f)
        SynapseUtilities.monitorJob(livyBatch, livyUrl)
      })

    try {
      val batchFutures: Array[Future[Any]] = livyBatches.map((batch: LivyBatch) => {
        Future {
          if (batch.state != "success") {
            if (batch.state == "error") {
              SynapseUtilities.postMortem(batch, livyUrl)
              throw new RuntimeException(s"${batch.id} returned with state ${batch.state}")
            }
            else {
              SynapseUtilities.retry(batch.id, livyUrl, SynapseUtilities.TimeoutInMillis, System.currentTimeMillis())
            }
          }
        }(ExecutionContext.global)
      })

      val failures = batchFutures
        .map(f => Await.ready(f, Duration(SynapseUtilities.TimeoutInMillis.toLong, TimeUnit.MILLISECONDS)).value.get)
        .filter(f => f.isFailure)
      assert(failures.isEmpty)
    }
    catch {
      case t: Throwable =>
        livyBatches.foreach { batch =>
          println(s"Cancelling job ${batch.id}")
          SynapseUtilities.cancelRun(livyUrl, batch.id)
        }
        throw t
    }
  }
}
