// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.nbtest
//TODO temp hack because ij picks up on it test classes by mistake

import java.util.concurrent.TimeUnit

import com.microsoft.ml.spark.core.test.base.TestBase

import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.language.existentials

/** Tests to validate fuzzing of modules. */
class NotebookTests extends TestBase {

  test("Databricks") {
    val clusterId = DatabricksUtilities.createClusterInPool(DatabricksUtilities.ClusterName, DatabricksUtilities.PoolId)
    try {
      println("Checking if cluster is active")
      tryWithRetries(Seq.fill(60*15)(1000).toArray){() =>
        assert(DatabricksUtilities.isClusterActive(clusterId))}
      println("Installing libraries")
      DatabricksUtilities.installLibraries(clusterId)
      tryWithRetries(Seq.fill(60*3)(1000).toArray){() =>
        assert(DatabricksUtilities.isClusterActive(clusterId))}
      val folder = DatabricksUtilities.Folder
      println(s"Creating folder $folder")
      DatabricksUtilities.workspaceMkDir(folder)
      println(s"Submitting jobs")
      val jobIds = DatabricksUtilities.NotebookFiles.map(DatabricksUtilities.uploadAndSubmitNotebook(clusterId, _))
      println(s"Submitted ${jobIds.length} for execution: ${jobIds.toList}")
      try {
        val monitors = jobIds.map((runId: Int) => DatabricksUtilities.monitorJob(
          runId,
          DatabricksUtilities.TimeoutInMillis,
          logLevel = 2))
        println(s"Monitoring Jobs...")
        val failures = monitors
          .map(Await.ready(_, Duration(DatabricksUtilities.TimeoutInMillis.toLong, TimeUnit.MILLISECONDS)).value.get)
          .filter(_.isFailure)
        assert(failures.isEmpty)
      } catch {
        case t: Throwable =>
          jobIds.foreach { jid =>
            println(s"Cancelling job $jid")
            DatabricksUtilities.cancelRun(jid)
          }
          throw t
      }
    } finally {
      DatabricksUtilities.deleteCluster(clusterId)
    }
  }

  test("SynapsePROD") {
    val workspaceName = "mmlsparkdemosynws"
    val poolName = "sparkpool1"
    val livyUrl = "https://" +
                  workspaceName +
                  ".dev.azuresynapse.net/livyApi/versions/2019-11-01-preview/sparkPools/" +
                  poolName +
                  "/batches"
    val livyBatches = LivyUtilities.NotebookFiles.map(LivyUtilities.uploadAndSubmitNotebook(livyUrl, _))
    println(s"Submitted ${livyBatches.length} jobs for execution: " +
      s"${livyBatches.map(batch => s"${batch.id} : ${batch.state}")}")
    try {
      val batchFutures = livyBatches.map((batch: LivyBatch) => {
        Future {
          if (batch.state != "success"){
            if (batch.state == "error") {
              LivyUtilities.postMortem(batch, livyUrl)
              throw new RuntimeException(s"${batch.id} returned with state ${batch.state}")
            }
            else{
              LivyUtilities.retry(batch.id, livyUrl, LivyUtilities.TimeoutInMillis, System.currentTimeMillis())
            }
          }
        }(ExecutionContext.global)
      })

      val failures = batchFutures
        .map(Await.ready(_, Duration(LivyUtilities.TimeoutInMillis.toLong, TimeUnit.MILLISECONDS)).value.get)
        .filter(_.isFailure)
      assert(failures.isEmpty)
    }
    catch {
      case t: Throwable =>
        livyBatches.foreach { batch =>
          println(s"Cancelling job ${batch.id}")
          LivyUtilities.cancelRun(livyUrl, batch.id)
        }
        throw t
    }
  }

  ignore("list running jobs for convenievce") {
    val obj = DatabricksUtilities.databricksGet("jobs/runs/list?active_only=true&limit=1000")
    println(obj)
  }

}
