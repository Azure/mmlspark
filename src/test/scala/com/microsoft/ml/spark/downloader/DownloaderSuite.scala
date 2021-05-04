// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.downloader

import java.io.File
import java.nio.file.Files

import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.commons.io.FileUtils

import scala.collection.JavaConverters._
import scala.concurrent.duration.Duration
import scala.util.Random

class DownloaderSuite extends TestBase {

  lazy val saveDir: File = Files.createTempDirectory("Models-").toFile
  lazy val d = new ModelDownloader(spark, saveDir.toURI)

  test("retry utility should catch flakiness"){
    (1 to 20).foreach { _ =>
      val result = FaultToleranceUtils.retryWithTimeout(20, Duration.apply(2, "seconds")) {
        val r = Random.nextDouble()
        if (r > .5) {
          println(s"$r failed")
          throw new IllegalArgumentException("Flakiness")
        } else if (r < .1){
          //Getting stuck
          val m = 3* 1e3.toLong
          println(s"$r Stuck for $m")
          Thread.sleep(m)
        }
        println(s"$r Success")
        5
      }
      assert(result === 5)
    }
  }

  test("A downloader should be able to download a model") {
    val m = d.remoteModels.asScala.filter(_.name == "CNN").next()
    val schema = d.downloadModel(m)
    println(schema)
    assert(m.size == new File(schema.uri).length())
    assert(d.localModels.asScala.toList.length == 1)
  }

  ignore("A downloader should be able to get all Models " +
    "and maybeDownload should be fast if models are downloaded") {
    val (modTimes, modTimes2) = FaultToleranceUtils.retryWithTimeout(10, Duration.apply(500, "seconds")) {
      d.downloadModels()
      val modTimes = d.localModels.asScala.map(s =>
        new File(s.uri).lastModified())

      d.downloadModels()
      val modTimes2 = d.localModels.asScala.map(s =>
        new File(s.uri).lastModified())
      (modTimes, modTimes2)
    }
    // No modification on second call because models are cached
    assert(modTimes.toList === modTimes2.toList)

    // the downloader's local models will reflect the change
    assert(d.localModels.asScala.toList.length == d.remoteModels.asScala.toList.length)

    // there will be a metadata file for every model
    assert(saveDir.list().count(_.endsWith(".meta")) == d.localModels.asScala.toList.length)
  }

  override def afterAll(): Unit = {
    if (saveDir.exists()) {
      FileUtils.forceDelete(saveDir)
    }
    super.afterAll()
  }

}
