// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.base

import java.nio.file.{Files, Path}

import org.apache.commons.io.FileUtils
import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, _}
import org.apache.spark.streaming.{StreamingContext, Seconds => SparkSeconds}
import org.scalactic.source.Position
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest._
import org.scalatest.concurrent.TimeLimits
import org.scalatest.time.{Seconds, Span}

import scala.concurrent._
import scala.reflect.ClassTag

// Common test tags
object TestBase {

  // Run only on Linux
  object LinuxOnly extends Tag("com.microsoft.ml.spark.test.tags.linuxonly")

  def sc: SparkContext = spark.sparkContext
  def ssc: StreamingContext = new StreamingContext(sc, SparkSeconds(1))

  private var SparkInternal: Option[SparkSession] = None
  def resetSparkSession(numRetries: Int = 1, numCores: Option[Int] = None): Unit = {
    SparkInternal.foreach(_.close())
    SparkInternal = Some(
      SparkSessionFactory.getSession(s"$this", numRetries=numRetries, numCores=numCores)
    )
  }

  def stopSparkSession(): Unit = {
    SparkInternal.foreach(_.close())
    SparkInternal = None
  }

  def spark: SparkSession = {
    if (SparkInternal.isEmpty){
      resetSparkSession()
    }
    SparkInternal.get
  }

}

trait LinuxOnly extends TestBase {
  override def test(testName: String, testTags: Tag*)(testFun: => Any)(implicit pos: Position): Unit =
    super.test(testName, testTags.toList.::(TestBase.LinuxOnly): _*)(testFun)
}

trait Flaky extends TestBase {

  val retyMillis: Array[Int] = Array(0, 1000, 5000)

  override def test(testName: String, testTags: Tag*)(testFun: => Any)(implicit pos: Position): Unit = {
    super.test(testName, testTags: _*) {
      tryWithRetries(retyMillis)(testFun _)
    }
  }

}

trait TimeLimitedFlaky extends TestBase with TimeLimits {

  val timeoutInSeconds: Int = 5 * 60

  val retyMillis: Array[Int] = Array(0, 100, 100)

  override def test(testName: String, testTags: Tag*)(testFun: => Any)(implicit pos: Position): Unit = {
    super.test(testName, testTags: _*) {
      tryWithRetries(retyMillis) {
        failAfter(Span(timeoutInSeconds, Seconds)) {
          println("Executing time-limited flaky function")
          testFun _
        }
      }
    }
  }

}

abstract class TestBase extends FunSuite with BeforeAndAfterEachTestData with BeforeAndAfterAll {

  lazy val spark: SparkSession = TestBase.spark
  lazy val sc: SparkContext = TestBase.sc
  lazy val ssc: StreamingContext = TestBase.ssc

  protected lazy val dir: String = SparkSessionFactory.WorkingDir

  private var tmpDirCreated = false
  protected lazy val tmpDir: Path = {
    tmpDirCreated = true
    Files.createTempDirectory("MML-Test-")
  }

  protected def normalizePath(path: String): String = SparkSessionFactory.customNormalize(path)

  // Timing info
  var suiteElapsed: Long = 0
  var testStart: Long = 0
  var testElapsed: Long = 0

  // Test Fixture Overrides
  protected override def beforeEach(td: TestData): Unit = {
    testStart = System.currentTimeMillis
    testElapsed = 0
    super.beforeEach(td)
  }

  protected override def afterEach(td: TestData): Unit = {
    try {
      super.afterEach(td)
    }
    finally {
      testElapsed = System.currentTimeMillis - testStart
      logTime(s"Test ${td.name}", testElapsed, 3000)
      suiteElapsed += testElapsed
    }
  }

  protected override def beforeAll(): Unit = {
    suiteElapsed = 0
  }

  protected override def afterAll(): Unit = {
    logTime(s"Suite $this", suiteElapsed, 10000)
    if (tmpDirCreated) {
      FileUtils.forceDelete(tmpDir.toFile)
    }
  }

  // Utilities

  def tryWithRetries[T](times: Array[Int] = Array(0, 100, 500, 1000, 3000, 5000))(block: () => T): T = {
    for ((t, i) <- times.zipWithIndex) {
      try {
        return block()
      } catch {
        case e: Exception if (i + 1) < times.length =>
          println(s"RETRYING after $t ms:  Caught error: $e ")
          blocking {
            Thread.sleep(t.toLong)
          }
      }
    }
    throw new RuntimeException("This error should not occur, bug has been introduced in tryWithRetries")
  }

  def withoutLogging[T](e: => T): T = {
    // This should really keep the old level, but there is no sc.getLogLevel, so
    // take the cheap way out for now: just use "WARN", and do something proper
    // when/if needed
    sc.setLogLevel("OFF")
    try e finally sc.setLogLevel("WARN")
  }

  def interceptWithoutLogging[E <: Throwable: ClassTag](e: => Any): Unit = {
    withoutLogging{intercept[E]{e}; ()}
  }

  def assertSparkException[E <: Throwable: ClassTag](stage: PipelineStage, data: DataFrame): Unit = {
    withoutLogging {
      intercept[E] {
        val transformer = stage match {
          case e: Estimator[_] => e.fit(data)
          case t: Transformer => t
          case _ => sys.error(s"Unknown PipelineStage value: $stage")
        }
        transformer.transform(data).foreachPartition { it: Iterator[Row] => it.toList; () }
      }
      ()
    }
  }

  import spark.implicits._

  def makeBasicDF(): DataFrame = {
    Seq(
      (0, 0.toDouble, "guitars", "drums", 1.toLong, true),
      (1, 1.toDouble, "piano", "trumpet", 2.toLong, false),
      (2, 2.toDouble, "bass", "cymbals", 3.toLong, true))
      .toDF("numbers", "doubles", "words", "more", "longs", "booleans")
  }

  def makeBasicNullableDF(): DataFrame = {
    Seq(
      (0, 2.5, "guitars", Some("drums"), Some(2.toLong), None),
      (1, Double.NaN, "piano", Some("trumpet"), Some(1.toLong), Some(true)),
      (2, 8.9, "bass", None, None, Some(false)))
      .toDF("numbers", "doubles", "words", "more", "longs", "booleans")
  }

  def verifyResult(expected: DataFrame, result: DataFrame): Boolean = {
    assert(expected.count === result.count)
    assert(expected.schema.length == result.schema.length)
    (expected.columns zip result.columns).forall { case (x, y) => x == y }
  }

  def time[R](block: => R): R = {
    val (result, t) = getTime(block)
    println(s"Elapsed time: ${t / 1e9} sec")
    result
  }

  def getTime[R](block: => R): (R, Long) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    (result, t1 - t0)
  }

  def getTime[R](n: Int)(block: => R): (Seq[R], Long) = {
    val t0 = System.nanoTime()
    val results = (1 to n).map(_ => block)
    val t1 = System.nanoTime()
    (results, (t1 - t0) / n)
  }

  private def logTime(name: String, time: Long, threshold: Long): Unit = {
    val msg = s"$name took ${time / 1000.0}s"
    if (time > threshold) {
      alert(msg)
    } else {
      info(msg)
    }
  }

}
