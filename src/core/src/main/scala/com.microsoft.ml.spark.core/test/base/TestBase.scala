// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.base

import java.io.File
import java.nio.file.Files

import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.ml.util.{MLReadable, MLWritable}
import org.apache.spark.sql.{DataFrame, _}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.col
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.scalactic.{Equality, TolerantNumerics}
import org.scalactic.source.Position
import org.scalatest._

import scala.reflect.ClassTag

// Common test tags
object TestBase {

  // Long, network, etc -- not running by default in local runs
  object Extended extends Tag("com.microsoft.ml.spark.test.tags.extended")

  // Depends on build environment (specifically, logged in through the az cli)
  object BuildServer extends Tag("com.microsoft.ml.spark.test.tags.buildserver")

  // Run only on Linux
  object LinuxOnly extends Tag("com.microsoft.ml.spark.test.tags.linuxonly")

}

trait LinuxOnly extends TestBase {
  override def test(testName: String, testTags: Tag*)(testFun: => Any)(implicit pos: Position): Unit =
    super.test(testName, testTags.toList.::(TestBase.LinuxOnly): _*)(testFun)
}

abstract class TestBase extends FunSuite with BeforeAndAfterEachTestData with BeforeAndAfterAll {

  println(s"\n>>>-------------------- $this --------------------<<<")

  // "This Is A Bad Thing" according to my research. However, this is
  // just for tests so maybe ok. A better design would be to break the
  // session stuff into TestSparkSession as a trait and have test suites
  // that need it "with TestSparkSession" instead, but that's a lot of
  // changes right now and maybe not desired.
  private var sessionInitialized = false
  protected lazy val session: SparkSession = {
    info(s"Creating a spark session for suite $this")
    sessionInitialized = true
    SparkSessionFactory.checkoutSession()
  }

  protected lazy val sc: SparkContext = session.sparkContext
  protected lazy val ssc: StreamingContext = new StreamingContext(sc, Seconds(1))

  protected lazy val dir = SparkSessionFactory.workingDir

  private var tmpDirCreated = false
  protected lazy val tmpDir = {
    tmpDirCreated = true
    Files.createTempDirectory("MML-Test-")
  }

  protected def normalizePath(path: String) = SparkSessionFactory.customNormalize(path)

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
    if (sessionInitialized) {
      info(s"Parallelism: ${session.sparkContext.defaultParallelism.toString}")
    }
    suiteElapsed = 0
  }

  protected override def afterAll(): Unit = {
    logTime(s"Suite $this", suiteElapsed, 10000)
    if (tmpDirCreated) {
      FileUtils.forceDelete(tmpDir.toFile)
    }
    if (sessionInitialized) {
      info("Shutting down spark session")
      // Only shut down sessions you yourself created
      SparkSessionFactory.returnSession()
    }
  }

  // Utilities

  def tryWithRetries[T](times: Array[Int] = Array(0, 100, 500, 1000, 3000, 5000))(block: () => T): T = {
    for ((t, i) <- times.zipWithIndex) {
      try {
        return block()
      } catch {
        case _: Exception if (i + 1) < times.length =>
          Thread.sleep(t.toLong)
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

  def interceptWithoutLogging[E <: Exception: ClassTag](e: => Any): Unit = {
    withoutLogging { intercept[E] { e }; () }
  }

  def assertSparkException[E <: Exception: ClassTag](stage: PipelineStage, data: DataFrame): Unit = {
    withoutLogging {
      intercept[E] {
        val transformer = stage match {
            case e: Estimator[_] => e.fit(data)
            case t: Transformer  => t
            case _ => sys.error(s"Unknown PipelineStage value: $stage")
          }
        // use .length to force the pipeline (.count might work, but maybe it's sometimes optimized)
        transformer.transform(data).foreach { r => r.length; () }
      }
      ()
    }
  }

  import session.implicits._

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
    assert(expected.count == result.count)
    assert(expected.schema.length == result.schema.length)
    (expected.columns zip result.columns).forall{ case (x,y) => x == y }
  }

  def time[R](block: => R): R = {
    val (result, t) = getTime(block)
    println(s"Elapsed time: ${t / 1e9} sec")
    result
  }

  def getTime[R](block: => R): (R, Long) = {
    val t0     = System.nanoTime()
    val result = block
    val t1     = System.nanoTime()
    (result, t1-t0)
  }

  def getTime[R](n: Int)(block: => R): (Seq[R], Long) = {
    val t0     = System.nanoTime()
    val results = (1 to n).map(i => block)
    val t1     = System.nanoTime()
    (results, (t1-t0)/n)
  }

  private def logTime(name: String, time: Long, threshold: Long) = {
    val msg = s"$name took ${time / 1000.0}s"
    if (time > threshold) {
      alert(msg)
    } else {
      info(msg)
    }
  }

}

trait DataFrameEquality extends TestBase {
  val epsilon = 1e-4
  implicit lazy val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(epsilon)
  implicit lazy val dvEq: Equality[DenseVector] = new Equality[DenseVector]{
    def areEqual(a: DenseVector, b: Any): Boolean = b match {
      case bArr:DenseVector =>
        a.values.zip(bArr.values).forall {case (x, y) => doubleEq.areEqual(x, y)}
    }
  }

  implicit lazy val rowEq: Equality[Row] = new Equality[Row]{
    def areEqual(a: Row, bAny: Any): Boolean = bAny match {
      case b:Row =>
        if (a.length != b.length) { return false }
        (0 until a.length).forall(j =>{
          a(j) match {
            case lhs: DenseVector =>
              lhs === b(j)
            case lhs: Double if lhs.isNaN =>
              b(j).asInstanceOf[Double].isNaN
            case lhs: Double =>
              b(j).asInstanceOf[Double] === lhs
            case lhs =>
              lhs === b(j)
          }
        })
    }
  }

  val sortInDataframeEquality = false

  val baseDfEq = new Equality[DataFrame]{
    def areEqual(a: DataFrame, bAny: Any): Boolean = bAny match {
      case ds:Dataset[_] =>
        val b = ds.toDF()
        if (a.columns !== b.columns) {
          return false
        }
        val (aList, bList) = if (sortInDataframeEquality) {
          (a.sort(a.columns.sorted.map(col):_*).collect(),
            b.sort(b.columns.sorted.map(col):_*).collect())
        } else {
          (a.collect(), b.collect())
        }

        if (aList.length != bList.length){
          return false
        }
        aList.zip(bList).forall { case (rowA, rowB) =>
          rowA === rowB
        }
    }
  }

  implicit lazy val dfEq: Equality[DataFrame] = baseDfEq

}
