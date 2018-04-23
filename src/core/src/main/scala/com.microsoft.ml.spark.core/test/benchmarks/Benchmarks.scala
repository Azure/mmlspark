// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.benchmarks

import java.io.PrintWriter

import com.microsoft.ml.spark.core.env.FileUtilities.File
import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.sql.Row
import org.scalatest.Assertion

import scala.collection.mutable.ListBuffer

case class Benchmark(name: String,
                     value: Double,
                     precision: Double,
                     higherIsBetter: Boolean = true) {
  def toCSVEntry: String = {
    s"$name,$value,$precision,$higherIsBetter"
  }
}

object Benchmark {
  def fromRow(r: Row): Benchmark = {
    Benchmark(r.getString(0), r.getDouble(1), r.getDouble(2), r.getBoolean(3))
  }

  def csvHeader: String = {
    "name,value,precision,higherIsBetter"
  }

}

abstract class Benchmarks extends TestBase {
  val moduleName: String
  val resourcesDirectory = new File(getClass.getResource("/").toURI)
  val oldBenchmarkFile = new File(resourcesDirectory, s"benchmarks_${this}.csv")
  val newBenchmarkFile = new File(resourcesDirectory, s"new_benchmarks_${this}.csv")
  val newBenchmarks: ListBuffer[Benchmark] = ListBuffer[Benchmark]()

  def addBenchmark(name: String,
                   value: Double,
                   precision: Double,
                   higherIsBetter: Boolean): Unit = {
    assert(!newBenchmarks.map(_.name).contains(name), s"Benchmark $name already exists")
    newBenchmarks.append(Benchmark(name, value, precision, higherIsBetter))
  }

  def addBenchmark(name: String,
                   value: Double,
                   precision: Int,
                   higherIsBetter: Boolean): Unit = {
    addBenchmark(name, value, scala.math.pow(10, -precision.toDouble), higherIsBetter)
  }

  def addBenchmark(name: String,
                   value: Double,
                   precision: Int): Unit = {
    addBenchmark(name, value, precision, higherIsBetter = true)
  }

  def addBenchmark(name: String,
                   value: Double,
                   precision: Double): Unit = {
    addBenchmark(name, value, precision, higherIsBetter = true)
  }

  def compareBenchmark(bmNew: Benchmark, bmOld: Benchmark): Assertion = {
    assert(bmNew.name == bmOld.name, "Benchmark names do not match")
    assert(bmNew.higherIsBetter == bmOld.higherIsBetter, "higherIsBetter does not match")
    assert(bmNew.precision === bmNew.precision, "precision does not match")
    assert(bmNew.precision >= 0, "precision needs to be positive")

    val diff = bmNew.value - bmOld.value
    assert(
      if (bmNew.higherIsBetter) {
        diff + bmNew.precision > 0
      } else {
        -1.0 * diff + bmNew.precision > 0
      }, s"new benchmark ${bmNew.value} and " +
        s"old benchmark ${bmOld.value} " +
        s"are not within ${bmNew.precision}")
  }

  def writeCSV(benchmarks: ListBuffer[Benchmark], file: File): Unit = {
    val lines = Seq(Benchmark.csvHeader) ++ benchmarks.map(_.toCSVEntry)
    StreamUtilities.using(new PrintWriter(file)) { pw =>
      pw.write(lines.mkString("\n"))
    }.get
  }

  def verifyBenchmarks(): Unit = {
    import session.implicits._
    val newBenchmarkDF = newBenchmarks.toDF()

    if (newBenchmarkFile.exists()) newBenchmarkFile.delete()
    writeCSV(newBenchmarks, newBenchmarkFile)

    val oldBenchmarks = session.read
      .option("header", true)
      .option("inferSchema", true)
      .csv(oldBenchmarkFile.getAbsolutePath)
      .collect().map(Benchmark.fromRow)
    val newMap = newBenchmarks.map(bm => (bm.name, bm)).toMap
    val oldMap = oldBenchmarks.map(bm => (bm.name, bm)).toMap

    assert(Set(newMap.keys) === Set(oldMap.keys))
    newMap.foreach { case (k, newBM) => compareBenchmark(newBM, oldMap(k)) }
  }

}

object DatasetUtils {

  def binaryTrainFile(name: String): File =
    new File(s"${sys.env("DATASETS_HOME")}/Binary/Train", name)

  def multiclassTrainFile(name: String): File =
    new File(s"${sys.env("DATASETS_HOME")}/Multiclass/Train", name)

  def regressionTrainFile(name: String): File =
    new File(s"${sys.env("DATASETS_HOME")}/Regression/Train", name)

}
