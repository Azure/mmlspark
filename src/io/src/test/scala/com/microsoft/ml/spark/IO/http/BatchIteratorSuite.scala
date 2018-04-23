// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.test.base.TestBase
import org.scalatest.Assertion

class BatchIteratorSuite extends TestBase {

  def delayedIterator(n: Int, wait: Int = 5): Iterator[Int] = {
    (1 to n).toIterator.map { x =>
      Thread.sleep(x * wait.toLong)
      x
    }
  }

  def standardIterator(n: Int): Iterator[Int] = (1 to n).toIterator

  def comparePerformance[T](iteratorCons: => Iterator[T],
                            ratioRequired: Double = 2.0,
                            maxBuffer: Int = Integer.MAX_VALUE,
                            numTrials: Int = 5): Assertion = {
    val (dresults, dtime) = getTime(numTrials) {
      new BatchIterator(iteratorCons, maxBuffer).toList
    }

    val (results, time) = getTime(numTrials) {
      iteratorCons.toList.map(x => List(x))
    }
    assert(dresults.head.flatten === results.head.flatten)
    val ratio = dtime/time.toDouble
    println(s"ratio: $ratio, Batched: ${dtime/1000000}ms, normal: ${time/1000000}ms")
    assert(ratio < ratioRequired)
  }

  def compareAbsolutePerformance[T](iteratorCons: => Iterator[T],
                                    extraTimeUpperBound: Double = 40,
                                    maxBuffer: Int = Integer.MAX_VALUE
                                   ): Assertion = {
    val (dresults, dtime) = getTime {
      new BatchIterator(iteratorCons, maxBuffer).toList
    }
    val (results, time) = getTime {
      iteratorCons.toList.map(x => List(x))
    }
    assert(dresults.flatten === results.flatten)
    val ratio = dtime/time.toDouble
    println(s"ratio: $ratio, Batched: ${dtime/1000000}ms, normal: ${time/1000000}ms")
    assert(dtime < time + extraTimeUpperBound * 1000000)
  }

  test("Performance and behavior") {
    comparePerformance(delayedIterator(30), numTrials = 1)
    comparePerformance(delayedIterator(10), numTrials = 1)
    comparePerformance(delayedIterator(30), maxBuffer = 10, numTrials = 1)
    comparePerformance(delayedIterator(10), maxBuffer = 10, numTrials = 1)
    compareAbsolutePerformance(standardIterator(300), maxBuffer = 10)
    compareAbsolutePerformance(standardIterator(3000), maxBuffer = 10)
  }

  test("no deadlocks") {
    (1 to 1000).foreach {_ =>
      val l = new BatchIterator((1 to 100).toIterator, 10).toList
      assert(l.flatten === (1 to 100))
    }
  }

}
