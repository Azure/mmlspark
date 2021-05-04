// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.utils

class StopWatch {
  private var elapsedTime: Long = 0L
  private var startTime: Long = 0L

  def restart(): Unit = {
    elapsedTime = 0

    start()
  }

  def start(): Unit = {
    startTime = System.nanoTime
  }

  def pause(): Unit = {
    elapsedTime += System.nanoTime - startTime
  }

  def measure[T](f: => T): T  = {
    start()

    val ret = f

    pause()

    ret
  }

  def elapsed(): Long = elapsedTime
}
