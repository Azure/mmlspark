// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.datagen

import java.sql.Timestamp
import java.sql.Date
import org.apache.commons.lang.RandomStringUtils
import scala.util.Random

/** Generates the specified random data type. */
class GenerateDataType(random: Random) extends Serializable {

  def nextTimestamp: Timestamp = new Timestamp(random.nextLong())

  def nextBoolean: Boolean = random.nextBoolean()

  def nextByte: Byte = {
    val byteArray = new Array[Byte](1)
    random.nextBytes(byteArray)
    byteArray(0)
  }

  def nextDouble: Double = random.nextDouble()

  def nextInt: Int = random.nextInt()

  def nextShort: Short = random.nextInt(Short.MaxValue).toShort

  def nextString: String = RandomStringUtils.random(random.nextInt(100), 0, 0, true, true, null,
    new java.util.Random(random.nextLong()))

  def nextDate: Date = new Date(random.nextLong())

}
