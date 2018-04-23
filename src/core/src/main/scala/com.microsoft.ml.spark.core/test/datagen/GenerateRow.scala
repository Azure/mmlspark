// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.datagen

import com.microsoft.ml.spark.core.test.datagen.ColumnOptions.ColumnOptions
import com.microsoft.ml.spark.core.test.datagen.DataOptions.DataOptions
import org.apache.spark.mllib.random.RandomDataGenerator
import org.apache.spark.sql.Row

import scala.util.Random

/** Combines an array of row generators into a single row generator.
  * @param generators
  */
class RandomRowGeneratorCombiner(generators: Array[RandomMMLGenerator[Row]]) extends RandomMMLGenerator[Row] {

  override def nextValue(): Row = Row.merge(generators.map(generator => generator.nextValue()): _*)

  override def copy(): RandomRowGeneratorCombiner = new RandomRowGeneratorCombiner(generators)

}

/** Randomly generates a row given the set space of data, column options.
  * @param col The column generation options specifying the column type to generate.
  * @param data The data generation options specifying the data to generate.
  */
class RandomRowGenerator(col: ColumnOptions, data: DataOptions) extends RandomMMLGenerator[Row] {

  override def nextValue(): Row = {
    if (data == DataOptions.Boolean)
      Row(random.nextBoolean)
    else if (data == DataOptions.Byte)
      Row(random.nextByte)
    else if (data == DataOptions.Double)
      Row(random.nextDouble)
    else if (data == DataOptions.Int)
      Row(random.nextInt)
    else if (data == DataOptions.Short)
      Row(random.nextShort)
    else if (data == DataOptions.String)
      Row(random.nextString)
    else if (data == DataOptions.Date)
      Row(random.nextDate)
    else if (data == DataOptions.Timestamp)
      Row(random.nextTimestamp)
    else throw new Exception("Selected type not supported: " + data)
  }

  override def copy(): RandomRowGenerator = new RandomRowGenerator(col, data)

}

/** Base abstract class for random generation of data.
  * @tparam T The data to generate.
  */
abstract class RandomMMLGenerator[T] extends RandomDataGenerator[T] {

  var seed: Long = 0
  var random: GenerateDataType = new GenerateDataType(new Random(seed))

  override def setSeed(seed: Long): Unit = {
    random = new GenerateDataType(new Random(seed))
    this.seed = seed
  }

}
