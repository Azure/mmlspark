// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.datagen

import breeze.stats.distributions.{Rand, RandBasis, Uniform}
import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}

import scala.util.Random

/** Specifies the trait for constraints on generating a dataset. */
trait HasDatasetGenerationConstraints {
  var numRows: Int
  var numCols: Int
  var numSlotsPerCol: Array[Int]
  var randomizeColumnNames: Boolean
}

/** Basic constraints for generating a dataset. */
class BasicDatasetGenerationConstraints(numberOfRows: Int, numberOfColumns: Int, numberOfSlotsPerColumn: Array[Int])
  extends HasDatasetGenerationConstraints {
  override var numRows: Int = numberOfRows
  override var numCols: Int = numberOfColumns
  override var numSlotsPerCol: Array[Int] = numberOfSlotsPerColumn
  override var randomizeColumnNames: Boolean = true
}

/** Contraints on generating a dataset where all parameters are randomly generated.
  * @param minRows The min number of rows.
  * @param maxRows The max number of rows.
  * @param minCols The min number of columns.
  * @param maxCols The max number of columns.
  * @param minSlots The min number of slots.
  * @param maxSlots The max number of slots.
  */
class RandomDatasetGenerationConstraints(minRows: Int,
                                         maxRows: Int,
                                         minCols: Int,
                                         maxCols: Int,
                                         minSlots: Int,
                                         maxSlots: Int)
  extends HasDatasetGenerationConstraints {

  override var numRows: Int = _
  override var numCols: Int = _
  override var numSlotsPerCol: Array[Int] = _
  override var randomizeColumnNames: Boolean = _

  /** Generates values for rows, columns and slots based on the given constraints using a random number generator.
    * @param random The random number generator.
    */
  def generateConstraints(random: Random): Unit = {
    val rand = new RandBasis(new MersenneTwister(random.nextInt()))
    val distributionRows = new Uniform(minRows.toDouble, maxRows.toDouble)(rand)
    val distributionCols = new Uniform(minCols.toDouble, maxCols.toDouble)(rand)
    val distributionSlots = new Uniform(minCols.toDouble, maxCols.toDouble)(rand)
    numRows = distributionRows.draw().toInt
    numCols = distributionCols.draw().toInt
    numSlotsPerCol = (1 to numCols).map(col => distributionSlots.draw().toInt).toArray
  }

}
