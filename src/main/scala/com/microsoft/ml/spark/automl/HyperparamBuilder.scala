// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.automl

import org.apache.spark.ml.param._

import scala.collection.{JavaConverters, mutable}
import scala.util.Random

abstract class RangeHyperParam[T](val min: T, val max: T, val seed: Long) extends Dist[T] {

  val random = new Random(seed)

}

class IntRangeHyperParam(min: Int, max: Int, seed: Long = 0)
    extends RangeHyperParam[Int](min, max, seed) {

  def getNext: Int = {
    val range = max - min
    random.nextInt(range) + min
  }

}

class LongRangeHyperParam(min: Long, max: Long, seed: Long = 0)
    extends RangeHyperParam[Long](min, max, seed) {

  def getNext: Long = {
    val range = max - min
    (random.nextLong() * range) + min
  }

}

class FloatRangeHyperParam(min: Float, max: Float, seed: Long = 0)
    extends RangeHyperParam[Float](min, max, seed) {

  val doubleRange = new DoubleRangeHyperParam(min.toDouble, max.toDouble)
  def getNext: Float = {
    doubleRange.getNext.toFloat
  }

}

class DoubleRangeHyperParam(min: Double, max: Double, seed: Long = 0)
    extends RangeHyperParam[Double](min, max, seed) {

  def getNext: Double = {
    val range = max - min
    (random.nextDouble() * range) + min
  }

}

object HyperParamUtils {
  /** Returns a range hyper param by matching to the given input type.
    * @param min The min value of the range
    * @param max The max value of the range
    * @param seed The random number seed.
    * @return A RangeHyperParam matched to the given type for min and max values.
    */
  def getRangeHyperParam(min: Any, max: Any, seed: Long = 0): RangeHyperParam[_] = {
    (min, max) match {
      case (minD: Double, maxD: Double) => new DoubleRangeHyperParam(minD, maxD, seed)
      case (minF: Float, maxF: Float) => new FloatRangeHyperParam(minF, maxF, seed)
      case (minL: Long, maxL: Long) => new LongRangeHyperParam(minL, maxL, seed)
      case (minI: Int, maxI: Int) => new IntRangeHyperParam(minI, maxI, seed)
      case _ => throw new Exception("Could not match RangeHyperParam constructor to the given type")
    }
  }

  /** Returns a discrete hyper param given a Java ArrayList through JavaConversions.
    * @param values The list of values from Java.
    * @param seed The random number seed.
    * @return A RangeHyperParam matched to the given type for min and max values.
    */
  def getDiscreteHyperParam(values: java.util.ArrayList[_], seed: Long = 0): DiscreteHyperParam[_] = {
    val valuesList = JavaConverters.asScalaBuffer(values).toList
    new DiscreteHyperParam(valuesList, seed)
  }
}

class DiscreteHyperParam[T](values: List[T], seed: Long = 0) extends Dist[T] {

  val random = new Random(seed)

  def getNext: T = {
    values(random.nextInt(values.length))
  }

}

/** Specifies the search space for hyperparameters.
  */
class HyperparamBuilder {

  private val hyperparams = mutable.Map.empty[Param[_], Dist[_]]

  /** Adds a param to the search space.
    */
  def addHyperparam[T](param: Param[T], values: Dist[T]): this.type = {
    hyperparams.put(param, values)
    this
  }

  /** Builds the search space of hyperparameters.
    * @return The map of hyperparameters to search through.
    */
  def build(): Array[(Param[_], Dist[_])] = hyperparams.toArray

}
