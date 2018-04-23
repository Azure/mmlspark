// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.MMLParams
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/** Constants for <code>PartitionSample</code>. */
object PSConstants {
    final val ModeRS = "RandomSample"
    final val ModeHead = "Head"
    final val ModeATP = "AssignToPartition"

    final val rsAbsolute = "Absolute"
    final val rsPercent = "Percentage"

    final val newColDefault = "Partition"
}

trait PartitionSampleParams extends MMLParams {

  // TODO: Convert to Enum
  /** Sampling mode. The options are:
    * - AssignToPartition
    *   - seed
    *   - numParts
    *   - newColName
    * - RandomSample
    *   - mode: Absolute or Percentage
    *     - count (when Absolute)
    *     - percent (when Percentage)
    *   - seed
    * - Head
    *   - count
    *
    * The default setting is RandomSample
    * @group param
    */
  final val mode = StringParam(this, "mode", "AssignToPartition, RandomSample, or Head")
  setDefault(mode, PSConstants.ModeRS)
  /** @group getParam */
  final def getMode: String = $(mode)
  /** @group setParam */
  def setMode(value: String): this.type = set(mode, value)

  // TODO: Convert to Enum
  /** RandomSample mode
    * - Absolute
    * - Percentage
    *
    * Default is Percentage
    * @group param
    */
  final val rsMode = StringParam(this, "rsMode", "Absolute or Percentage", PSConstants.rsPercent)
  /** @group getParam */
  final def getRandomSampleMode: String = $(rsMode)
  /** @group setParam */
  def setRandomSampleMode(value: String): this.type = set(rsMode, value)

  // TODO: We need to create Option[Int] idiom for params
  /** Seed for random operations (RandomSplit or AssignToPartition
    * Default is -1
    * @group param
    */
  final val seed = LongParam(this, "seed", "Seed for random operations", -1L)
  /** @group getParam */
  final def getSeed: Long = $(seed)
  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** Percentage of rows to return when Random Sampling
    * Default is .01
    * @group param
    */
  final val percent = DoubleParam(this, "percent", "Percent of rows to return", 0.01)
  /** @group getParam */
  final def getPercent: Double = $(percent)
  /** @group setParam */
  def setPercent(value: Double): this.type = set(percent, value)

  /** Number of rows to return. Must be specified when the mode is HEAD, or when the mode is RandomSample
    * and the sampling mode is Absolute.
    *
    * Default is 1000
    * @group param
    */
  final val count = LongParam(this, "count", "Number of rows to return", 1000L)
  /** @group getParam */
  final def getCount: Long = $(count)
  /** @group setParam */
  def setCount(value: Long): this.type = set(count, value)

  /** Name of the partition column. Specify a name when the mode is AssignToPartition
    *
    * Default is \"Partition\"
    * @group param
    */
  final val newColName = StringParam(this, "newColName", "Name of the partition column", PSConstants.newColDefault)
  /** @group getParam */
  final def getNewColName: String = $(newColName)
  /** @group setParam */
  def setNewColName(value: String): this.type = set(newColName, value)

  /** Number of partitions when the mode is AssignToPartition
    *
    * Default is 10
    * @group param
    */
  final val numParts = IntParam(this, "numParts", "Number of partitions", 10)
  /** @group getParam */
  final def getNumParts: Int = $(numParts)
  /** @group setParam */
  def setNumParts(value: Int): this.type = set(numParts, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    if (Seq(PSConstants.ModeHead, PSConstants.ModeRS).contains($(mode)))
      schema
    else
      ??? // schema + newCol
  }
}

object PartitionSample extends DefaultParamsReadable[PartitionSample]

// UID should be overridden by driver for controlled identification at the DAG level
/** @param uid The id of the module */
sealed class PartitionSample(override val uid: String) extends Transformer with PartitionSampleParams {

  def this() = this(Identifiable.randomUID("PartitionSample"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    $(mode) match {
      case PSConstants.ModeHead => dataset.limit(
        if ($(count) <= 2000000000) $(count).toInt else throw new Exception("Head limit 2b rows")).toDF
      case PSConstants.ModeRS => randomSample(dataset, $(rsMode), $(seed)).toDF
      case PSConstants.ModeATP => dataset.withColumn($(newColName), /* broken */ dataset.col("input"))
      case _ => ???
    }
  }

  def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  def copy(extra: ParamMap): PartitionSample = defaultCopy(extra)

  private def randomSample(
    ds: Dataset[_],
    rsMode: String,
    seed: Long,
    replace: Boolean = false): Dataset[_] = {
        val frac = rsMode match {
            case PSConstants.rsPercent => $(percent)
            case PSConstants.rsAbsolute => $(count).toDouble / ds.count
            case _ => ???
        }
        println(s"Sampling ${ds.count} rows by ${frac * 100}% to get ~${ds.count * frac} rows")
        return ds.sample(replace, frac, seed)
  }

}
