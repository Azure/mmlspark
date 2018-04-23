// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.serialize.{ConstructorReadable, ConstructorWritable}
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{broadcast, col, lit, max}
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.runtime.universe.{TypeTag, typeTag}

/** An estimator that calculates the weights for balancing a dataset.
  * For example, if the negative class is half the size of the positive class, the weights will be
  * 2 for rows with negative classes and 1 for rows with positive classes.
  * these weights can be used in weighted classifiers and regressors to correct for heavilty
  * skewed datasets. The inputCol should be the labels of the classes, and the output col will
  * be the requisite weights.
  *
  * @param uid
  */
class ClassBalancer(val uid: String) extends Estimator[ClassBalancerModel]
  with DefaultParamsWritable with HasInputCol with HasOutputCol {

  def this() = this(Identifiable.randomUID("ClassBalancer"))

  setDefault(outputCol -> "weight")

  val broadcastJoin = new BooleanParam(this, "broadcastJoin",
    "Whether to broadcast the class to weight mapping to the worker")

  def getBroadcastJoin: Boolean = $(broadcastJoin)

  def setBroadcastJoin(value: Boolean): this.type = set(broadcastJoin, value)

  setDefault(broadcastJoin -> true)

  def fit(dataset: Dataset[_]): ClassBalancerModel = {
    val counts = dataset.toDF().select(getInputCol).groupBy(getInputCol).count()
    val maxVal = counts.agg(max("count")).collect().head.getLong(0)
    val weights = counts
      .withColumn(getOutputCol, lit(maxVal) / col("count"))
      .select(getInputCol, getOutputCol)
    new ClassBalancerModel(uid, getInputCol, getOutputCol, weights, getBroadcastJoin).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[ClassBalancerModel] = defaultCopy(extra)

  def transformSchema(schema: StructType): StructType = schema.add(getOutputCol, DoubleType)

}

object ClassBalancer extends DefaultParamsReadable[ClassBalancer]

class ClassBalancerModel(val uid: String, val inputCol: String,
                         val outputCol: String, val weights: DataFrame, broadcastJoin: Boolean)
  extends Model[ClassBalancerModel] with ConstructorWritable[ClassBalancerModel] {

  override def copy(extra: ParamMap): ClassBalancerModel = defaultCopy(extra)

  val ttag: TypeTag[ClassBalancerModel] = typeTag[ClassBalancerModel]

  def objectsToSave: List[Any] = List(uid, inputCol, outputCol, weights, broadcastJoin)

  def transformSchema(schema: StructType): StructType = schema.add(outputCol, DoubleType)

  def transform(dataset: Dataset[_]): DataFrame = {
    val w = if (broadcastJoin) {
      broadcast(weights)
    } else {
      weights
    }
    dataset.toDF().join(w, inputCol)
  }

}

object ClassBalancerModel extends ConstructorReadable[ClassBalancerModel]
