// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.contracts.Wrappable
import com.microsoft.ml.spark.core.serialize.params.PipelineStageParam
import com.microsoft.ml.spark.core.serialize.{ConstructorReadable, ConstructorWritable}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PipelineStage, Transformer}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.runtime.universe.{TypeTag, typeTag}

object Timer extends ComplexParamsReadable[Timer]

trait TimerParams extends Wrappable {
  val stage = new PipelineStageParam(this, "stage", "The stage to time")

  def getStage: PipelineStage = $(stage)

  val logToScala = BooleanParam(this, "logToScala", "Whether to output the time to the scala console", true)

  def getLogToScala: Boolean = $(logToScala)

  def setLogToScala(v: Boolean): this.type = set(logToScala, v)

  val disableMaterialization =
    BooleanParam(this, "disableMaterialization",
                 "Whether to disable timing (so that one can turn it off for evaluation)",
                 false)

  def getDisableMaterialization: Boolean = $(disableMaterialization)

  def setDisable(v: Boolean): this.type = set(disableMaterialization, v)

  protected def formatTime(t: Long, isTransform: Boolean, count: Option[Long], stage: PipelineStage): String = {
    val time = {
      t / 1000000000.0
    }
    val action = if (isTransform) "transform" else "predict"
    val amount = count match {
      case Some(i) => s"$i rows"
      case _ => ""
    }
    s"${stage} took ${time}s to $action $amount"
  }

  protected def log(str: String): Unit = {
    if (str != "") println(str)
  }

}

class Timer(val uid: String) extends Estimator[TimerModel]
  with TimerParams with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("Timer"))

  def setStage(s: PipelineStage): this.type = set(stage, s)

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = getStage.transformSchema(schema)

  def fitWithTime(dataset: Dataset[_]): (TimerModel, String) = {
    val cachedDatasetBefore = if (getDisableMaterialization) dataset else dataset.cache()
    val countBeforeVal = if (getDisableMaterialization) None else Some(dataset.count())
    getStage match {
      case t: Transformer => (new TimerModel(uid, t).setParent(this), "")
      case e: Estimator[_] =>
        val t0 = System.nanoTime()
        val fitE = e.fit(dataset)
        val t1 = System.nanoTime()
        (new TimerModel(uid, fitE).setParent(this),
         formatTime(t1 - t0, false, countBeforeVal, e))
    }
  }

  def fit(dataset: Dataset[_]): TimerModel = {
    val (model, message) = fitWithTime(dataset)
    log(message)
    model
  }

}

object TimerModel extends ConstructorReadable[TimerModel]

class TimerModel(val uid: String, t: Transformer)
  extends Model[TimerModel] with TimerParams with ConstructorWritable[TimerModel]{

  override val objectsToSave = List(uid, t)
  override val ttag: TypeTag[TimerModel] = typeTag[TimerModel]

  override def copy(extra: ParamMap): TimerModel = {
    new TimerModel(uid, t.copy(extra)).setParent(parent)
  }

  def transformWithTime(dataset: Dataset[_]): (DataFrame, String) = {
    val cachedDatasetBefore = if (getDisableMaterialization) dataset else dataset.cache()
    val countBeforeVal = if (getDisableMaterialization) None else Some(dataset.count())

    val t0 = System.nanoTime()

    val res = t.transform(cachedDatasetBefore)

    val cachedDatasetAfter = if (getDisableMaterialization) res else res.cache()
    val countAfterVal = if (getDisableMaterialization) None else Some(cachedDatasetAfter.count())

    val t1 = System.nanoTime()

    (cachedDatasetAfter, formatTime(t1 - t0, true, countAfterVal, t))
  }

  def transformSchema(schema: StructType): StructType = t.transformSchema(schema)

  def transform(dataset: Dataset[_]): DataFrame = {
    val (model, message) = transformWithTime(dataset)
    log(message)
    model
  }

}
