// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.MMLParams
import com.microsoft.ml.spark.core.serialize.params.MapParam
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import spray.json.DefaultJsonProtocol._

import scala.collection.mutable

object EnsembleByKey extends DefaultParamsReadable[EnsembleByKey]

class EnsembleByKey(val uid: String) extends Transformer with MMLParams {
  def this() = this(Identifiable.randomUID("EnsembleByKey"))

  val keys = new StringArrayParam(this, "keys", "Keys to group by")

  def getKeys: Array[String] = $(keys)

  def setKeys(arr: Array[String]): this.type = set(keys, arr)

  def setKeys(arr: String*): this.type = set(keys, arr.toArray)

  def setKey(value: String): this.type = set(keys, Array(value))

  val cols = new StringArrayParam(this, "cols", "Cols to ensemble")

  def getCols: Array[String] = $(cols)

  def setCols(arr: Array[String]): this.type = set(cols, arr)

  def setCols(arr: String*): this.type = set(cols, arr.toArray)

  def setCol(value: String): this.type = set(cols, Array(value))

  val colNames = new StringArrayParam(this, "colNames", "Names of the result of each col")

  def getColNames: Array[String] = $(colNames)

  def setColNames(arr: Array[String]): this.type = set(colNames, arr)

  def setColNames(arr: String*): this.type = set(colNames, arr.toArray)

  def setColName(value: String): this.type = set(colNames, Array(value))

  val allowedStrategies = Set("mean")
  val strategy =
    StringParam(this, "strategy", "How to ensemble the scores, ex: mean",
                { x: String => allowedStrategies(x) })

  def getStrategy: String = $(strategy)

  def setStrategy(value: String): this.type = set(strategy, value)

  setDefault(strategy -> "mean")

  val collapseGroup =
    BooleanParam(this, "collapseGroup", "Whether to collapse all items in group to one entry")

  def getCollapseGroup: Boolean = $(collapseGroup)

  def setCollapseGroup(value: Boolean): this.type = set(collapseGroup, value)

  val vectorDims =new MapParam[String, Int](this, "vectorDims",
    "the dimensions of any vector columns, used to avoid materialization")

  def getVectorDims: Map[String, Int] = get(vectorDims).getOrElse(Map())

  def setVectorDims(value: Map[String, Int]): this.type = set(vectorDims, value)

  setDefault(collapseGroup -> true)

  override def transform(dataset: Dataset[_]): DataFrame = {

    if (get(colNames).isEmpty) {
      setDefault(colNames -> getCols.map(name => s"$getStrategy($name)"))
    }

    transformSchema(dataset.schema)

    val strategyToFloatFunction = Map(
      "mean" -> { (x: String, y: String) => mean(x).alias(y) }
    )

    val strategyToVectorFunction = Map(
      "mean" -> { (x: String, y: String) =>
        val dim = getVectorDims.getOrElse(x,
          dataset.select(x).take(1)(0).getAs[DenseVector](0).size)
        new VectorAvg(dim)(dataset(x)).asInstanceOf[Column].alias(y)
      }
    )

    val newCols = getCols.zip(getColNames).map { case (inColName, outColName) =>
      dataset.schema(inColName).dataType match {
        case _: DoubleType =>
          strategyToFloatFunction(getStrategy)(inColName, outColName)
        case _: FloatType =>
          strategyToFloatFunction(getStrategy)(inColName, outColName)
        case v if v == VectorType =>
          strategyToVectorFunction(getStrategy)(inColName, outColName)
        case t =>
          throw new IllegalArgumentException(s"Cannot operate on type $t with strategy $getStrategy")
      }
    }

    val aggregated = dataset.toDF()
      .groupBy(getKeys.head, getKeys.tail: _*)
      .agg(newCols.head, newCols.tail: _*)

    if (getCollapseGroup) {
      aggregated
    } else {
      val needToDrop = getColNames.toSet & dataset.columns.toSet
      dataset.drop(needToDrop.toList: _*).toDF().join(aggregated, getKeys)
    }

  }

  def transformSchema(schema: StructType): StructType = {
    val colSet = getCols.toSet
    val colToNewName = getCols.zip(getColNames).toMap

    val newFields = schema.fields.flatMap { f =>
      if (!colSet(f.name)) None
      else {
        val newField = StructField(colToNewName(f.name), f.dataType)
        f.dataType match {
          case _: DoubleType => Some(newField)
          case _: FloatType => Some(newField)
          case fdt if fdt == VectorType => Some(newField)
          case t => throw new IllegalArgumentException(s"Cannot operate on type $t with strategy $getStrategy")
        }
      }
    }

    val keyFields = schema.fields.filter(f => colSet(f.name))
    val fields =
      (if (getCollapseGroup) schema.fields else keyFields).++(newFields)

    new StructType(fields)
  }

  def copy(extra: ParamMap): this.type = defaultCopy(extra)

}

private class VectorAvg(n: Int) extends UserDefinedAggregateFunction {

  def inputSchema: StructType = new StructType().add("v", VectorType)

  def bufferSchema: StructType =
    new StructType().add("buff", ArrayType(DoubleType)).add("count", LongType)

  def dataType: DataType = VectorType

  def deterministic: Boolean = true

  def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer.update(0, Array.fill(n)(0.0))
    buffer.update(1, 0L)
  }

  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (!input.isNullAt(0)) {
      val buff = buffer.getAs[mutable.WrappedArray[Double]](0)
      val count = buffer.getLong(1)

      val v = input.getAs[Vector](0).toSparse
      for (i <- v.indices) {
        buff(i) += v(i)
      }
      buffer.update(0, buff)
      buffer.update(1, count + 1L)
    }
  }

  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val buff1 = buffer1.getAs[mutable.WrappedArray[Double]](0)
    val buff2 = buffer2.getAs[mutable.WrappedArray[Double]](0)
    val c1 = buffer1.getLong(1)
    val c2 = buffer2.getLong(1)

    for ((x, i) <- buff2.zipWithIndex) {
      buff1(i) += x
    }
    buffer1.update(0, buff1)
    buffer1.update(1, c1 + c2)
  }

  def evaluate(buffer: Row): Vector = {
    val c = buffer.getLong(1)
    Vectors.dense(buffer.getAs[Seq[Double]](0).map(_ / c).toArray)
  }

}
