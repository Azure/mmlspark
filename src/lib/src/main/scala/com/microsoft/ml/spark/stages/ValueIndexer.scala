// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import java.lang.{Boolean => JBoolean, Double => JDouble, Integer => JInt, Long => JLong}

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, MMLParams}
import com.microsoft.ml.spark.core.serialize.params.ArrayParam
import com.microsoft.ml.spark.schema._
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

import scala.math.Ordering
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

object ValueIndexer extends DefaultParamsReadable[ValueIndexer] {
  def validateAndTransformSchema(schema: StructType, outputCol: String): StructType = {
    val newField = NominalAttribute.defaultAttr.withName(outputCol).toStructField()
    if (schema.fieldNames.contains(outputCol)) {
      val index = schema.fieldIndex(outputCol)
      val fields = schema.fields
      fields(index) = newField
      StructType(fields)
    } else {
      schema.add(newField)
    }
  }
}

trait ValueIndexerParams extends MMLParams with HasInputCol with HasOutputCol

class NullOrdering[T] (ord: Ordering[T]) extends Ordering[T] {
  override def compare(x: T, y: T): Int =
    if (x == null && y == null) 0
    else if (x == null) -1
    else if (y == null) 1
    else ord.compare(x, y)
}

object NullOrdering {
  def apply[T](ord: Ordering[T]): NullOrdering[T] = new NullOrdering(ord)
}

/** Fits a dictionary of values from the input column.
  * Model then transforms a column to a categorical column of the given array of values.
  * Similar to StringIndexer except it can be used on any value types.
  */
class ValueIndexer(override val uid: String) extends Estimator[ValueIndexerModel]
  with ValueIndexerParams {

  def this() = this(Identifiable.randomUID("ValueIndexer"))

  /** Fits the dictionary of values from the input column.
    *
    * @param dataset The input dataset to train.
    * @return The model for transforming columns to categorical.
    */
  override def fit(dataset: Dataset[_]): ValueIndexerModel = {
    val dataType = dataset.schema(getInputCol).dataType
    val levels = dataset.select(getInputCol).distinct().collect().map(row => row(0))
    // Sort the levels
    val castSortLevels =
      dataType match {
        case _: IntegerType => sortLevels[JInt](levels)(NullOrdering[JInt](Ordering[JInt]))
        case _: LongType => sortLevels[JLong](levels)(NullOrdering[JLong](Ordering[JLong]))
        case _: DoubleType => sortLevels[JDouble](levels)(NullOrdering[JDouble](Ordering[JDouble]))
        case _: StringType => sortLevels[String](levels)(NullOrdering[String](Ordering[String]))
        case _: BooleanType => sortLevels[JBoolean](levels)(NullOrdering[JBoolean](Ordering[JBoolean]))
        case _ => throw new UnsupportedOperationException("Unsupported Categorical type " + dataType.toString)
      }
    // Create the indexer
    new ValueIndexerModel()
      .setInputCol(getInputCol)
      .setOutputCol(getOutputCol)
      .setLevels(castSortLevels)
      .setDataType(dataType)
  }

  private def sortLevels[T: TypeTag](levels: Array[_])
                        (ordering: Ordering[T])
                        (implicit ct: ClassTag[T]): Array[_] = {
    var castLevels = levels.map(_.asInstanceOf[T])
    castLevels.sorted(ordering)
  }

  override def copy(extra: ParamMap): Estimator[ValueIndexerModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    ValueIndexer.validateAndTransformSchema(schema, getOutputCol)
}

/** Model produced by [[ValueIndexer]]. */
class ValueIndexerModel(val uid: String)
    extends Model[ValueIndexerModel] with ValueIndexerParams with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("ValueIndexerModel"))

  /** Levels in categorical array
    * @group param
    */
  val levels = new ArrayParam(this, "levels", "Levels in categorical array")
  val emptyLevels = Array()
  /** @group getParam */
  def getLevels: Array[_] = if (isDefined(levels)) $(levels) else emptyLevels
  /** @group setParam */
  def setLevels(value: Array[_]): this.type = set(levels, value)

  /** The datatype of the levels as a jason string
    * @group param
    */
  val dataType = StringParam(this, "dataType", "The datatype of the levels as a Json string", "string")
  /** @group getParam */
  def getDataTypeStr: String = if ($(dataType) == "string") DataTypes.StringType.json else $(dataType)
  /** @group setParam */
  def setDataTypeStr(value: String): this.type = set(dataType, value)
  /** @group getParam */
  def getDataType: DataType = if ($(dataType) == "string") DataTypes.StringType else DataType.fromJson($(dataType))
  /** @group setParam */
  def setDataType(value: DataType): this.type = set(dataType, value.json)

  setDefault(inputCol -> "input", outputCol -> (uid + "_output"))

  override def copy(extra: ParamMap): ValueIndexerModel =
    new ValueIndexerModel(uid)
      .setLevels(getLevels)
      .setDataType(getDataType)
      .setInputCol(getInputCol)
      .setOutputCol(getOutputCol)

  /** Transform the input column to categorical */
  override def transform(dataset: Dataset[_]): DataFrame = {
    getDataType match {
      case _: IntegerType => addCategoricalColumn[Int](dataset)
      case _: LongType => addCategoricalColumn[Long](dataset)
      case _: DoubleType => addCategoricalColumn[Double](dataset)
      case _: StringType => addCategoricalColumn[String](dataset)
      case _: BooleanType => addCategoricalColumn[Boolean](dataset)
      case _ => throw new UnsupportedOperationException("Unsupported Categorical type " + getDataType.toString)
    }
  }

  private def addCategoricalColumn[T: TypeTag](dataset: Dataset[_])
                                              (implicit ct: ClassTag[T]): DataFrame = {
    val nonNullLevels = getLevels.filter(_ != null)
    val castLevels = nonNullLevels.map {
      case l: scala.math.BigInt => l.toInt.asInstanceOf[T]
      case l => l.asInstanceOf[T]
    }
    val hasNullLevel = getLevels.length != nonNullLevels.length
    val map = new CategoricalMap(castLevels, false, hasNullLevel)
    val unknownIndex =
      if (!map.hasNullLevel) {
        map.numLevels
      } else {
        map.numLevels + 1
      }
    val getIndex = udf((level: T) => {
      // Treat nulls and NaNs specially
      if (level == null || (level.isInstanceOf[Double] && level.asInstanceOf[Double].isNaN)) {
        map.numLevels
      } else {
        map.getIndexOption(level).getOrElse(unknownIndex)
      }
    })
    // Add the MML style and MLLIB style metadata for categoricals
    val metadata = map.toMetadata(map.toMetadata(dataset.schema(getInputCol).metadata, true), false)
    val inputColIndex = getIndex(dataset(getInputCol))
    dataset.withColumn(getOutputCol, inputColIndex.as(getOutputCol, metadata))
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    ValueIndexer.validateAndTransformSchema(schema, getOutputCol)
}

object ValueIndexerModel extends ComplexParamsReadable[ValueIndexerModel]
