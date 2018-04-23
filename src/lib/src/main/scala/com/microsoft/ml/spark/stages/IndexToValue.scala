// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, MMLParams}
import com.microsoft.ml.spark.core.schema.SchemaConstants.MMLTag
import com.microsoft.ml.spark.schema.{CategoricalColumnInfo, CategoricalUtilities}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

object IndexToValue extends DefaultParamsReadable[IndexToValue]

/** This class takes in a categorical column with MML style attibutes and then transforms
  * it back to the original values.  This extends MLLIB IndexToString by allowing the transformation
  * back to any types of values.
  */

class IndexToValue(val uid: String) extends Transformer with HasInputCol with HasOutputCol with MMLParams {
  def this() = this(Identifiable.randomUID("IndexToValue"))

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from column selection
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val info = new CategoricalColumnInfo(dataset.toDF(), getInputCol)
    require(info.isCategorical, "column " + getInputCol + "is not Categorical")
    val dataType = info.dataType
    var getLevel =
      dataType match {
        case _: IntegerType => getLevelUDF[Int](dataset)
        case _: LongType => getLevelUDF[Long](dataset)
        case _: DoubleType => getLevelUDF[Double](dataset)
        case _: StringType => getLevelUDF[String](dataset)
        case _: BooleanType => getLevelUDF[Boolean](dataset)
        case _ => throw new Exception("Unsupported type " + dataType.toString)
      }
    dataset.withColumn(getOutputCol, getLevel(dataset(getInputCol)).as(getOutputCol))
  }

  private class Default[T] {var value: T = _ }

  def getLevelUDF[T: TypeTag](dataset: Dataset[_])(implicit ct: ClassTag[T]): UserDefinedFunction = {
    val map = CategoricalUtilities.getMap[T](dataset.schema(getInputCol).metadata)
    udf((index: Int) => {
      if (index == map.numLevels && map.hasNullLevel) {
        new Default[T].value
      } else {
        map.getLevelOption(index)
          .getOrElse(throw new IndexOutOfBoundsException(
            "Invalid metadata: Index greater than number of levels in metadata, " +
              s"index: $index, levels: ${map.numLevels}"))
      }
    })
  }

  def transformSchema(schema: StructType): StructType = {
    val metadata = schema(getInputCol).metadata
    val dataType =
      if (metadata.contains(MMLTag)) {
        CategoricalColumnInfo.getDataType(metadata, true).get
      } else {
        schema(getInputCol).dataType
      }
    val newField = StructField(getOutputCol, dataType)
    if (schema.fieldNames.contains(getOutputCol)) {
      val index = schema.fieldIndex(getOutputCol)
      val fields = schema.fields
      fields(index) = newField
      StructType(fields)
    } else {
      schema.add(newField)
    }
  }

  def copy(extra: ParamMap): this.type = defaultCopy(extra)
}
