// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.feature

import scala.collection.mutable.ArrayBuilder
import org.apache.spark.SparkException
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/** A fast vector assembler.  The columns given must be ordered such that categorical columns come first
  * (otherwise spark learners will give categorical attributes to the wrong index).
  * Does not keep spurious numeric data which can significantly slow down computations when there are
  * millions of columns.
  */
class FastVectorAssembler (override val uid: String)
  extends Transformer with HasInputCols with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("FastVectorAssembler"))

  /** @group setParam */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    // Schema transformation.
    val schema = dataset.schema
    lazy val first = dataset.toDF.first()
    var addedNumericField = false

    // Propagate only nominal (categorical) attributes (others only slow down the code)
    val attrs: Array[Attribute] = $(inputCols).flatMap { c =>
      val field = schema(c)
      val index = schema.fieldIndex(c)
      field.dataType match {
        case _: NumericType | BooleanType =>
          val attr = Attribute.fromStructField(field)
          if (attr.isNominal) {
            if (addedNumericField) {
              throw new SparkException("Categorical columns must precede all others, column out of order: " + c)
            }
            Some(attr.withName(c))
          } else {
            addedNumericField = true
            None
          }
        case _: VectorUDT =>
          val group = AttributeGroup.fromStructField(field)
          if (group.attributes.isDefined) {
            // If attributes are defined, copy them with updated names.
            group.attributes.get.zipWithIndex.map { case (attr, i) =>
              if (attr.isNominal && attr.name.isDefined) {
                if (addedNumericField) {
                  throw new SparkException("Categorical columns must precede all others, column out of order: " + c)
                }
                attr.withName(c + "_" + attr.name.get)
              } else if (attr.isNominal) {
                if (addedNumericField) {
                  throw new SparkException("Categorical columns must precede all others, column out of order: " + c)
                }
                attr.withName(c + "_" + i)
              } else {
                addedNumericField = true
                null
              }
            }.filter(attr => attr != null)
          } else {
            addedNumericField = true
            None
          }
        case otherType =>
          throw new SparkException(s"FastVectorAssembler does not support the $otherType type")
      }
    }
    val metadata = new AttributeGroup($(outputCol), attrs).toMetadata()

    // Data transformation.
    val assembleFunc = udf { r: Row =>
      FastVectorAssembler.assemble(r.toSeq: _*)
    }
    val args = $(inputCols).map { c =>
      schema(c).dataType match {
        case DoubleType => dataset(c)
        case _: VectorUDT => dataset(c)
        case _: NumericType | BooleanType => dataset(c).cast(DoubleType).as(s"${c}_double_$uid")
      }
    }

    dataset.select(col("*"), assembleFunc(struct(args: _*)).as($(outputCol), metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputColNames = $(inputCols)
    val outputColName = $(outputCol)
    val inputDataTypes = inputColNames.map(name => schema(name).dataType)
    inputDataTypes.foreach {
      case _: NumericType | BooleanType =>
      case t if t.isInstanceOf[VectorUDT] =>
      case other =>
        throw new IllegalArgumentException(s"Data type $other is not supported.")
    }
    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }
    StructType(schema.fields :+ new StructField(outputColName, new VectorUDT, true))
  }

  override def copy(extra: ParamMap): FastVectorAssembler = defaultCopy(extra)

}

object FastVectorAssembler extends DefaultParamsReadable[FastVectorAssembler] {

  override def load(path: String): FastVectorAssembler = super.load(path)

  private[feature] def assemble(vv: Any*): Vector = {
    val indices = ArrayBuilder.make[Int]
    val values = ArrayBuilder.make[Double]
    var cur = 0
    vv.foreach {
      case v: Double =>
        if (v != 0.0) {
          indices += cur
          values += v
        }
        cur += 1
      case vec: Vector =>
        vec.foreachActive { case (i, v) =>
          if (v != 0.0) {
            indices += cur + i
            values += v
            ()
          }
        }
        cur += vec.size
      case null =>
        throw new SparkException("Values to assemble cannot be null.")
      case o =>
        throw new SparkException(s"$o of type ${o.getClass.getName} is not supported.")
    }
    Vectors.sparse(cur, indices.result(), values.result()).compressed
  }

}
