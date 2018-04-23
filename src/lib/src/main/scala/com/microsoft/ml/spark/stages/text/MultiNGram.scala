// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.text

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, MMLParams}
import com.microsoft.ml.spark.core.schema.DatasetExtensions
import com.microsoft.ml.spark.core.serialize.params.ArrayParam
import org.apache.spark.ml.NamespaceInjections.pipelineModel
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable

object MultiNGram extends DefaultParamsReadable[MultiNGram]

/** Extracts several ngrams
  *
  * @param uid The id of the module
  */
class MultiNGram(override val uid: String)
    extends Transformer with HasInputCol with HasOutputCol with MMLParams {

  def this() = this(Identifiable.randomUID("MultiNGram"))

  setDefault(outputCol, uid + "_output")

  val lengths =
    new ArrayParam(this, "lengths",
                   "the collection of lengths to use for ngram extraction")

  def getLengths:Array[Int] = $(lengths)
    .toArray.map {
      case i: scala.math.BigInt => i.toInt
      case i: java.lang.Integer => i.toInt
    }

  def setLengths(v: Array[Int]): this.type = set(lengths, v)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val intermediateOutputCols = getLengths.map( n =>
      DatasetExtensions.findUnusedColumnName(s"ngram_$n")(dataset.columns.toSet)
    )
    val models = getLengths.zip(intermediateOutputCols).map { case (n, out) =>
      new NGram().setN(n).setInputCol(getInputCol).setOutputCol(out)
    }
    val intermediateDF = pipelineModel(models).transform(df)
    intermediateDF.map {row =>
      val mergedNGrams = intermediateOutputCols
        .map(col => row.getAs[mutable.WrappedArray[String]](col))
        .reduce(_ ++ _)
      Row.merge(row, Row(mergedNGrams))
    } (RowEncoder(intermediateDF.schema.add(getOutputCol, ArrayType(StringType))))
      .drop(intermediateOutputCols:_*)
  }

  override def copy(extra: ParamMap): MultiNGram =
    defaultCopy(extra)

  def transformSchema(schema: StructType): StructType = {
    assert(schema(getInputCol).dataType==ArrayType(StringType))
    schema.add(getOutputCol, ArrayType(StringType))
  }

}
