// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lime

import com.microsoft.ml.spark.core.spark.FluentAPI._
import com.microsoft.ml.spark.codegen.Wrappable
import com.microsoft.ml.spark.core.schema.DatasetExtensions
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{ComplexParamsReadable, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

object TextLIME extends ComplexParamsReadable[TextLIME]

/** Distributed implementation of
  * Local Interpretable Model-Agnostic Explanations (LIME)
  *
  * https://arxiv.org/pdf/1602.04938v1.pdf
  */
class TextLIME(val uid: String) extends Model[TextLIME]
  with LIMEBase with Wrappable with BasicLogging {
  logClass()

  setDefault(nSamples -> 1000, regularization -> 0.0, samplingFraction -> 0.3)

  def this() = this(Identifiable.randomUID("TextLIME"))

  protected def maskText(tokens: Seq[String], states: Seq[Boolean]): String = {
    tokens.zip(states).filter { case (_, state) => state }.map(_._1).mkString(" ")
  }

  val tokenCol = new Param[String](this,
    "tokenCol", "The column holding the token")

  def getTokenCol: String = $(tokenCol)

  def setTokenCol(v: String): this.type = set(tokenCol, v)

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val df = dataset.toDF

      val idCol = DatasetExtensions.findUnusedColumnName("id", df)
      val statesCol = DatasetExtensions.findUnusedColumnName("states", df)
      val inputCol2 = DatasetExtensions.findUnusedColumnName("inputCol2", df)

      val tokenizer = new Tokenizer().setInputCol(getInputCol).setOutputCol(getTokenCol)

      val tokenDF = tokenizer.transform(df)

      // Indices of the columns containing each image and image's superpixels
      val inputType = df.schema(getInputCol).dataType
      val maskUDF = UDFUtils.oldUdf(maskText _, StringType)

      val mapped = tokenDF.withColumn(idCol, monotonically_increasing_id())
        .withColumnRenamed(getInputCol, inputCol2)
        .withColumn(statesCol, explode_outer(getSampleUDF(size(col(getTokenCol)))))
        .withColumn(getInputCol, maskUDF(col(getTokenCol), col(statesCol)))
        .withColumn(statesCol, UDFUtils.oldUdf(
          { barr: Seq[Boolean] => new DenseVector(barr.map(b => if (b) 1.0 else 0.0).toArray) },
          VectorType)(col(statesCol)))
        .mlTransform(getModel)
        .drop(getInputCol)

      LIMEUtils.localAggregateBy(mapped, idCol, Seq(statesCol, getPredictionCol))
        .withColumn(statesCol, arrToMatUDF(col(statesCol)))
        .withColumn(getPredictionCol, arrToVectUDF(col(getPredictionCol)))
        .withColumn(getOutputCol, fitLassoUDF(col(statesCol), col(getPredictionCol), lit(getRegularization)))
        .drop(statesCol, getPredictionCol)
        .withColumnRenamed(inputCol2, getInputCol)
    })
  }

  override def copy(extra: ParamMap): TextLIME = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getTokenCol, ArrayType(StringType)).add(getOutputCol, VectorType)
  }

}

