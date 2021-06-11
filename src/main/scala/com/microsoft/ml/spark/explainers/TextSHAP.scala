// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.explainers

import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.ComplexParamsReadable
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.shared.HasInputCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, explode}
import org.apache.spark.sql.types._

trait TextSHAPParams extends KernelSHAPParams with HasInputCol with HasTokensCol {
  self: TextSHAP =>

  def setInputCol(value: String): this.type = this.set(inputCol, value)

  setDefault(tokensCol -> "tokens")
}

class TextSHAP(override val uid: String)
  extends KernelSHAPBase(uid)
    with TextSHAPParams
    with TextExplainer {

  logClass()

  def this() = {
    this(Identifiable.randomUID("TextSHAP"))
  }

  override protected def createSamples(df: DataFrame,
                                       idCol: String,
                                       coalitionCol: String): DataFrame = {
    val numSamplesOpt = this.getNumSamplesOpt

    val samplesUdf = UDFUtils.oldUdf(
      {
        (tokens: Seq[String]) =>
          val effectiveNumSamples = KernelSHAPBase.getEffectiveNumSamples(numSamplesOpt, tokens.size)
          val sampler = new KernelSHAPTextSampler(tokens, effectiveNumSamples)
          (1 to effectiveNumSamples).map {
            _ =>
              val (sampleTokens, features, distance) = sampler.sample
              val sampleText = sampleTokens.mkString(" ")
              (sampleText, features, distance)
          }
      },
      getSampleSchema(StringType)
    )

    df.withColumn("samples", explode(samplesUdf(col(getTokensCol))))
      .select(
        col(idCol),
        col("samples.coalition").alias(coalitionCol),
        col("samples.sample").alias(getInputCol)
      )
  }

  override protected def validateSchema(schema: StructType): Unit = {
    super.validateSchema(schema)

    require(
      schema(getInputCol).dataType == StringType,
      s"Field $getInputCol is expected to be string type, but got ${schema(getInputCol).dataType} instead."
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    this.validateSchema(schema)
    schema
      .add(getTokensCol, ArrayType(StringType))
      .add(getOutputCol, VectorType)
  }
}

object TextSHAP extends ComplexParamsReadable[TextSHAP]