// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.schema.DatasetExtensions.{findUnusedColumnName => newCol}
import com.microsoft.ml.spark.core.serialize.params.TransformerParam
import com.microsoft.ml.spark.stages.basic.DropColumns
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.ml.{NamespaceInjections, PipelineModel, Transformer}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

object SimpleHTTPTransformer extends ComplexParamsReadable[SimpleHTTPTransformer]

class SimpleHTTPTransformer(val uid: String)
    extends Transformer with HTTPParams with HasMaxBatchSize
    with HasInputCol with HasOutputCol with ComplexParamsWritable {

  def this() = this(Identifiable.randomUID("SimpleHTTPTransformer"))

  val flattenOutputBatches: Param[Boolean] = BooleanParam(
      this, "flattenOutputBatches",
      "whether to flatten the output batches")

  /** @group getParam */
  def getFlattenOutputBatches: Boolean = $(flattenOutputBatches)

  /** @group setParam */
  def setFlattenOutputBatches(value: Boolean): this.type = set(flattenOutputBatches, value)

  val inputParser: Param[Transformer] = new TransformerParam(
      this, "inputParser", "format to parse the column to",
      { case _: HTTPInputParser => true; case _ => false })

  /** @group getParam */
  def getInputParser: HTTPInputParser = $(inputParser).asInstanceOf[HTTPInputParser]

  /** @group setParam */
  def setInputParser(value: HTTPInputParser): this.type = set(inputParser, value)

  setDefault(inputParser -> new JSONInputParser())

  def setUrl(url: String): SimpleHTTPTransformer.this.type = {
    getInputParser match {
      case jip: JSONInputParser => setInputParser(jip.setUrl(url))
      case _ => throw new IllegalArgumentException("this setting is only availible when using a JSONInputParser")
    }
  }

  val outputParser: Param[Transformer] = new TransformerParam(
    this, "outputParser", "format to parse the column to",
    { case _: HTTPOutputParser => true; case _ => false })

  /** @group getParam */
  def getOutputParser: HTTPOutputParser = $(outputParser).asInstanceOf[HTTPOutputParser]

  /** @group setParam */
  def setOutputParser(value: HTTPOutputParser): this.type = set(outputParser, value)

  private def makePipeline(schema: StructType): PipelineModel = {
    val colsToAvoid = schema.fieldNames.toSet ++ Set(getOutputCol)

    val parsedInputCol = newCol("parsedInput")(colsToAvoid)
    val unparsedOutputCol = newCol("unparsedOutput")(colsToAvoid)

    val mb = get(maxBatchSize).map(n =>
      new MiniBatchTransformer().setMaxBatchSize(getMaxBatchSize)
    )

    val inputParser =
      Some(getInputParser
             .setInputCol(getInputCol)
             .setOutputCol(parsedInputCol))

    val client =
      Some(new HTTPTransformer()
             .setHandlingStrategy(getHandlingStrategy)
             .setConcurrency(getConcurrency)
             .setConcurrentTimeout(getConcurrentTimeout)
             .setInputCol(parsedInputCol)
             .setOutputCol(unparsedOutputCol))

    val outputParser =
      Some(getOutputParser
             .setInputCol(unparsedOutputCol)
             .setOutputCol(getOutputCol))

    val dropCols = Some(new DropColumns().setCols(Array(parsedInputCol, unparsedOutputCol)))

    val flatten = get(maxBatchSize).flatMap(_ =>
      get(flattenOutputBatches) match {
        case None => Some(new FlattenBatch())
        case Some(true) => Some(new FlattenBatch())
        case Some(false) => None
      }
    )

    NamespaceInjections.pipelineModel(Array(mb, inputParser, client, outputParser, dropCols, flatten)
                                        .flatten)

  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    makePipeline(dataset.schema).transform(dataset.toDF())
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    makePipeline(schema).transformSchema(schema)
  }

}
