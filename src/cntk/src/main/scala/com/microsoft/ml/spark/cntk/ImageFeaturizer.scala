// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import com.microsoft.CNTK.{SerializableFunction => CNTKFunction}
import com.microsoft.CNTK.CNTKExtensions._
import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, Wrappable}
import com.microsoft.ml.spark.core.env.InternalWrapper
import com.microsoft.ml.spark.core.schema.DatasetExtensions
import com.microsoft.ml.spark.core.serialize.params.TransformerParam
import com.microsoft.ml.spark.downloader.ModelSchema
import com.microsoft.ml.spark.opencv.{ImageTransformer, UnrollImage}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

object ImageFeaturizer extends ComplexParamsReadable[ImageFeaturizer]

/** The <code>ImageFeaturizer</code> relies on a CNTK model to do the featurization, one can set
  * this model using the <code>modelLocation</code> parameter. To map the nodes of the CNTK model
  * onto the standard "layers" structure of a feed forward neural net, one needs to supply a list of
  * node names that range from the output node, back towards the input node of the CNTK Function.
  * This list does not need to be exhaustive, and is provided to you if you use a model downloaded
  * from the <code>ModelDownloader</code>, one can find this layer list in the schema of the
  * downloaded model.
  *
  * The <code>ImageFeaturizer</code> takes an input column of images (the type returned by the
  * <code>ImageReader</code>), and automatically resizes them to fit the CMTKModel's inputs.  It
  * then feeds them through a pre-trained CNTK model.  One can truncate the model using the <code>
  * cutOutputLayers </code> parameter that determines how many layers to truncate from the output of
  * the network.  For example, layer=0 means that no layers are removed, layer=2 means that the
  * image featurizer returns the activations of the layer that is two layers from the output layer.
  *
  * @param uid the uid of the image transformer
  */
@InternalWrapper
class ImageFeaturizer(val uid: String) extends Transformer with HasInputCol with HasOutputCol
  with Wrappable with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("ImageFeaturizer"))

  // Parameters related to the inner model

  val cntkModel: TransformerParam =
    new TransformerParam(
      this,
      "cntkModel", "The internal CNTK model used in the featurizer",
      { t => t.isInstanceOf[CNTKModel] })

  /** @group setParam */
  def setCntkModel(value: CNTKModel): this.type = set(cntkModel, value)

  val emptyCntkModel = new CNTKModel()
  /** @group getParam */
  def getCntkModel: CNTKModel =
    if (isDefined(cntkModel)) $(cntkModel).asInstanceOf[CNTKModel] else emptyCntkModel

  /** @group setParam */
  def setMiniBatchSize(value: Int): this.type = set(cntkModel, getCntkModel.setMiniBatchSize(value))

  /** @group getParam */
  def getMiniBatchSize: Int = getCntkModel.getMiniBatchSize

  /** @group setParam */
  def setInputNode(value: Int): this.type = set(cntkModel, getCntkModel.setInputNode(value))

  /** @group getParam */
  def getInputNode: Int = getCntkModel.getInputNode

  def setModelLocation(spark: SparkSession, path: String): this.type = {
    set(cntkModel, getCntkModel.setModelLocation(spark, path))
  }

  def setModel(spark: SparkSession, modelSchema: ModelSchema): this.type = {
    setLayerNames(modelSchema.layerNames)
      .setInputNode(modelSchema.inputNode)
      .setModelLocation(spark, modelSchema.uri.toString)
  }

  /** @group setParam */
  def setModel(model: CNTKFunction): this.type = set(cntkModel, getCntkModel.setModel(model))

  /** @group getParam */
  def getModel: CNTKFunction = getCntkModel.getModel

  // Parameters for just the ImageFeaturizer

  /** The number of layer to cut off the endof the network; 0 leaves the network intact, 1 rmoves
    * the output layer, etc.
    *
    * @group param
    */
  val cutOutputLayers: IntParam = IntParam(this, "cutOutputLayers", "The number of layers to cut " +
    "off the end of the network, 0 leaves the network intact," +
    " 1 removes the output layer, etc", ParamValidators.gtEq(0))

  /** @group setParam */
  def setCutOutputLayers(value: Int): this.type = set(cutOutputLayers, value)

  /** @group getParam */
  def getCutOutputLayers: Int = $(cutOutputLayers)

  /** Array with valid CNTK nodes to choose from; the first entries of this array should be closer
    * to the output node.
    *
    * @group param
    */
  val layerNames: StringArrayParam = new StringArrayParam(this, "layerNames",
    "Array with valid CNTK nodes to choose from, the first entries of" +
      " this array should be closer to the output node")

  /** @group setParam */
  def setLayerNames(value: Array[String]): this.type = set(layerNames, value)

  /** @group getParam */
  def getLayerNames: Array[String] = $(layerNames)

  setDefault(cutOutputLayers -> 1, outputCol -> (uid + "_output"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = dataset.sparkSession

    val resizedCol = DatasetExtensions.findUnusedColumnName("resized")(dataset.columns.toSet)

    val cntkModel = getCntkModel
      .setOutputNodeName(getLayerNames.apply(getCutOutputLayers))
      .setInputCol(resizedCol)
      .setOutputCol(getOutputCol)

    val requiredSize = cntkModel.getModel.getArguments.get(0).getShape().getDimensions

    val prepare = new ImageTransformer()
      .setInputCol($(inputCol))
      .resize(requiredSize(0).toInt, requiredSize(1).toInt)

    val unroll = new UnrollImage()
      .setInputCol(prepare.getOutputCol)
      .setOutputCol(resizedCol)

    val resizedDF = prepare.transform(dataset)
    val unrolledDF = unroll.transform(resizedDF).drop(prepare.getOutputCol)
    val featurizedDF = cntkModel.transform(unrolledDF).drop(resizedCol)
    featurizedDF
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  /** Add the features column to the schema
    *
    * @param schema
    * @return schema with features column
    */
  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, VectorType)
  }

}
