// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.nio.file.Paths
import java.nio.FloatBuffer
import java.util

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.{ArrayParam, Param, ParamMap}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{FloatType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._
import scala.collection.mutable

object TFModel extends ComplexParamsReadable[TFModel]

@InternalWrapper
class TFModel(override val uid: String) extends Model[TFModel] with ComplexParamsWritable
  with HasInputCol with HasOutputCol with Wrappable{

  def this() = this(Identifiable.randomUID("TFModel"))

  /** Path to the tensorflow model
    * @group param
    */
  val modelPath: Param[String] = new Param(this, "modelPath", "Path to the tensorflow model")

  /** @group setParam */
  def setModelPath(value: String): this.type = set(modelPath, value)

  /** @group getParam */
  def getModelPath: String                   = $(modelPath)

  /** Name of protobuf file
    * @group param
    */
  val graphFile: Param[String] = new Param(this, "graphFile", "Name of protobuf file")

  /** @group setParam */
  def setGraphFile(value: String): this.type = set(graphFile, value)

  /** @group getParam */
  def getGraphFile: String                   = $(graphFile)

  /** Name of text file containing labels
    * @group param
    */
  val labelFile: Param[String] = new Param(this, "labelFile", "Name of text file containing labels")

  /** @group setParam */
  def setLabelFile(value: String): this.type =
    set(labelFile, value)

  /** @group getParam */
  def getLabelFile: String                   = $(labelFile)

  /** preprocessing info: width, height, mean and scale
    * @group param
    */
  val expectedDims: ArrayParam = new ArrayParam(this, "expectedDims", "Preprocessing info in array")

  /** @group setParam */
  def setExpectedDims(value: Array[Double]): this.type = set(expectedDims, value)

  /** @group getParam */
  def getExpectedDims: Array[Double]                   = {
    /*$(expectedDims) match {
      case arr:Array[Double] => arr
      case arr => throw new IllegalArgumentException(s"HERE ${arr.getClass}")
    }*/
    Array(224,224,117,1)
  }

  /** Name of input node in TF graph - set to 'input' by default
    * @group param
    */
  val inputTensorName: Param[String] = new Param(this, "inputTensorName", "Name of input node in TF graph")

  /** @group setParam */
  def setInputTensorName(value: String): this.type = set(inputTensorName, value)

  /** @group getParam */
  def getInputTensorName: String                   = $(inputTensorName)

  /** Name of output node in TF graph - set to 'output' by default
    * @group param
    */
  val outputTensorName: Param[String] = new Param(this, "outputTensorName", "Name of output node in TF graph")

  /** @group setParam */
  def setOutputTensorName(value: String): this.type = set(outputTensorName, value)

  /** @group getParam */
  def getOutputTensorName: String                   = $(outputTensorName)

  /** Type of the transformation. 2 options: image OR other:
    *   - image provides preprocessing and postprocessing methods increasing the productivity of the user
    *   - other is more general, expects a DF of arrays of floats and returns a VectorType
    * @group param
    */
  val transformationType: Param[String] = new Param(this, "transformationType",
                                                    "Type of the transformation the model is going to perform")

  /** @group setParam */
  def setTransformationType(value: String): this.type = set(transformationType, value)

  /** @group getParam */
  def getTransformationType: String                   = $(transformationType)

  //Set default values for some Params that don't have to be specified.
//  setDefault(expectedDims -> Array[F(224f,224f,128f,255f)) //for image type transformation only
  setDefault(inputTensorName -> "input")
  setDefault(outputTensorName -> "output")
  setDefault(transformationType -> "image")

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  def transformSchema(schema: StructType): StructType = schema.add(getOutputCol, StringType)

  def flatten[A](arr: Array[A]): Array[Float] =
    arr.flatMap {
      case s: Float => Array(s)
      case a: Array[_] => flatten(a)
    }

  /** Evaluate the model (on images)
    * @param dataset the dataset to featurize
    * @return featurized dataset - StringType
    */
  def transform(dataset: Dataset[_]): DataFrame = {
    val toReturn: DataFrame = if (getTransformationType == "image") {

      //Start of Set-up for evaluation code
      val executer = new TFModelExecutor()
      val labels = executer.readAllLinesOrExit(Paths.get(getModelPath, getLabelFile))
      val graph = executer.readAllBytesOrExit(Paths.get(getModelPath, getGraphFile))
      //End of set-up for evaluation code

      val df = dataset.toDF()

      val encoder = RowEncoder(new StructType().add(StructField("Output labels", StringType)))
      val output = df.mapPartitions { it =>
        it.map { r =>
          val rawData = r.toSeq.toArray
          val rawDataDouble = rawData(0).asInstanceOf[Array[Byte]]
          val height = rawData(1).asInstanceOf[Int]
          val width = rawData(2).asInstanceOf[Int]
          val typeForEncode = rawData(3).asInstanceOf[Int]
          val prediction: String = executer.evaluateForSpark(graph, labels, rawDataDouble, height, width,
            typeForEncode, getExpectedDims.map(_.toFloat), getInputTensorName, getOutputTensorName)
          Row.fromSeq(Array(prediction).toSeq)
        }
      }(encoder)
      output
      }
      else {
        //Start of Set-up for evaluation code
        val executer = new TFModelExecutor()
        val graph = executer.readAllBytesOrExit(Paths.get(getModelPath, getGraphFile))
        //End of set-up for evaluation code
        val df = dataset.toDF()
        val encoder = RowEncoder(new StructType().add(StructField("Output vector", VectorType)))
//        df.select("input").rdd.map(row => row.get(0).asInstanceOf[mutable.WrappedArray[Float]].array)
        val output = df.mapPartitions { it =>
          it.map { r =>

            val rawData: mutable.WrappedArray[Float] = r.toSeq.get(0).asInstanceOf[mutable.WrappedArray[Float]]
            val y  = rawData.toArray
            val x = y.map(x => x.asInstanceOf[Float])
            val buffered_data: FloatBuffer = FloatBuffer.wrap(x)

            val prediction: Array[Float] = executer.evaluateForSparkAny(graph, buffered_data,
              getExpectedDims.map(_.toFloat), getInputTensorName, getOutputTensorName)
            Row(Vectors.dense(prediction.map(i => i.toDouble)))
          }
        }(encoder)
        output
      }
    toReturn
  }
}
