// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.io.IOException
import java.nio.{DoubleBuffer, FloatBuffer}
import java.nio.charset.Charset
import java.nio.file.{Files, Path, Paths}
import java.util
import java.util.{Arrays, List}

import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.types.UInt8
import StreamUtilities.using

/**
  * Master Class responsible for loading tensorflow .pb graphs and doing inference on inputs
  */
class TFModelExecutor extends Serializable {

  /**
    * Evaluates a tensorflow graph on the provided input and prints the best matched label with confidence
    * TODO: change that to inputs
    *       @param args: array containing the path to the protobuf file (first arg),
    *                  and the paths to the inputs to be eval
    *       @param expectedShape:the expected shape of the images (Height, Width, Mean, Scale)
    *       Height and Width will try to be deduced from graph definition, mean and scale default value are
    *       128f and 1f (i.e no scaling and simple substraction of half max rgb value as normalization)
    *       @param inputTensorName: name of the input Tensor. "input" by default
    *       @param outputTensorName: name of the output Tensor. "output" by default
    *
    */
  def main(args: Array[String],
           expectedShape: Array[Float] = Array[Float](128,128,128f,1f),
           inputTensorName: String = "input",
           outputTensorName: String = "output"): Unit = {

    //Magic number explanation --> proxy test that user provided all arguments needed
    if (args.length != 4) {
      System.out.println(System.err)
      System.exit(1)
    }

    val modelDir = args(0)
    val imageFile = args(1)
    val graphDef = readAllBytesOrExit(Paths.get(modelDir, args(2)))
    val labels = readAllLinesOrExit(Paths.get(modelDir, args(3)))
    val imageBytes = readAllBytesOrExit(Paths.get(imageFile))

    val prediction = evaluate(graphDef, labels, imageBytes, expectedShape, inputTensorName, outputTensorName)
    println(prediction)
  }

  /**
    * Method used for classifying an input (image for now) using a TF graph [for Spark]
    * @param graphDef bytes from the protobuf file describing the model
    * @param labels lists of labels/classes
    * @param imageBytes bytes from the image file
    * @param height [for openCV encoding back to image format] height of input image
    * @param width  [for openCV encoding back to image format] width of input image
    * @param typeForEncode [for openCV encoding back to image format] type
    * @param expectedShape contains preprocessing info: expected width, height of input to graph
    *                      AND normalization info (mean to substract from values - 0 if none- and scale to divide by -
    *                      1 if none)
    * @param inputTensorName Name of input node from graph. If unknown and default "input" does not work,
    *                        visualize graph using tensorboard to find answer. If model is provided by MMLSpark,
    *                        "input" should work.
    * @param outputTensorName Name of output node from graph. If unknown and default "output" does not work,
    *                        visualize graph using tensorboard to find answer. If model is provided by MMLSpark,
    *                        "output" should work.
    * @return String of the best fitting label with confidence
    */
  def evaluateForSpark(graphDef: Array[Byte], labels: List[String], imageBytes: Array[Byte],
                       height: Int, width: Int, typeForEncode: Int,
               expectedShape: Array[Float] = Array[Float](128,128,128f,1f),
               inputTensorName: String = "input",
               outputTensorName: String = "output"): String = {
    //Check if graph contains info on expected shape of input
      val g = new Graph
      g.importGraphDef(graphDef)
      val shapeToUse: Array[Float] = expectedShape

      //for now, will need to change this later to make it more flexible
      val inputShape = g.operation(inputTensorName).output(0).shape()
      if(inputShape.numDimensions() != -1){
        var i = 0
        for (i <- 1 to 2){
          //second and third indices only because we only want H and W!!
          shapeToUse(i-1) = inputShape.size(i).asInstanceOf[Float] //returns size of ith dimension
        }
      }

      val transformer = new InputToTensor("image_inception", shapeToUse)
      val image = transformer.constructAndExecuteGraphToNormalizeImage(imageBytes, height, width, typeForEncode)

      try {
        val labelProbabilities = executeInceptionGraph(g, image, inputTensorName, outputTensorName)
        val bestLabelIdx = labelProbabilities.indexOf(labelProbabilities.max)
        val bestPredictedLabel = labels.get(bestLabelIdx)
        val highestProb = labelProbabilities(bestLabelIdx) * 100f
        val prediction = s"${bestPredictedLabel} (${highestProb}% likely)"
        System.out.println(prediction)
        prediction
      } finally if (image != null) image.close()
  }

  def evaluateForSparkAny(graphDef: Array[Byte], inputArray: FloatBuffer,
                       expectedShape: Array[Float] = Array[Float](1,224,224,3),
                       inputTensorName: String = "input",
                       outputTensorName: String = "output"): Array[Float] = {
      val g = new Graph
      g.importGraphDef(graphDef)
      val shape: Array[Long] = expectedShape.map(e => e.toLong)
      val inputTensor = Tensor.create(shape, inputArray)
//      val inputTensorInt = inputTensor.
      executeInceptionGraph(g, inputTensor, inputTensorName, outputTensorName)
  }

  /**
    * Method used for classifying an input (image for now) using a TF graph [when not in Spark]
    * @param graphDef bytes from the protobuf file describing the model
    * @param labels lists of labels/classes
    * @param imageBytes bytes from the image file
    * @param expectedShape contains preprocessing info: expected width, height of input to graph
    *                      AND normalization info (mean to substract from values - 0 if none- and scale to divide by -
    *                      1 if none)
    * @param inputTensorName Name of input node from graph. If unknown and default "input" does not work,
    *                        visualize graph using tensorboard to find answer. If model is provided by MMLSpark,
    *                        "input" should work.
    * @param outputTensorName Name of output node from graph. If unknown and default "output" does not work,
    *                        visualize graph using tensorboard to find answer. If model is provided by MMLSpark,
    *                        "output" should work.
    * @return String of the best fitting label with confidence
    */
  def evaluate(graphDef: Array[Byte], labels: List[String], imageBytes: Array[Byte],
               expectedShape: Array[Float] = Array[Float](128,128,128f,1f),
               inputTensorName: String = "input",
               outputTensorName: String = "output"): String = {
    this.evaluateForSpark(graphDef, labels, imageBytes, -1,-1, -1, expectedShape, inputTensorName, outputTensorName)
  }

  /**
    * Inputs tensor to TF graph and outputs probabilities. Private for now but will probably change to public for
    * generalization to other types of inputs than images
    * @param g Graph object holding the graph definition and info
    * @param image input Tensor
    * @param inputTensorName
    * @param outputTensorName
    * @return Array of probabilities of match between input and classes/labels
    */
  private def executeInceptionGraph(g: Graph,
                                    image: Tensor[java.lang.Float],
                                    inputTensorName: String = "input",
                                    outputTensorName: String = "output"): Array[Float] = {
    val s = new Session(g)
    val result = s.runner.feed(inputTensorName, image).fetch(outputTensorName).
      run.get(0).expect(classOf[java.lang.Float])
    try {
      val rshape = result.shape
      if (result.numDimensions != 2 || rshape(0) != 1)
        throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is " +
          "the number of labels, instead it produced one with shape %s", util.Arrays.toString(rshape)))
      val nlabels = rshape(1).toInt
      result.copyTo(Array.ofDim[Float](1, nlabels))(0)
    } finally {
      if (s != null) s.//      g.importGraphDef(graphDef)
close()
      if (result != null) result.close()
    }
  }

  def readAllBytesOrExit(path: Path): Array[Byte] = {
    try
      return Files.readAllBytes(path)
    catch {
      case e: IOException =>
        System.err.println("Failed to read [" + path + "]: " + e.getMessage)
        System.exit(1)
    }
    null
  }

  def readAllLinesOrExit(path: Path): List[String] = {
    try
      return Files.readAllLines(path, Charset.forName("UTF-8"))
    catch {
      case e: IOException =>
        System.err.println("Failed to read [" + path + "]: " + e.getMessage)
        System.exit(0)
    }
    null
  }
}
