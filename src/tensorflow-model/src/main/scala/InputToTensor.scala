// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import com.microsoft.ml.spark.StreamUtilities.using
import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs
import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.types.UInt8

import collection.JavaConverters._

/**
  * Class responsible for turning inputs (text, images, etc.) to Tensor objects
  * For now, it is going to support images only (based on the tensorflow example for the inception model)
  */

class InputToTensor(input_type: String, expectedShape: Array[Float]) {
  //Constructor
  var itype: String = input_type

  // Probably input.shape().size is what you need to play with
  // TODO: Figure out if this encoding back to jpeg format from openCV format is the right way of doing this
  // - seems convoluted
  /**
    * Method for preprocessing images. Returns a Tensor object representing the preprocessing image
    * (decoding the bytes + substracting the mean + the dividing by the scale
    * @param imageBytes image to preprocess in Array of bytes
    * @param height [For Spark purposes] height of image, to use to encode to jpeg from openCV bytecode
    * @param width [For Spark purposes] width of image, to use to encode to jpeg from openCV bytecode
    * @param typeForEncode [For Spark purposes] type of image, to use to encode to jpeg from openCV bytecode
    * @return Input Tensor to feed to the computational graph
    */
  def constructAndExecuteGraphToNormalizeImage(imageBytes: Array[Byte],
                                               height: Int = -1, width: Int = -1,
                                               typeForEncode: Int = -1): Tensor[java.lang.Float] = {
    //This if statement has been added to generalize this method to all types of inputs later - might be easier to give
    //the preprocessing responsibility to the MMLSpark User
    if(itype == "image_inception")
    {
      using(new Graph) { g =>
        val b = new TensorflowGraphBuilder(g)

        //Now we are shifting these constants to variables that are either provided or using general default values
        var expectedDim: Array[Float] = Array()
        expectedShape.length match {
          case x if x == 0 => expectedDim = expectedShape ++ Array(128f, 128f, 128f, 1f)
          case x if x == 1 => expectedDim = expectedShape ++ Array(128f, 128f, 1f)
          case x if x == 2 => expectedDim = expectedShape ++ Array(128f, 1f)
          case x if x == 3 => expectedDim = expectedShape ++ Array(1f)
          case _ => expectedDim = expectedShape
        }

        val H: Int = expectedDim(0).asInstanceOf[Int]
        val W: Int = expectedDim(1).asInstanceOf[Int]
        val mean = expectedDim(2)
        val scale = expectedDim(3)

        // Since the graph is being constructed once per execution here, we can use a constant for the
        // input image. If the graph were to be re-used for multiple input images, a placeholder would
        // have been more appropriate. TODO: to make this more efficient on partitions

        //Check if we are being passed width and height --> change opencv bytes into image bytes
        val imageToPass: Array[Byte] = if (width != -1 && height != -1 && typeForEncode != -1) {
          val mat = new MatOfByte()
          val xmat = new Mat(height, width, typeForEncode)
          xmat.put(0, 0, imageBytes)
          Imgcodecs.imencode(".jpeg", xmat, mat)
          mat.toArray
        }
        else {
          imageBytes
        }

        val input: Output[String] = b.constant("input", imageToPass)
        val output: Output[java.lang.Float] = b.div(
          b.sub(
            b.resizeBilinear(
              b.expandDims(
                b.cast(
                  b.decodeJpeg(input, 3),
                  classOf[java.lang.Float]
                ),
                b.constant("make_batch", 0)
              ),
              b.constant("size", Array[Int](H, W))
            ),
            b.constant("mean", mean)
          ),
          b.constant("scale", scale))

        using(new Session(g)) { s =>
          s.runner.fetch(output.op.name).run.get(0).expect(classOf[java.lang.Float])
        }.get
      }.get
    }
    else
    {
      val t: Tensor[java.lang.Float] = Tensor.create(1000f).expect(classOf[java.lang.Float])
      t
    }
  }

  /**
    * Method assuming the preprocessing is done already on Spark. Best used on inputs other than images, or images
    * requiring preprocessing beyond simple normalization (substraction + division). Transforms the preprocessed
    * array of float to an input Tensor ready to be fed to the TF graph
    * @param input Array[Float] representing the preprocessed input
    * @return outputTensor
    */
  def arrayToTensor(input: Array[Float]): Tensor[java.lang.Float] = {
    using(new Graph) { g =>
      val b = new TensorflowGraphBuilder(g)

      val inputHandler: Output[java.lang.Float] = b.constant("input", input)
      val outputHandler: Output[java.lang.Float] = b.cast(inputHandler, classOf[java.lang.Float])

      using(new Session(g)) { s =>
        s.runner.fetch(outputHandler.op.name).run.get(0).expect(classOf[java.lang.Float])
      }.get
    }.get
  }
}
