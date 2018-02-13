// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.nio.file.Paths

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.{TensorFlow => tf}
//import com.microsoft.ml.spark.TFModelExecutioner

class TensorflowSuite extends TestBase {
//
//  test("foo"){
//    println("here")
//  }
//
//  test("cleaner tensorflow hello test"){
//    val g = new Graph()
//    val value = "Hello from " + tf.version
//    val t = Tensor.create(value.getBytes("UTF-8"))
//    g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType).setAttr("value", t).build
//    val s = new Session(g)
//    val output = s.runner.fetch("MyConst").run.get(0)
//    System.out.println(new String(output.bytesValue, "UTF-8"))
//  }

  test("Inception model executed in scala (Java version tested in TensorflowTest.java"){
    val executer = new TFModelExecutor()
    val modelPath = "/home/houssam/externship/mmlspark/src/tensorflow-model/src/test/LabelImage_data/inception5h"
    val graphFile = "tensorflow_inception_graph.pb"
    val labelFile = "imagenet_comp_graph_label_strings.txt"
    val applePath = "/home/houssam/externship/mmlspark/src/tensorflow-model/src/test/LabelImage_data/RedApple.jpg"
    val pineapplePath = "/home/houssam/externship/mmlspark/src/tensorflow-model/" +
      "src/test/LabelImage_data/olpineapple.jpeg"
    //Only the first arg is required if the input shape is provided by the graph protobuf file
    // and the input placeholder is called "input"
    executer.main(Array[String](modelPath, applePath, graphFile, labelFile), Array[Float](224f,224f,117f,1f))
    executer.main(Array[String](modelPath, pineapplePath, graphFile, labelFile), Array[Float](224f,224f,117f,1f))

  }

  test("Another model: inceptionv3 trained on imageNet"){
    val inceptionv3 = "/home/houssam/externship/mmlspark/src/tensorflow-model/src/test/LabelImage_data/inceptionv3"
    val graphv3 = "inception_v3_2016_08_28_frozen.pb"
    val labelsv3 = "imagenet_slim_labels.txt"
    val pineapplePath = "/home/houssam/externship/mmlspark/src/tensorflow-model/" +
      "src/test/LabelImage_data/olpineapple.jpeg"
    val jackfruit = "/home/houssam/externship/mmlspark/src/tensorflow-model/src/test/LabelImage_data/jackfruit.jpg"
    val executer = new TFModelExecutor()

    executer.main(Array[String](inceptionv3, pineapplePath, graphv3, labelsv3),
                  Array[Float](0f,0f,128f,255f),
                  outputTensorName = "InceptionV3/Predictions/Reshape_1")
    executer.main(Array[String](inceptionv3, jackfruit, graphv3, labelsv3),
                  Array[Float](0f,0f,128f,255f),
                  outputTensorName = "InceptionV3/Predictions/Reshape_1")
  }

  test("Street view model test, trained by Abishkar"){
    val multiDigitModelPath = "/home/houssam/externship/mmlspark/src/tensorflow-model/" +
      "src/test/LabelImage_data/streetview"
    val graphName = "meter_reading_new.pb"
    val labelsDigits = "labels.txt"
    val imageTestPath = "/home/houssam/externship/mmlspark/src/tensorflow-model/" +
      "src/test/LabelImage_data/streetview/695.jpeg"
    val executer = new TFModelExecutor()

    executer.main(Array[String](multiDigitModelPath, imageTestPath, graphName, labelsDigits),
                  Array[Float](32f,32f,0f,255f),
                  outputTensorName = "output_node0",
                  inputTensorName = "input_1")
//    val graphDef = executer.readAllBytesOrExit(Paths.get(multiDigitModelPath, graphName))
//    val g = new Graph
//    g.importGraphDef(graphDef)
//
//    println("successful so far")
//
//    val shapeToUse : Array[Float] = expectedShape
//
//    //for now, will need to change this later to make it more flexible
//    val inputShape = g.operation("input").output(0).shape()
//    if(inputShape.numDimensions() != -1){
//      var i = 0
//      for (i <- 1 to 2){
//        //second and third indices only because we only want H and W!!
//
//        shapeToUse(i-1) = inputShape.size(i).asInstanceOf[Float] //returns size of ith dimension
//      }
//    }

  }

}
