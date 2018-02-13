// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.types.UInt8

/**
  * Helper class to build a graph for preprocessing images and other inputs. Heavily inspired by the example code
  * for the java API for tensorflow.
  * @param graph Graph object to build upon
  */
class TensorflowGraphBuilder(graph: Graph) {

    var g: Graph = graph

    def div(x: Output[java.lang.Float], y: Output[java.lang.Float]): Output[java.lang.Float] = binaryOp("Div", x, y)

    def sub[T](x: Output[T], y: Output[T]): Output[T] = binaryOp("Sub", x, y)

    def resizeBilinear[T](images: Output[T], size: Output[Integer]): Output[java.lang.Float] =
      binaryOp3("ResizeBilinear", images, size)

    def expandDims[T](input: Output[T], dim: Output[Integer]): Output[T] = binaryOp3("ExpandDims", input, dim)

    def cast[T, U](value: Output[T], `type`: Class[U]): Output[U] = {
      val dtype: DataType = DataType.fromClass(`type`)
      g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build.output[U](0)
    }

    def decodeJpeg(contents: Output[String], channels: Long): Output[UInt8] =
      g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents).setAttr("channels", channels).build.output[UInt8](0)

    def constant[T](name: String, value: Any, `type`: Class[T]): Output[T] = {
      val t: Tensor[T] = Tensor.create[T](value, `type`)
      try
        g.opBuilder("Const", name).setAttr("dtype", DataType.fromClass(`type`)).setAttr("value", t).build.output[T](0)
      finally if (t != null) t.close()
    }

    def constant(name: String, value: Array[Byte]): Output[String] = this.constant(name, value, classOf[String])

    def constant(name: String, value: Int): Output[Integer] = this.constant(name, value, classOf[Integer])

    def constant(name: String, value: Array[Int]): Output[Integer] = this.constant(name, value, classOf[Integer])

    def constant(name: String, value: Array[Float]): Output[java.lang.Float] =
      this.constant(name, value, classOf[java.lang.Float])

    def constant(name: String, value: Float): Output[java.lang.Float] =
      this.constant(name, value, classOf[java.lang.Float])

    private def binaryOp[T](`type`: String, in1: Output[T], in2: Output[T]): Output[T] =
      g.opBuilder(`type`, `type`).addInput(in1).addInput(in2).build.output[T](0)

    private def binaryOp3[T, U, V](`type`: String, in1: Output[U], in2: Output[V]): Output[T] =
      g.opBuilder(`type`, `type`).addInput(in1).addInput(in2).build.output[T](0)
}
