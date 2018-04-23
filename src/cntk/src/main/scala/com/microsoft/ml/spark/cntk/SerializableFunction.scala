// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.CNTK

import java.io._
import java.util.UUID._

import com.microsoft.ml.spark.core.env.StreamUtilities.using
import org.apache.commons.io.FileUtils.{forceDelete, getTempDirectoryPath, writeByteArrayToFile}
import org.apache.commons.io.IOUtils

import scala.language.implicitConversions

object CNTKExtensions {
  implicit def toSerializable(in: Function): SerializableFunction = new SerializableFunction(in)

  implicit def fromSerializable(in: SerializableFunction): Function = in.fvar
}

object CNTKUtils {

  def loadModelFromBytes(bytes: Array[Byte],
                         device: DeviceDescriptor =
                         DeviceDescriptor.useDefaultDevice): Function = {
    try {
      Function.load(bytes, device)
    } catch {
      case _: IllegalArgumentException =>
        // println("WARNING: model is a legacy model," +
        //           " consider using a more recent model for faster loading")
        import java.util.UUID._
        val modelFile = new File(s"$getTempDirectoryPath/$randomUUID.model")
        writeByteArrayToFile(modelFile, bytes)
        val model = try {
          Function.load(modelFile.getPath, device)
        } finally forceDelete(modelFile)
        model
    }
  }
}

object SerializableFunction {

  import CNTKExtensions._

  def load(filepath: String): SerializableFunction = Function.load(filepath)

  def load(buffer: Array[Byte], device: DeviceDescriptor): SerializableFunction =
    Function.load(buffer, device)

  def load(filepath: String, device: DeviceDescriptor): SerializableFunction =
    Function.load(filepath, device)

  def loadModelFromBytes(bytes: Array[Byte],
                         device: DeviceDescriptor = DeviceDescriptor.useDefaultDevice()): SerializableFunction =
    CNTKUtils.loadModelFromBytes(bytes, device)
}

class SerializableFunction(f: Function) extends Serializable {
  var fvar: Function = f

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    val modelFile = new File(s"$getTempDirectoryPath/$randomUUID.model")
    fvar.save(modelFile.toString)
    val bytes = try {
      using(new FileInputStream(modelFile)){ fis =>
        IOUtils.toByteArray(fis)
      }.get
    } finally {
      forceDelete(modelFile)
    }
    out.writeObject(bytes)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    val bytes = in.readObject().asInstanceOf[Array[Byte]]
    val device = DeviceDescriptor.useDefaultDevice()
    fvar = CNTKUtils.loadModelFromBytes(bytes, device)
  }

}
