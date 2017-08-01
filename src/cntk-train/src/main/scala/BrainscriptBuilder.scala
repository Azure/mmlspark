// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import scala.collection.mutable.ListBuffer

case class InputShape(dim: Int, form: String)

case class InputData(format: String, path: String, shapes: Map[String, InputShape])

case class BrainScriptConfig(name: String, text: Seq[String])

/**
  * Utility methods for manipulating the BrainScript and overrides configs output to disk.
  */
class BrainScriptBuilder {

  var modelName = "ModelOut"

  var inData: Option[InputData] = None

  var rootDir: String = ""
  var outDir: String = ""
  var weightPrecision: String = "float"

  var commands = ListBuffer[String]("trainNetwork")
  var testModel = false

  def setInputFile(path: String, format: String, shapes: Map[String, InputShape]): this.type = {
    inData = Some(InputData(format, path, shapes))
    this
  }

  def setModelName(n: String): this.type = {
    modelName = n
    this
  }

  def getModelPath: String = {
    s"""file://$getLocalModelPath"""
  }

  def getLocalModelPath: String = {
    s"""$outDir/Models/$modelName"""
  }

  def setRootDir(p: String): this.type = {
    outDir = p
    this
  }

  def setOutputRoot(p: String): this.type = {
    outDir = p
    this
  }

  private def getInputString: String = {
    val ips = inData.get.shapes
                .map { case(name, shape) => name + " = [ dim = " +
                       shape.dim.toString + " ; format = \"" + shape.form + "\" ]" }
                .mkString("; ")
    s"input = [ $ips ]"
  }

  def setCommands(c: String*): this.type = {
    this
  }

  def setTestModel(b: Boolean): this.type = {
    if (!testModel && b) {
      commands.append("testNetwork")
    }
    this
  }

  def toReaderConfig: String = {
    val ipstring = getInputString
    val loc = inData.get.path
    val form = inData.get.format match {
      case "text" => "CNTKTextFormatReader"
    }
    s"""reader = [ readerType = $form ; file = "$loc" ; $ipstring ]"""
  }

  def toOverrideConfig: Seq[String] = {
    val rootOverrides = Seq(
      s"""command = ${ commands.mkString(":") }""",
      s"precision=$weightPrecision",
      "traceLevel=1",
      "deviceId=\"auto\"",
      s"""rootDir="$rootDir" """,
      s"""outputDir="$outDir" """,
      s"""modelPath="${getLocalModelPath}" """)
    val commandReaders = commands.map(c => s"$c = [ ${toReaderConfig} ]")

    rootOverrides ++ commandReaders
  }

}
