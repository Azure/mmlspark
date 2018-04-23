// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.downloader

import java.io.InputStream
import java.net.URI

import org.apache.commons.codec.digest.DigestUtils
import spray.json._

import scala.collection.JavaConversions._

private[spark] object NamingConventions {

  def canonicalModelFilename(name: String, dataset: String): String =
    s"${name}_$dataset.model"

  def canonicalModelFilename(model: ModelSchema): String =
    s"${model.name}_${model.dataset}.model"

}

/** Abstract representation of a schema for an item that can be held in a repository
  *
  * @param uri  location of the underlying file (local, HDFS, or HTTP)
  * @param hash sha256 hash of the underlying file
  * @param size size in bytes of the underlying file
  */
abstract class Schema(val uri: URI, val hash: String, val size: Long) {

  def updateURI(newURI: URI): this.type

  def assertMatchingHash(bytes: InputStream): Unit = {
    val newHash = DigestUtils.sha256Hex(bytes)
    if (newHash != hash) {
      throw new IllegalArgumentException(s"downloaded hash: $newHash does not match given hash: $hash")
    }
  }

}

/** Class representing the schema of a CNTK model
  * @param name      name of the model architecture
  * @param dataset   dataset the model was trained on
  * @param modelType type of problem the model is suited for eg: (image, text, sound, sentiment etc)
  * @param uri       location of the underlying file (local, HDFS, or HTTP)
  * @param hash      sha256 hash of the underlying file
  * @param size      size in bytes of the underlying file
  * @param inputNode the node which represents the input
  * @param numLayers the number of layers of the model
  * @param layerNames the names nodes that represent layers in the network
  */
case class ModelSchema(name: String,
                       dataset: String,
                       modelType: String,
                       override val uri: URI,
                       override val hash: String,
                       override val size: Long,
                       inputNode: Int,
                       numLayers: Int,
                       layerNames: Array[String])
  extends Schema(uri, hash, size) {

  def this(name: String, dataset: String, modelType: String,
           uri: URI, hash: String, size: Long, inputNode: Int, numLayers: Int,
           layerNames: java.util.ArrayList[String]) = {
    this(name, dataset, modelType, uri, hash, size,
      inputNode, numLayers, layerNames.toList.toArray)
  }

  override def updateURI(newURI: URI): this.type = this.copy(uri = newURI).asInstanceOf[this.type]

}

private[spark] object SchemaJsonProtocol extends DefaultJsonProtocol {

  implicit object URIJsonFormat extends JsonFormat[URI] {
    def write(u: URI): JsValue = {
      JsString(u.toString)
    }

    def read(value: JsValue): URI = new URI(value.asInstanceOf[JsString].value)
  }

  implicit val modelSchemaFormat: RootJsonFormat[ModelSchema] =
    jsonFormat(ModelSchema.apply,
      "name", "dataset", "modelType", "uri", "hash", "size", "inputNode", "numLayers", "layerNames")

}
