// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize.params

import org.apache.spark.ml.param.{NamespaceInjections, Param, ParamPair, Params}
import spray.json._

import scala.collection.JavaConverters._
import scala.collection.immutable.Map
import scala.collection.mutable

/** Param for Map of String to Seq of String. */
class MapParam[K, V](parent: Params, name: String, doc: String, isValid: Map[K, V] => Boolean)
                    (implicit val fk: JsonFormat[K], implicit val fv: JsonFormat[V])
  extends Param[Map[K, V]](parent, name, doc, isValid) with CollectionFormats {

  def this(parent: Params, name: String, doc: String)(implicit fk: JsonFormat[K], fv: JsonFormat[V]) =
    this(parent, name, doc, NamespaceInjections.alwaysTrue)

  /** Creates a param pair with the given value (for Java). */
  def w(value: java.util.HashMap[K, V]): ParamPair[Map[K, V]] = {
    val mutMap = mutable.Map[K, V]()
    for (key <- value.keySet().asScala) {
      mutMap(key) = value.get(key)
    }
    w(mutMap.toMap)
  }

  override def jsonEncode(value: Map[K, V]): String = {
    value.toJson.prettyPrint
  }

  override def jsonDecode(json: String): Map[K, V] = {
    json.parseJson.convertTo[Map[K, V]]
  }

}
