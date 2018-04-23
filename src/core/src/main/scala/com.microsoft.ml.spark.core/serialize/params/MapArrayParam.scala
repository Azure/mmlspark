// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize.params

import org.apache.spark.ml.param._
import spray.json.{DefaultJsonProtocol, _}

import scala.collection.JavaConverters._
import scala.collection.immutable.Map
import scala.collection.mutable

object MapArrayJsonProtocol extends DefaultJsonProtocol {

  implicit object MapJsonFormat extends JsonFormat[Map[String, Seq[String]]] {
    def write(m: Map[String, Seq[String]]): JsValue = {
      JsObject(m.mapValues {
        case v: Seq[String] => seqFormat[String].write(v)
        case default => serializationError(s"Unable to serialize $default")
      })
    }

    def read(value: JsValue): Map[String, Seq[String]] = value.asInstanceOf[JsObject].fields.map(kvp => {
      val convValue = kvp._2 match {
        case v: JsValue => seqFormat[String].read(v)
        case default => deserializationError(s"Unable to deserialize $default")
      }
      (kvp._1, convValue)
    })
  }

}

/** Param for Map of String to Seq of String. */
class MapArrayParam(parent: Params, name: String, doc: String, isValid: Map[String, Seq[String]] => Boolean)
  extends Param[Map[String, Seq[String]]](parent, name, doc, isValid) {
    import MapArrayJsonProtocol._

    def this(parent: Params, name: String, doc: String) =
      this(parent, name, doc, NamespaceInjections.alwaysTrue)

    /** Creates a param pair with the given value (for Java). */
    def w(value: java.util.HashMap[String, java.util.List[String]]): ParamPair[Map[String, Seq[String]]] = {
      val mutMap = mutable.Map[String, Seq[String]]()
      for (key <- value.keySet().asScala) {
        val list = value.get(key).asScala
        mutMap(key) = list
      }
      w(mutMap.toMap)
    }

    override def jsonEncode(value: Map[String, Seq[String]]): String = {
      val convertedMap = value.map(kvp => (kvp._1, kvp._2.toArray))
      val json = convertedMap.toJson
      json.prettyPrint
    }

    override def jsonDecode(json: String): Map[String, Seq[String]] = {
      val jsonValue = json.parseJson
      jsonValue.convertTo[Map[String, Seq[String]]]
    }

  }
