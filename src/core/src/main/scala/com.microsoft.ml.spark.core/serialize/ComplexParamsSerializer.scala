// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.util

import com.microsoft.ml.spark.core.serialize.ComplexParam
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.Utils
import org.json4s.JsonDSL._
import org.json4s.{JObject, _}
import org.json4s.jackson.JsonMethods._

trait ComplexParamsWritable extends MLWritable {
  self: Params =>

  override def write: MLWriter = new ComplexParamsWriter(this)
}

trait ComplexParamsReadable[T] extends MLReadable[T] {

  override def read: MLReader[T] = new ComplexParamsReader[T]
}

/** Default [[MLWriter]] implementation for transformers and estimators that contain basic
  * (json4s-serializable) params and no data. This will not handle more complex params or types with
  * data (e.g., models with coefficients).
  *
  * @param instance object to save
  */
private[ml] class ComplexParamsWriter(instance: Params) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    val complexParamLocs = ComplexParamsWriter.getComplexParamLocations(instance, path)
    val complexParamJson = ComplexParamsWriter.getComplexMetadata(complexParamLocs)
    ComplexParamsWriter.saveMetadata(instance, path, sc, complexParamJson)
    ComplexParamsWriter.saveComplexParams(path, complexParamLocs, shouldOverwrite)
  }
}

private[ml] object ComplexParamsWriter {

  def getComplexMetadata(complexParamLocs: Map[ParamPair[_], Path]): Option[JObject] = {
    if (complexParamLocs.nonEmpty) {
      Some(JObject("complexParamLocs" -> complexParamLocs.map {
        case (ParamPair(p, _), path) => JObject(p.name -> render(path.toString))
      }.reduce(_ ~ _)))
    } else {
      None
    }
  }

  def getComplexParamLocations(instance: Params, path: String): Map[ParamPair[_], Path] = {
    val complexParams = instance.extractParamMap().toSeq.filter {
      case ParamPair(_: ComplexParam[_], _) => true
      case _ => false
    }
    complexParams.map {
      pp => pp -> new Path("complexParams", pp.param.name)
    }.toMap
  }

  def saveComplexParams(basePath: String, complexParamLocs: Map[ParamPair[_], Path],
                        shouldOverwrite: Boolean): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    complexParamLocs.foreach {
      case (ParamPair(p: ComplexParam[Any], v), loc) =>
        p.save(v, spark, new Path(basePath, loc), shouldOverwrite)
    }
  }

  /** Saves metadata + Params to: path + "/metadata"
    *  - class
    *  - timestamp
    *  - sparkVersion
    *  - uid
    *  - paramMap
    *  - (optionally, extra metadata)
    *
    * @param extraMetadata Extra metadata to be saved at same level as uid, paramMap, etc.
    * @param paramMap      If given, this is saved in the "paramMap" field.
    *                      Otherwise, all [[org.apache.spark.ml.param.Param]]s are encoded using
    *                      [[org.apache.spark.ml.param.Param.jsonEncode()]].
    */
  def saveMetadata(instance: Params,
                   path: String,
                   sc: SparkContext,
                   extraMetadata: Option[JObject] = None,
                   paramMap: Option[JValue] = None): Unit = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(instance, sc, extraMetadata, paramMap)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  /** Helper for [[saveMetadata()]] which extracts the JSON to save.
    * This is useful for ensemble models which need to save metadata for many sub-models.
    *
    * @see [[saveMetadata()]] for details on what this includes.
    */
  def getMetadataToSave(instance: Params,
                        sc: SparkContext,
                        extraMetadata: Option[JObject] = None,
                        paramMap: Option[JValue] = None): String = {
    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.extractParamMap().toSeq
      .filter(!_.param.isInstanceOf[ComplexParam[_]]).asInstanceOf[Seq[ParamPair[Any]]]
    val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) =>
        basicMetadata ~ jObject
      case None =>
        basicMetadata
    }
    val metadataJson: String = compact(render(metadata))
    metadataJson
  }

}

/** Default [[MLReader]] implementation for transformers and estimators that contain basic
  * (json4s-serializable) params and no data. This will not handle more complex params or types with
  * data (e.g., models with coefficients).
  *
  * @tparam T ML instance type
  *           TODO: Consider adding check for correct class name.
  */
private[ml] class ComplexParamsReader[T] extends MLReader[T] {

  override def load(path: String): T = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc)
    val cls = Utils.classForName(metadata.className)
    val instance =
      cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
    DefaultParamsReader.getAndSetParams(instance, metadata)
    ComplexParamsReader.getAndSetComplexParams(instance, metadata, path)
    instance.asInstanceOf[T]
  }
}

private[ml] object ComplexParamsReader {

  def getAndSetComplexParams(instance: Params, metadata: Metadata, basePath: String): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    implicit val format = DefaultFormats
    val complexParamLocs = (metadata.metadata \ "complexParamLocs") match {
      case JNothing =>
        Map[String, Path]()
      case j =>
        j.extract[Map[String, String]].map { case (name, path) => (name, new Path(basePath, path)) }
    }

    instance.params.foreach {
      case p: ComplexParam[_] =>
        complexParamLocs.get(p.name) match {
          case Some(loc) =>
            instance.set(p, p.load(spark,loc))
          case None =>
        }
      case _ =>
    }
  }

}
