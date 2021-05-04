// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive

import java.net.URI
import com.microsoft.ml.spark.core.schema.DatasetExtensions
import com.microsoft.ml.spark.io.http.SimpleHTTPTransformer
import com.microsoft.ml.spark.logging.BasicLogging
import com.microsoft.ml.spark.stages.{DropColumns, Lambda, UDFTransformer}
import org.apache.http.client.methods.{HttpPost, HttpRequestBase}
import org.apache.http.entity.{AbstractHttpEntity, StringEntity}
import org.apache.spark.injections.UDFUtils
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{BooleanParam, Param, ServiceParam}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{ComplexParamsReadable, NamespaceInjections, PipelineModel, Transformer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, _}
import org.apache.spark.sql.Row
import spray.json._
import spray.json.DefaultJsonProtocol._

abstract class TextAnalyticsBase(override val uid: String) extends CognitiveServicesBase(uid)
  with HasCognitiveServiceInput with HasInternalJsonOutputParser with HasSetLocation {

  val text = new ServiceParam[Seq[String]](this, "text", "the text in the request body", isRequired = true)

  def setTextCol(v: String): this.type = setVectorParam(text, v)

  def setText(v: Seq[String]): this.type = setScalarParam(text, v)

  def setText(v: String): this.type = setScalarParam(text, Seq(v))

  setDefault(text -> Right("text"))

  val language = new ServiceParam[Seq[String]](this, "language",
    "the language code of the text (optional for some services)")

  def setLanguageCol(v: String): this.type = setVectorParam(language, v)

  def setLanguage(v: Seq[String]): this.type = setScalarParam(language, v)

  def setLanguage(v: String): this.type = setScalarParam(language, Seq(v))

  setDefault(language -> Left(Seq("en")))

  protected def innerResponseDataType: StructType =
    responseDataType("documents").dataType match {
      case ArrayType(idt: StructType, _) => idt
      case _ =>
        throw new IllegalArgumentException("response data types should have a inner type")
    }

  override protected def responseDataType: StructType = {
    new StructType()
      .add("documents", ArrayType(innerResponseDataType))
      .add("errors", ArrayType(TAError.schema))
  }

  override protected def inputFunc(schema: StructType): Row => Option[HttpRequestBase] = {
    { row: Row =>
      if (shouldSkip(row)) {
        None
      } else if (getValue(row, text).forall(Option(_).isEmpty)) {
        None
      } else {
        import com.microsoft.ml.spark.cognitive.TAJSONFormat._
        val post = new HttpPost(getUrl)
        getValueOpt(row, subscriptionKey).foreach(post.setHeader("Ocp-Apim-Subscription-Key", _))
        post.setHeader("Content-Type", "application/json")
        val texts = getValue(row, text)

        val languages: Option[Seq[String]] = getValueOpt(row, language) match {
          case Some(Seq(lang)) => Some(Seq.fill(texts.size)(lang))
          case s => s
        }

        val documents = texts.zipWithIndex.map { case (t, i) =>
          TADocument(languages.flatMap(ls => Option(ls(i))), i.toString, Option(t).getOrElse(""))
        }
        val json = TARequest(documents).toJson.compactPrint
        post.setEntity(new StringEntity(json, "UTF-8"))
        Some(post)
      }
    }
  }

  override protected def prepareEntity: Row => Option[AbstractHttpEntity] = { _ => None }

  override protected def getInternalTransformer(schema: StructType): PipelineModel = {
    val dynamicParamColName = DatasetExtensions.findUnusedColumnName("dynamic", schema)

    def reshapeToArray(parameterName: String): Option[(Transformer, String, String)] = {
      val reshapedColName = DatasetExtensions.findUnusedColumnName(parameterName, schema)
      getVectorParamMap.get(parameterName).flatMap {
        case c if schema(c).dataType == StringType =>
          Some((Lambda(_.withColumn(reshapedColName, array(col(getVectorParam(parameterName))))),
            getVectorParam(parameterName),
            reshapedColName))
        case _ => None
      }
    }

    val reshapeCols = Seq(reshapeToArray("text"), reshapeToArray("language")).flatten

    val newColumnMapping = reshapeCols.map {
      case (_, oldCol, newCol) => (oldCol, newCol)
    }.toMap

    val columnsToGroup = getVectorParamMap.map { case (_, oldCol) =>
      val newCol = newColumnMapping.getOrElse(oldCol, oldCol)
      col(newCol).alias(oldCol)
    }.toSeq

    val innerFields = innerResponseDataType.fields.filter(_.name != "id")

    val unpackBatchUDF = UDFUtils.oldUdf({ rowOpt: Row =>
      Option(rowOpt).map { row =>
        val documents = row.getSeq[Row](1).map(doc =>
          (doc.getString(0).toInt, doc)).toMap
        val errors = row.getSeq[Row](2).map(err => (err.getString(0).toInt, err)).toMap
        val rows: Seq[Row] = (0 until (documents.size + errors.size)).map(i =>
          documents.get(i)
            .map(doc => Row.fromSeq(doc.toSeq.tail ++ Seq(None)))
            .getOrElse(Row.fromSeq(
              Seq.fill(innerFields.length)(None) ++ Seq(errors.get(i).map(_.getString(1)).orNull)))
        )
        rows
      }
    }, ArrayType(
      innerResponseDataType.fields.filter(_.name != "id").foldLeft(new StructType()) { case (st, f) =>
        st.add(f.name, f.dataType)
      }.add("error-message", StringType)
    )
    )

    val stages = reshapeCols.map(_._1).toArray ++ Array(
      Lambda(_.withColumn(
        dynamicParamColName,
        struct(columnsToGroup: _*))),
      new SimpleHTTPTransformer()
        .setInputCol(dynamicParamColName)
        .setOutputCol(getOutputCol)
        .setInputParser(getInternalInputParser(schema))
        .setOutputParser(getInternalOutputParser(schema))
        .setHandler(getHandler)
        .setConcurrency(getConcurrency)
        .setConcurrentTimeout(get(concurrentTimeout))
        .setErrorCol(getErrorCol),
      new UDFTransformer()
        .setInputCol(getOutputCol)
        .setOutputCol(getOutputCol)
        .setUDF(unpackBatchUDF),
      new DropColumns().setCols(Array(
        dynamicParamColName) ++ newColumnMapping.values.toArray.asInstanceOf[Array[String]])
    )

    NamespaceInjections.pipelineModel(stages)
  }

}

trait HasLanguage extends HasServiceParams {
  val language = new ServiceParam[String](this, "language", "the language to use", isURLParam = true)

  def setLanguageCol(v: String): this.type = setVectorParam(language, v)

  def setLanguage(v: String): this.type = setScalarParam(language, v)
}

object TextSentimentV2 extends ComplexParamsReadable[TextSentimentV2]

class TextSentimentV2(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("TextSentimentV2"))

  override def responseDataType: StructType = SentimentResponseV2.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment")

}

object LanguageDetectorV2 extends ComplexParamsReadable[LanguageDetectorV2]

class LanguageDetectorV2(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LanguageDetectorV2"))

  override def responseDataType: StructType = DetectLanguageResponseV2.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v2.0/languages")

}

object EntityDetectorV2 extends ComplexParamsReadable[EntityDetectorV2]

class EntityDetectorV2(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("EntityDetectorV2"))

  override def responseDataType: StructType = DetectEntitiesResponseV2.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v2.0/entities")

}

object NERV2 extends ComplexParamsReadable[NERV2]

class NERV2(override val uid: String) extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("NERV2"))

  override def responseDataType: StructType = NERResponseV2.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v2.1/entities")
}

object KeyPhraseExtractorV2 extends ComplexParamsReadable[KeyPhraseExtractorV2]

class KeyPhraseExtractorV2(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("KeyPhraseExtractorV2"))

  override def responseDataType: StructType = KeyPhraseResponseV2.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v2.0/keyPhrases")

}

object TextSentiment extends ComplexParamsReadable[TextSentiment]

class TextSentiment(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("TextSentiment"))

  val showStats = new ServiceParam[Boolean](this, "showStats",
    "if set to true, response will contain input and document level statistics.", isURLParam = true)

  def setShowStats(v: Boolean): this.type = setScalarParam(showStats, v)

  val modelVersion = new ServiceParam[String](this, "modelVersion",
    "This value indicates which model will be used for scoring." +
      " If a model-version is not specified, the API should default to the latest," +
      " non-preview version.", isURLParam = true)

  def setModelVersion(v: String): this.type = setScalarParam(modelVersion, v)

  override def responseDataType: StructType = SentimentResponseV3.schema

  def setLocation(v: String): this.type = {
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v3.0/sentiment")
  }

  override def inputFunc(schema: StructType): Row => Option[HttpRequestBase] = { r: Row =>
    super.inputFunc(schema)(r).map { request =>
      request.setURI(new URI(prepareUrl(r)))
      request
    }
  }

}

object KeyPhraseExtractor extends ComplexParamsReadable[KeyPhraseExtractor]

class KeyPhraseExtractor(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("KeyPhraseExtractor"))

  override def responseDataType: StructType = KeyPhraseResponseV3.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v3.0/keyPhrases")
}

object NER extends ComplexParamsReadable[NER]

class NER(override val uid: String) extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("NER"))

  override def responseDataType: StructType = NERResponseV3.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v3.0/entities/recognition/general")
}

object LanguageDetector extends ComplexParamsReadable[LanguageDetector]

class LanguageDetector(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LanguageDetector"))

  override def responseDataType: StructType = DetectLanguageResponseV3.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v3.0/languages")

}

object EntityDetector extends ComplexParamsReadable[EntityDetector]

class EntityDetector(override val uid: String)
  extends TextAnalyticsBase(uid) with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("EntityDetector"))

  override def responseDataType: StructType = DetectEntitiesResponseV3.schema

  def setLocation(v: String): this.type =
    setUrl(s"https://$v.api.cognitive.microsoft.com/text/analytics/v3.0/entities/linking")
}
