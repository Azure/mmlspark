// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive

import java.net.URL
import com.microsoft.ml.spark.core.utils.AsyncUtils
import com.microsoft.ml.spark.logging.BasicLogging
import com.microsoft.ml.spark.stages.Lambda
import org.apache.commons.io.IOUtils
import org.apache.http.client.methods.{HttpGet, HttpRequestBase}
import org.apache.http.entity.AbstractHttpEntity
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.ComplexParamsReadable
import org.apache.spark.ml.param.ServiceParam
import org.apache.spark.ml.util._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions.{col, explode}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row}

import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Future}
import spray.json.DefaultJsonProtocol._

object BingImageSearch extends ComplexParamsReadable[BingImageSearch] with Serializable {

  def getUrlTransformer(imageCol: String, urlCol: String): Lambda = {
    val fromRow = BingImagesResponse.makeFromRowConverter
    Lambda(_
      .withColumn(urlCol, explode(
        UDFUtils.oldUdf({ rOpt: Row =>
          Option(rOpt).map(r => fromRow(r).value.map(_.contentUrl))
        }, ArrayType(StringType))(col(imageCol))))
      .select(urlCol)
    )
  }

  def downloadFromUrls(pathCol: String,
                       bytesCol: String,
                       concurrency: Int,
                       timeout: Int
                      ): Lambda = {
    Lambda({ df =>
      val outputSchema = df.schema.add(bytesCol, BinaryType, nullable = true)
      val encoder = RowEncoder(outputSchema)
      df.toDF().mapPartitions { rows =>
        val futures = rows.map { row: Row =>
          (Future {
            IOUtils.toByteArray(new URL(row.getAs[String](pathCol)))
          }(ExecutionContext.global), row)
        }
        AsyncUtils.bufferedAwaitSafeWithContext(
          futures, concurrency, Duration.fromNanos(timeout * 1e6.toLong))(ExecutionContext.global)
          .map {
            case (bytesOpt, row) => Row.merge(row, Row(bytesOpt.getOrElse(null))) //scalastyle:ignore null
          }
      }(encoder)
    })

  }
}

class BingImageSearch(override val uid: String)
  extends CognitiveServicesBase(uid)
  with HasCognitiveServiceInput with HasInternalJsonOutputParser with BasicLogging {
  logClass()

  override protected lazy val pyInternalWrapper = true

  def this() = this(Identifiable.randomUID("BingImageSearch"))

  setDefault(url -> "https://api.cognitive.microsoft.com/bing/v7.0/images/search")

  override def prepareMethod(): HttpRequestBase = new HttpGet()

  override def responseDataType: DataType = BingImagesResponse.schema

  val q = new ServiceParam[String](this, "q",
    "The user's search query string",
    isRequired = true, isURLParam = true)
  def setQuery(v: String): this.type = setScalarParam(q, v)
  def setQueryCol(v: String): this.type = setVectorParam(q, v)
  def setQ(v: String): this.type = setScalarParam(q, v)
  def setQCol(v: String): this.type = setVectorParam(q, v)

  val count = new ServiceParam[Int](this, "count",
    "The number of image results to return in the response." +
      " The actual number delivered may be less than requested.",
    isURLParam = true)
  def setCount(v: Int): this.type = setScalarParam(count, v)
  def setCountCol(v: String): this.type = setVectorParam(count, v)

  val offset = new ServiceParam[Int](this, "offset",
    "The zero-based offset that indicates the" +
      " number of image results to skip before returning results",
    isURLParam = true)
  def setOffsetCol(v: String): this.type = setVectorParam(offset, v)
  def setOffset(v: Int): this.type = setScalarParam(offset, v)

  val mkt = new ServiceParam[String](this, "mkt",
    "The market where the results come from." +
      " Typically, this is the country where the user " +
      "is making the request from; however, it could be a different" +
      " country if the user is not located in a country where Bing " +
      "delivers results. The market must be in the form -." +
      " For example, en-US. Full list of supported markets: " +
      "es-AR,en-AU,de-AT,nl-BE,fr-BE,pt-BR,en-CA," +
      "fr-CA,es-CL,da-DK,fi-FI,fr-FR,de-DE,zh-HK," +
      "en-IN,en-ID,en-IE,it-IT,ja-JP,ko-KR,en-MY," +
      "es-MX,nl-NL,en-NZ,no-NO,zh-CN,pl-PL,pt-PT," +
      "en-PH,ru-RU,ar-SA,en-ZA,es-ES,sv-SE,fr-CH," +
      "de-CH,zh-TW,tr-TR,en-GB,en-US,es-US",
    isURLParam = true)
  def setMarket(v: String): this.type = setScalarParam(mkt, v)
  def setMarketCol(v: String): this.type = setVectorParam(mkt, v)
  def setMkt(v: String): this.type = setScalarParam(mkt, v)
  def setMktCol(v: String): this.type = setVectorParam(mkt, v)

  val imageType = new ServiceParam[String](this, "imageType",
    "Filter images by the following image types:" +
      "AnimatedGif: return animated gif images" +
      "AnimatedGifHttps: return animated gif images that are from an https address" +
      "Clipart: Return only clip art images" +
      "Line: Return only line drawings" +
      "Photo: Return only photographs " +
      "(excluding line drawings, animated Gifs, and clip art)" +
      "Shopping: Return only images that contain items where" +
      " Bing knows of a merchant that is selling the items. " +
      "This option is valid in the en-US market only. " +
      "Transparent: Return only images with a transparent background.",
    isURLParam = true)
  def setImageType(v: String): this.type = setScalarParam(imageType, v)
  def setImageTypeCol(v: String): this.type = setVectorParam(imageType, v)

  val aspect = new ServiceParam[String](this, "aspect",
    "Filter images by the following aspect ratios: " +
      "Square: Return images with standard aspect ratio" +
      "Wide: Return images with wide screen aspect ratio" +
      "Tall: Return images with tall aspect ratio" +
      "All: Do not filter by aspect. Specifying this value " +
      "is the same as not specifying the aspect parameter.",
    isURLParam = true)
  def setAspect(v: String): this.type = setScalarParam(aspect, v)
  def setAspectCol(v: String): this.type = setVectorParam(aspect, v)

  val color = new ServiceParam[String](this, "color",
    "Filter images by the following color options:" +
      "ColorOnly: Return color images" +
      "Monochrome: Return black and white images" +
      "Return images with one of the following dominant colors:" +
      "Black,Blue,Brown,Gray,Green,Orange,Pink,Purple,Red,Teal,White,Yellow",
    isURLParam = true)
  def setColor(v: String): this.type = setScalarParam(color, v)
  def setColorCol(v: String): this.type = setVectorParam(color, v)

  val freshness = new ServiceParam[String](this, "freshness",
    "Filter images by the following discovery options:" +
      "Day: Return images discovered by Bing within the last 24 hours" +
      "Week: Return images discovered by Bing within the last 7 days" +
      "Month: Return images discovered by Bing within the last 30 days" +
      "Year: Return images discovered within the last year" +
      "2017-06-15..2018-06-15: Return images discovered within" +
      " the specified range of dates",
      isURLParam = true)
  def setFreshness(v: String): this.type = setScalarParam(freshness, v)
  def setFreshnessCol(v: String): this.type = setVectorParam(freshness, v)

  val height = new ServiceParam[Int](this, "height",
    "Filter images that have the specified height, in pixels." +
      "You may use this filter with the size filter to return small" +
      " images that have a height of 150 pixels.",
    isURLParam = true)
  def setHeight(v: Int): this.type = setScalarParam(height, v)
  def setHeightCol(v: String): this.type = setVectorParam(height, v)

  val width = new ServiceParam[Int](this, "width",
    "Filter images that have the specified width, in pixels." +
      "You may use this filter with the size filter to return small" +
      " images that have a width of 150 pixels.",
    isURLParam = true)
  def setWidth(v: Int): this.type = setScalarParam(width, v)
  def setWidthCol(v: String): this.type = setVectorParam(width, v)

  val size = new ServiceParam[String](this, "size",
    "Filter images by the following sizes:" +
      "Small: Return images that are less than 200x200 pixels" +
      "Medium: Return images that are greater than or equal to 200x200 " +
      "pixels but less than 500x500 pixels" +
      "Large: Return images that are 500x500 pixels or larger" +
      "Wallpaper: Return wallpaper images." +
      "AllDo not filter by size. Specifying this value" +
      " is the same as not specifying the size parameter." +
      "You may use this parameter along with the height or width parameters. " +
      "For example, you may use height and size to request " +
      "small images that are 150 pixels tall.",
    isURLParam = true)
  def setSize(v: String): this.type = setScalarParam(size, v)
  def setSizeCol(v: String): this.type = setVectorParam(size, v)

  val imageContent = new ServiceParam[String](this, "imageContent",
    "Filter images by the following content types:" +
      "Face: Return images that show only a person's face" +
      "Portrait: Return images that show only a person's head and shoulders",
    isURLParam = true)
  def setImageContent(v: String): this.type = setScalarParam(imageContent, v)
  def setImageContentCol(v: String): this.type = setVectorParam(imageContent, v)

  val license = new ServiceParam[String](this, "license",
    "Filter images by the following license types:" +
      "Any: Return images that are under any license type. " +
      "The response doesn't include images that do not specify a " +
      "license or the license is unknown." +
      "Public: Return images where the creator has waived their " +
      "exclusive rights, to the fullest extent allowed by law." +
      "Share: Return images that may be shared with others. " +
      "Changing or editing the image might not be allowed." +
      " Also, modifying, sharing, and using the image for commercial " +
      "purposes might not be allowed. Typically, this " +
      "option returns the most images." +
      "ShareCommercially: Return images that may be shared " +
      "with others for personal or commercial purposes. " +
      "Changing or editing the image might not be allowed." +
      "Modify: Return images that may be modified, shared, and used." +
      " Changing or editing the image might not be allowed." +
      " Modifying, sharing, and using the image for commercial" +
      " purposes might not be allowed. " +
      "ModifyCommercially: Return images that may be modified, shared," +
      " and used for personal or commercial purposes." +
      " Typically, this option returns the fewest images." +
      "All: Do not filter by license type. Specifying this value " +
      "is the same as not specifying the license parameter. " +
      "For more information about these license types, " +
      "see Filter Images By License Type.",
    isURLParam = true)
  def setLicense(v: String): this.type = setScalarParam(license, v)
  def setLicenseCol(v: String): this.type = setVectorParam(license, v)

  val maxFileSize = new ServiceParam[Int](this, "maxFileSize",
    "Filter images that are less than or equal to the specified file size." +
      "The maximum file size that you may specify is 520,192 bytes. " +
      "If you specify a larger value, the API uses 520,192. " +
      "It is possible that the response may include images that are slightly " +
      "larger than the specified maximum." +
      "You may specify this filter and minFileSize to filter images " +
      "within a range of file sizes.",
    isURLParam = true)
  def setMaxFileSize(v: Int): this.type = setScalarParam(maxFileSize, v)
  def setMaxFileSizeCol(v: String): this.type = setVectorParam(maxFileSize, v)

  val maxHeight = new ServiceParam[Int](this, "maxHeight",
    "Filter images that have a height that is less than" +
      " or equal to the specified height. Specify the height in pixels." +
      "You may specify this filter and minHeight to filter images " +
      "within a range of heights. This filter and the " +
      "height filter are mutually exclusive.",
    isURLParam = true)
  def setMaxHeight(v: Int): this.type = setScalarParam(maxHeight, v)
  def setMaxHeightCol(v: String): this.type = setVectorParam(maxHeight, v)

  val maxWidth = new ServiceParam[Int](this, "maxWidth",
    "Filter images that have a width that is less than or equal " +
      "to the specified width. Specify the width in pixels." +
      "You may specify this filter and maxWidth to filter images " +
      "within a range of widths. This filter and the width " +
      "filter are mutually exclusive.",
    isURLParam = true)
  def setMaxWidth(v: Int): this.type = setScalarParam(maxWidth, v)
  def setMaxWidthCol(v: String): this.type = setVectorParam(maxWidth, v)

  val minFileSize = new ServiceParam[Int](this, "minFileSize",
    "Filter images that are greater than or equal to the specified file size. " +
      "The maximum file size that you may specify is 520,192 bytes." +
      " If you specify a larger value, the API uses 520,192. " +
      "It is possible that the response may include images that " +
      "are slightly smaller than the specified minimum. " +
      "You may specify this filter and maxFileSize to filter images " +
      "within a range of file sizes.",
    isURLParam = true)
  def setMinFileSize(v: Int): this.type = setScalarParam(minFileSize, v)
  def setMinFileSizeCol(v: String): this.type = setVectorParam(minFileSize, v)

  val minHeight = new ServiceParam[Int](this, "minHeight",
    "Filter images that have a height that is greater than or equal" +
      " to the specified height. Specify the height in pixels." +
      "You may specify this filter and maxHeight to filter images " +
      "within a range of heights. This filter and the height " +
      "filter are mutually exclusive.",
    isURLParam = true)
  def setMinHeight(v: Int): this.type = setScalarParam(minHeight, v)
  def setMinHeightCol(v: String): this.type = setVectorParam(minHeight, v)

  val minWidth = new ServiceParam[Int](this, "minWidth",
    "Filter images that have a width that is greater than or equal" +
      " to the specified width. Specify the width in pixels. " +
      "You may specify this filter and maxWidth to filter images " +
      "within a range of widths. This filter and the width " +
      "filter are mutually exclusive.",
    isURLParam = true)
  def setMinWidth(v: Int): this.type = setScalarParam(minWidth, v)
  def setMinWidthCol(v: String): this.type = setVectorParam(minWidth, v)

  override protected def prepareEntity: Row => Option[AbstractHttpEntity] = {_ => None}
}
