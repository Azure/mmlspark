// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split1

import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.cognitive.BingImageSearch
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import com.microsoft.ml.spark.stages.Lambda
import org.apache.spark.ml.NamespaceInjections.pipelineModel
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.{DataFrame, Row}
import org.scalactic.Equality

trait HasImageSearchKey {
  lazy val imageSearchKey: String = sys.env.getOrElse("BING_IMAGE_SEARCH_KEY", Secrets.BingImageSearchKey)
}

class ImageSearchSuite extends TransformerFuzzing[BingImageSearch]
  with HasImageSearchKey {

  import spark.implicits._

  lazy val offsets: Seq[Int] = (0 to 1).map(_ * 10)
  lazy val searchQueries = List("Elephant", "African Elephant",
    "Asian Elephant", "Rhino", "rhinoceros")
  lazy val requestParameters: DataFrame = searchQueries
    .flatMap { q: String => offsets.map { o: Int => (q, o) } }
    .toDF("queries", "offsets")

  lazy val bis: BingImageSearch = new BingImageSearch()
    .setSubscriptionKey(imageSearchKey)
    .setOffsetCol("offsets")
    .setQueryCol("queries")
    .setCount(10)
    .setImageType("photo")
    .setOutputCol("images")

  lazy val getURLs: Lambda = BingImageSearch.getUrlTransformer("images", "url")

  test("Elephant Detection") {
    val pipe = pipelineModel(Array(bis, getURLs))
    val resultsDF = pipe.transform(requestParameters)
    val results = resultsDF.collect()
    assert(results.length === 100)
    results.foreach(r => assert(r.getString(0).startsWith("http")))
    val bytesDF = BingImageSearch
      .downloadFromUrls("url", "bytes", 4, 10000)
      .transform(resultsDF.limit(15))
    val numSucesses = bytesDF.collect().count(row =>
      Option(row.getAs[Array[Byte]](1)).getOrElse(Array()).length > 100)
    assert(numSucesses>3)
  }

  test("All Parameters") {
    val row = (10,"microsoft", 10, "all","black","Year",0, 520192, 0, 0,2000,2000, "Face","Photo","All", "en-US")

    val df = Seq(row).toDF()

    val staticBis = new BingImageSearch()
      .setSubscriptionKey(imageSearchKey)
      .setOffset(row._1)
      .setQuery(row._2)
      .setCount(row._3)
      .setAspect(row._4)
      .setColor(row._5)
      .setFreshness(row._6)
      .setMinFileSize(row._7)
      .setMaxFileSize(row._8)
      .setMinWidth(row._9)
      .setMinHeight(row._10)
      .setMaxWidth(row._11)
      .setMaxHeight(row._12)
      .setImageContent(row._13)
      .setImageType(row._14)
      .setLicense(row._15)
      .setMarket(row._16)
      .setOutputCol("images")

    val sdf = staticBis.transform(df).cache()
    assert(sdf.collect().head.getAs[Row]("images") != null)

    val dynamicBis = new BingImageSearch()
      .setSubscriptionKey(imageSearchKey)
      .setOffsetCol("_1")
      .setQueryCol("_2")
      .setCountCol("_3")
      .setAspectCol("_4")
      .setColorCol("_5")
      .setFreshnessCol("_6")
      .setMinFileSizeCol("_7")
      .setMaxFileSizeCol("_8")
      .setMinWidthCol("_9")
      .setMinHeightCol("_10")
      .setMaxWidthCol("_11")
      .setMaxHeightCol("_12")
      .setImageContentCol("_13")
      .setImageTypeCol("_14")
      .setLicenseCol("_15")
      .setMarketCol("_16")
      .setOutputCol("images")

    val ddf = dynamicBis.transform(df).cache()
    assert(ddf.collect().head.getAs[Row]("images") != null)
  }

  override lazy val dfEq: Equality[DataFrame] = new Equality[DataFrame] {
    def areEqual(a: DataFrame, b: Any): Boolean =
      (a.schema === b.asInstanceOf[DataFrame].schema) &&
        (a.count() === b.asInstanceOf[DataFrame].count()) // BIS is nondeterminisic
  }

  override def testObjects(): Seq[TestObject[BingImageSearch]] =
    Seq(new TestObject(bis, requestParameters))

  override def reader: MLReadable[_] = BingImageSearch

}
