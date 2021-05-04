// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.featurize.text

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class PageSplitterSpec extends TransformerFuzzing[PageSplitter] {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "words words  words     wornssaa ehewjkdiw weijnsikjn xnh",
    "s s  s   s     s           s",
    "hsjbhjhnskjhndwjnbvckjbnwkjwenbvfkjhbnwevkjhbnwejhkbnvjkhnbndjkbnd",
    "hsjbhjhnskjhndwjnbvckjbnwkjwenbvfkjhbnwevkjhbnwejhkbnvjkhnbndjkbnd " +
      "190872340870271091309831097813097130i3u709781",
    "",
    null //scalastyle:ignore null
  ).toDF("text")

  lazy val t: PageSplitter = new PageSplitter()
    .setInputCol("text")
    .setMaximumPageLength(20)
    .setMinimumPageLength(10)
    .setOutputCol("pages")

  test("Basic usage") {
    val resultList = t.transform(df).collect().toList
    resultList.dropRight(1).foreach { row =>
      val pages = row.getSeq[String](1).toList
      val text = row.getString(0)
      assert(pages.mkString("") === text)
      assert(pages.forall(_.length <= 20))
      assert(pages.dropRight(1).forall(_.length >= 10))
    }
  }

  override def testObjects(): Seq[TestObject[PageSplitter]] =
    List(new TestObject(t, df))

  override def reader: MLReadable[_] = PageSplitter

}
