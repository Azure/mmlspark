// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.spark.injections.UDFUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, udf}

case class Foo(a: Int, b: String, c: Seq[Bar])

object Foo extends SparkBindings[Foo]

case class Bar(a: Int, c: Seq[Byte])

object Bar extends SparkBindings[Bar]

class SparkBindingsTest2 extends TestBase {

  import spark.implicits._

  test("Test to make sure there are no strange memory leaks") {
    (1 to 40).foreach { _ =>
      val foos = (0 to 40).map(i => Tuple1(Foo(i, i.toString, Seq(Bar(i, "foo".getBytes)))))
      val converter = Foo.makeFromRowConverter
      val df = foos.toDF("foos")
        .repartition(2)
        .withColumn("mapped2",
          UDFUtils.oldUdf({ r: Row => converter(r) }, Foo.schema)(col("foos")))
      val results = df.collect().toList
      println(results.head)
    }
  }

}
