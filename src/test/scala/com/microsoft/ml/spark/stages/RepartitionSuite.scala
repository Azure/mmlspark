// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class RepartitionSuite extends TestBase with TransformerFuzzing[Repartition] {

  import spark.implicits._

  lazy val input: DataFrame = Seq(
    (0, "guitars", "drums"),
    (1, "piano", "trumpet"),
    (2, "bass", "cymbals"),
    (3, "guitars", "drums"),
    (4, "piano", "trumpet"),
    (5, "bass", "cymbals"),
    (6, "guitars", "drums"),
    (7, "piano", "trumpet"),
    (8, "bass", "cymbals"),
    (9, "guitars", "drums"),
    (10, "piano", "trumpet"),
    (11, "bass", "cymbals")
  ).toDF("numbers", "words", "more")

  test("Work for several values of n") {

    def test(n: Int): Unit = {
      val result = new Repartition()
        .setN(n)
        .transform(input)
      assert(result.rdd.getNumPartitions == n)
      ()
    }
    List(1, 2, 3, 10).foreach(test)

  }

  test("Should allow a user to set the partitions specifically in pipeline transform") {
    val r = new Repartition().setN(1)
    val pipe = new Pipeline().setStages(Array(r))
    val fitPipe = pipe.fit(input)
    assert(fitPipe.transform(input).rdd.getNumPartitions==1)
    assert(fitPipe.transform(input, ParamMap(r.n->5)).rdd.getNumPartitions ==5)
  }

  def testObjects(): Seq[TestObject[Repartition]] = List(new TestObject(
    new Repartition().setN(1), input))

  def reader: MLReadable[_] = Repartition
}
