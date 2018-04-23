// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class TimerSuite extends EstimatorFuzzing[Timer] {

  val df: DataFrame = session
    .createDataFrame(Seq((0, "Hi I"),
                         (1, "I wish for snow today"),
                         (2, "we Cant go to the park, because of the snow!"),
                         (3, "")))
    .toDF("label", "sentence")

  test("Work with transformers and estimators") {

    val tok = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("tokens")

    val df2 = new Timer().setStage(tok).fit(df).transform(df)

    val df3 = new HashingTF().setInputCol("tokens").setOutputCol("hash").transform(df2)

    val idf = new IDF().setInputCol("hash").setOutputCol("idf")

    val df4 = new Timer().setStage(idf).fit(df3).transform(df3)

  }

  test("should work within pipelines") {
    val tok = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("tokens")
    val ttok = new Timer().setStage(tok)
    val hash = new HashingTF().setInputCol("tokens").setOutputCol("hash")
    val idf  = new IDF().setInputCol("hash").setOutputCol("idf")
    val tidf = new Timer().setStage(idf)
    val pipe = new Pipeline().setStages(Array(ttok, hash, tidf))
    pipe.fit(df).transform(df)
  }

  test("should be able to turn off timing") {
    val tok = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("tokens")
    val ttok = new Timer().setStage(tok)
    val hash = new HashingTF().setInputCol("tokens").setOutputCol("hash")
    val idf  = new IDF().setInputCol("hash").setOutputCol("idf")
    val tidf = new Timer().setStage(idf)
    val pipe = new Pipeline().setStages(Array(ttok, hash, tidf))
    val model = pipe.fit(df)

    println("Transforming")
    println(model.stages(0).params.foreach(println(_)))
    model.stages(0).asInstanceOf[TimerModel].setDisable(true)
    model.stages(2).asInstanceOf[TimerModel].setDisable(true)

    println("here")
    println(model.stages(0).getParam("disableMaterialization"))

    model.stages(0).params.foreach(p =>println("foo: " + p.toString))

    model.transform(df)
  }

  val reader: MLReadable[_] = Timer
  val modelReader: MLReadable[_] = TimerModel

  override def testObjects(): Seq[TestObject[Timer]] = Seq(new TestObject[Timer]({
    val tok = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("tokens")
    new Timer().setStage(tok)
  }, df))
}
