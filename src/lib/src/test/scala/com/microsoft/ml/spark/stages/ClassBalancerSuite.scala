// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class ClassBalancerSuite extends EstimatorFuzzing[ClassBalancer] {
  lazy val df: DataFrame = session
    .createDataFrame(Seq((0, 1.0, "Hi I"),
                         (1, 1.0, "I wish for snow today"),
                         (2, 2.0, "I wish for snow today"),
                         (3, 2.0, "I wish for snow today"),
                         (4, 2.0, "I wish for snow today"),
                         (5, 2.0, "I wish for snow today"),
                         (6, 0.0, "I wish for snow today"),
                         (7, 1.0, "I wish for snow today"),
                         (8, 0.0, "we Cant go to the park, because of the snow!"),
                         (9, 2.0, "")))
    .toDF("index", "label", "sentence")

  val reader: MLReadable[_] = ClassBalancer
  val modelReader: MLReadable[_] = ClassBalancerModel
  override def testObjects(): Seq[TestObject[ClassBalancer]] = Seq(new TestObject[ClassBalancer](new ClassBalancer()
    .setInputCol("label"), df))

  test("yield proper weights") {
    val model = new ClassBalancer()
      .setInputCol("label").fit(df)
    val df2 = model.transform(df)
    df2.show()

    assert(df2.collect()(8).getDouble(3) == 2.5)
    assert(df2.schema.fields.toSet == model.transformSchema(df.schema).fields.toSet)
  }
}
