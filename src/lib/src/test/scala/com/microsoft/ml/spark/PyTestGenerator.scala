// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.text

import com.microsoft.ml.spark.core.test.pytest.PyTestGeneratorBase
import com.microsoft.ml.spark.stages.VerifyFindBestModel
import com.microsoft.ml.spark.stages.basic.{MultiColumnAdapterSpec, TimerSuite, UDFTransformerSuite}
import com.microsoft.ml.spark.stages.featurize.{VerifyTrainClassifier, VerifyTrainRegressor}

object PyTestGenerator extends PyTestGeneratorBase {
  override def exceptions: Set[Class[_]] = Set(
    classOf[VerifyFindBestModel],
    classOf[MultiColumnAdapterSpec],
    classOf[TimerSuite],
    classOf[UDFTransformerSuite],
    classOf[VerifyTrainClassifier],
    classOf[VerifyTrainRegressor]
  )

}
