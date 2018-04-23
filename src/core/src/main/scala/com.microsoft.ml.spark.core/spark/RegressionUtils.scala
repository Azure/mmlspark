// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml
import org.apache.spark.ml.regression.Regressor

object RegressionUtils {

  def isRegressor(stage: PipelineStage): Boolean = {
    stage match {
      case _: Regressor[_, _, _] => true
      case default => false
    }
  }

}
