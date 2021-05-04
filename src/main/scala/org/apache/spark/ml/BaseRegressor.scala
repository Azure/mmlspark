// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml

import org.apache.spark.ml.regression.{RegressionModel, Regressor}

/** Temporary hack to expose private Regressor class in SparkML as a developer API
  */
abstract class BaseRegressor[F, R <: Regressor[F, R, M], M <: RegressionModel[F, M]] extends Regressor[F, R, M]
