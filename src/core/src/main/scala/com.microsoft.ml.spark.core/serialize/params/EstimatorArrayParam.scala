// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize.params

import com.microsoft.ml.spark.core.serialize.ComplexParam
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{NamespaceInjections, ParamPair, Params}

import scala.collection.JavaConverters._

/** Param for Array of Models. */
class EstimatorArrayParam(parent: Params, name: String, doc: String, isValid: Array[Estimator[_]] => Boolean)
    extends ComplexParam[Array[Estimator[_]]](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, NamespaceInjections.alwaysTrue)

  /** Creates a param pair with the given value (for Java). */
  def w(value: java.util.List[Estimator[_]]): ParamPair[Array[Estimator[_]]] = w(value.asScala.toArray)

}
