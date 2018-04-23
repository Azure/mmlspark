// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk.train

import org.apache.spark.ml._
import org.apache.spark.sql.types._

object TypeMapping {
  val mmlTypes = Seq[DataType](
    BooleanType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    StringType)
}

trait TypeConversion {
  def conversionMap: Map[DataType, DataType]
}

// This is the root of Featurize stage 1: type mapping, where stage 2 is assembly tactic
// There is one problem I cannot resolve: Can type mapping be dependent on assembly strategy?
// If so, the lines are a bit blurry and it's likely not going to a 2 stage pipeline, but
// rather a single estimator configurable (params) by an ITypeMapping and IAssemblyStrategy, to use
// C# terminology for clarity.
abstract class SingleTypeReducer(target: DataType) extends Transformer with TypeConversion {
  private lazy val map = TypeMapping.mmlTypes.map(t => t -> target).toMap
  def conversionMap: Map[DataType, DataType] = map
}

abstract class VectorAssembler()

class SingleVectorAssembler() extends VectorAssembler
class MultiVectorAssembler extends VectorAssembler
object MultiVectorAssembler {
  def create(groups: Map[String, Seq[Int]]): Unit = {}
}
