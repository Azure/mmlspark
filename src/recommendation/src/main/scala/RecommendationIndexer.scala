// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param.{Param, ParamMap, Params, TransformerParam}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{NumericType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

class RecommendationIndexer(override val uid: String)
  extends Estimator[RecommendationIndexerModel] with RecommendationIndexerBase {

  def this() = this(Identifiable.randomUID("RecommendationIndexer"))

  override def fit(dataset: Dataset[_]): RecommendationIndexerModel = {
    val userIndexModel: StringIndexerModel = new StringIndexer()
      .setInputCol(getUserInputCol)
      .setOutputCol(getUserOutputCol)
      .setHandleInvalid(getHandleInvalid)
      .fit(dataset)

    val itemIndexModel: StringIndexerModel = new StringIndexer()
      .setInputCol(getItemInputCol)
      .setOutputCol(getItemOutputCol)
      .setHandleInvalid(getHandleInvalid)
      .fit(dataset)

    new RecommendationIndexerModel(uid)
      .setParent(this)
      .setUserIndexModel(userIndexModel)
      .setItemIndexModel(itemIndexModel)
      .setUserInputCol(getUserInputCol)
      .setUserOutputCol(getUserOutputCol)
      .setItemInputCol(getItemInputCol)
      .setItemOutputCol(getItemOutputCol)
  }

  override def copy(extra: ParamMap): Estimator[RecommendationIndexerModel] = defaultCopy(extra)

}

object RecommendationIndexer extends ComplexParamsReadable[RecommendationIndexer]

class RecommendationIndexerModel(override val uid: String) extends Model[RecommendationIndexerModel] with
  RecommendationIndexerBase with Wrappable {
  override def copy(extra: ParamMap): RecommendationIndexerModel = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    getItemIndexModel.transform(getUserIndexModel.transform(dataset))
  }

  def this() = this(Identifiable.randomUID("RecommendationIndexerModel"))

  val userIndexModel = new TransformerParam(this, "userIndexModel", "userIndexModel", {
    case _: StringIndexerModel => true
    case _ => false
  })

  def setUserIndexModel(m: StringIndexerModel): this.type = set(userIndexModel, m)

  def getUserIndexModel: StringIndexerModel = $(userIndexModel).asInstanceOf[StringIndexerModel]

  val itemIndexModel = new TransformerParam(this, "itemIndexModel", "itemIndexModel", {
    case _: StringIndexerModel => true
    case _ => false
  })

  def setItemIndexModel(m: StringIndexerModel): this.type = set(itemIndexModel, m)

  def getItemIndexModel: StringIndexerModel = $(itemIndexModel).asInstanceOf[StringIndexerModel]

  def getUserIndex: Map[Int, String] = {
    getUserIndexModel
      .labels
      .zipWithIndex
      .map(t => (t._2, t._1))
      .toMap
  }

  def getItemIndex: Map[Int, String] = {
    getItemIndexModel
      .labels
      .zipWithIndex
      .map(t => (t._2, t._1))
      .toMap
  }

  def recoverUser(): UserDefinedFunction = udf((userID: Integer) => getUserIndex.getOrElse[String](userID, "-1"))

  def recoverItem(): UserDefinedFunction = udf((itemID: Integer) => getItemIndex.getOrElse[String](itemID, "-1"))

}

object RecommendationIndexerModel extends ComplexParamsReadable[RecommendationIndexerModel]

trait RecommendationIndexerBase extends Params with ComplexParamsWritable {

  /** @group setParam */
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  /** @group getParam */
  def getHandleInvalid: String = $(handleInvalid)

  val handleInvalid = new Param[String](this, "handleInvalid", "Handle Invalid")

  /** @group setParam */
  def setUserInputCol(value: String): this.type = set(userInputCol, value)

  /** @group getParam */
  def getUserInputCol: String = $(userInputCol)

  val userInputCol = new Param[String](this, "userInputCol", "User Input Col")

  /** @group setParam */
  def setUserOutputCol(value: String): this.type = set(userOutputCol, value)

  /** @group getParam */
  def getUserOutputCol: String = $(userOutputCol)

  val userOutputCol = new Param[String](this, "userOutputCol", "User Output Col")

  /** @group setParam */
  def setItemInputCol(value: String): this.type = set(itemInputCol, value)

  /** @group getParam */
  def getItemInputCol: String = $(itemInputCol)

  val itemInputCol = new Param[String](this, "itemInputCol", "Item Input Col")

  /** @group setParam */
  def setItemOutputCol(value: String): this.type = set(itemOutputCol, value)

  /** @group getParam */
  def getItemOutputCol: String = $(itemOutputCol)

  val itemOutputCol = new Param[String](this, "itemOutputCol", "Item Output Col")

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group getParam */
  def getRatingCol: String = $(ratingCol)

  val ratingCol = new Param[String](this, "ratingCol", "Rating Col")

  def transformSchema(schema: StructType): StructType = {
    val userInputColName = getUserInputCol
    val userInputDataType = schema(userInputColName).dataType
    require(userInputDataType == StringType || userInputDataType.isInstanceOf[NumericType],
      s"The input column $userInputColName must be either string type or numeric type, " +
        s"but got $userInputDataType .")
    val itemInputColName = getItemInputCol
    val itemInputDataType = schema(userInputColName).dataType
    require(itemInputDataType == StringType || itemInputDataType.isInstanceOf[NumericType],
      s"The input column $itemInputColName must be either string type or numeric type, " +
        s"but got $itemInputDataType .")

    val inputFields = schema.fields
    val userOutputColName = getUserOutputCol
    require(inputFields.forall(_.name != userOutputColName),
      s"Output column $userOutputColName already exists.")
    val userAttr = NominalAttribute.defaultAttr.withName(getUserOutputCol)
    val itemOutputColName = getItemOutputCol
    require(inputFields.forall(_.name != itemOutputColName),
      s"Output column $itemOutputColName already exists.")
    val itemAttr = NominalAttribute.defaultAttr.withName(getItemOutputCol)

    val outputFields = inputFields :+ userAttr.toStructField() :+ itemAttr.toStructField()
    StructType(outputFields)

    setDefault(handleInvalid -> "error")
  }
}
