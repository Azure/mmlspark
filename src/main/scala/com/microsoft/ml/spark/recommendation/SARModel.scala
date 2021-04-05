// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.recommendation

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.DenseVector
import com.microsoft.ml.spark.core.contracts.Wrappable
import com.microsoft.ml.spark.core.env.InternalWrapper
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Model}
import org.apache.spark.ml.param.{DataFrameParam, ParamMap}
import org.apache.spark.ml.recommendation.{BaseRecommendationModel, Constants}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField => SF, _}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/** SAR Model
  *
  * @param uid The id of the module
  */
@InternalWrapper
class SARModel(override val uid: String) extends Model[SARModel]
  with BaseRecommendationModel with Wrappable with SARParams with ComplexParamsWritable {

  /** @group setParam */
  def setUserDataFrame(value: DataFrame): this.type = set(userDataFrame, value)

  val userDataFrame = new DataFrameParam(this, "userDataFrame", "Time of activity")

  /** @group getParam */
  def getUserDataFrame: DataFrame = $(userDataFrame)

  /** @group setParam */
  def setItemDataFrame(value: DataFrame): this.type = set(itemDataFrame, value)

  val itemDataFrame = new DataFrameParam(this, "itemDataFrame", "Time of activity")

  /** @group getParam */
  def getItemDataFrame: DataFrame = $(itemDataFrame)

  def this() = this(Identifiable.randomUID("SARModel"))

  /**
    * Returns top `numItems` items recommended for each user, for all users.
    *
    * @param numItems max number of recommendations for each user
    * @return a DataFrame of (userCol: Int, recommendations), where recommendations are
    *         stored as an array of (itemCol: Int, rating: Float) Rows.
    */
  def recommendForAllUsers(numItems: Int): DataFrame = {
    recommendForAll(getUserDataFrame, getItemDataFrame, getUserCol, getItemCol, numItems)
  }

  /**
    * Returns top `numItems` items recommended for each user id in the input data set. Note that if
    * there are duplicate ids in the input dataset, only one set of recommendations per unique id
    * will be returned.
    *
    * @param dataset  a Dataset containing a column of user ids. The column name must match `userCol`.
    * @param numItems max number of recommendations for each user.
    * @return a DataFrame of (userCol: Int, recommendations), where recommendations are
    *         stored as an array of (itemCol: Int, rating: Float) Rows.
    */
  def recommendForUserSubset(dataset: Dataset[_], numItems: Int): DataFrame = {
    val srcFactorSubset = getSourceFactorSubset(dataset, getUserDataFrame, getUserCol)
    recommendForAll(srcFactorSubset, getItemDataFrame, getUserCol, getItemCol, numItems)
  }

  /**
    * Returns a subset of a factor DataFrame limited to only those unique ids contained
    * in the input dataset.
    *
    * @param dataset input Dataset containing id column to user to filter factors.
    * @param factors factor DataFrame to filter.
    * @param column  column name containing the ids in the input dataset.
    * @return DataFrame containing factors only for those ids present in both the input dataset and
    *         the factor DataFrame.
    */
  private def getSourceFactorSubset(
    dataset: Dataset[_],
    factors: DataFrame,
    column: String): DataFrame = {
    factors
      .join(dataset.select(column), factors(getUserCol) === dataset(column), joinType = "left_semi")
      .select(factors(getUserCol), factors("flatList"))
  }

  /**
    * Personalized recommendations for a single user are obtained by multiplying the Item-to-Item similarity matrix
    * with a user affinity vector. The user affinity vector is simply a transposed row of the affinity matrix
    * corresponding to that user.
    *
    * @param num
    * @return
    */
  private def recommendForAll(
    srcFactors: DataFrame,
    dstFactors: DataFrame,
    srcOutputColumn: String,
    dstOutputColumn: String,
    num: Int): DataFrame = {

    def dfToRDDMatrxEntry(dataframe: DataFrame) = {
      dataframe.rdd
        .flatMap(row =>
          row.getAs[Seq[Float]](1).zipWithIndex.map { case (list, index) => Row(row.getDouble(0), index, list) })
        .map(item => MatrixEntry(item.getDouble(0).toLong, item.getInt(1).toLong, item.getFloat(2).toDouble))
    }

    val sourceMatrix = new CoordinateMatrix(dfToRDDMatrxEntry(srcFactors)).toBlockMatrix().toLocalMatrix().asML
    val destMatrix = new CoordinateMatrix(dfToRDDMatrxEntry(dstFactors)).toBlockMatrix().toLocalMatrix().asML
//    val sourceMatrix = new CoordinateMatrix(dfToRDDMatrxEntry(srcFactors)).toBlockMatrix()//.cache()
//    val destMatrix = new CoordinateMatrix(dfToRDDMatrxEntry(dstFactors)).toBlockMatrix()//.cache()

    val userToItemMatrix = sourceMatrix
      .multiply(destMatrix)
      .toIndexedRowMatrix()
      .rows.map(indexedRow => (indexedRow.index.toInt, indexedRow.vector))

    val orderAndTakeTopK = udf((vector: DenseVector) => {
      vector.toArray.zipWithIndex
        .map { case (list, index) => (index, list) }
        .sortWith(_._2 > _._2)
        .take(num)
    })

    val recommendationArrayType =
      ArrayType(new StructType(Array(SF(dstOutputColumn, IntegerType), SF(Constants.RatingCol, FloatType))))

    val product = sourceMatrix.multiply(destMatrix.toDense)
      .rowIter
      .zipWithIndex
      .map { case (vector, index) => (index, vector) }
      .toSeq

    getUserDataFrame.sparkSession.createDataFrame(product)
      .toDF(id, ratings).withColumn(recommendations, orderAndTakeTopK(col(ratings))).select(id, recommendations)
      .select(col(id).as(getUserCol), col(recommendations).cast(recommendationArrayType))
  }

  private val id = Constants.IdCol
  private val ratings = Constants.RatingCol + "s"
  private val recommendations = Constants.Recommendations

  override def copy(extra: ParamMap): SARModel = {
    val copied = new SARModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transform($(rank), $(userDataFrame), $(itemDataFrame), dataset)
  }

  override def transformSchema(schema: StructType): StructType = {
    checkNumericType(schema, $(userCol))
    checkNumericType(schema, $(itemCol))
    schema
  }

  /**
    * Check whether the given schema contains a column of the numeric data type.
    *
    * @param colName column name
    */
  private def checkNumericType(
    schema: StructType,
    colName: String,
    msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(actualDataType.isInstanceOf[NumericType], s"Column $colName must be of type " +
      s"NumericType but was actually of type $actualDataType.$message")
  }
}

object SARModel extends ComplexParamsReadable[SARModel]
