// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.basic

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasInputCols, HasOutputCol, Wrappable}
import com.microsoft.ml.spark.core.env.InternalWrapper
import com.microsoft.ml.spark.core.serialize.params.{UDFParam, UDPyFParam}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.execution.python.UserDefinedPythonFunction
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, Dataset}

object UDFTransformer extends ComplexParamsReadable[UDFTransformer]

/** <code>UDFTransformer</code> takes as input input column, output column, and a UserDefinedFunction
  * returns a dataframe comprised of the original columns with the output column as the result of the
  * udf applied to the input column
  */
@InternalWrapper
class UDFTransformer(val uid: String) extends Transformer with Wrappable with ComplexParamsWritable
  with HasInputCol with HasInputCols with HasOutputCol {
  def this() = this(Identifiable.randomUID("UDFTransformer"))
  val udfScalaKey = "udfScala"
  val udfPythonKey = "udf"
  val invalidColumnSetError = s"If using multiple columns, only use set inputCols"

  override def setInputCol(value: String): this.type = {
    require(!isSet(inputCols), invalidColumnSetError)
    set(inputCol, value)
  }

  override def setInputCols(value: Array[String]): this.type = {
    require(!isSet(inputCol), invalidColumnSetError)
    set(inputCols, value)
  }

  val udfScala = new UDFParam(this, udfScalaKey, "User Defined Function to be applied to the DF input col")
  val udfPython = new UDPyFParam(this, udfPythonKey,
    "User Defined Python Function to be applied to the DF input col")
  val udfParams = Seq(udfScala, udfPython)

  /** @group getParam */
  def getUDF: UserDefinedFunction = $(udfScala)

  /** @group getParam */
  def getUDPyF: UserDefinedPythonFunction = $(udfPython)

  /** @group setParam */
  def setUDF(value: UserDefinedFunction): this.type = {
    udfParams.filter(isSet).foreach(clear)
    set(udfScalaKey, value)
  }

  /** @group setParam */
  def setUDF(value: UserDefinedPythonFunction): this.type = {
    udfParams.filter(isSet).foreach(clear)
    set(udfPythonKey, value)
  }

  def applyUDF(col: Column): Column =  {
    if (isSet(udfScala)) getUDF(col)
    else getUDPyF(col)
  }

  def applyUDFOnCols(cols: Column*): Column =  {
    if (isSet(udfScala)) getUDF(cols: _*)
    else getUDPyF(cols: _*)
  }

  def getDataType: DataType =  {
    if (isSet(udfScala)) getUDF.dataType
    else getUDPyF.dataType
  }

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from applying the udf to the inputted dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    if (isSet(inputCol)) {
      dataset.withColumn(getOutputCol, applyUDF(dataset.col(getInputCol)))
    } else {
      dataset.withColumn(getOutputCol, applyUDFOnCols(getInputCols.map(dataset.col(_)): _*))
    }
  }

  def validateAndTransformSchema(schema: StructType): StructType = {
    if (isSet(inputCol)) schema(getInputCol) else schema(Set(getInputCols: _*))
    schema.add(StructField(getOutputCol, getDataType))
  }

  def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  def copy(extra: ParamMap): UDFTransformer = defaultCopy(extra)

}

