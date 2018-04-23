// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.lightgbm._
import com.microsoft.ml.spark.lightgbm.LightGBMUtils.{intToPtr, newDoubleArray, newIntArray}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

/** Represents a LightGBM Booster learner
  * @param model The string serialized representation of the learner
  */
class LightGBMBooster(val model: String) extends Serializable {
  /** Transient variable containing local machine's pointer to native booster
    */
  @transient
  var boosterPtr: SWIGTYPE_p_void = null

  def numClasses(): Int = {
    if (boosterPtr == null) {
      LightGBMUtils.initializeNativeLibrary()
      boosterPtr = getModel()
    }
    val numClasses = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterGetNumClasses(boosterPtr, numClasses),
      "Booster GetNumClasses")
    lightgbmlib.intp_value(numClasses)
  }

  def score(features: Vector, raw: Boolean): Double = {
    // Reload booster on each node
    if (boosterPtr == null) {
      LightGBMUtils.initializeNativeLibrary()
      boosterPtr = getModel()
    }
    val kind =
      if (raw) lightgbmlibConstants.C_API_PREDICT_RAW_SCORE
      else lightgbmlibConstants.C_API_PREDICT_NORMAL
    features match {
      case dense: DenseVector => predictForMat(dense.toArray, kind)
      case sparse: SparseVector => predictForCSR(sparse, kind)
    }
  }

  protected def predictForCSR(sparseVector: SparseVector, kind: Int): Double = {
    var values: Option[(SWIGTYPE_p_void, SWIGTYPE_p_double)] = None
    var indexes: Option[(SWIGTYPE_p_int32_t, SWIGTYPE_p_int)] = None
    var indptrNative: Option[(SWIGTYPE_p_int32_t, SWIGTYPE_p_int)] = None
    var scoredDataOutPtr: Option[SWIGTYPE_p_double] = None
    try {
      val numRows = 1
      val valuesArray = sparseVector.values
      values = Some(newDoubleArray(valuesArray))
      val indexesArray = sparseVector.indices
      indexes = Some(newIntArray(indexesArray))
      val indptr = new Array[Int](numRows + 1)
      indptr(1) = sparseVector.numNonzeros
      indptrNative = Some(newIntArray(indptr))
      val numCols = sparseVector.size

      val datasetParams = "max_bin=255 is_pre_partition=True"
      val dataInt32bitType = lightgbmlibConstants.C_API_DTYPE_INT32
      val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64
      scoredDataOutPtr = Some(lightgbmlib.new_doubleArray(numRows))

      val scoredDataLengthLongPtr = lightgbmlib.new_longp()
      lightgbmlib.longp_assign(scoredDataLengthLongPtr, numRows)
      val scoredDataLength_int64_tPtr = lightgbmlib.long_to_int64_t_ptr(scoredDataLengthLongPtr)

      LightGBMUtils.validate(CSRUtils.LGBM_BoosterPredictForCSR(boosterPtr, indptrNative.get._1, dataInt32bitType,
        indexes.get._1, values.get._1, data64bitType, intToPtr(indptr.length), intToPtr(numRows + 1), intToPtr(numCols),
        kind, -1, datasetParams,
        scoredDataLength_int64_tPtr, scoredDataOutPtr.get), "Booster Predict")
      lightgbmlib.doubleArray_getitem(scoredDataOutPtr.get, 0)
    } finally {
      // Delete the input row
      if (values.isDefined) lightgbmlib.delete_doubleArray(values.get._2)
      if (indexes.isDefined) lightgbmlib.delete_intArray(indexes.get._2)
      if (indptrNative.isDefined) lightgbmlib.delete_intArray(indptrNative.get._2)
      // Delete the output scored row
      if (scoredDataOutPtr.isDefined) lightgbmlib.delete_doubleArray(scoredDataOutPtr.get)
    }
  }

  protected def predictForMat(row: Array[Double], kind: Int): Double = {
    var data: Option[(SWIGTYPE_p_void, SWIGTYPE_p_double)] = None
    var scoredDataOutPtr: Option[SWIGTYPE_p_double] = None
    try {
      val numRows = 1
      data = Some(LightGBMUtils.generateData(numRows, Array(row)))
      val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64

      val numRowsIntPtr = lightgbmlib.new_intp()
      lightgbmlib.intp_assign(numRowsIntPtr, numRows)
      val numRows_int32_tPtr = lightgbmlib.int_to_int32_t_ptr(numRowsIntPtr)
      val scoredDataLengthLongPtr = lightgbmlib.new_longp()
      lightgbmlib.longp_assign(scoredDataLengthLongPtr, numRows)
      val scoredDataLength_int64_tPtr = lightgbmlib.long_to_int64_t_ptr(scoredDataLengthLongPtr)

      val numCols = row.length
      val isRowMajor = 1
      val numColsIntPtr = lightgbmlib.new_intp()
      lightgbmlib.intp_assign(numColsIntPtr, numCols)
      val numCols_int32_tPtr = lightgbmlib.int_to_int32_t_ptr(numColsIntPtr)
      scoredDataOutPtr = Some(lightgbmlib.new_doubleArray(numRows))
      val datasetParams = "max_bin=255"
      LightGBMUtils.validate(
        lightgbmlib.LGBM_BoosterPredictForMat(
          boosterPtr, data.get._1, data64bitType, numRows_int32_tPtr,
          numCols_int32_tPtr, isRowMajor, kind,
          -1, datasetParams, scoredDataLength_int64_tPtr, scoredDataOutPtr.get),
        "Booster Predict")
      lightgbmlib.doubleArray_getitem(scoredDataOutPtr.get, 0)
    } finally {
      // Delete the input row
      if (data.isDefined) lightgbmlib.delete_doubleArray(data.get._2)
      // Delete the output scored row
      if (scoredDataOutPtr.isDefined) lightgbmlib.delete_doubleArray(scoredDataOutPtr.get)
    }
  }

  protected def getModel(): SWIGTYPE_p_void = {
    val boosterOutPtr = lightgbmlib.voidpp_handle()
    val numItersOut = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterLoadModelFromString(model, numItersOut, boosterOutPtr),
      "Booster LoadFromString")
    lightgbmlib.voidpp_value(boosterOutPtr)
  }

  def mergeModels(newModel: LightGBMBooster): LightGBMBooster = {
    // Merges models to first handle
    if (model != null) {
      LightGBMUtils.validate(
        lightgbmlib.LGBM_BoosterMerge(getModel(), newModel.getModel()),
        "Booster merge")
    }
    new LightGBMBooster(model)
  }
}
