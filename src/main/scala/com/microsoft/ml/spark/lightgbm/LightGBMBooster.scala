// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.lightgbm._
import com.microsoft.ml.spark.lightgbm.LightGBMUtils.getBoosterPtrFromModelString
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.{SaveMode, SparkSession}

/** Represents a LightGBM Booster learner
  * @param model The string serialized representation of the learner
  */
@SerialVersionUID(777L)
class LightGBMBooster(val model: String) extends Serializable {
  /** Transient variable containing local machine's pointer to native booster
    */
  @transient
  var boosterPtr: SWIGTYPE_p_void = _

  def score(features: Vector, raw: Boolean, classification: Boolean): Array[Double] = {
    // Reload booster on each node
    if (boosterPtr == null) {
      LightGBMUtils.initializeNativeLibrary()
      boosterPtr = getBoosterPtrFromModelString(model)
    }
    val kind =
      if (raw) lightgbmlibConstants.C_API_PREDICT_RAW_SCORE
      else lightgbmlibConstants.C_API_PREDICT_NORMAL
    features match {
      case dense: DenseVector => predictForMat(dense.toArray, kind, classification)
      case sparse: SparseVector => predictForCSR(sparse, kind, classification)
    }
  }

  def featuresShap(features: Vector): Array[Double] = {
    // Reload booster on each node
    if (boosterPtr == null) {
      LightGBMUtils.initializeNativeLibrary()
      boosterPtr = getBoosterPtrFromModelString(model)
    }
    features match {
      case dense: DenseVector => shapForMat(dense.toArray)
      case sparse: SparseVector => shapForCSR(sparse)
    }
  }

  lazy val numClasses: Int = getNumClasses()

  lazy val numFeatures: Int = getNumFeatures()

  @transient
  var scoredDataOutPtr: SWIGTYPE_p_double = _

  @transient
  var scoredDataLengthLongPtr: SWIGTYPE_p_long_long = _

  @transient
  var shapDataOutPtr: SWIGTYPE_p_double = _

  @transient
  var shapDataLengthLongPtr: SWIGTYPE_p_long_long = _

  def ensureScoredDataCreated(): Unit = {
    if (scoredDataLengthLongPtr != null)
      return

    scoredDataOutPtr = lightgbmlib.new_doubleArray(numClasses)
    scoredDataLengthLongPtr = lightgbmlib.new_int64_tp()
    lightgbmlib.int64_tp_assign(scoredDataLengthLongPtr, 1)
  }

  def ensureShapDataCreated(): Unit = {
    if (shapDataLengthLongPtr != null)
      return

    shapDataOutPtr = lightgbmlib.new_doubleArray(numFeatures)
    shapDataLengthLongPtr = lightgbmlib.new_int64_tp()
    lightgbmlib.int64_tp_assign(shapDataLengthLongPtr, numFeatures)
  }

  override protected def finalize(): Unit = {
    if (scoredDataLengthLongPtr != null)
      lightgbmlib.delete_int64_tp(scoredDataLengthLongPtr)
    if (scoredDataOutPtr == null)
      lightgbmlib.delete_doubleArray(scoredDataOutPtr)
    if (shapDataLengthLongPtr != null)
      lightgbmlib.delete_int64_tp(shapDataLengthLongPtr)
    if (shapDataOutPtr == null)
      lightgbmlib.delete_doubleArray(shapDataOutPtr)
  }

  protected def shapForCSR(sparseVector: SparseVector): Array[Double] = {
    val numCols = sparseVector.size
    val kind = lightgbmlibConstants.C_API_PREDICT_CONTRIB

    val datasetParams = "max_bin=255 is_pre_partition=True"
    val dataInt32bitType = lightgbmlibConstants.C_API_DTYPE_INT32
    val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64

    ensureShapDataCreated()

    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterPredictForCSRSingle(
        sparseVector.indices, sparseVector.values,
        sparseVector.numNonzeros,
        boosterPtr, dataInt32bitType, data64bitType, 2, numCols,
        kind, -1, datasetParams,
        shapDataLengthLongPtr, shapDataOutPtr), "Booster Predict")

    predToArray(false, shapDataOutPtr, kind)
  }

  protected def predictForCSR(sparseVector: SparseVector, kind: Int, classification: Boolean): Array[Double] = {
    val numCols = sparseVector.size

    val datasetParams = "max_bin=255 is_pre_partition=True"
    val dataInt32bitType = lightgbmlibConstants.C_API_DTYPE_INT32
    val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64

    ensureScoredDataCreated()

    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterPredictForCSRSingle(
        sparseVector.indices, sparseVector.values,
        sparseVector.numNonzeros,
        boosterPtr, dataInt32bitType, data64bitType, 2, numCols,
        kind, -1, datasetParams,
        scoredDataLengthLongPtr, scoredDataOutPtr), "Booster Predict")

    predToArray(classification, scoredDataOutPtr, kind)
  }

  protected def shapForMat(row: Array[Double]): Array[Double] = {
    val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64
    val kind = lightgbmlibConstants.C_API_PREDICT_CONTRIB

    val numCols = row.length
    val isRowMajor = 1

    val datasetParams = "max_bin=255"

    ensureScoredDataCreated()

    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterPredictForMatSingle(
        row, boosterPtr, data64bitType,
        numCols,
        isRowMajor, kind,
        -1, datasetParams, scoredDataLengthLongPtr, scoredDataOutPtr),
      "Booster Predict")
    predToArray(false, scoredDataOutPtr, kind)
  }

  protected def predictForMat(row: Array[Double], kind: Int, classification: Boolean): Array[Double] = {
    val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64

    val numCols = row.length
    val isRowMajor = 1

    val datasetParams = "max_bin=255"

    ensureScoredDataCreated()

    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterPredictForMatSingle(
        row, boosterPtr, data64bitType,
        numCols,
        isRowMajor, kind,
        -1, datasetParams, scoredDataLengthLongPtr, scoredDataOutPtr),
      "Booster Predict")
    predToArray(classification, scoredDataOutPtr, kind)
  }

  def saveNativeModel(session: SparkSession, filename: String, overwrite: Boolean): Unit = {
    if (filename == null || filename.isEmpty) {
      throw new IllegalArgumentException("filename should not be empty or null.")
    }
    val rdd = session.sparkContext.parallelize(Seq(model))
    import session.sqlContext.implicits._
    val dataset = session.sqlContext.createDataset(rdd)
    val mode = if (overwrite) SaveMode.Overwrite else SaveMode.ErrorIfExists
    dataset.coalesce(1).write.mode(mode).text(filename)
  }

  /**
    * Calls into LightGBM to retrieve the feature importances.
    * @param importanceType Can be "split" or "gain"
    * @return The feature importance values as an array.
    */
  def getFeatureImportances(importanceType: String): Array[Double] = {
    val importanceTypeNum = if (importanceType.toLowerCase.trim == "gain") 1 else 0
    if (boosterPtr == null) {
      LightGBMUtils.initializeNativeLibrary()
      boosterPtr = getBoosterPtrFromModelString(model)
    }
    val numFeaturesOut = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterGetNumFeature(boosterPtr, numFeaturesOut),
      "Booster NumFeature")
    val numFeatures = lightgbmlib.intp_value(numFeaturesOut)
    val featureImportances = lightgbmlib.new_doubleArray(numFeatures)
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterFeatureImportance(boosterPtr, -1, importanceTypeNum, featureImportances),
      "Booster FeatureImportance")
    (0 until numFeatures).map(lightgbmlib.doubleArray_getitem(featureImportances, _)).toArray
  }

  /**
    * Retrieve the number of classes from LightGBM Booster
    * @return The number of classes.
    */
  def getNumClasses(): Int = {
    if (boosterPtr == null) {
      LightGBMUtils.initializeNativeLibrary()
      boosterPtr = getBoosterPtrFromModelString(model)
    }
    val numClassesOut = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterGetNumClasses(boosterPtr, numClassesOut),
      "Booster NumClasses")
    lightgbmlib.intp_value(numClassesOut)
  }

  private def getNumFeatures(): Int = {
    if (boosterPtr == null) {
      LightGBMUtils.initializeNativeLibrary()
      boosterPtr = getBoosterPtrFromModelString(model)
    }
    val numFeaturesOut = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterGetNumFeature(boosterPtr, numFeaturesOut),
      "Booster NumFeature")
    lightgbmlib.intp_value(numFeaturesOut)
  }

  private def predToArray(classification: Boolean, scoredDataOutPtr: SWIGTYPE_p_double, kind: Int): Array[Double] = {
    if (kind == lightgbmlibConstants.C_API_PREDICT_CONTRIB) {
      (0 until numFeatures).map(featNum =>
        lightgbmlib.doubleArray_getitem(shapDataOutPtr, featNum)).toArray
    } else if (classification && numClasses == 1) {
      // Binary classification scenario - LightGBM only returns the value for the positive class
      val pred = lightgbmlib.doubleArray_getitem(scoredDataOutPtr, 0)
      if (kind == lightgbmlibConstants.C_API_PREDICT_RAW_SCORE) {
        // Return the raw score for binary classification
        Array(-pred, pred)
      } else {
        // Return the probability for binary classification
        Array(1 - pred, pred)
      }
    } else {
      (0 until numClasses).map(classNum =>
        lightgbmlib.doubleArray_getitem(scoredDataOutPtr, classNum)).toArray
    }
  }
}
