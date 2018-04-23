// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.lightgbm.{SWIGTYPE_p_float, SWIGTYPE_p_void, lightgbmlib, lightgbmlibConstants}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.Row
import org.slf4j.Logger

private object TrainUtils extends java.io.Serializable {

  def translate(labelColumn: String, featuresColumn: String, log: Logger, trainParams: TrainParams,
                inputRows: Iterator[Row]): Iterator[LightGBMBooster] = {
    if (!inputRows.hasNext)
      List[LightGBMBooster]().toIterator

    val rows = inputRows.toArray
    val numRows = rows.length
    val labels = rows.map(row => row.getDouble(row.fieldIndex(labelColumn)))
    val hrow = rows.head
    var datasetPtr: Option[SWIGTYPE_p_void] = None
    try {
      datasetPtr =
        if (hrow.get(hrow.fieldIndex(featuresColumn)).isInstanceOf[DenseVector]) {
          val rowsAsDoubleArray = rows.map(row => row.get(row.fieldIndex(featuresColumn)) match {
            case dense: DenseVector => dense.toArray
            case sparse: SparseVector => sparse.toDense.toArray
          })
          Some(LightGBMUtils.generateDenseDataset(numRows, rowsAsDoubleArray))
        } else {
          val rowsAsSparse = rows.map(row => row.get(row.fieldIndex(featuresColumn)) match {
            case dense: DenseVector => dense.toSparse
            case sparse: SparseVector => sparse
          })
          Some(LightGBMUtils.generateSparseDataset(rowsAsSparse))
        }

      // Validate generated dataset has the correct number of rows and cols
      validateDataset(datasetPtr.get)

      // Generate the label column and add to dataset
      var labelColArray: Option[SWIGTYPE_p_float] = None
      try {
        labelColArray = Some(lightgbmlib.new_floatArray(numRows))
        labels.zipWithIndex.foreach(ri =>
          lightgbmlib.floatArray_setitem(labelColArray.get, ri._2, ri._1.toFloat))
        val labelAsVoidPtr = lightgbmlib.float_to_voidp_ptr(labelColArray.get)
        val data32bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT32
        LightGBMUtils.validate(
          lightgbmlib.LGBM_DatasetSetField(datasetPtr.get, "label", labelAsVoidPtr, numRows, data32bitType),
          "DatasetSetField")
      } finally {
        // Free label column
        if (labelColArray.isDefined) {
          lightgbmlib.delete_floatArray(labelColArray.get)
        }
      }

      var boosterPtr: Option[SWIGTYPE_p_void] = None
      try {
        // Create the booster
        val boosterOutPtr = lightgbmlib.voidpp_handle()
        val parameters = trainParams.toString()
        LightGBMUtils.validate(lightgbmlib.LGBM_BoosterCreate(datasetPtr.get, parameters, boosterOutPtr), "Booster")
        boosterPtr = Some(lightgbmlib.voidpp_value(boosterOutPtr))
        val isFinishedPtr = lightgbmlib.new_intp()
        var isFinised = 0
        var iters = 0
        while (isFinised == 0 && iters < trainParams.numIterations) {
          val result = lightgbmlib.LGBM_BoosterUpdateOneIter(boosterPtr.get, isFinishedPtr)
          LightGBMUtils.validate(result, "Booster Update One Iter")
          isFinised = lightgbmlib.intp_value(isFinishedPtr)
          log.info("LightGBM running iteration: " + iters + " with result: " +
            result + " and is finished: " + isFinised)
          iters = iters + 1
        }
        val bufferLength = LightGBMConstants.defaultBufferLength
        val bufferLengthPtr = lightgbmlib.new_longp()
        lightgbmlib.longp_assign(bufferLengthPtr, bufferLength)
        val bufferLengthPtrInt64 = lightgbmlib.long_to_int64_t_ptr(bufferLengthPtr)
        val bufferOutLengthPtr = lightgbmlib.new_int64_tp()
        val tempM =
          lightgbmlib.LGBM_BoosterSaveModelToStringSWIG(
            boosterPtr.get, -1, bufferLengthPtrInt64,
            bufferOutLengthPtr)
        val bufferOutLength = lightgbmlib.longp_value(lightgbmlib.int64_t_to_long_ptr(bufferOutLengthPtr))
        // TODO: Move the reallocation logic inside the SWIG wrapper
        val model =
          if (bufferOutLength > bufferLength) {
            lightgbmlib.LGBM_BoosterSaveModelToStringSWIG(
              boosterPtr.get, -1, bufferOutLengthPtr, bufferOutLengthPtr)
          } else tempM
        log.info("Buffer output length for model: " + bufferOutLength)
        List[LightGBMBooster](new LightGBMBooster(model)).toIterator
      } finally {
        // Free booster
        if (boosterPtr.isDefined) {
          LightGBMUtils.validate(lightgbmlib.LGBM_BoosterFree(boosterPtr.get),
                                 "Finalize Booster")
        }
      }
    } finally {
      // Free dataset
      if (datasetPtr.isDefined) {
        LightGBMUtils.validate(lightgbmlib.LGBM_DatasetFree(datasetPtr.get),
                               "Finalize Dataset")
      }
    }
  }

  private def validateDataset(datasetPtr: SWIGTYPE_p_void): Unit = {
    // Validate num rows
    val numDataPtr = lightgbmlib.new_intp()
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetGetNumData(datasetPtr, numDataPtr), "DatasetGetNumData")
    val numData = lightgbmlib.intp_value(numDataPtr)
    if (numData <= 0) {
      throw new Exception("Unexpected num data: " + numData)
    }

    // Validate num cols
    val numFeaturePtr = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_DatasetGetNumFeature(datasetPtr, numFeaturePtr),
      "DatasetGetNumFeature")
    val numFeature = lightgbmlib.intp_value(numFeaturePtr)
    if (numFeature <= 0) {
      throw new Exception("Unexpected num feature: " + numFeature)
    }
  }

  def trainLightGBM(nodes: String, numNodes: Int, labelColumn: String, featuresColumn: String,
                    defaultListenPort: Int, log: Logger, trainParams: TrainParams)
                   (inputRows: Iterator[Row]): Iterator[LightGBMBooster] = {
    // Initialize the native library
    LightGBMUtils.initializeNativeLibrary()
    // Initialize the network communication
    val localListenPort = defaultListenPort + LightGBMUtils.getId
    log.info("LightGBM worker listening on: " + localListenPort)
    try {
      LightGBMUtils.validate(lightgbmlib.LGBM_NetworkInit(nodes, localListenPort,
        LightGBMConstants.defaultListenTimeout, numNodes), "Network init")
      translate(labelColumn, featuresColumn, log, trainParams, inputRows)
    } finally {
      // Finalize network when done
      LightGBMUtils.validate(lightgbmlib.LGBM_NetworkFree(), "Finalize network")
    }
  }

}
