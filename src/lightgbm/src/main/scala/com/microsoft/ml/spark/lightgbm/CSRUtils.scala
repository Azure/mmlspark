// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.lightgbm

/** Temporary class that accepts int32_t arrays instead of void pointer arguments.
  * TODO: Need to generate a new lightGBM jar with utility to convert int array
  * to void pointer and then remove this file.
  */
object CSRUtils {
  // scalastyle:off parameter.number
  def LGBM_DatasetCreateFromCSR(var0: SWIGTYPE_p_int32_t, var1: Int, var2: SWIGTYPE_p_int32_t, var3: SWIGTYPE_p_void,
                                var4: Int, var5: SWIGTYPE_p_int64_t, var6: SWIGTYPE_p_int64_t,
                                var7: SWIGTYPE_p_int64_t, var8: String, var9: SWIGTYPE_p_void,
                                var10: SWIGTYPE_p_p_void): Int = {
    lightgbmlibJNI.LGBM_DatasetCreateFromCSR(SWIGTYPE_p_int32_t.getCPtr(var0), var1, SWIGTYPE_p_int32_t.getCPtr(var2),
      SWIGTYPE_p_void.getCPtr(var3), var4, SWIGTYPE_p_int64_t.getCPtr(var5), SWIGTYPE_p_int64_t.getCPtr(var6),
      SWIGTYPE_p_int64_t.getCPtr(var7), var8, SWIGTYPE_p_void.getCPtr(var9), SWIGTYPE_p_p_void.getCPtr(var10))
  }

  def LGBM_BoosterPredictForCSR(var0: SWIGTYPE_p_void, var1: SWIGTYPE_p_int32_t, var2: Int, var3: SWIGTYPE_p_int32_t,
                                var4: SWIGTYPE_p_void, var5: Int, var6: SWIGTYPE_p_int64_t, var7: SWIGTYPE_p_int64_t,
                                var8: SWIGTYPE_p_int64_t, var9: Int, var10: Int, var11: String,
                                var12: SWIGTYPE_p_int64_t, var13: SWIGTYPE_p_double): Int = {
    lightgbmlibJNI.LGBM_BoosterPredictForCSR(SWIGTYPE_p_void.getCPtr(var0), SWIGTYPE_p_int32_t.getCPtr(var1),
      var2, SWIGTYPE_p_int32_t.getCPtr(var3), SWIGTYPE_p_void.getCPtr(var4), var5, SWIGTYPE_p_int64_t.getCPtr(var6),
      SWIGTYPE_p_int64_t.getCPtr(var7), SWIGTYPE_p_int64_t.getCPtr(var8), var9, var10, var11,
      SWIGTYPE_p_int64_t.getCPtr(var12), SWIGTYPE_p_double.getCPtr(var13))
  }
  // scalastyle:on parameter.number
}
