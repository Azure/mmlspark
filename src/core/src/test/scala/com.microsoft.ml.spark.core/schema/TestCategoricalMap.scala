// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.schema._
import org.apache.spark.sql.types._

import scala.reflect.{ClassTag, classTag}

class TestCategoricalMap extends TestBase {

  /** Basic asserts that should be true for all Categorical Maps
    *
    * @param levels       sorted categorical levels
    * @param wrong_level  a value that is not a level
    * @param dataType     corresponding Spark datatype
    * @param isOrdinal    whether levels are Ordinal or not
    * @param mmlStyle     save to MML (true, default) or MLlib (false) metadata
    */
  private def testMapBasic[T: ClassTag](levels: Array[T], wrong_level: T, dataType: DataType,
                                        isOrdinal: Boolean, mmlStyle: Boolean = true): Unit = {

    val map = new CategoricalMap(levels, isOrdinal)
    val s   = " " + classTag[T]; // to idenfity which type throws the error

    assert(map.numLevels == levels.length, "numLevels" + s)
    assert(map.isOrdinal == isOrdinal, "isOrdinal" + s)
    assert(map.dataType == dataType, "dataType" + s)
    assert(map.getIndex(levels.head) == 0 & map.getIndex(levels.last) == levels.length - 1, "getIndex" + s)
    assert(map.getIndexOption(wrong_level).isEmpty & map.getIndexOption(levels(1)).contains(1), "getIndexOption" + s)
    assert(map.hasLevel(levels(1)) & !map.hasLevel(wrong_level), "hasLevel" + s)
    assert(map.getLevel(1) == levels(1), "getLevel" + s)
    assert(map.getLevelOption(1).contains(levels(1)) & map.getLevelOption(-1).isEmpty, "getLevelOption" + s)

    val mml_meta = map.toMetadata(mmlStyle) //TODO: check metadata for correctness
  }

  /** Test CategoricalMap for different undelying types */
  test("Test: Create basic CategoricalMap") {
    for (mmlStyle <- List(true, false)) {
      val isOrdinal = mmlStyle
      testMapBasic(Array("as", "", "efe"),
                   "wrong_level", StringType, isOrdinal, mmlStyle)
      testMapBasic(Array[Int](34, 54747, -346, 756, 0),
                   -45, IntegerType, isOrdinal, mmlStyle)
      testMapBasic(Array[Long](34, 54747, -346, 756, 0),
                   -45: Long, LongType, isOrdinal, mmlStyle)
      testMapBasic(Array[Double](34.45, 54.747, -3.46, 7.56, 0),
                   -45: Double, DoubleType, isOrdinal, mmlStyle)
    }
  }

}
