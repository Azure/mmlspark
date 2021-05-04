// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize

import java.io.File

import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Transformer}
import org.apache.spark.ml.param.{ByteArrayParam, Param, ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class TestEstimatorBase(val uid: String) extends Transformer {
  def this() = this(Identifiable.randomUID("TestEstimatorBase"))

  def transform(dataset: Dataset[_]): DataFrame = dataset.toDF()

  def copy(extra: ParamMap): this.type = defaultCopy(extra)

  def transformSchema(schema: StructType): StructType = schema

}

trait HasByteArrayParam extends Params {
  val byteArray = new ByteArrayParam(this, "byteArray", "bar")

  def getByteArray: Array[Byte] = $(byteArray)

  def setByteArray(value: Array[Byte]): this.type = set(byteArray, value)
}

trait HasStringParam extends Params {
  val stringParam = new Param[String](this, "stringParam", "bar")

  def getStringParam: String = $(stringParam)

  def setStringParam(value: String): this.type = set(stringParam, value)
}

class ComplexParamTest(override val uid: String) extends TestEstimatorBase(uid)
  with HasByteArrayParam with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("ComplexParamTest"))
}

object ComplexParamTest extends ComplexParamsReadable[ComplexParamTest]

class StandardParamTest(override val uid: String) extends TestEstimatorBase(uid)
  with HasStringParam with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("StandardParamTest"))
}

object StandardParamTest extends ComplexParamsReadable[StandardParamTest]

class MixedParamTest(override val uid: String) extends TestEstimatorBase(uid)
  with HasStringParam with HasByteArrayParam with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("MixedParamTest"))
}

object MixedParamTest extends ComplexParamsReadable[MixedParamTest]

class ValidateComplexParamSerializer extends TestBase {
  val saveFile: String = new File(tmpDir.toFile, "m1.model").toString
  val saveFile2: String = new File(tmpDir.toFile, "m2.model").toString

  test("Complex Param serialization should work on all complex, all normal, or mixed") {
    spark

    val bytes = "foo".toCharArray.map(_.toByte)
    val s = "foo"

    val cpt1 = new ComplexParamTest("foo").setByteArray(bytes)
    cpt1.write.overwrite().save(saveFile)
    val cpt2 = ComplexParamTest.load(saveFile)
    assert(cpt1.getByteArray === cpt2.getByteArray)

    val spt1 = new StandardParamTest("foo").setStringParam(s)
    spt1.write.overwrite().save(saveFile)
    val spt2 = StandardParamTest.load(saveFile)
    assert(spt1.getStringParam === spt2.getStringParam)

    val mpt1 = new MixedParamTest("foo").setByteArray(bytes).setStringParam(s)
    mpt1.write.overwrite().save(saveFile)
    val mpt2 = MixedParamTest.load(saveFile)
    assert(mpt1.getByteArray === mpt2.getByteArray)
    assert(mpt1.getStringParam === mpt2.getStringParam)
  }

  test("Complex Param serialization should yield portable models") {
    spark
    val bytes = "foo".toCharArray.map(_.toByte)
    val s = "foo"

    val mpt1 = new MixedParamTest("foo").setByteArray(bytes).setStringParam(s)
    mpt1.write.overwrite().save(saveFile)

    FileUtils.moveDirectory(new File(saveFile), new File(saveFile2))

    val mpt2 = MixedParamTest.load(saveFile2)
    assert(mpt1.getByteArray === mpt2.getByteArray)
    assert(mpt1.getStringParam === mpt2.getStringParam)
  }

  override def afterAll(): Unit = {
    new File(saveFile).delete()
    new File(saveFile2).delete()
    super.afterAll()
  }
}
