// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import com.microsoft.ml.spark.codegen.Wrappable
import com.microsoft.ml.spark.core.contracts.{HasInputCols, HasOutputCol}
import com.microsoft.ml.spark.logging.BasicLogging
import com.microsoft.ml.spark.vw.featurizer._
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Transformer}
import org.apache.spark.ml.param.{BooleanParam, IntParam, ParamMap, StringArrayParam}
import org.apache.spark.sql.types.{StringType, _}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, struct, udf}
import org.vowpalwabbit.spark.VowpalWabbitMurmur
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

import scala.collection.mutable

object VowpalWabbitFeaturizer extends ComplexParamsReadable[VowpalWabbitFeaturizer]

class VowpalWabbitFeaturizer(override val uid: String) extends Transformer
  with HasInputCols with HasOutputCol with HasNumBits with HasSumCollisions
  with Wrappable with ComplexParamsWritable with BasicLogging
{
  logClass()
  def this() = this(Identifiable.randomUID("VowpalWabbitFeaturizer"))

  setDefault(inputCols -> Array())
  setDefault(outputCol -> "features")

  val seed = new IntParam(this, "seed", "Hash seed")
  setDefault(seed -> 0)

  def getSeed: Int = $(seed)
  def setSeed(value: Int): this.type = set(seed, value)

  val stringSplitInputCols = new StringArrayParam(this, "stringSplitInputCols",
    "Input cols that should be split at word boundaries")
  setDefault(stringSplitInputCols -> Array())

  def getStringSplitInputCols: Array[String] = $(stringSplitInputCols)
  def setStringSplitInputCols(value: Array[String]): this.type = set(stringSplitInputCols, value)

  val preserveOrderNumBits = new IntParam(this, "preserveOrderNumBits",
    "Number of bits used to preserve the feature order. This will reduce the hash size. " +
      "Needs to be large enough to fit count the maximum number of words",
    (value: Int) => value >= 0 && value < 29)
  setDefault(preserveOrderNumBits -> 0)

  def getPreserveOrderNumBits: Int = $(preserveOrderNumBits)
  def setPreserveOrderNumBits(value: Int): this.type = set(preserveOrderNumBits, value)

  val prefixStringsWithColumnName = new BooleanParam(this, "prefixStringsWithColumnName",
    "Prefix string features with column name")
  setDefault(prefixStringsWithColumnName -> true)

  def getPrefixStringsWithColumnName: Boolean = $(prefixStringsWithColumnName)
  def setPrefixStringsWithColumnName(value: Boolean): this.type = set(prefixStringsWithColumnName, value)

  private def getAllInputCols = getInputCols ++ getStringSplitInputCols

  private def getFeaturizer(name: String,
                            dataType: DataType,
                            nullable: Boolean,
                            idx: Int,
                            namespaceHash: Int): Featurizer = {

    val prefixName = if (getPrefixStringsWithColumnName) name else ""
    dataType match {
      case DoubleType => getNumericFeaturizer[Double](prefixName, nullable, idx, namespaceHash, 0)
      case FloatType => getNumericFeaturizer[Float](prefixName, nullable, idx, namespaceHash, 0)
      case IntegerType => getNumericFeaturizer[Int](prefixName, nullable, idx, namespaceHash, 0)
      case LongType => getNumericFeaturizer[Long](prefixName, nullable, idx, namespaceHash, 0)
      case ShortType => getNumericFeaturizer[Short](prefixName, nullable, idx, namespaceHash, 0)
      case ByteType => getNumericFeaturizer[Byte](prefixName, nullable, idx, namespaceHash, 0)
      case BooleanType => new BooleanFeaturizer(idx, prefixName, namespaceHash, getMask)
      case StringType => getStringFeaturizer(name, prefixName, idx, namespaceHash)
      case ArrayType(t: StringType, _) => getArrayFeaturizer("", ArrayType(t), nullable, idx, namespaceHash)
      // Arrays of strings never use a prefix and use the column name namespace hash
      case arr: ArrayType => getArrayFeaturizer(name, arr, nullable, idx)
      case struct: StructType => getStructFeaturizer(struct, name, nullable, idx)
      case m: MapType => getMapFeaturizer(prefixName, m, idx, namespaceHash)
      case m: Any => getOtherFeaturizer(m, prefixName, idx)
    }
  }

  private def getNumericFeaturizer[T](prefixName: String,
                                                                       nullable: Boolean,
                                                                       idx: Int,
                                                                       namespaceHash: Int,
                                                                       zero: T)(implicit n: Numeric[T]): Featurizer = {
    if (nullable)
      new NullableNumericFeaturizer[T](idx, prefixName, namespaceHash, getMask, n)
    else
      new NumericFeaturizer[T](idx, prefixName, namespaceHash, getMask, n)
  }

  private def getArrayFeaturizer(name: String, dataType: ArrayType, nullable: Boolean, idx: Int,
                                 namespaceHash: Int = this.getSeed): Featurizer = {
    new SeqFeaturizer(idx, name, getFeaturizer(name, dataType.elementType, nullable, idx, namespaceHash))
  }

  private def getOtherFeaturizer(dataType: Any, prefixName: String, idx: Int): Featurizer =
    if (dataType == VectorType) // unfortunately the type is private
      new VectorFeaturizer(idx, prefixName, getMask)
    else
      throw new RuntimeException(s"Unsupported data type: $dataType")

  private def getStringFeaturizer(name: String, prefixName: String, idx: Int, namespaceHash: Int): Featurizer =
    if (getStringSplitInputCols.contains(name))
      new StringSplitFeaturizer(idx, prefixName, namespaceHash, getMask)
    else
      new StringFeaturizer(idx, prefixName, namespaceHash, getMask)

  private def getStructFeaturizer(dataType: StructType,
                                  name: String,
                                  nullable: Boolean,
                                  idx: Int): Featurizer = {
    val namespaceHash = VowpalWabbitMurmur.hash(name, this.getSeed)
    val subFeaturizers = dataType.fields
      .zipWithIndex
      .map { case (f, i) => getFeaturizer(f.name, f.dataType, f.nullable, i, namespaceHash) }

    if (nullable)
      new NullableStructFeaturizer(idx, name, subFeaturizers)
    else
      new StructFeaturizer(idx, name, subFeaturizers)
  }

  private def getMapFeaturizer(prefixName: String, dataType: MapType, idx: Int, namespaceHash: Int): Featurizer = {
    if (dataType.keyType != DataTypes.StringType)
      throw new RuntimeException(s"Unsupported map key type: $dataType")

    dataType.valueType match {
      case StringType => new MapStringFeaturizer(idx, prefixName, namespaceHash, getMask)
      case DoubleType => new MapFeaturizer[Double](idx, prefixName, namespaceHash, getMask, v => v)
      case FloatType => new MapFeaturizer[Float](idx, prefixName, namespaceHash, getMask, v => v.toDouble)
      case IntegerType => new MapFeaturizer[Int](idx, prefixName, namespaceHash, getMask, v => v.toDouble)
      case LongType => new MapFeaturizer[Long](idx, prefixName, namespaceHash, getMask, v => v.toDouble)
      case ShortType => new MapFeaturizer[Short](idx, prefixName, namespaceHash, getMask, v => v.toDouble)
      case ByteType => new MapFeaturizer[Byte](idx, prefixName, namespaceHash, getMask, v => v.toDouble)
      case _ => throw new RuntimeException(s"Unsupported map value type: $dataType")
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      if (getPreserveOrderNumBits + getNumBits > 30)
        throw new IllegalArgumentException(
          s"Number of bits used for hashing (${getNumBits} and " +
            s"number of bits used for order preserving (${getPreserveOrderNumBits}) must be less than 30")

      val maxFeaturesForOrdering = 1 << getPreserveOrderNumBits

      val inputColsList = getAllInputCols
      val namespaceHash: Int = VowpalWabbitMurmur.hash(this.getOutputCol, this.getSeed)

      val fieldSubset = dataset.schema.fields
        .filter(f => inputColsList.contains(f.name))

      val featurizers: Array[Featurizer] = fieldSubset.zipWithIndex
        .map { case (field, idx) => getFeaturizer(field.name, field.dataType, field.nullable, idx, namespaceHash) }

      // TODO: list types
      // BinaryType
      // CalendarIntervalType
      // DateType
      // NullType
      // TimestampType

      val mode = udf((r: Row) => {
        val indices = mutable.ArrayBuilder.make[Int]
        val values = mutable.ArrayBuilder.make[Double]

        // educated guess on size
        indices.sizeHint(featurizers.length)
        values.sizeHint(featurizers.length)

        // apply all featurizers
        for (f <- featurizers)
          if (!r.isNullAt(f.fieldIdx))
            f.featurize(r, indices, values)

        val indicesArray = indices.result
        if (getPreserveOrderNumBits > 0) {
          val idxPrefixBits = 30 - getPreserveOrderNumBits

          if (indicesArray.length > maxFeaturesForOrdering)
            throw new IllegalArgumentException(
              s"Too many features ${indicesArray.length} for " +
                s"number of bits used for order preserving (${getPreserveOrderNumBits})")

          // prefix every feature index with a counter value
          // will be stripped when passing to VW
          for (i <- 0 until indicesArray.length) {
            val idxPrefix = i << idxPrefixBits
            indicesArray(i) = indicesArray(i) | idxPrefix
          }
        }

        // if we use the highest order bits to preserve the ordering
        // the maximum index size is larger
        val size = if (getPreserveOrderNumBits > 0) 1 << 30 else 1 << getNumBits

        // sort by indices and remove duplicate values
        // Warning:
        //   - due to SparseVector limitations (which doesn't allow duplicates) we need filter
        //   - VW command line allows for duplicate features with different values (just updates twice)
        val (indicesSorted, valuesSorted) = VectorUtils.sortAndDistinct(indicesArray, values.result, getSumCollisions)

        Vectors.sparse(size, indicesSorted, valuesSorted)
      })

      dataset.toDF.withColumn(getOutputCol, mode.apply(struct(fieldSubset.map(f => col(f.name)): _*)))
    })
  }

  override def copy(extra: ParamMap): VowpalWabbitFeaturizer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val fieldNames = schema.fields.map(_.name)
    for (f <- getAllInputCols)
      if (!fieldNames.contains(f))
        throw new IllegalArgumentException(s"missing input column $f")

    schema.add(StructField(getOutputCol, VectorType, true))
  }
}
