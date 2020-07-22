package com.microsoft.ml.spark.cyber.feature

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, Wrappable}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Estimator, Model}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, max, min, struct, udf, mean => meanFunc, stddev_pop => stdFunc, lit}
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, RelationalGroupedDataset, Row}

trait HasPartitionKey extends Params {
  val partitionKey = new Param[String](this, "partitionKey",
    "The name of the column to partition by, i.e., scale the values of inputCol within each partition. ")

  def getPartitionKey: String = $(partitionKey)

  def setPartitionKey(v: String): this.type = set(partitionKey, v)
}

trait HasPerGroupStatsCol extends Params {
  val perGroupStatsCol = new Param[String](this, "perGroupStatsCol",
    "the column name of the stats in the perGroupStats dataframe")

  def getPerGroupStatsCol: String = $(perGroupStatsCol)

  def setPerGroupStatsCol(v: String): this.type = set(perGroupStatsCol, v)

  setDefault(perGroupStatsCol -> "stats")

}

abstract class PartitionedScalerModel[M <: PartitionedScalerModel[M]]
  extends Model[M] with HasInputCol with HasOutputCol
    with HasPartitionKey with HasPerGroupStatsCol with ComplexParamsWritable {

  //Note: This type of parameter requires the use of ComplexParamsWritable and Readable
  val perGroupStats = new DataFrameParam(this, "perGroupStats",
    "the statistics used for scaling each group")

  def getPerGroupStats: DataFrame = $(perGroupStats)

  def setPerGroupStats(v: DataFrame): this.type = set(perGroupStats, v)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val partitionKeyOrDefault = get(partitionKey).getOrElse(uid + "_partitionKey")
    val dfWithGroup = get(partitionKey).map(_ => df).getOrElse(
      df.withColumn(partitionKeyOrDefault, lit(1))
    )

    assert(getPerGroupStats.schema(partitionKeyOrDefault).dataType ==
      dfWithGroup.schema(partitionKeyOrDefault).dataType)
    assert(getPerGroupStats.schema.fieldNames.contains(getPerGroupStatsCol))

    val resultDF = dfWithGroup
      .join(getPerGroupStats, Seq(partitionKeyOrDefault),"left")
      .withColumn(getOutputCol, scaleValueUDF(col(getInputCol), col(getPerGroupStatsCol)))
      .drop(getPerGroupStatsCol)

    get(partitionKey).map(_ => resultDF).getOrElse(resultDF.drop(partitionKeyOrDefault))
  }

  private[ml] def scaleValue(input: Double, stats: Row): Double

  private[ml] def scaleValueGeneric(input: Any, stats: Row): Double = {
    val newInput = input match {
      case n: Double => n
      case n: Int => n.toDouble
      case n: Float => n.toDouble
      case n: Long => n.toDouble
      case n: BigInt => n.toDouble
      case n => n.asInstanceOf[Double]
    }
    scaleValue(newInput, stats)
  }

  private[ml] def scaleValueUDF: UserDefinedFunction = udf(scaleValueGeneric _, DoubleType)

  override def copy(extra: ParamMap): M = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, schema(getInputCol).dataType)
  }

}

abstract class PartitionedScaler[M <: PartitionedScalerModel[M]] extends Estimator[M]
  with HasPartitionKey with HasInputCol
  with HasOutputCol with HasPerGroupStatsCol with DefaultParamsWritable {

  private[ml] def computeStats(groupedDF: RelationalGroupedDataset): DataFrame

  private[ml] def constructor(uid: String): M

  override def fit(dataset: Dataset[_]): M = {
    val df = dataset.toDF()
    val partitionKeyOrDefault = get(partitionKey).getOrElse(uid + "_partitionKey")
    val dfWithGroup = get(partitionKey).map(_ => df).getOrElse(
      df.withColumn(partitionKeyOrDefault, lit(1))
    )

    val perGroupStats = computeStats(dfWithGroup.groupBy(col(partitionKeyOrDefault)))
    extractParamMap().toSeq
      .foldLeft(constructor(uid).setPerGroupStats(perGroupStats)) { case (m, pp) =>
        m.set(pp.param, pp.value)
      }
  }

  override def copy(extra: ParamMap): Estimator[M] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, schema(getInputCol).dataType)
  }

}

trait HasEpsilon extends Params {
  val epsilon = new DoubleParam(this, "epsilon",
    "The number used to stabilize scaling")

  def getEpsilon: Double = $(epsilon)

  def setEpsilon(v: Double): this.type = set(epsilon, v)

  setDefault(epsilon -> .000001)
}

trait HasMinAndMax extends Params {
  val minValue = new DoubleParam(this, "minValue",
    "The minimum value to scale to column to")

  def getMinValue: Double = $(minValue)

  def setMinValue(v: Double): this.type = set(minValue, v)

  val maxValue = new DoubleParam(this, "maxValue",
    "The maximum value to scale to column to")

  def getMaxValue: Double = $(maxValue)

  def setMaxValue(v: Double): this.type = set(maxValue, v)

  setDefault(minValue -> 0.0, maxValue -> 1.0)
}

object PartitionedMinMaxScaler extends DefaultParamsReadable[PartitionedMinMaxScaler]

class PartitionedMinMaxScaler(override val uid: String)
  extends PartitionedScaler[PartitionedMinMaxScalerModel]
    with HasMinAndMax with Wrappable with HasEpsilon {
  def this() = this(Identifiable.randomUID("PartitionedMinMaxScaler"))

  override private[ml] def computeStats(groupedDF: RelationalGroupedDataset) = {
    groupedDF.agg(struct(
      min(col(getInputCol)).alias("min"),
      max(col(getInputCol)).alias("max")).alias(getPerGroupStatsCol))
  }

  override private[ml] def constructor(uid: String) = new PartitionedMinMaxScalerModel(uid)
}

object PartitionedMinMaxScalerModel extends ComplexParamsReadable[PartitionedMinMaxScalerModel]

class PartitionedMinMaxScalerModel(override val uid: String)
  extends PartitionedScalerModel[PartitionedMinMaxScalerModel] with HasMinAndMax with HasEpsilon {
  def this() = this(Identifiable.randomUID("PartitionedMinMaxScalerModel"))

  override private[ml] def scaleValue(input: Double, stats: Row) = {
    val observedMin = stats.getAs[Double]("min")
    val observedMax = stats.getAs[Double]("max")
    val observedDelta = observedMax - observedMin
    val delta = getMaxValue - getMinValue
    if (observedDelta <= getEpsilon) {
      (getMaxValue + getMinValue) / 2.0
    } else {
      ((input - observedMin) / observedDelta) * delta + getMinValue
    }
  }

}

trait HasMeanAndStd extends Params {
  val mean = new DoubleParam(this, "mean",
    "The target mean to match the data to")

  def getMean: Double = $(mean)

  def setMean(v: Double): this.type = set(mean, v)

  val std = new DoubleParam(this, "std",
    "The target standard deviation to match the data to")

  def getStd: Double = $(std)

  def setStd(v: Double): this.type = set(std, v)

  setDefault(mean -> 0, std -> 1)
}

object PartitionedStandardScaler extends DefaultParamsReadable[PartitionedMinMaxScaler]

class PartitionedStandardScaler(override val uid: String)
  extends PartitionedScaler[PartitionedStandardScalerModel] with HasMeanAndStd with HasEpsilon with Wrappable {
  def this() = this(Identifiable.randomUID("PartitionedStandardScaler"))

  override private[ml] def computeStats(groupedDF: RelationalGroupedDataset) = {
    groupedDF.agg(struct(
      meanFunc(col(getInputCol)).alias("mean"),
      stdFunc(col(getInputCol)).alias("std")).alias(getPerGroupStatsCol))
  }

  override private[ml] def constructor(uid: String) = new PartitionedStandardScalerModel(uid)
}

object PartitionedStandardScalerModel extends ComplexParamsReadable[PartitionedStandardScalerModel]

class PartitionedStandardScalerModel(override val uid: String)
  extends PartitionedScalerModel[PartitionedStandardScalerModel]
    with HasMeanAndStd with HasEpsilon {
  def this() = this(Identifiable.randomUID("PartitionedStandardScalerModel"))

  override private[ml] def scaleValue(input: Double, stats: Row) = {
    val observedMean = stats.getAs[Double]("mean")
    val observedStd = {
      val obStd = stats.getAs[Double]("std")
      Math.max(if (obStd.isNaN) 0.0 else obStd, getEpsilon)
    }
    val targetStd = Math.max(getStd, getEpsilon)
    (input - observedMean) * (targetStd / observedStd) + getMean
  }

}
