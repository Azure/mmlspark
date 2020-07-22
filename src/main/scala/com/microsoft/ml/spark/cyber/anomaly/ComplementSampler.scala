package com.microsoft.ml.spark.cyber.anomaly

import com.microsoft.ml.spark.core.contracts.{HasInputCols, Wrappable}
import com.microsoft.ml.spark.cyber.feature.HasPartitionKey
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, IntegerType, LongType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.Random

object ComplementSampler extends DefaultParamsReadable[ComplementSampler]

/**
  * Given a dataframe it returns a new dataframe with access patterns sampled from
  * the set of possible access patterns which did not occur in the given dataframe
  * (i.e., it returns a sample from the complement set).
  *
  * @param uid uid of the model
  */
class ComplementSampler(override val uid: String)
  extends Transformer with HasInputCols
    with HasPartitionKey with DefaultParamsWritable with Wrappable {

  def this() = this(Identifiable.randomUID("ComplementAccess"))

  //Note: This type of parameter requires the use of ComplexParamsWritable and Readable
  val samplingFactor = new DoubleParam(this, "samplingFactor",
    "The approximate relative size of the complement set to generate")

  def getSamplingFactor: Double = $(samplingFactor)

  def setSamplingFactor(v: Double): this.type = set(samplingFactor, v)

  setDefault(samplingFactor -> 1.0)

  def sample(seed: String, mins: Seq[Long], maxes: Seq[Long], n: Int): Seq[Seq[Long]] = {
    val rng = new Random(seed.hashCode)
    (0 until n).map(_ =>
      mins.zip(maxes).map { case (min: Long, max: Long) => rng.nextInt((max - min + 1).toInt).toLong + min }
    )
  }

  private def verifyIncomingSchema(schema: StructType): Unit = {
    getInputCols.map(schema(_)).foreach(sf =>
      assert(sf.dataType match {
        case _: LongType => true
        case _: IntegerType => true
        case _ => false
      }, s"InputColumn ${sf.name} needs to be LongType got ${sf.dataType}"))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val partitionKeyOrDefault = get(partitionKey).getOrElse(uid + "_partitionKey")
    val dfWithPartition = get(partitionKey).map(_ => df).getOrElse(
      df.withColumn(partitionKeyOrDefault, lit("1"))
    )
    verifyIncomingSchema(df.schema)

    val sampleDF = dfWithPartition
      .select(Seq(col(partitionKeyOrDefault)) ++ getInputCols.map(ic => col(ic).cast(LongType).alias(ic)): _*)
      .groupBy(partitionKeyOrDefault)
      .agg(
        array(getInputCols.map(min): _*).alias("mins"),
        array(getInputCols.map(max): _*).alias("maxes"),
        count("*").alias("count"))
      .select(
        col(partitionKeyOrDefault),
        explode(
          udf(sample _, ArrayType(ArrayType(LongType)))(
            col(partitionKeyOrDefault),
            col("mins"),
            col("maxes"),
            (col("count") * lit(getSamplingFactor)).cast(IntegerType)
          )
        ).alias("samples"))
      .select(
        Seq(col(partitionKeyOrDefault)) ++
          getInputCols.zipWithIndex.map { case (name, i) => col("samples").getItem(i).alias(name) }: _*
      )

    val resultDF = sampleDF
      .join(
        dfWithPartition,
        Seq(partitionKeyOrDefault) ++ getInputCols,
        "left_anti"
      ).select(Seq(col(partitionKeyOrDefault)) ++
      getInputCols.map(ic => col(ic).cast(df.schema(ic).dataType).alias(ic)): _*)

    get(partitionKey).map(_ => resultDF).getOrElse(resultDF.drop(partitionKeyOrDefault))
  }

  override def copy(extra: ParamMap): ComplementSampler = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    verifyIncomingSchema(schema)
    (get(partitionKey).toList ++ getInputCols)
      .foldLeft(new StructType()) { case (st, colName) => st.add(schema(colName)) }
  }

}
