package com.microsoft.ml.spark.cyber.feature

import com.microsoft.ml.spark.core.contracts.{HasAdditionalPythonMethods, HasInputCol, HasOutputCol, Wrappable}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Estimator, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, IntegerType, LongType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

trait HasPerGroupIndexCol extends Params {
  val perGroupIndexCol = new Param[String](this, "perGroupIndexCol",
    "the column name of the index in the perGroupIndex dataframe")

  def getPerGroupIndexCol: String = $(perGroupIndexCol)

  def setPerGroupIndexCol(v: String): this.type = set(perGroupIndexCol, v)

  setDefault(perGroupIndexCol -> "index")

}

object PartitionedStringIndexerModel extends ComplexParamsReadable[PartitionedStringIndexerModel]

class PartitionedStringIndexerModel(override val uid: String)
  extends Model[PartitionedStringIndexerModel] with HasInputCol with HasOutputCol
    with HasPartitionKey with HasPerGroupIndexCol with ComplexParamsWritable with HasAdditionalPythonMethods {

  override def additionalPythonMethods(): String = {
    """
      |    def inverseTransform(self, df):
      |        ctx = SparkContext._active_spark_context
      |        sql_ctx = SQLContext.getOrCreate(ctx)
      |        return  DataFrame(self._java_obj.inverseTransform(df._jdf), sql_ctx)
      |""".stripMargin + super.additionalPythonMethods()
  }

  def this() = this(Identifiable.randomUID("PartitionedStringIndexerModel"))

  //Note: This type of parameter requires the use of ComplexParamsWritable and Readable
  val perGroupIndex = new DataFrameParam(this, "perGroupIndex",
    "the mapping used to index each partition")

  def getPerGroupIndex: DataFrame = $(perGroupIndex)

  def setPerGroupIndex(v: DataFrame): this.type = set(perGroupIndex, v)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val partitionKeyOrDefault = get(partitionKey).getOrElse(uid + "_partitionKey")
    val dfWithPartition = get(partitionKey).map(_ => df).getOrElse(
      df.withColumn(partitionKeyOrDefault, lit(1))
    )

    assert(getPerGroupIndex.schema(partitionKeyOrDefault).dataType ==
      dfWithPartition.schema(partitionKeyOrDefault).dataType)
    assert(getPerGroupIndex.schema.fieldNames.contains(getPerGroupIndexCol))
    assert(getPerGroupIndex.schema.fieldNames.contains(getInputCol))

    val resultDF = dfWithPartition
      .join(getPerGroupIndex, Seq(partitionKeyOrDefault, getInputCol), "left")
      .withColumnRenamed("index", getOutputCol)

    get(partitionKey).map(_ => resultDF).getOrElse(resultDF.drop(partitionKeyOrDefault))
  }

  def inverseTransform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val partitionKeyOrDefault = get(partitionKey).getOrElse(uid + "_partitionKey")
    val dfWithPartition = get(partitionKey).map(_ => df).getOrElse(
      df.withColumn(partitionKeyOrDefault, lit(1))
    )

    assert(getPerGroupIndex.schema(partitionKeyOrDefault).dataType ==
      dfWithPartition.schema(partitionKeyOrDefault).dataType)
    assert(getPerGroupIndex.schema.fieldNames.contains(getPerGroupIndexCol))
    assert(getPerGroupIndex.schema.fieldNames.contains("index"))

    val resultDF = dfWithPartition
      .join(
        getPerGroupIndex.withColumnRenamed("index", getOutputCol),
        Seq(partitionKeyOrDefault, getOutputCol), "left")

    get(partitionKey).map(_ => resultDF).getOrElse(resultDF.drop(partitionKeyOrDefault))
  }

  override def copy(extra: ParamMap): PartitionedStringIndexerModel = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, LongType)
  }

}

object PartitionedStringIndexer extends DefaultParamsReadable[PartitionedStringIndexer]

class PartitionedStringIndexer(override val uid: String) extends Estimator[PartitionedStringIndexerModel]
  with HasPartitionKey with HasInputCol with HasOutputCol
  with HasPerGroupIndexCol with DefaultParamsWritable with Wrappable {

  val overlappingIndicies = new BooleanParam(this, "overlappingIndicies",
    "whether the indicies should start at 0 for each partition," +
      " or be separate per partition ")

  def setOverlappingIndicies(v: Boolean): this.type = set(overlappingIndicies, v)

  def getOverlappingIndicies: Boolean = $(overlappingIndicies)

  setDefault(overlappingIndicies -> true)

  def this() = this(Identifiable.randomUID("PartitionedStringIndexer"))

  private def makePerGroupIndex(dfWithGroup: DataFrame, partitionKeyOrDefault: String): DataFrame = {
    dfWithGroup
      .groupBy(col(partitionKeyOrDefault), col(getInputCol))
      .count()
      .groupBy(partitionKeyOrDefault)
      .agg(sort_array(
        collect_list(struct(col("count"), col(getInputCol))),
        asc = false).alias("sorted_strings"))
      .withColumn(
        "strings_and_indicies",
        udf({ arr: Seq[Row] =>
          arr.map(r => r.getString(1)).zipWithIndex.map(p => (p._1, p._2.toLong))
        },
          ArrayType(new StructType()
            .add(getInputCol, StringType)
            .add("index", LongType))
        )(col("sorted_strings")))
      .select(
        explode(col("strings_and_indicies")).alias("strings_and_indicies"),
        col(partitionKeyOrDefault))
      .select(
        col(partitionKeyOrDefault),
        col("strings_and_indicies").getField(getInputCol).alias(getInputCol),
        col("strings_and_indicies").getField("index").alias("index")
      )
  }

  private[ml] def asLong(in: Any): Long = in match {
    case x: Int => x.toLong
    case x: Long => x
    case x: BigInt => x.toLong
    case x: Float => x.toLong
    case x: Double => x.toLong
  }

  override def fit(dataset: Dataset[_]): PartitionedStringIndexerModel = {
    val df = dataset.toDF()

    import df.sparkSession.implicits._

    val partitionKeyOrDefault = get(partitionKey).getOrElse(uid + "_partitionKey")
    val dfWithGroup = get(partitionKey).map(_ => df).getOrElse(
      df.withColumn(partitionKeyOrDefault, lit(1))
    )

    val perGroupIndex = makePerGroupIndex(dfWithGroup, partitionKeyOrDefault)

    val perGroupIndex2 = if (getOverlappingIndicies) {
      perGroupIndex
    } else {
      val keysAndCounts = perGroupIndex
        .groupBy(partitionKeyOrDefault)
        .count()
        .collect()
        .map(r => (r.getString(0), asLong(r.get(1))))

      val cumCounts = keysAndCounts.map(_._2).scanLeft(0L) { case (sum, next) => sum + next }

      perGroupIndex.join(
        keysAndCounts.map(_._1).zip(cumCounts).toSeq.toDF(partitionKeyOrDefault, "offset"),
        Seq(partitionKeyOrDefault),
        "left")
        .select(
          (col("index") + col("offset")).cast(IntegerType).alias("index"),
          col(partitionKeyOrDefault),
          col(getInputCol)
        )
    }

    extractParamMap().toSeq.filterNot(_.param == overlappingIndicies)
      .foldLeft(
        new PartitionedStringIndexerModel(uid).setPerGroupIndex(perGroupIndex2)
      ) { case (m, pp) =>
        m.set(pp.param, pp.value)
      }
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, LongType)
  }

}

