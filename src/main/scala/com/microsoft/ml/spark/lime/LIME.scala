// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lime

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.Rand
import com.microsoft.ml.spark.core.spark.FluentAPI._
import com.microsoft.ml.spark.codegen.Wrappable
import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.schema.{DatasetExtensions, ImageSchemaUtils}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.injections.UDFUtils
import org.apache.spark.internal.{Logging => SLogging}
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.SQLDataTypes.{MatrixType, VectorType}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable
import scala.util.Random

object LIMEUtils extends SLogging {

  def randomMasks(decInclude: Double, numSlots: Int): Iterator[Array[Boolean]] =
    new Iterator[Array[Boolean]] {
      Random.setSeed(0)

      override def hasNext: Boolean = true

      override def next(): Array[Boolean] = {
        Array.fill(numSlots) {
          Random.nextDouble() > decInclude
        }
      }
    }

  def localAggregateBy(df: DataFrame, groupByCol: String, colsToSquish: Seq[String]): DataFrame = {
    val schema = new StructType(df.schema.fields.map {
      case field if colsToSquish.contains(field.name) => StructField(field.name, ArrayType(field.dataType))
      case f => f
    })
    val encoder = RowEncoder(schema)
    val indiciesToSquish = colsToSquish.map(df.schema.fieldIndex)
    df.mapPartitions { it =>
      val isEmpty = it.isEmpty
      if (isEmpty) {
        (Nil: Seq[Row]).toIterator
      } else {
        // Current Id, What we have accumulated, Previous Row
        val accumulator = mutable.ListBuffer[Seq[Any]]()
        var currentId: Option[Any] = None
        var prevRow: Option[Row] = None

        def returnState(accumulated: List[Seq[Any]], prevRow: Row): Row = {
          Row.fromSeq(prevRow.toSeq.zipWithIndex.map {
            case (_, i) if indiciesToSquish.contains(i) =>
              accumulated.map(_.apply(indiciesToSquish.indexOf(i)))
            case (v, _) => v
          })
        }

        def enqueueAndMaybeReturn(row: Row): Option[Row] = {
          val id = row.getAs[Any](groupByCol)
          if (currentId.isEmpty) {
            currentId = Some(id)
            prevRow = Some(row)
            accumulator += colsToSquish.map(row.getAs[Any])
            None
          } else if (id != currentId.get) {
            val accumulated = accumulator.toList
            accumulator.clear()
            accumulator += colsToSquish.map(row.getAs[Any])
            val modified = returnState(accumulated, prevRow.get)
            currentId = Some(id)
            prevRow = Some(row)
            Some(modified)
          } else {
            prevRow = Some(row)
            accumulator += colsToSquish.map(row.getAs[Any])
            None
          }
        }

        val it1 = it
          .flatMap(row => enqueueAndMaybeReturn(row))
          .map(r => Left(r): Either[Row, Null]) //scalastyle:ignore null
          .++(Seq(Right(null): Either[Row, Null])) //scalastyle:ignore null

        val ret = it1.map {
          case Left(r) => r: Row
          case Right(_) => returnState(accumulator.toList, prevRow.getOrElse {
            logWarning("Could not get previous row in local aggregator, this is an error that should be fixed")
            null
          }): Row
        }
        ret
      }

    }(encoder)
  }
}

trait LIMEParams extends HasInputCol with HasOutputCol with HasPredictionCol {

  def setPredictionCol(v: String): this.type = set(predictionCol, v)

  val model = new TransformerParam(this, "model", "Model to try to locally approximate")

  def getModel: Transformer = $(model)

  def setModel(v: Transformer): this.type = set(model, v)

  val nSamples = new IntParam(this, "nSamples", "The number of samples to generate")

  def getNSamples: Int = $(nSamples)

  def setNSamples(v: Int): this.type = set(nSamples, v)

  val samplingFraction = new DoubleParam(this, "samplingFraction", "The fraction of superpixels to keep on")

  def getSamplingFraction: Double = $(samplingFraction)

  def setSamplingFraction(d: Double): this.type = set(samplingFraction, d)

  val regularization = new DoubleParam(this, "regularization", "regularization param for the lasso")

  def getRegularization: Double = $(regularization)

  def setRegularization(v: Double): this.type = set(regularization, v)
}

trait LIMEBase extends LIMEParams with ComplexParamsWritable {

  protected def getSamples(n: Int): Seq[Seq[Boolean]] = {
    LIMEUtils.randomMasks(getSamplingFraction, n)
      .take(getNSamples).map(_.toSeq).toSeq
  }

  protected def arrToMat(dvs: Seq[DenseVector]): DenseMatrix = {
    val mat = BDM(dvs.map(_.values): _*)
    new DenseMatrix(mat.rows, mat.cols, mat.data)
  }

  protected val arrToMatUDF: UserDefinedFunction = UDFUtils.oldUdf(arrToMat _, MatrixType)

  protected def arrToVect(ds: Seq[Double]): DenseVector = {
    new DenseVector(ds.toArray)
  }

  protected val arrToVectUDF: UserDefinedFunction = UDFUtils.oldUdf(arrToVect _, VectorType)

  protected val fitLassoUDF: UserDefinedFunction = UDFUtils.oldUdf(LimeNamespaceInjections.fitLasso _, VectorType)

  protected val getSampleUDF: UserDefinedFunction = UDFUtils.oldUdf(getSamples _, ArrayType(ArrayType(BooleanType)))

}

object TabularLIME extends ComplexParamsReadable[TabularLIME]

class TabularLIME(val uid: String) extends Estimator[TabularLIMEModel]
  with LIMEParams with Wrappable with ComplexParamsWritable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("TabularLIME"))

  setDefault(nSamples -> 1000, regularization -> 0.0, samplingFraction -> 0.3)

  override def fit(dataset: Dataset[_]): TabularLIMEModel = {
    logFit({
      val fitScaler = new StandardScaler()
        .setInputCol(getInputCol)
        .setOutputCol(getOutputCol)
        .setWithStd(true)
        .setWithMean(true)
        .fit(dataset)

      extractParamMap().toSeq.foldLeft(new TabularLIMEModel()) { case (m, pp) =>
        m.set(m.getParam(pp.param.name), pp.value)
      }
        .setColumnMeans(fitScaler.mean.toArray)
        .setColumnSTDs(fitScaler.std.toArray)
    })
  }

  override def copy(extra: ParamMap): TabularLIME = super.defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, VectorType)

  }
}

object TabularLIMEModel extends ComplexParamsReadable[TabularLIMEModel]

class TabularLIMEModel(val uid: String) extends Model[TabularLIMEModel]
  with LIMEBase with Wrappable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("TabularLIMEModel"))

  val columnMeans = new DoubleArrayParam(this, "columnMeans", "the means of each of the columns for perturbation")

  def getColumnMeans: Array[Double] = $(columnMeans)

  def setColumnMeans(v: Array[Double]): this.type = set(columnMeans, v)

  val columnSTDs = new DoubleArrayParam(this, "columnSTDs",
    "the standard deviations of each of the columns for perturbation")

  def getColumnSTDs: Array[Double] = $(columnSTDs)

  def setColumnSTDs(v: Array[Double]): this.type = set(columnSTDs, v)

  private def perturbedDenseVectors(v: DenseVector): Seq[DenseVector] = {
    Seq.fill(getNSamples) {
      val perturbed = BDV.rand(v.size, Rand.gaussian) * BDV(getColumnSTDs) + BDV(getColumnMeans)
      new DenseVector(perturbed.toArray)
    }
  }

  private val perturbedDenseVectorsUDF: UserDefinedFunction =
    UDFUtils.oldUdf(perturbedDenseVectors _, ArrayType(VectorType, true))

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val df = dataset.toDF
      val idCol = DatasetExtensions.findUnusedColumnName("id", df)
      val statesCol = DatasetExtensions.findUnusedColumnName("states", df)
      val inputCol2 = DatasetExtensions.findUnusedColumnName("inputCol2", df)

      val mapped = df.withColumn(idCol, monotonically_increasing_id())
        .withColumnRenamed(getInputCol, inputCol2)
        .withColumn(getInputCol, explode_outer(perturbedDenseVectorsUDF(col(inputCol2))))
        .mlTransform(getModel)

      LIMEUtils.localAggregateBy(mapped, idCol, Seq(getInputCol, getPredictionCol))
        .withColumn(getInputCol, arrToMatUDF(col(getInputCol)))
        .withColumn(getPredictionCol, arrToVectUDF(col(getPredictionCol)))
        .withColumn(getOutputCol, fitLassoUDF(col(getInputCol), col(getPredictionCol), lit(getRegularization)))
        .drop(statesCol, getPredictionCol, idCol, getInputCol)
        .withColumnRenamed(inputCol2, getInputCol)
    })
  }

  override def copy(extra: ParamMap): TabularLIMEModel = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, VectorType)
  }

}

object ImageLIME extends ComplexParamsReadable[ImageLIME]

/** Distributed implementation of
  * Local Interpretable Model-Agnostic Explanations (LIME)
  *
  * https://arxiv.org/pdf/1602.04938v1.pdf
  */
class ImageLIME(val uid: String) extends Transformer with LIMEBase
  with Wrappable with HasModifier with HasCellSize with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("ImageLIME"))

  val superpixelCol = new Param[String](this, "superpixelCol", "The column holding the superpixel decompositions")

  def getSuperpixelCol: String = $(superpixelCol)

  def setSuperpixelCol(v: String): this.type = set(superpixelCol, v)

  setDefault(nSamples -> 900, cellSize -> 16, modifier -> 130, regularization -> 0.0,
    samplingFraction -> 0.3, superpixelCol -> "superpixels")

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val df = dataset.toDF
      val idCol = DatasetExtensions.findUnusedColumnName("id", df)
      val statesCol = DatasetExtensions.findUnusedColumnName("states", df)
      val inputCol2 = DatasetExtensions.findUnusedColumnName("inputCol2", df)

      // Data frame with new column containing superpixels (Array[Cluster]) for each row (image)
      val spt = new SuperpixelTransformer()
        .setCellSize(getCellSize)
        .setModifier(getModifier)
        .setInputCol(getInputCol)
        .setOutputCol(getSuperpixelCol)

      val spDF = spt.transform(df)

      // Indices of the columns containing each image and image's superpixels
      val inputType = df.schema(getInputCol).dataType
      val maskUDF = inputType match {
        case BinaryType => Superpixel.MaskBinaryUDF
        case t if ImageSchemaUtils.isImage(t) => Superpixel.MaskImageUDF
      }

      val mapped = spDF.withColumn(idCol, monotonically_increasing_id())
        .withColumnRenamed(getInputCol, inputCol2)
        .withColumn(statesCol, explode_outer(getSampleUDF(size(col(getSuperpixelCol).getField("clusters")))))
        .withColumn(getInputCol, maskUDF(col(inputCol2), col(spt.getOutputCol), col(statesCol)))
        .withColumn(statesCol, UDFUtils.oldUdf(
          { barr: Seq[Boolean] => new DenseVector(barr.map(b => if (b) 1.0 else 0.0).toArray) },
          VectorType)(col(statesCol)))
        .mlTransform(getModel)
        .drop(getInputCol)

      LIMEUtils.localAggregateBy(mapped, idCol, Seq(statesCol, getPredictionCol))
        .withColumn(statesCol, arrToMatUDF(col(statesCol)))
        .withColumn(getPredictionCol, arrToVectUDF(col(getPredictionCol)))
        .withColumn(getOutputCol, fitLassoUDF(col(statesCol), col(getPredictionCol), lit(getRegularization)))
        .drop(statesCol, getPredictionCol)
        .withColumnRenamed(inputCol2, getInputCol)
    })
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getSuperpixelCol, SuperpixelData.Schema).add(getOutputCol, VectorType)
  }

}


