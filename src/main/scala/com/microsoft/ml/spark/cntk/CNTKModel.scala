// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import com.microsoft.CNTK.CNTKExtensions._
import com.microsoft.CNTK.CNTKUtils._
import com.microsoft.CNTK.{CNTKExtensions, DataType => CNTKDataType, SerializableFunction => CNTKFunction, _}
import com.microsoft.ml.spark.cntk.ConversionUtils.GVV
import com.microsoft.ml.spark.codegen.Wrappable
import com.microsoft.ml.spark.core.schema.DatasetExtensions.findUnusedColumnName
import com.microsoft.ml.spark.logging.BasicLogging
import com.microsoft.ml.spark.stages.{FixedMiniBatchTransformer, FlattenBatch, HasMiniBatcher}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast._
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Model}
import org.apache.spark.ml.linalg.{SQLDataTypes, Vectors, Vector => SVector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import spray.json.DefaultJsonProtocol._

import scala.collection.JavaConverters._

private object CNTKModelUtils extends java.io.Serializable {

  def applyCNTKFunction(model: CNTKFunction,
                        feedDict: Map[Variable, GVV],
                        outputVars: List[Variable],
                        device: DeviceDescriptor): List[GVV] = {

    val valueMap = feedDict.map { case (v, gvv) =>
      gvv match {
        case Left(fvv) => v -> Value.createDenseFloat(v.getShape, fvv, device)
        case Right(dvv) => v -> Value.createDenseDouble(v.getShape, dvv, device)
      }
    }

    val inputDataMap = new UnorderedMapVariableValuePtr()
    valueMap.foreach { case (vr, vl) => inputDataMap.add(vr, vl) }

    val outputDataMap = new UnorderedMapVariableValuePtr()

    outputVars.foreach(ov => outputDataMap.add(ov, null)) //scalastyle:ignore null
    model.evaluate(inputDataMap, outputDataMap, device)

    val out = outputVars.map { ov: Variable =>
      ov.getDataType match {
        case CNTKDataType.Float =>
          val fvv = new FloatVectorVector() //TODO try re-using
        val value = outputDataMap.getitem(ov)
          value.copyVariableValueToFloat(ov, fvv)
          value.delete()
          Left(fvv)
        case CNTKDataType.Double =>
          val dvv = new DoubleVectorVector() //TODO try re-using
        val value = outputDataMap.getitem(ov)
          value.copyVariableValueToDouble(ov, dvv)
          value.delete()
          Right(dvv)
      }
    }

    valueMap.values.foreach(_.delete())
    out
  }

  private def makeInputExtractors(inputMapVar: Map[Int, Variable]) = {
    inputMapVar.map {
      case (colNum, variable) => variable -> {
        variable.getDataType match {
          case CNTKDataType.Float =>
            r: Row => Left(r.getAs[Seq[Seq[Float]]](colNum))
          case CNTKDataType.Double =>
            r: Row => Right(r.getAs[Seq[Seq[Double]]](colNum))
        }
      }
    }
  }

  def applyModel(inputMap: Map[String, Int],
                 broadcastedModel: Broadcast[CNTKFunction],
                 outputMap: Map[String, String])
                (inputRows: Iterator[Row]): Iterator[Row] = {

    if (!inputRows.hasNext) {
      Iterator() // Quickly skip empty partitions
    } else {
      val device = DeviceDescriptor.useDefaultDevice
      //CNTKLib.SetMaxNumCPUThreads(1)
      val m = CNTKExtensions.fromSerializable(broadcastedModel.value).clone(ParameterCloningMethod.Share)

      val inputMapVar = inputMap.map { case (k, v) => v -> m.getInputVar(k) }
      val outputMapVar = outputMap.map { case (k, v) => m.getOutputVar(v) -> k }
      val inputExtractors = makeInputExtractors(inputMapVar)

      val inputGVVs = inputMapVar.map {
        case (_, variable) => variable -> {
          variable.getDataType match {
            case CNTKDataType.Float =>
              Left(new FloatVectorVector())
            case CNTKDataType.Double =>
              Right(new DoubleVectorVector())
          }
        }
      }

      // WARNING: DO NOT simplify this to mapValues,
      // for some reason it calls the inner function more than it should
      val preprocessFunction: Row => Map[Variable, GVV] = {
        { row: Row =>
          inputExtractors.map { case (k, f) =>
            k -> ConversionUtils.toGVV(f(row), inputGVVs(k))
          }
        }
      }

      val outputVars = outputMapVar.keys.toList

      val outputVarVector = new VariableVector()
      outputVars.foreach(outputVarVector.add)
      val of = CNTKLib.Combine(outputVarVector)

      inputRows.map { row =>
        val feedDict = preprocessFunction(row)
        val outputGVVs = applyCNTKFunction(of, feedDict, outputVars, device)
        val resultRow = Row(outputGVVs.map(ConversionUtils.convertGVV): _*)
        val outputRow = Row.merge(row, resultRow)
        outputGVVs.foreach(ConversionUtils.deleteGVV)
        outputRow
      }
    }
  }

}

object CNTKModel extends ComplexParamsReadable[CNTKModel]

class CNTKModel(override val uid: String) extends Model[CNTKModel] with ComplexParamsWritable
  with HasMiniBatcher with Wrappable with BasicLogging {
  logClass()

  override protected lazy val pyInternalWrapper = true

  def this() = this(Identifiable.randomUID("CNTKModel"))

  /** Array of bytes containing the serialized CNTK <code>Function</code>
    *
    * @group param
    */
  val model: CNTKFunctionParam =
    new CNTKFunctionParam(this, "model", "Array of bytes containing the serialized CNTKModel")

  private var broadcastedModelOption: Option[Broadcast[CNTKFunction]] = None

  /** @group setParam */
  def setModel(value: CNTKFunction): this.type = {
    // Free up memory used by the previous model
    // TODO: investigate using destroy()
    broadcastedModelOption.foreach(_.unpersist())
    broadcastedModelOption = None
    set(model, value)
  }

  /** @group getParam */
  def getModel: CNTKFunction = $(model)

  /** @group setParam */
  def setModelLocation(path: String): this.type = {
    val modelBytes = SparkContext.getOrCreate().binaryFiles(path).first()._2.toArray
    setModel(CNTKFunction.loadModelFromBytes(modelBytes))
  }

  val batchInput = new BooleanParam(this, "batchInput",
    "whether to use a batcher")

  setDefault(batchInput -> true)

  def setBatchInput(v: Boolean): this.type = set(batchInput, v)

  def getBatchInput: Boolean = $(batchInput)

  val shapeOutput = new BooleanParam(this, "shapeOutput",
    "whether to shape the output")

  setDefault(shapeOutput -> false)

  def setShapeOutput(v: Boolean): this.type = set(shapeOutput, v)

  def getShapeOutput: Boolean = $(shapeOutput)

  val convertOutputToDenseVector = new BooleanParam(this, "convertOutputToDenseVector",
    "whether to convert the output to dense vectors")

  setDefault(convertOutputToDenseVector -> true)

  def setConvertOutputToDenseVector(v: Boolean): this.type = set(convertOutputToDenseVector, v)

  def getConvertOutputToDenseVector: Boolean = $(convertOutputToDenseVector)

  val feedDict: MapParam[String, String] = new MapParam[String, String](this, "feedDict",
    " Map of CNTK Variable names (keys) and Column Names (values)")

  setDefault(feedDict -> Map((ArgumentPrefix + 0) -> (ArgumentPrefix + 0)))

  def setFeedDict(value: Map[String, String]): this.type = set(feedDict, value)

  def setFeedDict(k: String, v: String): this.type = set(feedDict, Map(k -> v))

  def getFeedDict: Map[String, String] = $(feedDict)

  val fetchDict: MapParam[String, String] = new MapParam[String, String](this, "fetchDict",
    " Map of Column Names (keys) and CNTK Variable names (values)")
  setDefault(fetchDict -> Map((OutputPrefix + 0) -> (OutputPrefix + 0)))

  def setFetchDict(value: Map[String, String]): this.type = set("fetchDict", value)

  def setFetchDict(k: String, v: String): this.type = set(fetchDict, Map(k -> v))

  def getFetchDict: Map[String, String] = $(fetchDict)

  // Alternative Input APIs

  def setInputNodeIndex(value: Int): this.type = {
    val fd = getFeedDict
    if (fd.isEmpty) {
      setFeedDict(ArgumentPrefix + value, ArgumentPrefix + value)
    } else if (fd.size == 1) {
      setFeedDict(ArgumentPrefix + value, fd.values.head)
    } else {
      throw new IllegalArgumentException("existing feed dict has too many elements," +
        " consider using the more expressive feedDict param directly")
    }
  }

  def getInputNodeIndex: Int = {
    val fd = getFeedDict
    if (fd.size == 1) {
      fd.keys.head match {
        case node if node.startsWith(ArgumentPrefix) =>
          node.stripPrefix(ArgumentPrefix).toInt
        case _ => throw new RuntimeException("Feed dict did not have the proper structure")
      }
    } else {
      throw new IllegalArgumentException("existing feed dict has too many elements," +
        " consider using the more expressive feedDict param directly")
    }
  }

  def setInputNode(value: String): this.type = {
    val fd = getFeedDict
    if (fd.isEmpty) {
      setFeedDict(value, value)
    } else if (fd.size == 1) {
      setFeedDict(value, fd.values.head)
    } else {
      throw new IllegalArgumentException("existing feed dict has too many elements," +
        " consider using the more expressive feedDict param directly")
    }
  }

  def getInputNode: String = {
    val fd = getFeedDict
    if (fd.size == 1) {
      fd.keys.head
    } else {
      throw new IllegalArgumentException("existing feed dict has too many elements," +
        " consider using the more expressive feedDict param directly")
    }
  }

  def setInputCol(value: String): this.type = {
    val fd = getFeedDict
    if (fd.isEmpty) {
      setFeedDict(value, value)
    } else if (fd.size == 1) {
      setFeedDict(fd.keys.head, value)
    } else {
      throw new IllegalArgumentException("existing feed dict has too many elements," +
        " consider using the more expressive feedDict param directly")
    }
  }

  def getInputCol: String = {
    val fd = getFeedDict
    if (fd.size == 1) {
      fd.values.head
    } else {
      throw new IllegalArgumentException("existing feed dict has too many elements," +
        " consider using the more expressive feedDict param directly")
    }
  }

  // Alternative Output APIs

  def setOutputNodeIndex(value: Int): this.type = {
    val fd = getFetchDict
    if (fd.isEmpty) {
      setFetchDict(OutputPrefix + value, OutputPrefix + value)
    } else if (fd.size == 1) {
      setFetchDict(fd.keys.head, OutputPrefix + value)
    } else {
      throw new IllegalArgumentException("existing fetch dict has too many elements," +
        " consider using the more expressive fetchDict param directly")
    }
  }

  def getOutputNodeIndex: Int = {
    val fd = getFetchDict
    if (fd.size == 1) {
      fd.values.head match {
        case node if node.startsWith(OutputPrefix) =>
          node.stripPrefix(OutputPrefix).toInt
        case _ => throw new RuntimeException("Fetch dict did not have the proper structure")
      }
    } else {
      throw new IllegalArgumentException("existing fetch dict has too many elements," +
        " consider using the more expressive fetchDict param directly")
    }
  }

  def setOutputNode(value: String): this.type = {
    val fd = getFetchDict
    if (fd.isEmpty) {
      setFetchDict(value, value)
    } else if (fd.size == 1) {
      setFetchDict(fd.keys.head, value)
    } else {
      throw new IllegalArgumentException("existing fetch dict has too many elements," +
        " consider using the more expressive fetchDict param directly")
    }
  }

  def getOutputNode: String = {
    val fd = getFetchDict
    if (fd.size == 1) {
      fd.values.head
    } else {
      throw new IllegalArgumentException("existing fetch dict has too many elements," +
        " consider using the more expressive fetchDict param directly")
    }
  }

  def setOutputCol(value: String): this.type = {
    val fd = getFetchDict
    if (fd.isEmpty) {
      setFetchDict(value, value)
    } else if (fd.size == 1) {
      setFetchDict(value, fd.values.head)
    } else {
      throw new IllegalArgumentException("existing fetch dict has too many elements," +
        " consider using the more expressive fetchDict param directly")
    }
  }

  def getOutputCol: String = {
    val fd = getFetchDict
    if (fd.size == 1) {
      fd.keys.head
    } else {
      throw new IllegalArgumentException("existing fetch dict has too many elements," +
        " consider using the more expressive fetchDict param directly")
    }
  }

  /** Returns the dimensions of the required input */
  def getInputShapes: List[Array[Int]] = {
    getModel.getArguments.asScala.map(_.getShape.getDimensions.map(_.toInt)).toList
  }

  setDefault(miniBatcher -> new FixedMiniBatchTransformer().setBatchSize(10)) //scalastyle:ignore magic.number

  private def getElementType(t: DataType): DataType = {
    t match {
      case ArrayType(et, _) => getElementType(et)
      case et => et
    }
  }

  def transformSchema(schema: StructType): StructType = {
    getFeedDict.foreach { case (_, colName) =>
      val colType = schema(colName).dataType
      val innerTypes = Set(VectorType,
        ArrayType(DoubleType, true),
        ArrayType(FloatType, true),
        ArrayType(DoubleType, false),
        ArrayType(FloatType, false))
      val allowedTypes: Set[DataType] = if (getBatchInput) {
        innerTypes
      } else {
        innerTypes.map(ArrayType(_))
      }
      assert(allowedTypes.contains(colType))
    }

    if (getConvertOutputToDenseVector) {
      getFetchDict.toList.sorted
        .foldLeft(schema) { case (st, (_, colName)) => st.add(colName, VectorType) }
    } else {
      getModel.getOutputSchema(getFetchDict)
        .foldLeft(schema) { case (st, sf) => st.add(sf) }
    }

  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  def rebroadcastCNTKModel(spark: SparkSession): Unit = {
    broadcastedModelOption = Some(spark.sparkContext.broadcast(getModel))
  }

  private val coercionPrefix = s"coerced_$uid"

  private def coerceType(schema: StructType, colName: String, targetElementType: DataType):
  Option[(UserDefinedFunction, String)] = {
    val colType = schema(colName).dataType match {
      case ArrayType(dt, _) => dt
    }

    val funcOpt = (colType, targetElementType) match {
      case (VectorType, DoubleType) =>
        Some({ av: Seq[SVector] => av.map(_.toArray) })
      case (VectorType, FloatType) =>
        Some({ av: Seq[SVector] => av.map(_.toArray.map(_.toFloat)) })
      case (ArrayType(FloatType, _), DoubleType) =>
        Some({ av: Seq[Seq[Float]] => av.map(_.map(_.toDouble)) })
      case (ArrayType(DoubleType, _), FloatType) =>
        Some({ av: Seq[Seq[Double]] => av.map(_.map(_.toFloat)) })
      case (ArrayType(DoubleType, _), DoubleType) =>
        None
      case (ArrayType(FloatType, _), FloatType) =>
        None
      case _ =>
        throw new IllegalArgumentException(s"unsupported column and element type: $colType and $targetElementType")
    }

    funcOpt.map { f =>
      (UDFUtils.oldUdf(f, ArrayType(ArrayType(targetElementType))),
        findUnusedColumnName(coercionPrefix, schema))
    }
  }

  private def coerceDFAndFeedDict(df: DataFrame,
                                  feedDict: Map[String, String]
                                 ): (DataFrame, Map[String, String]) = {
    feedDict.foldLeft((df, feedDict)) {
      case ((dfInternal, fdInternal), (varName, colName)) =>
        val elementType = variableToElementType(getModel.getInputVar(varName))
        coerceType(dfInternal.schema, colName, elementType) match {
          case Some((udfVal, newColName)) =>
            (
              dfInternal.withColumn(newColName, udfVal(col(colName))),
              fdInternal.updated(varName, newColName)
            )
          case None =>
            (dfInternal, feedDict)
        }
    }
  }

  private def coerceOutputDF(unbatchedDF: DataFrame): DataFrame = {
    val floatToDV = UDFUtils.oldUdf({ v: Seq[Float] =>
      Vectors.dense(v.map(_.toDouble).toArray) }, SQLDataTypes.VectorType)
    val doubleToDV = UDFUtils.oldUdf({ v: Seq[Double] =>
      Vectors.dense(v.toArray) }, SQLDataTypes.VectorType)

    if (getConvertOutputToDenseVector) {
      val outputSchema = getModel.getOutputSchema(getFetchDict)
      val outputColumnNames = outputSchema.map(_.name).toSet
      val colsToSelect = unbatchedDF.schema.map {
        case sf if outputColumnNames(sf.name) =>
          sf match {
            case StructField(name, ArrayType(FloatType, _), _, _) =>
              floatToDV(col(name)).alias(name)
            case StructField(name, ArrayType(DoubleType, _), _, _) =>
              doubleToDV(col(name)).alias(name)
            case _ =>
              throw new MatchError("Improper column type")
          }
        case sf => col(sf.name)
      }
      unbatchedDF.select(colsToSelect: _*)
    } else {
      unbatchedDF
    }
  }

  /** Evaluate the model
    *
    * @param dataset the dataset to featurize
    * @return featurized dataset
    */
  def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val spark = dataset.sparkSession
      val df = dataset.toDF()

      transformSchema(df.schema) // Check if the schema is correct

      val batchedDF = if (getBatchInput) {
        getMiniBatcher.transform(df)
      } else {
        df
      }

      val (preprocessedDF, coercedFeedDict) =
        coerceDFAndFeedDict(batchedDF, getFeedDict)

      val columnIndexToVar = coercedFeedDict.map { case (k, v) =>
        k -> preprocessedDF.schema.fieldIndex(v)
      }

      if (broadcastedModelOption.isEmpty) rebroadcastCNTKModel(spark)

      val encoder = RowEncoder(getModel.getOutputSchema(getFetchDict)
        .foldLeft(preprocessedDF.schema) { case (st, sf) => st.add(sf.name, ArrayType(sf.dataType)) }
      )

      val outputDF = preprocessedDF.mapPartitions { it =>
        CNTKModelUtils.applyModel(
          columnIndexToVar,
          broadcastedModelOption.get,
          getFetchDict)(it)
      }(encoder)

      val droppedDF = outputDF.drop(outputDF.columns.filter(_.startsWith(coercionPrefix)): _*)

      val unbatchedDF = if (getBatchInput) {
        new FlattenBatch().transform(droppedDF)
      } else {
        droppedDF
      }
      coerceOutputDF(unbatchedDF)
    })
  }

}
