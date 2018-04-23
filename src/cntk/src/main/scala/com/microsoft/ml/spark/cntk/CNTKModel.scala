// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import com.microsoft.CNTK.CNTKExtensions._
import com.microsoft.CNTK.{DataType => CNTKDataType, SerializableFunction => CNTKFunction, _}
import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, Wrappable}
import com.microsoft.ml.spark.core.env.InternalWrapper
import com.microsoft.ml.spark.core.schema.DatasetExtensions
import org.apache.spark.broadcast._
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

private object CNTKModelUtils extends java.io.Serializable {

  def applyCNTKFunction(model: CNTKFunction,
                        inputFVV: FloatVectorVector,
                        inputVar: Variable,
                        device: DeviceDescriptor): FloatVectorVector = {
    val inputShape    = inputVar.getShape
    val inputVal      = Value.createDenseFloat(inputShape, inputFVV, device)
    val inputDataMap  = new UnorderedMapVariableValuePtr()
    inputDataMap.add(inputVar, inputVal)

    val outputDataMap = new UnorderedMapVariableValuePtr()
    val outputVar     = model.getOutputs.get(0)
    outputDataMap.add(outputVar, null)

    model.evaluate(inputDataMap, outputDataMap, device)

    val outputFVV = new FloatVectorVector()
    outputDataMap.getitem(outputVar).copyVariableValueToFloat(outputVar, outputFVV)
    outputFVV
  }

  private def deleteFVV(fvv: FloatVectorVector): Unit = {
    val size = fvv.size().toInt
    (0 until size).foreach{i =>
      fvv.get(i).delete()
    }
    fvv.delete()
  }

  /** This defines and instantiates an iterator, hasNext and next are the abstract methods that
    * define the interface and inputBuffer and outputBuffer hold the input and output rows so that
    * they can be joined and returned.
    * The logic inside next checks to see if the buffer is empty, and if so sends a new batch off
    * to be evaluated.
    */
  def minibatchIterator(model: CNTKFunction,
                    inputVar: Variable,
                    device: DeviceDescriptor,
                    inputRows: Iterator[Row],
                    minibatchSize: Int,
                    inputIndex: Int): Iterator[Row] ={
    new Iterator[Row] {
      val inputBuffer    = new ListBuffer[Row]()
      val outputBuffer   = new ListBuffer[Row]()
      val inputSize: Int = inputVar.getShape.getTotalSize().toInt

      def hasNext: Boolean = inputRows.hasNext || outputBuffer.nonEmpty

      def next(): Row = {
        if (outputBuffer.isEmpty) {
          inputBuffer ++= inputRows.take(minibatchSize)
          val inputFVV = new FloatVectorVector(inputBuffer.length.toLong)
          inputBuffer.zipWithIndex.foreach { case (row, i) =>
            val fv = new FloatVector(inputSize.toLong)
            row.getSeq[Float](inputIndex).view.zipWithIndex.foreach { case (x, j) =>
              fv.set(j, x)
            }
            inputFVV.set(i, fv)
            fv
          }

          val outputFVV = applyCNTKFunction(model,inputFVV, inputVar, device)
          assert(outputBuffer.isEmpty,
                 "The output row buffer should be empty before new elements are added.")
          outputBuffer ++= toSeqSeq(outputFVV)
            .map(fs => Row(Vectors.dense(fs.map(_.toDouble).toArray)))

          deleteFVV(inputFVV)
          deleteFVV(outputFVV)
        }
        val ret = Row.merge(inputBuffer.head, outputBuffer.head)
        inputBuffer.remove(0)
        outputBuffer.remove(0)
        ret
      }
    }
  }

  def applyModel(inputIndex: Int,
                 broadcastedModel: Broadcast[CNTKFunction],
                 minibatchSize: Int,
                 inputNode: Int,
                 outputNodeName: Option[String],
                 outputNodeIndex: Option[Int])(inputRows: Iterator[Row]): Iterator[Row] = {
    if (!inputRows.hasNext) {
      Iterator() // Quickly skip empty partitions
    } else {
      val device = DeviceDescriptor.useDefaultDevice
      //CNTKLib.SetMaxNumCPUThreads(1)
      val m = fromSerializable(broadcastedModel.value).clone(ParameterCloningMethod.Share)
      val outputNode: Option[CNTKFunction] = (outputNodeName, outputNodeIndex) match {
        case (Some(name), None) =>
          Some(Option(m.findByName(name)).getOrElse(
            throw new IllegalArgumentException(s"Node $name does not exist")))
        case (None, Some(index)) => Some(m.getOutputs.get(index).getOwner)
        case (Some(_), Some(_)) =>
          throw new Exception("Must specify one and only one of outputNodeName or outputNodeIndex")
        case _ => None
      }

      val model = outputNode.map(CNTKLib.AsComposite(_)).getOrElse(m)

      val inputVar = model.getArguments.get(inputNode)
      require(inputVar.getDataType() == CNTKDataType.Float,
        "input variable type is not Float input type")
      minibatchIterator(model, inputVar, device, inputRows, minibatchSize, inputIndex)
    }
  }

  private def toSeqSeq(fvv: FloatVectorVector): Seq[Seq[Float]] = {
    (0 until fvv.size.toInt)
      .map(i => (0 until fvv.get(i).size.toInt).map(j => fvv.get(i).get(j)))
  }
}

object CNTKModel extends ComplexParamsReadable[CNTKModel]

@InternalWrapper
class CNTKModel(override val uid: String) extends Model[CNTKModel] with ComplexParamsWritable
  with HasInputCol with HasOutputCol with Wrappable{

  def this() = this(Identifiable.randomUID("CNTKModel"))

  /** Array of bytes containing the serialized CNTK <code>Function</code>
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
  def setModelLocation(spark: SparkSession, path: String): this.type = {
    val modelBytes = spark.sparkContext.binaryFiles(path).first()._2.toArray
    setModel(CNTKFunction.loadModelFromBytes(modelBytes))
  }

  /** Index of the input node
    * @group param
    */
  val inputNode: IntParam                 = new IntParam(this, "inputNode", "Index of the input node")

  /** @group setParam */
  def setInputNode(value: Int): this.type = set(inputNode, value)

  /** @group getParam */
  def getInputNode: Int                   = $(inputNode)
  setDefault(inputNode -> 0)

  /** Returns the dimensions of the required input */
  def getInputShape: Array[Int] =
    getModel.getInputs.get(getInputNode).getShape.getDimensions.map(_.toInt)

  /** Index of the output node
    * @group param
    */
  val outputNodeIndex: IntParam = new IntParam(this, "outputNodeIndex", "Index of the output node")

  /** @group setParam */
  def setOutputNodeIndex(value: Int): this.type = set(outputNodeIndex, value)

  /** @group getParam */
  def getOutputNodeIndex: Int                   = $(outputNodeIndex)

  /** Name of the output node
    * @group param
    */
  val outputNodeName: Param[String] = new Param(this, "outputNodeName", "Name of the output node")

  /** @group setParam */
  def setOutputNodeName(value: String): this.type = set(outputNodeName, value)

  /** @group getParam */
  def getOutputNodeName: String                   = $(outputNodeName)

  /** Size of minibatches. Must be greater than 0; default is 10
    * @group param
    */
  val miniBatchSize: IntParam =
    new IntParam(this, "miniBatchSize", "Size of minibatches", ParamValidators.gt(0))

  /** @group setParam */
  def setMiniBatchSize(value: Int): this.type = set(miniBatchSize, value)

  /** @group getParam */
  def getMiniBatchSize: Int                   = $(miniBatchSize)
  setDefault(miniBatchSize -> 10)

  def transformSchema(schema: StructType): StructType = schema.add(getOutputCol, VectorType)

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  def rebroadcastCNTKModel(spark: SparkSession): Unit = {
    broadcastedModelOption = Some(spark.sparkContext.broadcast(getModel))
  }

  /** Evaluate the model
    * @param dataset the dataset to featurize
    * @return featurized dataset
    */
  def transform(dataset: Dataset[_]): DataFrame = {
    val spark      = dataset.sparkSession
    val inputIndex = dataset.columns.indexOf(getInputCol)

    if (inputIndex == -1)
      throw new IllegalArgumentException(s"Input column $getInputCol does not exist")

    val setByName  = get(outputNodeName)
    val setByIndex = get(outputNodeIndex)
    if ((setByName.isDefined && setByIndex.isDefined) ||
          (setByName.isEmpty && setByIndex.isEmpty))
      throw new Exception("Must specify one and only one of outputNodeName or outputNodeIndex")

    val coersionOptionUDF = dataset.schema.fields(inputIndex).dataType match {
      case ArrayType(tp, _) =>
        tp match {
          case DoubleType => Some(udf((x: mutable.WrappedArray[Double]) => x.map(_.toFloat)))
          case FloatType  => None
          case _ =>
            throw new IllegalArgumentException(s"improper column type: $tp, need Array[Float]")
        }
      case VectorType => Some(udf((x: DenseVector) => x.toArray.map(_.toFloat)))
    }

    val coercedCol = DatasetExtensions.findUnusedColumnName("coerced")(dataset.columns.toSet)
    val (df, selectedIndex) = coersionOptionUDF match {
      case Some(coersionUDF) =>
        val coercedDF = dataset.toDF().withColumn(coercedCol, coersionUDF(col(getInputCol)))
        (coercedDF, coercedDF.columns.indexOf(coercedCol))
      case None => (dataset.toDF(), inputIndex)
    }

    val inputType = df.schema($(inputCol)).dataType
    if (broadcastedModelOption.isEmpty) rebroadcastCNTKModel(spark)
    val broadcastedModel = broadcastedModelOption.get
    val encoder = RowEncoder(df.schema.add(StructField(getOutputCol, VectorType)))
    val output = df.mapPartitions(
      CNTKModelUtils.applyModel(selectedIndex,
                                broadcastedModel,
                                getMiniBatchSize,
                                getInputNode,
                                get(outputNodeName),
                                get(outputNodeIndex)))(encoder)

    coersionOptionUDF match {
      case Some(_) => output.drop(coercedCol)
      case None    => output
    }
  }

}
