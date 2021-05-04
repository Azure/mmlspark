// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import com.microsoft.ml.spark.core.utils.ClusterUtil
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.shared.{HasFeaturesCol => HasFeaturesColSpark, HasLabelCol => HasLabelColSpark}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}

import scala.concurrent.Await
import scala.concurrent.duration.{Duration, SECONDS}
import scala.language.existentials
import scala.math.min
import scala.util.matching.Regex

trait LightGBMBase[TrainedModel <: Model[TrainedModel]] extends Estimator[TrainedModel]
  with LightGBMParams with HasFeaturesColSpark with HasLabelColSpark with BasicLogging {

  /** Trains the LightGBM model.  If batches are specified, breaks training dataset into batches for training.
    *
    * @param dataset The input dataset to train.
    * @return The trained model.
    */
  protected def train(dataset: Dataset[_]): TrainedModel = {
    logTrain({
      if (getNumBatches > 0) {
        val ratio = 1.0 / getNumBatches
        val datasets = dataset.randomSplit((0 until getNumBatches).map(_ => ratio).toArray)
        datasets.zipWithIndex.foldLeft(None: Option[TrainedModel]) { (model, datasetWithIndex) =>
          if (model.isDefined) {
            setModelString(stringFromTrainedModel(model.get))
          }

          val dataset = datasetWithIndex._1
          val batchIndex = datasetWithIndex._2

          beforeTrainBatch(batchIndex, dataset, model)

          val newModel = innerTrain(dataset, batchIndex)
          afterTrainBatch(batchIndex, dataset, newModel)

          Some(newModel)
        }.get
      } else {
        innerTrain(dataset, batchIndex = 0)
      }
    })
  }

  def beforeTrainBatch(batchIndex: Int, dataset: Dataset[_], model: Option[TrainedModel]): Unit = {
    if (getDelegate.isDefined) {
      val previousBooster: Option[LightGBMBooster] = model match {
        case Some(_) => Option(new LightGBMBooster(stringFromTrainedModel(model.get)))
        case None => None
      }

      getDelegate.get.beforeTrainBatch(batchIndex, log, dataset, previousBooster)
    }
  }

  def afterTrainBatch(batchIndex: Int, dataset: Dataset[_], model: TrainedModel): Unit = {
    if (getDelegate.isDefined) {
      val booster = new LightGBMBooster(stringFromTrainedModel(model))
      getDelegate.get.afterTrainBatch(batchIndex, log, dataset, booster)
    }
  }

  protected def castColumns(dataset: Dataset[_], trainingCols: Array[(String, Seq[DataType])]): DataFrame = {
    val schema = dataset.schema
    // Cast columns to correct types
    dataset.select(
      trainingCols.map {
        case (name, datatypes: Seq[DataType]) => {
          val index = schema.fieldIndex(name)
          // Note: We only want to cast if original column was of numeric type

          schema.fields(index).dataType match {
            case numericDataType: NumericType =>
              // If more than one datatype is allowed, see if any match
              if (datatypes.contains(numericDataType)) {
                dataset(name)
              } else {
                // If none match, cast to the first option
                dataset(name).cast(datatypes.head)
              }

            case _ => dataset(name)
          }
        }
      }: _*
    ).toDF()
  }

  protected def prepareDataframe(dataset: Dataset[_], trainingCols: Array[(String, Seq[DataType])],
                                 numTasks: Int): DataFrame = {
    val df = castColumns(dataset, trainingCols)
    // Reduce number of partitions to number of executor tasks
    /* Note: with barrier execution mode we must use repartition instead of coalesce when
     * running on spark standalone.
     * Using coalesce, we get the error:
     *
     * org.apache.spark.scheduler.BarrierJobUnsupportedRDDChainException:
     * [SPARK-24820][SPARK-24821]: Barrier execution mode does not allow the following
     * pattern of RDD chain within a barrier stage:
     * 1. Ancestor RDDs that have different number of partitions from the resulting
     * RDD (eg. union()/coalesce()/first()/take()/PartitionPruningRDD). A workaround
     * for first()/take() can be barrierRdd.collect().head (scala) or barrierRdd.collect()[0] (python).
     * 2. An RDD that depends on multiple barrier RDDs (eg. barrierRdd1.zip(barrierRdd2)).
     *
     * Without repartition, we may hit the error:
     * org.apache.spark.scheduler.BarrierJobSlotsNumberCheckFailed: [SPARK-24819]:
     * Barrier execution mode does not allow run a barrier stage that requires more
     * slots than the total number of slots in the cluster currently. Please init a
     * new cluster with more CPU cores or repartition the input RDD(s) to reduce the
     * number of slots required to run this barrier stage.
     *
     * Hence we still need to estimate the number of tasks and repartition even when using
     * barrier execution, which is unfortunate as repartition is more expensive than coalesce.
     */
    if (getUseBarrierExecutionMode) {
      val numPartitions = df.rdd.getNumPartitions
      if (numPartitions > numTasks) {
        df.repartition(numTasks)
      } else {
        df
      }
    } else {
      df.coalesce(numTasks)
    }
  }

  protected def getTrainingCols: Array[(String, Seq[DataType])] = {
    val colsToCheck: Array[(Option[String], Seq[DataType])] = Array(
      (Some(getLabelCol), Seq(DoubleType)),
      (Some(getFeaturesCol), Seq(VectorType)),
      (get(weightCol), Seq(DoubleType)),
      (getOptGroupCol, Seq(IntegerType, LongType, StringType)),
      (get(validationIndicatorCol), Seq(BooleanType)),
      (get(initScoreCol), Seq(DoubleType, VectorType)))

    colsToCheck.flatMap { case (col: Option[String], colType: Seq[DataType]) => {
      if (col.isDefined) Some(col.get, colType) else None
    }
    }
  }

  /**
    * Retrieves the categorical indexes in the features column.
    * @param df The dataset to train on.
    * @return the categorical indexes in the features column.
    */
  private def getCategoricalIndexes(df: DataFrame): Array[Int] = {
    val categoricalSlotIndexesArr = get(categoricalSlotIndexes).getOrElse(Array.empty[Int])
    val categoricalSlotNamesArr = get(categoricalSlotNames).getOrElse(Array.empty[String])
    LightGBMUtils.getCategoricalIndexes(df, getFeaturesCol, getSlotNames,
      categoricalSlotIndexesArr, categoricalSlotNamesArr)
  }

  private def validateSlotNames(df: DataFrame, columnParams: ColumnParams, trainParams: TrainParams): Unit = {
    val schema = df.schema
    val featuresSchema = schema.fields(schema.fieldIndex(getFeaturesCol))
    val metadata = AttributeGroup.fromStructField(featuresSchema)
    if (metadata.attributes.isDefined) {
      val slotNamesOpt = TrainUtils.getSlotNames(df.schema,
        columnParams.featuresColumn, metadata.attributes.get.length, trainParams)
      val pattern = new Regex("[\",:\\[\\]{}]")
      slotNamesOpt.foreach(slotNames => {
        val badSlotNames = slotNames.flatMap(slotName =>
          if (pattern.findFirstIn(slotName).isEmpty) None else Option(slotName))
        if (!badSlotNames.isEmpty) {
          val errorMsg = s"Invalid slot names detected in features column: ${badSlotNames.mkString(",")}"
          throw new IllegalArgumentException(errorMsg)
        }
      })
    }
  }

  /**
    * Inner train method for LightGBM learners.  Calculates the number of workers,
    * creates a driver thread, and runs mapPartitions on the dataset.
    * @param dataset The dataset to train on.
    * @param batchIndex In running in batch training mode, gets the batch number.
    * @return The LightGBM Model from the trained LightGBM Booster.
    */
  protected def innerTrain(dataset: Dataset[_], batchIndex: Int): TrainedModel = {
    val sc = dataset.sparkSession.sparkContext
    val numTasksPerExec = ClusterUtil.getNumTasksPerExecutor(dataset, log)
    // By default, we try to intelligently calculate the number of executors, but user can override this with numTasks
    val numTasks =
      if (getNumTasks > 0) getNumTasks
      else {
        val numExecutorTasks = ClusterUtil.getNumExecutorTasks(dataset, numTasksPerExec, log)
        min(numExecutorTasks, dataset.rdd.getNumPartitions)
      }
    // Only get the relevant columns
    val trainingCols = getTrainingCols

    val df = prepareDataframe(dataset, trainingCols, numTasks)

    val (inetAddress, port, future) =
      LightGBMUtils.createDriverNodesThread(numTasks, df, log, getTimeout, getUseBarrierExecutionMode,
        getDriverListenPort)

    /* Run a parallel job via map partitions to initialize the native library and network,
     * translate the data to the LightGBM in-memory representation and train the models
     */
    val encoder = Encoders.kryo[LightGBMBooster]

    val trainParams = getTrainParams(numTasks, getCategoricalIndexes(df), dataset)
    log.info(s"LightGBM parameters: ${trainParams.toString()}")
    val networkParams = NetworkParams(getDefaultListenPort, inetAddress, port, getUseBarrierExecutionMode)
    val (trainingData, validationData) =
      if (get(validationIndicatorCol).isDefined && dataset.columns.contains(getValidationIndicatorCol))
        (df.filter(x => !x.getBoolean(x.fieldIndex(getValidationIndicatorCol))),
          Some(sc.broadcast(preprocessData(df.filter(x =>
            x.getBoolean(x.fieldIndex(getValidationIndicatorCol)))).collect())))
      else (df, None)
    val preprocessedDF = preprocessData(trainingData)
    val schema = preprocessedDF.schema
    val columnParams = ColumnParams(getLabelCol, getFeaturesCol, get(weightCol), get(initScoreCol), getOptGroupCol)
    validateSlotNames(preprocessedDF, columnParams, trainParams)
    val mapPartitionsFunc = TrainUtils.trainLightGBM(batchIndex, networkParams, columnParams, validationData, log,
      trainParams, numTasksPerExec, schema)(_)
    val lightGBMBooster =
      if (getUseBarrierExecutionMode) {
        preprocessedDF.rdd.barrier().mapPartitions(mapPartitionsFunc).reduce((booster1, _) => booster1)
      } else {
        preprocessedDF.mapPartitions(mapPartitionsFunc)(encoder).reduce((booster1, _) => booster1)
      }
    // Wait for future to complete (should be done by now)
    Await.result(future, Duration(getTimeout, SECONDS))
    getModel(trainParams, lightGBMBooster)
  }

  /** Optional group column for Ranking, set to None by default.
    *
    * @return None
    */
  protected def getOptGroupCol: Option[String] = None

  /** Gets the trained model given the train parameters and booster.
    *
    * @return trained model.
    */
  protected def getModel(trainParams: TrainParams, lightGBMBooster: LightGBMBooster): TrainedModel

  /** Gets the training parameters.
    *
    * @return train parameters.
    */
  protected def getTrainParams(numTasks: Int, categoricalIndexes: Array[Int], dataset: Dataset[_]): TrainParams

  protected def stringFromTrainedModel(model: TrainedModel): String

  /** Allow algorithm specific preprocessing of dataset.
    *
    * @param dataset The dataset to preprocess prior to training.
    * @return The preprocessed data.
    */
  protected def preprocessData(dataset: DataFrame): DataFrame = dataset
}
