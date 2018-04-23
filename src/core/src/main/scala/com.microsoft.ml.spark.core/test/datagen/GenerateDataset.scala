// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.datagen

import com.microsoft.ml.spark.core.test.datagen.ColumnOptions.ColumnOptions
import com.microsoft.ml.spark.core.test.datagen.DataOptions.DataOptions
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import scala.util.Random

/** Defines methods to generate a random spark DataFrame dataset based on given options. */
object GenerateDataset {

  /** Generates a random Spark DataFrame given a set of dataset generation constraints.
    * @param sparkSession The spark session.
    * @param datasetGenerationConstraints The dataset generation constraints to use.
    * @param seed The random seed.
    * @return A randomly generated dataset.
    */
  def generateDataset(sparkSession: SparkSession,
                      datasetGenerationConstraints: HasDatasetGenerationConstraints,
                      seed: Long): DataFrame = {
    generateDatasetFromOptions(sparkSession, Map[Int, DatasetOptions](), datasetGenerationConstraints, seed)
  }

  /** Generates a random Spark DataFrame given a map of index to DataGenerationOptions.
    * @param sparkSession The spark session.
    * @param indexToOptions The map of indexes to DataGenerationOptions.
    * @param datasetGenerationConstraints The constraints for generating the dataset.
    * @param seed The random seed.
    * @return The randomly generated dataset.
    */
  def generateDatasetFromOptions(sparkSession: SparkSession,
                                 indexToOptions: Map[Int, DatasetOptions],
                                 datasetGenerationConstraints: HasDatasetGenerationConstraints,
                                 seed: Long): DataFrame = {

    val random = new Random(seed)
    val numCols: Int = datasetGenerationConstraints.numCols
    val datasetGenerationOptions = (1 to numCols).
      map(index =>
        if (indexToOptions.contains(index)) indexToOptions(index)
        else new DatasetOptions(ColumnOptions.values,
                                          DataOptions.values,
                                          new DatasetMissingValuesGenerationOptions(0.5,
                                                                                    ColumnOptions.values,
                                                                                    DataOptions.values)))
    // Get random options chosen from given valid space-dimension complex
    val chosenOptions:Array[(ColumnOptions, DataOptions)] =
      datasetGenerationOptions.toArray.map(option => chooseOptions(option, random))

    val rdd = RandomRDDs.randomRDD[Row](sparkSession.sparkContext,
      new RandomRowGeneratorCombiner(chosenOptions.map(option => new RandomRowGenerator(option._1, option._2))),
      datasetGenerationConstraints.numRows.toLong, 1, random.nextLong())
    sparkSession.createDataFrame(rdd, getSchemaFromOptions(chosenOptions, random))
  }

  def getOptionsFromSchema(schema: StructType): Map[Int, DatasetOptions] = {
    val datasetOptions = schema.map(sf => DatasetOptions(ColumnOptions.Scalar, getOptionsFromDataType(sf.dataType)))
    datasetOptions.zipWithIndex.map(kvp => (kvp._2 + 1, kvp._1)).toMap
  }

  private def chooseOptions(options: DatasetOptions, random: Random) = {
    val (optionsColumnArray, optionsDataArray) = (options.columnTypes.toArray, options.dataTypes.toArray)
    (optionsColumnArray(random.nextInt(optionsColumnArray.length)),
      optionsDataArray(random.nextInt(optionsDataArray.length)))
  }

  private def getSchemaFromOptions(chosenOptions: Array[(ColumnOptions, DataOptions)],
                                   random: Random): StructType = {
    val generateDataType = new GenerateDataType(random)
    new StructType(
      chosenOptions
        .map(option => getDataTypeFromOptions(option._2))
        .map(dataType => StructField(generateDataType.nextString, dataType)))
  }

  lazy val dataTypeToOptions: Map[DataOptions, DataType] = Map(
    DataOptions.String -> StringType,
    DataOptions.Timestamp -> TimestampType,
    DataOptions.Short -> ShortType,
    DataOptions.Int -> IntegerType,
    DataOptions.Boolean -> BooleanType,
    DataOptions.Byte -> ByteType,
    DataOptions.Date -> DateType,
    DataOptions.Double -> DoubleType
  )

  lazy val optionsToDataType: Map[DataType, DataOptions] = dataTypeToOptions.map(kvp => (kvp._2, kvp._1))

  private def getDataTypeFromOptions(data: DataOptions): DataType = {
    if (dataTypeToOptions.contains(data)) {
      dataTypeToOptions(data)
    } else {
      throw new Exception("The type does not exist in spark: " + data)
    }
  }

  private def getOptionsFromDataType(data: DataType): DataOptions = {
    if (optionsToDataType.contains(data)) {
      optionsToDataType(data)
    } else {
      throw new Exception("The corresponding option does not exist for spark data type: " + data)
    }
  }

}
