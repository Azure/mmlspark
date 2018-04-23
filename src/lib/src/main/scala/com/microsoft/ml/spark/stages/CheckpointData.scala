// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.MMLParams
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage._

trait CheckpointDataParams extends MMLParams {

  /** Persist to disk as well as memory. Storage level is MEMORY_AND_DISK if true, else MEMORY_ONLY.
    * Default is false (MEMORY_ONLY)
    * @group param
    */
  val diskIncluded: BooleanParam = BooleanParam(this, "diskIncluded", "Persist to disk as well as memory", false)

  /** @group getParam */
  final def getDiskIncluded: Boolean = $(diskIncluded)

  /** @group setParam */
  def setDiskIncluded(value: Boolean): this.type = set(diskIncluded, value)

  /** Reverse the cache operatation; unpersist a cached dataset. Default is false
    * @group param
    */
  val removeCheckpoint: BooleanParam = BooleanParam(this, "removeCheckpoint", "Unpersist a cached dataset", false)

  /** @group getParam */
  final def getRemoveCheckpoint: Boolean = $(removeCheckpoint)

  /** @group setParam */
  def setRemoveCheckpoint(value: Boolean): this.type = set(removeCheckpoint, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }

}

/** Cache the dataset at this point to memory or memory and disk
  * @param uid The id of the module
  */
class CheckpointData(override val uid: String) extends Transformer with CheckpointDataParams {

  def this() = this(Identifiable.randomUID("CheckpointData"))

  /** Apply the transform to the dataset to persist or unpersist the data
    * @param dataset
    * @return dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    if ($(removeCheckpoint)) {
      CheckpointData.clearCache(dataset, false)
    } else {
      CheckpointData.cache(dataset, $(diskIncluded), false)
    }
  }

  /** Transform the schema
    * @param schema
    * @return new schema
    */
  def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  def copy(extra: ParamMap): CheckpointData = defaultCopy(extra)

}

/** Cache the dataset to memory or memory and disk. */
object CheckpointData extends DefaultParamsReadable[CheckpointData]{

  /** Clear (Unpersist) the cached dataset
    * @param ds The dataset to be unpersisted
    * @param blocking True if this operation should be a blocking operation
    * @return
    */
  def clearCache(ds: Dataset[_], blocking: Boolean): DataFrame = {
    ds.unpersist(blocking)
    ds.toDF
  }

  /** Cache the dataset
    * @param ds dataset to be cahced
    * @param disk save to disk as well as memory when true
    * @param serialized serialize data
    * @return
    */
  def cache(ds: Dataset[_], disk: Boolean, serialized: Boolean): DataFrame = {
    ds.persist(if (disk && serialized) StorageLevel.MEMORY_AND_DISK_SER
               else if (serialized)    StorageLevel.MEMORY_ONLY_SER
               else if (disk)          StorageLevel.MEMORY_AND_DISK
               else                    StorageLevel.MEMORY_ONLY)
    ds.toDF
  }

  /** Cache to Hive
    * @param ds dataset to be cached
    * @param dbName Hive db name
    * @param tableName Hive table name
    * @return
    */
  def persistToHive(ds: Dataset[_], dbName: String, tableName: String): DataFrame = {
    ds.write.mode("overwrite").saveAsTable(dbName + "." + tableName)
    ds.toDF
  }

}
