// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.TaskContext
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}

class StratifiedRepartitionSuite extends TestBase with TransformerFuzzing[StratifiedRepartition] {

  import spark.implicits._

  val values = "values"
  val colors = "colors"
  val const = "const"

  lazy val input: DataFrame = Seq(
    (0, "Blue", 2),
    (0, "Red", 2),
    (0, "Green", 2),
    (1, "Purple", 2),
    (1, "Orange", 2),
    (1, "Indigo", 2),
    (2, "Violet", 2),
    (2, "Black", 2),
    (2, "White", 2),
    (3, "Gray", 2),
    (3, "Yellow", 2),
    (3, "Cerulean", 2)
  ).toDF(values, colors, const)

  test("Assert doing a stratified repartition will ensure all keys exist across all partitions") {
    val inputSchema = new StructType()
      .add(values, IntegerType).add(colors, StringType).add(const, IntegerType)
    val inputEnc = RowEncoder(inputSchema)
    val valuesFieldIndex = inputSchema.fieldIndex(values)
    val numPartitions = 3
    val trainData = input.repartition(numPartitions).select(values, colors, const)
      .mapPartitions(iter => {
        val ctx = TaskContext.get
        val partId = ctx.partitionId
        // Remove all instances of 0 class on partition 1
        if (partId == 1) {
          iter.flatMap(row => {
            if (row.getInt(valuesFieldIndex) <= 0)
              None
            else Some(row)
          })
        } else {
          // Add back at least 3 instances on other partitions
          val oneOfEachExample = List(Row(0, "Blue", 2), Row(1, "Purple", 2), Row(2, "Black", 2), Row(3, "Gray", 2))
          iter.toList.union(oneOfEachExample).union(oneOfEachExample).union(oneOfEachExample).toIterator
        }
      })(inputEnc).cache()
    // Some debug to understand what data is on which partition
    trainData.foreachPartition { rows: Iterator[Row] =>
      rows.foreach { row =>
        val ctx = TaskContext.get
        val partId = ctx.partitionId
        println(s"Row: $row partition id: $partId")
      }
    }
    val stratifiedInputData = new StratifiedRepartition().setLabelCol(values)
      .setMode(SPConstants.Equal).transform(trainData)
    // Assert stratified data contains all keys across all partitions, with extra count
    // for it to be evaluated
    stratifiedInputData
      .mapPartitions(iter => {
        val actualLabels = iter.map(row => row.getInt(valuesFieldIndex))
          .toArray.distinct.sorted.toList
        val expectedLabels = (0 to 3).toList
        if (actualLabels != expectedLabels)
          throw new Exception(s"Missing labels, actual: $actualLabels, expected: $expectedLabels")
        iter
      })(inputEnc).count()
    val stratifiedMixedInputData = new StratifiedRepartition().setLabelCol(values)
      .setMode(SPConstants.Mixed).transform(trainData)
    assert(stratifiedMixedInputData.count() >= trainData.count())
    val stratifiedOriginalInputData = new StratifiedRepartition().setLabelCol(values)
      .setMode(SPConstants.Original).transform(trainData)
    assert(stratifiedOriginalInputData.count() == trainData.count())
  }

  def testObjects(): Seq[TestObject[StratifiedRepartition]] = List(new TestObject(
    new StratifiedRepartition().setLabelCol(values).setMode(SPConstants.Equal), input))

  def reader: MLReadable[_] = StratifiedRepartition
}
