// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.vowpalwabbit.spark.{VowpalWabbitExample, VowpalWabbitMurmur, VowpalWabbitNative}

object VowpalWabbitUtil {
  /**
    * Generate namespace info (hash, feature group, field index) for supplied columns.
    *
    * @param schema data frame schema to lookup column indices.
    * @return
    */
  def generateNamespaceInfos(schema: StructType, hashSeed: Int, cols: Seq[String]): Array[NamespaceInfo] =
    cols
      .map(col => generateNamespaceInfo(schema, hashSeed, col))
      .toArray

  def generateNamespaceInfo(schema: StructType, hashSeed: Int, col: String): NamespaceInfo =
    NamespaceInfo(VowpalWabbitMurmur.hash(col, hashSeed), col.charAt(0), schema.fieldIndex(col))

  def addFeaturesToExample(featureColIndices: Seq[NamespaceInfo],
                           row: Row,
                           ex: VowpalWabbitExample): Unit =
    for (ns <- featureColIndices)
      addFeaturesToExample(row.getAs[Vector](ns.colIdx), ns, ex)

  private def addFeaturesToExample(features: Vector,
                                   ns: NamespaceInfo,
                                   ex: VowpalWabbitExample): Unit =
    features match {
      case dense: DenseVector => ex.addToNamespaceDense(ns.featureGroup,
        ns.hash, dense.values)
      case sparse: SparseVector => ex.addToNamespaceSparse(ns.featureGroup,
        sparse.indices, sparse.values)
    }

  private def createSharedExample(row: Row,
                                  sharedNamespaces: Array[NamespaceInfo],
                                  exampleStack: ExampleStack): VowpalWabbitExample = {
    val sharedExample = exampleStack.getOrCreateExample()
    // Mark example as shared.
    sharedExample.setSharedLabel()
    // Add all of the namespaces into this VW example.
    addFeaturesToExample(sharedNamespaces, row, sharedExample)

    sharedExample
  }

  def prepareMultilineExample[T](row: Row,
                                 actionNamespaceInfos: Array[NamespaceInfo],
                                 sharedNamespaceInfos: Array[NamespaceInfo],
                                 vw: VowpalWabbitNative,
                                 exampleStack: ExampleStack,
                                 fun: Array[VowpalWabbitExample] => T): T = {
    // first example is the shared feature example
    val sharedExample = createSharedExample(row, sharedNamespaceInfos, exampleStack)

    // transfer actions
    val actionsForZerothNamespace = row.getAs[Seq[Vector]](actionNamespaceInfos.head.colIdx)

    // Seq[Seq[Vector]] - each features column is a Seq[Vector]
    // first index  ... namespaces
    // second index ... actions
    val actionFeaturesForEachNamespace = actionNamespaceInfos.map(
      namespaceInfo => row.getAs[Seq[Vector]](namespaceInfo.colIdx).toArray)

    // loop over actions
    val examples = (for (actionIdx <- actionsForZerothNamespace.indices) yield {
      val vwExample = exampleStack.getOrCreateExample()

      // loop over namespaces
      for ((namespace, namespaceIndex) <- actionNamespaceInfos.zipWithIndex)
        addFeaturesToExample(actionFeaturesForEachNamespace(namespaceIndex)(actionIdx), namespace, vwExample)

      vwExample
    }).toArray // make sure it materializes

    val allExamples = sharedExample +: examples

    try {
      // execute function
      fun(allExamples)
    }
    finally {
      // perform cleanup
      for (ex <- allExamples)
        exampleStack.returnExample(ex)
    }
  }
}
