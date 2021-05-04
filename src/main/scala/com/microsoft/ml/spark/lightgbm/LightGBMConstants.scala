// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

object LightGBMConstants {
  /** The port for LightGBM Driver server, 0 (random)
    */
  val DefaultDriverListenPort = 0
  /** The default port for LightGBM network initialization
    */
  val DefaultLocalListenPort = 12400
  /** Max port range available on machine, 65535
    */
  val MaxPort: Int = (2 << 15) - 1
  /** The default timeout for LightGBM network initialization
    */
  val DefaultListenTimeout = 120
  /** Default buffer length for model
    */
  val DefaultBufferLength: Int = 10000
  /** Default top_k value for LightGBM voting parallel
    */
  val DefaultTopK: Int = 20
  /** Lambdarank ranking objective
    */
  val RankObjective: String = "lambdarank"
  /** Binary classification objective
    */
  val BinaryObjective: String = "binary"
  /** Multiclass classification objective
    */
  val MulticlassObjective: String = "multiclass"
  /** Ignore task status, used to ignore tasks that get empty partitions
    */
  val IgnoreStatus: String = "ignore"
  /** Barrier execution flag telling driver that all tasks have completed
    * sending port and host information
    */
  val FinishedStatus: String = "finished"
  /** The default start iteration
    */
  val DefaultStartIteration: Int = 0
  /** The default num iterations for prediction
    */
  val DefaultNumIterations: Int = -1
}
