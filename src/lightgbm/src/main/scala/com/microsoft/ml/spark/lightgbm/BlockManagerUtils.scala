// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark

import org.apache.spark.sql.Dataset
import org.apache.spark.storage.BlockManager

object BlockManagerUtils {
  /** Returns the block manager from the dataframe's spark context.
    * @param data The dataframe to get the block manager from.
    * @return The block manager.
    */
  def getBlockManager(data: Dataset[_]): BlockManager = {
    data.sparkSession.sparkContext.env.blockManager
  }
}
