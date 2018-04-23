// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.test.datagen

import com.microsoft.ml.spark.core.test.base.TestBase

/** Verifies generating a dataset using the api. */
class VerifyGenerateDataset extends TestBase {

  test("Smoke test to verify that generating a dataset works") {
    val numRows = 10
    val numCols = 20
    val numSlotsPerVectorCol = Array(15, 15)
    val seed = 1337
    val df = GenerateDataset
      .generateDataset(session, new BasicDatasetGenerationConstraints(numRows, numCols, numSlotsPerVectorCol),
        seed.toLong)
    assert(df.columns.length == numCols)
    assert(df.count == numRows)
  }

  test("Verify that the generated dataset is always the same") {
    val numRows = 10
    val numCols = 20
    val numSlotsPerVectorCol = Array(15, 15)
    val seed = 1337

    val datasets = (0 to 10).map(i => GenerateDataset
      .generateDataset(session, new BasicDatasetGenerationConstraints(numRows, numCols, numSlotsPerVectorCol),
        seed.toLong))

    assert(datasets.forall(df => verifyResult(df, datasets(0))), "Datasets must be equal")
  }

  test("Verify that for different seed, you will get different datasets") {
    val numRows = 25
    val numCols = 10

    val datasets = (0 to 10).map(i => GenerateDataset
      .generateDataset(session, new BasicDatasetGenerationConstraints(numRows, numCols, Array()), i.toLong))

    assert(!datasets.forall(df => verifyResult(df, datasets(0))), "Datasets must not be equal for different seeds")
  }

}
