// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.DataFrame

import scala.language.existentials

trait RankingTestBase extends TestBase {
  lazy val userCol = "customerIDOrg"
  lazy val itemCol = "itemIDOrg"
  lazy val ratingCol = "rating"

  lazy val userColIndex = "customerID"
  lazy val itemColIndex = "itemID"

  lazy val ratings: DataFrame = session
    .createDataFrame(Seq(
      ("11", "Movie 01", 2),
      ("11", "Movie 03", 1),
      ("11", "Movie 04", 5),
      ("11", "Movie 05", 3),
      ("11", "Movie 06", 4),
      ("11", "Movie 07", 1),
      ("11", "Movie 08", 5),
      ("11", "Movie 09", 3),
      ("22", "Movie 01", 4),
      ("22", "Movie 02", 5),
      ("22", "Movie 03", 1),
      ("22", "Movie 05", 3),
      ("22", "Movie 06", 3),
      ("22", "Movie 07", 5),
      ("22", "Movie 08", 1),
      ("22", "Movie 10", 3),
      ("33", "Movie 01", 4),
      ("33", "Movie 03", 1),
      ("33", "Movie 04", 5),
      ("33", "Movie 05", 3),
      ("33", "Movie 06", 4),
      ("33", "Movie 08", 1),
      ("33", "Movie 09", 5),
      ("33", "Movie 10", 3),
      ("44", "Movie 01", 4),
      ("44", "Movie 02", 5),
      ("44", "Movie 03", 1),
      ("44", "Movie 05", 3),
      ("44", "Movie 06", 4),
      ("44", "Movie 07", 5),
      ("44", "Movie 08", 1),
      ("44", "Movie 10", 3)))
    .toDF(userCol, itemCol, ratingCol)
    .dropDuplicates()
    .cache()

  lazy val recommendationIndexer: RecommendationIndexer = new RecommendationIndexer()
    .setUserInputCol(userCol)
    .setUserOutputCol(userColIndex)
    .setItemInputCol(itemCol)
    .setItemOutputCol(itemColIndex)
    .setRatingCol(ratingCol)
    .setHandleInvalid("keep")

  val als = new ALS()
  als
    .setUserCol(recommendationIndexer.getUserOutputCol)
    .setItemCol(recommendationIndexer.getItemOutputCol)
    .setRatingCol(ratingCol)

  val sar = new SAR()
  sar.setUserCol(recommendationIndexer.getUserOutputCol)
    .setItemCol(recommendationIndexer.getItemOutputCol)
    .setRatingCol(ratingCol)
    .setTimeCol("timestamp")

  lazy val paramGrid: Array[ParamMap] = new ParamGridBuilder()
    .addGrid(als.regParam, Array(1.0))
    .build()

  lazy val evaluator: RankingEvaluator = new RankingEvaluator()
    .setK(3)
    .setNItems(10)

  lazy val transformedDf: DataFrame = recommendationIndexer.fit(ratings).transform(ratings)

  lazy val adapter: RankingAdapter = new RankingAdapter()
    .setK(evaluator.getK)
    .setRecommender(als)

  lazy val rankingTrainValidationSplit: RankingTrainValidationSplit = new RankingTrainValidationSplit()
    .setEvaluator(evaluator)
    .setEstimator(als)
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)
    .setUserCol(recommendationIndexer.getUserOutputCol)
    .setItemCol(recommendationIndexer.getItemOutputCol)
    .setRatingCol(ratingCol)
}
