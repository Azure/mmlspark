// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import scala.language.existentials

class SARSpec extends RankingTestBase with EstimatorFuzzing[SAR] {
  override def testObjects(): List[TestObject[SAR]] = {
    List(
      new TestObject(new SAR()
        .setUserCol(recommendationIndexer.getUserOutputCol)
        .setItemCol(recommendationIndexer.getItemOutputCol)
        .setRatingCol(ratingCol), transformedDf)
    )
  }

  override def reader: SAR.type = SAR

  override def modelReader: SARModel.type = SARModel

  test("SAR") {

    val algo = sar
      .setSupportThreshold(1)
      .setSimilarityFunction("jacccard")
      .setActivityTimeFormat("EEE MMM dd HH:mm:ss Z yyyy")

    val adapter: RankingAdapter = new RankingAdapter()
      .setK(5)
      .setRecommender(algo)

    val recopipeline = new Pipeline()
      .setStages(Array(recommendationIndexer, adapter))
      .fit(ratings)

    val output = recopipeline.transform(ratings)

    val evaluator: RankingEvaluator = new RankingEvaluator()
      .setK(5)
      .setNItems(10)

    assert(evaluator.setMetricName("ndcgAt").evaluate(output) == 0.7168486344464263)
    assert(evaluator.setMetricName("fcp").evaluate(output) == 0.05000000000000001)
    assert(evaluator.setMetricName("mrr").evaluate(output) == 1.0)

    val users: DataFrame = session
      .createDataFrame(Seq(("0", "0"), ("1", "1")))
      .toDF(userColIndex, itemColIndex)

    val recs = recopipeline.stages(1).asInstanceOf[RankingAdapterModel].getRecommenderModel
        .asInstanceOf[SARModel].recommendForUserSubset(users, 10)
    assert(recs.count == 2)
  }

  val testFile: String = getClass.getResource("/demoUsage.csv.gz").getPath
  val demoCatalog: String = getClass.getResource("/demoCatalog.csv.gz").getPath
  val sim_count1: String = getClass.getResource("/sim_count1.csv.gz").getPath
  val sim_lift1: String = getClass.getResource("/sim_lift1.csv.gz").getPath
  val sim_jac1: String = getClass.getResource("/sim_jac1.csv.gz").getPath
  val sim_count3: String = getClass.getResource("/sim_count3.csv.gz").getPath
  val sim_lift3: String = getClass.getResource("/sim_lift3.csv.gz").getPath
  val sim_jac3: String = getClass.getResource("/sim_jac3.csv.gz").getPath
  val user_aff: String = getClass.getResource("/user_aff.csv.gz").getPath
  val userpred_count3: String = getClass.getResource("/userpred_count3_userid_only.csv.gz").getPath
  val userpred_lift3: String = getClass.getResource("/userpred_lift3_userid_only.csv.gz").getPath
  val userpred_jac3: String = getClass.getResource("/userpred_jac3_userid_only.csv.gz").getPath

  private lazy val tlcSampleData: DataFrame = session.read
    .option("header", "true") //reading the headers
    .option("inferSchema", "true")
    .csv(testFile).na.drop.cache

  private lazy val demoCatalogDataFrame: DataFrame = session.read
    .option("header", "true") //reading the headers
    .option("inferSchema", "true")
    .csv(demoCatalog).na.drop.cache

  test("tlc test sim count1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "cooc", sim_count1, user_aff))

  test("tlc test sim lift1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "lift", sim_lift1, user_aff))

  test("tlc test sim jac1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "jaccard", sim_jac1, user_aff))

  test("tlc test sim count3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "cooc", sim_count3, user_aff))

  test("tlc test sim lift3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "lift", sim_lift3, user_aff))

  test("tlc test sim jac3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "jaccard", sim_jac3, user_aff))

  test("tlc test userpred count3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "cooc", sim_count3, user_aff, userpred_count3))

  test("tlc test userpred lift3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "lift", sim_lift3, user_aff, userpred_lift3))

  test("tlc test userpred jac3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "jaccard", sim_jac3, user_aff, userpred_jac3))

  test("tlc test sim+cold count3")(
    SarTLCSpec.test_affinity_matrices_cold(tlcSampleData, 3, "cooc", sim_count3, user_aff, demoCatalogDataFrame))

class SARModelSpec extends RankingTestBase with TransformerFuzzing[SARModel] {
  override def testObjects(): Seq[TestObject[SARModel]] = {
    List(
      new TestObject(new SAR()
        .setUserCol(recommendationIndexer.getUserOutputCol)
        .setItemCol(recommendationIndexer.getItemOutputCol)
        .setRatingCol(ratingCol)
        .fit(transformedDf), transformedDf)
    )
  }

    override def reader: MLReadable[_] = SARModel
  }

  object SarTLCSpec extends RankingTestBase {
    override lazy val userCol = "userId"
    override lazy val itemCol = "productId"
    override lazy val ratingCol = "rating"

    override lazy val userColIndex = "customerID"
    override lazy val itemColIndex = "itemID"

  def test_affinity_matrices(tlcSampleData: DataFrame, threshold: Int, similarityFunction: String, simFile: String,
    user_aff: String):
  (SARModel, RecommendationIndexerModel) = {

      val ratings = tlcSampleData

    val recommendationIndexerModel = recommendationIndexer.fit(ratings)
    val transformedDf = recommendationIndexerModel.transform(ratings)

    val itemMap = recommendationIndexerModel.getItemIndex

      val model = sar
        .setSupportThreshold(threshold)
        .setSimilarityFunction(similarityFunction)
        .setStartTime("2015/06/09T19:39:37")
        .setStartTimeFormat("yyyy/MM/dd'T'h:mm:ss")
        .fit(transformedDf)

      val simMap = model.getItemDataFrame.collect().map(row => {
        val itemI = itemMap.getOrElse(row.getDouble(0).toInt, "-1")
        val similarityVectorMap = row.getList(1).toArray.zipWithIndex.map(t => (itemMap.getOrElse(t._2, "-1"), t._1))
          .toMap
        itemI -> similarityVectorMap
      }).toMap

    val itemAff = session.read.option("header", "true").csv(simFile)
    itemAff.collect().foreach(row => {
      val itemI = row.getString(0)
      itemAff.drop("_c0").schema.fieldNames.foreach(itemJ => {
        val groundTrueScore = row.getAs[String](itemJ).toFloat
        val sparkSarScore = simMap.getOrElse(itemI, Map()).getOrElse(itemJ, "-1.0")
        assert(groundTrueScore == sparkSarScore)
      })
    })
    (model, recommendationIndexerModel)
  }

    def test_product_recommendations(tlcSampleData: DataFrame, threshold: Int, similarityFunction: String,
      simFile: String, user_aff: String,
      userPredFile: String): Unit = {

    val (model, recommendationIndexerModel) = test_affinity_matrices(tlcSampleData, threshold, similarityFunction,
      simFile,
      user_aff)

    val recoverUser = recommendationIndexerModel.recoverUser()
    val recoverItem = recommendationIndexerModel.recoverItem()

      val usersProducts = tlcSampleData
        .filter(col("userId") === "0003000098E85347")
        .select("productId")
        .distinct()
        .collect()
        .map(_.getString(0))

      val usersProductsBC = session.sparkContext.broadcast(usersProducts)

    val itemMapBC = session.sparkContext.broadcast(recommendationIndexerModel.getItemIndex)

    val filterScore = udf((items: Seq[Int], ratings: Seq[Float]) => {
      items.zipWithIndex
        .filter(p => {
          val itemId = itemMapBC.value.getOrElse[String](p._1, "-1")
          val bol = usersProductsBC.value.contains(itemId)
          !bol
        }).map(p => (p._1, ratings.toList(p._2)))
    })

      val row = model.recommendForAllUsers(10 + usersProducts.length)
        .select(col("customerID"), filterScore(col("recommendations.itemID"), col("recommendations.rating")) as
          "recommendations")
        .select(col("customerID"), col("recommendations._1") as "itemID", col("recommendations._2") as "rating")
        .select(
          recoverUser(col("customerID")) as "customerID",
          recoverItem(col("itemID")(0)) as "rec1",
          recoverItem(col("itemID")(1)) as "rec2",
          recoverItem(col("itemID")(2)) as "rec3",
          recoverItem(col("itemID")(3)) as "rec4",
          recoverItem(col("itemID")(4)) as "rec5",
          recoverItem(col("itemID")(5)) as "rec6",
          recoverItem(col("itemID")(6)) as "rec7",
          recoverItem(col("itemID")(7)) as "rec8",
          recoverItem(col("itemID")(8)) as "rec9",
          recoverItem(col("itemID")(9)) as "rec10",
          col("rating")(0) as "score1",
          col("rating")(1) as "score2",
          col("rating")(2) as "score3",
          col("rating")(3) as "score4",
          col("rating")(4) as "score5",
          col("rating")(5) as "score6",
          col("rating")(6) as "score7",
          col("rating")(7) as "score8",
          col("rating")(8) as "score9",
          col("rating")(9) as "score10")
        .filter(col("customerID") === "0003000098E85347")
        .take(1)

      val answer = session.read.option("header", "true").csv(userPredFile).collect()

      assert(row(0).getString(0) == "0003000098E85347", "Assert Customer ID's Match")
      (0 to 10).foreach(i => assert(row(0).getString(i) == answer(0).getString(i)))
      (11 to 20).foreach(i => assert("%.3f".format(row(0).getFloat(i)) == "%.3f".format(answer(0).getString(i)
        .toFloat)))
      ()
    }

    def test_affinity_matrices_cold(tlcSampleData: DataFrame, threshold: Int, similarityFunction: String,
      simFile: String,
      user_aff: String, demoCatalogDataFrame: DataFrame):
    (SARModel, Broadcast[Map[Int, String]], Broadcast[Map[Int, String]]) = {

      val session = tlcSampleData.sparkSession
      val sparkContext = session.sparkContext

      val recommendationIndexerModel = recommendationIndexer.fit(ratings)
      val transformedDf = recommendationIndexerModel.transform(ratings)

      val itemindexModel = new StringIndexer()
        .setInputCol(itemCol)
        .setOutputCol(itemColIndex).setHandleInvalid("skip")
        .fit(tlcSampleData)

      val tagIndexModel = new StringIndexer()
        .setInputCol("tagId_org")
        .setOutputCol("tagId")
        .fit(demoCatalogDataFrame)

      val demoCatalogDataFrameIndex =
        tagIndexModel.transform(itemindexModel.transform(demoCatalogDataFrame))

      sar
        .setSupportThreshold(threshold)
        .setSimilarityFunction(similarityFunction)
        .setStartTime("2015/06/09T19:39:37")
        .setStartTimeFormat("yyyy/MM/dd'T'h:mm:ss")
        .setItemFeatures(demoCatalogDataFrameIndex)

      val model = sar.fit(transformedDf)

      val userMap = recommendationIndexerModel.getUserIndex
      val itemMap = recommendationIndexerModel.getItemIndex

      val userMapBC = sparkContext.broadcast(userMap)
      val itemMapBC = sparkContext.broadcast(itemMap)

      val itemDF = model.getItemDataFrame
      val simMap = itemDF.collect().map(row => {
        val itemI = itemMap.getOrElse(row.getDouble(0).toInt, "-1")
        val similarityVectorMap = row.getList(1).toArray.zipWithIndex.map(t => (itemMap.getOrElse(t._2, "-1"), t._1))
          .toMap
        itemI -> similarityVectorMap
      }).toMap
      itemDF.count

      val itemAff = session.read.option("header", "true").csv(simFile)
      itemAff.count
      itemAff.collect().foreach(row => {
        val itemI = row.getString(0)
        itemAff.drop("_c0").schema.fieldNames.foreach(itemJ => {
          val groundTrueScore = row.getAs[String](itemJ).toFloat
          val sparkSarScore = simMap.getOrElse(itemI, Map()).getOrElse(itemJ, "-1").asInstanceOf[Float]
          if (groundTrueScore > 0)
            assert(sparkSarScore >= groundTrueScore, "Item I: " + itemI + " Item J: " + itemJ)
          else
            assert(sparkSarScore > groundTrueScore, "Item I: " + itemI + " Item J: " + itemJ)
        })
      })
      (model, userMapBC, itemMapBC)
    }
  }

}
