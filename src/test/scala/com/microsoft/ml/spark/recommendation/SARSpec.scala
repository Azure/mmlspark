// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.recommendation

import com.microsoft.ml.spark.core.test.fuzzing.{EstimatorFuzzing, TestObject, TransformerFuzzing}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.Pipeline
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

  override val epsilon = .3

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

    assert(evaluator.setMetricName("ndcgAt").evaluate(output) === 0.602819875812812)
    assert(evaluator.setMetricName("fcp").evaluate(output) === 0.05 ||
      evaluator.setMetricName("fcp").evaluate(output) === 0.1)
    assert(evaluator.setMetricName("mrr").evaluate(output) === 1.0)

    val users: DataFrame = spark
      .createDataFrame(Seq(("0","0"),("1","1")))
      .toDF(userColIndex, itemColIndex)

    val recs = recopipeline.stages(1).asInstanceOf[RankingAdapterModel].getRecommenderModel
      .asInstanceOf[SARModel].recommendForUserSubset(users, 10)
    assert(recs.count == 2)
  }

  lazy val testFile: String = getClass.getResource("/demoUsage.csv.gz").getPath
  lazy val demoCatalog: String = getClass.getResource("/demoCatalog.csv.gz").getPath
  lazy val simCount1: String = getClass.getResource("/sim_count1.csv.gz").getPath
  lazy val simLift1: String = getClass.getResource("/sim_lift1.csv.gz").getPath
  lazy val simJac1: String = getClass.getResource("/sim_jac1.csv.gz").getPath
  lazy val simCount3: String = getClass.getResource("/sim_count3.csv.gz").getPath
  lazy val simLift3: String = getClass.getResource("/sim_lift3.csv.gz").getPath
  lazy val simJac3: String = getClass.getResource("/sim_jac3.csv.gz").getPath
  lazy val userAff: String = getClass.getResource("/user_aff.csv.gz").getPath
  lazy val userpredCount3: String = getClass.getResource("/userpred_count3_userid_only.csv.gz").getPath
  lazy val userpredLift3: String = getClass.getResource("/userpred_lift3_userid_only.csv.gz").getPath
  lazy val userpredJac3: String = getClass.getResource("/userpred_jac3_userid_only.csv.gz").getPath

  private lazy val tlcSampleData: DataFrame = spark.read
    .option("header", "true") //reading the headers
    .option("inferSchema", "true")
    .csv(testFile).na.drop.cache

  private lazy val demoCatalogDataFrame: DataFrame = session.read
    .option("header", "true") //reading the headers
    .option("inferSchema", "true")
    .csv(demoCatalog).na.drop.cache

  test("tlc test sim count1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "cooc", simCount1, userAff))

  test("tlc test sim+cold count1")(
    SarTLCSpec.test_affinity_matrices_cold(tlcSampleData, 1, "cooc", sim_count1, user_aff, demoCatalogDataFrame))

  test("tlc test sim lift1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "lift", simLift1, userAff))

  test("tlc test sim jac1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "jaccard", simJac1, userAff))

  test("tlc test sim count3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "cooc", simCount3, userAff))

  test("tlc test sim lift3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "lift", simLift3, userAff))

  test("tlc test sim jac3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "jaccard", simJac3, userAff))

  test("tlc test userpred count3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "cooc", simCount3, userAff, userpredCount3))

  test("tlc test userpred lift3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "lift", simLift3, userAff, userpredLift3))

  test("tlc test userpred jac3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "jaccard", simJac3, userAff, userpredJac3))

}

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
  //scalastyle:off field.name
  override lazy val userCol = "userId"
  override lazy val itemCol = "productId"
  override lazy val ratingCol = "rating"
  override lazy val userColIndex = "customerID"
  override lazy val itemColIndex = "itemID"
  //scalastyle:on field.name

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

    val itemAff = spark.read.option("header", "true").csv(simFile)
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

    val usersProductsBC = spark.sparkContext.broadcast(usersProducts)

    val itemMapBC = spark.sparkContext.broadcast(recommendationIndexerModel.getItemIndex)

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

    val answer = spark.read.option("header", "true").csv(userPredFile).collect()

    assert(row(0).getString(0) == "0003000098E85347", "Assert Customer ID's Match")
    (0 to 10).foreach(i => assert(row(0).getString(i) == answer(0).getString(i)))
    (11 to 20).foreach(i => assert("%.3f".format(row(0).getFloat(i)) == "%.3f".format(answer(0).getString(i).toFloat)))
    ()
  }

  def test_affinity_matrices_cold(tlcSampleData: DataFrame, threshold: Int, similarityFunction: String, simFile: String,
    user_aff: String, demoCatalogDataFrame: DataFrame):
  (SARModel, Broadcast[Map[Int, String]], Broadcast[Map[Int, String]]) = {

    val session = tlcSampleData.sparkSession
    val sparkContext = session.sparkContext

    val pipelineModel: PipelineModel = pipeline.fit(ratings)
    val transformedDf = pipelineModel.transform(ratings)

    val tagIndexModel = new StringIndexer()
      .setInputCol("tagId_org")
      .setOutputCol("tagId")
      .fit(demoCatalogDataFrame)

    val demoCatalogDataFrameIndex = tagIndexModel.transform(pipelineModel.transform(demoCatalogDataFrame))

    val sar = new SAR()
      .setUserCol(customerIndex.getOutputCol)
      .setItemCol(itemIndex.getOutputCol)
      .setTimeCol("timestamp")
      .setSupportThreshold(threshold)
      .setSimilarityFunction(similarityFunction)
      .setStartTime("2015/06/09T19:39:37")
      .setStartTimeFormat("yyyy/MM/dd'T'h:mm:ss")
      .setItemFeatures(demoCatalogDataFrameIndex)

    val model = sar.fit(transformedDf)

    val userMap = pipelineModel
      .stages(0).asInstanceOf[StringIndexerModel]
      .labels
      .zipWithIndex
      .map(t => (t._2, t._1))
      .toMap

    val itemMap = pipelineModel
      .stages(1).asInstanceOf[StringIndexerModel]
      .labels
      .zipWithIndex
      .map(t => (t._2, t._1))
      .toMap

    val userMapBC = sparkContext.broadcast(userMap)
    val itemMapBC = sparkContext.broadcast(itemMap)

    val recoverUser = udf((userID: Integer) => userMapBC.value.getOrElse[String](userID, "-1"))
    val recoverItem = udf((itemID: Integer) => itemMapBC.value.getOrElse[String](itemID, "-1"))

    val row = model
      .getUserDataFrame
      .select(recoverUser(col("customerID")) as "customerID", col("flatList"))
      .filter(col("customerID") === "0003000098E85347")
      .select("flatList")
      .collect()(0)

    val list = row.getList(0)

    val map = list.toArray.zipWithIndex.map(t => (itemMap.getOrElse(t._2, "-1"), t._1)).toMap

    val userAff = session.read.option("header", "true").csv(user_aff).drop("_c0")
    val affinityRow = userAff.collect()(0)
    val columnNames = userAff.schema.fieldNames

    val itemDF = model.getItemDataFrame
    val simMap = itemDF.collect().map(row => {
      val itemI = itemMap.getOrElse(row.getInt(0), "-1")
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
        val sparkSarScore = simMap.getOrElse(itemI, Map()).getOrElse(itemJ, "-1")
        assert(groundTrueScore == sparkSarScore)
      })
    })
    (model, userMapBC, itemMapBC)
  }
}
