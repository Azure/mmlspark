// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split1

import java.util.UUID

import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.functions.{col, explode, lit}
import org.apache.spark.sql.{DataFrame, Row}
import org.scalactic.Equality
import org.scalatest.Assertion

class DetectFaceSuite extends TransformerFuzzing[DetectFace] with CognitiveKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test2.jpg"
  ).toDF("url")

  lazy val face: DetectFace = new DetectFace()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("face")
    .setReturnFaceId(true)
    .setReturnFaceLandmarks(true)
    .setReturnFaceAttributes(Seq(
      "age", "gender", "headPose", "smile", "facialHair", "glasses", "emotion",
      "hair", "makeup", "occlusion", "accessories", "blur", "exposure", "noise"))

  override def assertDFEq(df1: DataFrame, df2: DataFrame)(implicit eq: Equality[DataFrame]): Unit = {
    def prep(df: DataFrame) = df.select(explode(col("face"))).select("col.*").drop("faceId")
    super.assertDFEq(prep(df1), prep(df2))(eq)
  }

  test("Basic Usage") {
    face.transform(df)
    val results = face.transform(df)
    val fromRow = Face.makeFromRowConverter

    val f1 = fromRow(results.select("face").collect().head.getSeq[Row](0).head)
    assert(f1.faceAttributes.get.age.get > 20)

    results.show(truncate = false)
  }

  override def testObjects(): Seq[TestObject[DetectFace]] =
    Seq(new TestObject(face, df))

  override def reader: MLReadable[_] = DetectFace
}

class FindSimilarFaceSuite extends TransformerFuzzing[FindSimilarFace] with CognitiveKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test1.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test2.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test3.jpg"
  ).toDF("url")

  lazy val detector: DetectFace = new DetectFace()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("detected_faces")
    .setReturnFaceId(true)
    .setReturnFaceLandmarks(false)
    .setReturnFaceAttributes(Seq())

  lazy val fromRow: Row => Face = Face.makeFromRowConverter

  lazy val faceIdDF: DataFrame = detector.transform(df)
    .select(
      col("detected_faces").getItem(0).getItem("faceId").alias("id"))
    .cache()

  lazy val faceIds: Array[String] = faceIdDF.collect().map(row =>
    row.getAs[String]("id"))

  lazy val findSimilar: FindSimilarFace = new FindSimilarFace()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setOutputCol("similar")
    .setFaceIdCol("id")
    .setFaceIds(faceIds)

  test("Basic Usage") {
    val similarFaceDF = findSimilar.transform(faceIdDF)
    val numMatches = similarFaceDF.select("similar").collect().map(row =>
      row.getSeq[Row](0).length
    )
    assert(numMatches === List(1, 2, 2))
  }

  override def testObjects(): Seq[TestObject[FindSimilarFace]] =
    Seq(new TestObject(findSimilar, faceIdDF))

  override def reader: MLReadable[_] = FindSimilarFace
}

class GroupFacesSuite extends TransformerFuzzing[GroupFaces] with CognitiveKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test1.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test2.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test3.jpg"
  ).toDF("url")

  lazy val detector: DetectFace = new DetectFace()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("detected_faces")
    .setReturnFaceId(true)
    .setReturnFaceLandmarks(false)
    .setReturnFaceAttributes(Seq())

  lazy val fromRow: Row => Face = Face.makeFromRowConverter

  lazy val faceIdDF: DataFrame = detector.transform(df)
    .select(
      col("detected_faces").getItem(0).getItem("faceId").alias("id"))
    .cache()

  lazy val faceIds: Array[String] = faceIdDF.collect().map(row =>
    row.getAs[String]("id"))

  lazy val group: GroupFaces = new GroupFaces()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setOutputCol("grouping")
    .setFaceIds(faceIds)

  test("Basic Usage") {
    val groupDF = group.transform(faceIdDF)
    val numMatches = groupDF.select(col("grouping.groups").getItem(0)).collect().map(row =>
      row.getSeq[Row](0).length
    )
    assert(numMatches === List(2, 2, 2))
  }

  override def testObjects(): Seq[TestObject[GroupFaces]] =
    Seq(new TestObject(group, faceIdDF))

  override def reader: MLReadable[_] = GroupFaces
}

class IdentifyFacesSuite extends TransformerFuzzing[IdentifyFaces] with CognitiveKey {

  import spark.implicits._

  lazy val satyaFaces = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test1.jpg"
  )

  lazy val bradFaces = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test2.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test3.jpg"
  )

  lazy val pgName: String = "group" + UUID.randomUUID().toString

  lazy val pgId: String = {
    PersonGroup.create(pgName, pgName)
    PersonGroup.list().find(_.name == pgName).get.personGroupId
  }

  lazy val satyaId: String = Person.create("satya", pgId)
  lazy val bradId: String = Person.create("brad", pgId)

  lazy val satyaFaceIds: Seq[String] = satyaFaces.map(Person.addFace(_, pgId, satyaId))
  lazy val bradFaceIds: Seq[String] = bradFaces.map(Person.addFace(_, pgId, bradId))

  lazy val detector: DetectFace = new DetectFace()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("detected_faces")
    .setReturnFaceId(true)
    .setReturnFaceLandmarks(false)
    .setReturnFaceAttributes(Seq())

  lazy val otherFaceIds: Seq[String] = detector.transform((satyaFaces ++bradFaces).toDF("url"))
    .select(col("detected_faces").getItem(0).getItem("faceId"))
    .collect()
    .map(r => r.getString(0)).toSeq

  override def beforeAll(): Unit = {
    super.beforeAll()
    println(satyaFaceIds ++ bradFaceIds)
    PersonGroup.train(pgId)
    tryWithRetries() { () =>
      assert(PersonGroup.getTrainingStatus(pgId).status === "succeeded")
      ()
    }
    println("done training face group")
  }

  override def afterAll(): Unit = {
    PersonGroup.list().find(_.name == pgName).foreach { pgi =>
      PersonGroup.delete(pgi.personGroupId)
      println("deleted group")
    }
    super.afterAll()
  }

  lazy val id: IdentifyFaces = new IdentifyFaces()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setFaceIdsCol("faces")
    .setPersonGroupId(pgId)
    .setOutputCol("identified_faces")

  lazy val df: DataFrame = otherFaceIds.map(Seq[String](_)).toDF("faces")

  test("Basic Usage") {
    Person.list(pgId).foreach(println)
    val matches = id.transform(df).select(col("identified_faces")
      .getItem(0).getItem("candidates").getItem(0).getItem("personId"))
      .collect().map(_.getString(0))
    assert(matches === List(satyaId, bradId, bradId))
  }

  override def testObjects(): Seq[TestObject[IdentifyFaces]] = Seq(new TestObject(id, df))

  override def reader: MLReadable[_] = IdentifyFaces
}

class VerifyFacesSuite extends TransformerFuzzing[VerifyFaces] with CognitiveKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test1.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test2.jpg",
    "https://mmlspark.blob.core.windows.net/datasets/DSIR/test3.jpg"
  ).toDF("url")

  lazy val detector: DetectFace = new DetectFace()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setImageUrlCol("url")
    .setOutputCol("detected_faces")
    .setReturnFaceId(true)
    .setReturnFaceLandmarks(false)
    .setReturnFaceAttributes(Seq())

  lazy val fromRow: Row => Face = Face.makeFromRowConverter

  lazy val faceIdDF: DataFrame = detector.transform(df)
    .select(col("detected_faces").getItem(0).getItem("faceId").alias("faceId1"))
    .cache()

  lazy val faceIdDF2: DataFrame = faceIdDF.withColumn(
    "faceId2", lit(faceIdDF.take(1).head.getString(0)))

  lazy val verify: VerifyFaces = new VerifyFaces()
    .setSubscriptionKey(cognitiveKey)
    .setLocation("eastus")
    .setOutputCol("same")
    .setFaceId1Col("faceId1")
    .setFaceId2Col("faceId2")

  test("Basic Usage") {
    val verifyDF = verify.transform(faceIdDF2)
    assert(verifyDF
      .select("same.isIdentical")
      .collect().map(_.getBoolean(0)) === List(true, false, false))
  }

  override def testObjects(): Seq[TestObject[VerifyFaces]] =
    Seq(new TestObject(verify, faceIdDF2))

  override def reader: MLReadable[_] = VerifyFaces

}
