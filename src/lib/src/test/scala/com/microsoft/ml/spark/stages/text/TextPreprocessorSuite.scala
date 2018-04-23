// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.text

import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable

class TextPreprocessorSuite extends TestBase with TransformerFuzzing[TextPreprocessor] {
  val toMap1 = "The happy sad boy drank sap"
  val toMap2 = "The hater sad doy drank sap"
  val toMap3 = "The hater sad doy"
  val INPUT_COL = "words1"
  val OUTPUT_COL = "out"

  lazy val expectedResult = session.createDataFrame(Seq(
    (toMap1, "The sad sap boy drank sap"),
    (toMap2, "The sap sap drank sap"),
    ("foo", "foo"),
    (s"$toMap3 aABc0123456789Zz_", "The sap sap")))
    .toDF(INPUT_COL, OUTPUT_COL)

  lazy val wordDF = expectedResult.drop(OUTPUT_COL)

  lazy val testMap = Map[String, String] (
    "happy"   -> "sad",
    "hater"   -> "sap",
    "sad"     -> "sap",
    "sad doy" -> "sap"
  )

  lazy val testTrie1 = Trie(Map[String, String]("happy" -> "sad", "hater" -> "sap"))
  var testTrie1Pivot: Trie = testTrie1
  for (letter <- "ha") testTrie1Pivot = testTrie1Pivot.get(letter).get

  test("Check for value of words with intersection in trie") {
      var copyHappy: Trie = testTrie1
      for (letter <- "happy") copyHappy = copyHappy.get(letter).get
      assert(copyHappy.value.mkString("") == "sad")

      var copyHater: Trie = testTrie1
      for (letter <- "hater") copyHater = copyHater.get(letter).get
      assert(copyHater.value.mkString("") == "sap")
    }

    test("Check continuation from child node in trie") {
      var copyHaHater: Trie = testTrie1Pivot
      for (letter <- "ppy") copyHaHater = copyHaHater.get(letter).get
      assert(copyHaHater.value.mkString("") == "sad")

      var copyHaHappy: Trie = testTrie1Pivot
      for (letter <- "ter") copyHaHappy = copyHaHappy.get(letter).get
      assert(copyHaHappy.value.mkString("") == "sap")
  }

  test("Check trie put vs putAll equality") {
    val test2 = Trie(Map[String,String]("happy" -> "sad", "hater" -> "sap"))
    val testString = "happy hater"
    assert(testTrie1.mapText(testString).equals(test2.mapText(testString)))
  }

  test("Check trie text mapper") {
    val mappings = Map[String, String]("happy" -> "sad", "hater" -> "sap", "sad" -> "sap", "sad doy" -> "sap")
    val test = Trie(mappings)
    print(test.mapText(toMap1))
    assert(test.mapText(toMap1).equals("The sad sap boy drank sap"))
    assert(test.mapText(toMap2).equals("The sap sap drank sap"))
  }

  test("Check trie multiple puts") {
    var test = new Trie(normFunction = Character.toUpperCase)
    test = test.put("happy", "sad").put("hater", "sap")
    test = test.put("sad", "sap").put("sad doy", "sap")
    test = test.put("the", "sat")
    val item = test.mapText(toMap1)
    val item1 = test.mapText(toMap2)
    assert(item.equals("sat sad sap boy drank sap"))
    assert(item1.equals("sat sap sap drank sap"))
  }

  test("Check TextPreprocessor text normalizers valid") {
    new TextPreprocessor()
      .setMap(testMap)
      .setInputCol(INPUT_COL)
      .setOutputCol(OUTPUT_COL)
    new TextPreprocessor()
      .setMap(testMap).setInputCol(INPUT_COL)
      .setOutputCol(OUTPUT_COL)
      .setNormFunc("identity")
    new TextPreprocessor()
      .setMap(testMap)
      .setInputCol(INPUT_COL)
      .setOutputCol(OUTPUT_COL)
      .setNormFunc("lowerCase")
  }

  test("Check TextPreprocessor text normalizers invalid") {
    assertThrows[IllegalArgumentException] {
      new TextPreprocessor()
        .setMap(testMap)
        .setNormFunc("p")
        .setInputCol(INPUT_COL)
        .setOutputCol(OUTPUT_COL)
    }
  }

  test("Check trie text df") {
    val textPreprocessor = new TextPreprocessor()
      .setNormFunc("lowerCase")
      .setMap(testMap)
      .setInputCol(INPUT_COL)
      .setOutputCol(OUTPUT_COL)
    val result = textPreprocessor.transform(wordDF)
    assert(verifyResult(result, expectedResult))
  }

  def testObjects(): Seq[TestObject[TextPreprocessor]] = List(new TestObject(
    new TextPreprocessor().setInputCol("words").setOutputCol("out"), makeBasicDF()))

  override def reader: MLReadable[_] = TextPreprocessor
}
