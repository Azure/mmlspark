// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages.text

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, Wrappable}
import com.microsoft.ml.spark.core.serialize.params.MapParam
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import spray.json.DefaultJsonProtocol._

class Trie(map: Map[Char, Trie] = Map.empty,
           textValue: Seq[Char] = Seq.empty,
           normFunction: Char => Char = identity) extends Serializable {
  val trieMap: Map[Char, Trie] = map
  val value: Seq[Char] = textValue
  val normFunc: Char => Char = normFunction
  def getValueString: String = this.value.toString

  def getNodeValue(key: Seq[Char], value: Seq[Char]): Seq[Char] = (key, value) match {
    case (k, v) if k.tail.isEmpty => v
    case (_, _) => Seq.empty[Char]
  }
  def put(key: String, value: String): Trie = this.put(key.toCharArray, value)

  def put(key: Seq[Char], value: String, isNormalized: Boolean = false): Trie = (key, value) match {
    case (_, _) if key.isEmpty => this.copy()
    case (k, v) =>
      val normalizedKey = if (isNormalized) k else k.map(this.normFunc)
      val newLeaf = new Trie(textValue = getNodeValue(normalizedKey, v.toCharArray), normFunction = this.normFunc)
      val next = this.trieMap.getOrElse(normalizedKey.head, newLeaf).put(normalizedKey.tail, v, true)
      val newMap = this.trieMap + (normalizedKey.head -> next)
      new Trie(newMap, this.value, this.normFunc)
  }

  def get(letter: Char): Option[Trie] = this.trieMap.get(this.normFunc(letter))

  def putAll(stringMap: Map[String, String]): Trie = {
    stringMap.foldLeft(this){
      case (out, (k, v)) => out.put(k, v)
    }
  }

  def copy(): Trie = new Trie(this.trieMap, this.value, this.normFunc)

  def mapText(inputText: String): String = {
    val chars = inputText.toCharArray
    val outputText: StringBuilder = new StringBuilder("")
    def isAlpha(char: Char): Boolean = {
      char.isLetterOrDigit || char.equals("_".charAt(0))
    }
    def skipAlphas(rest: Seq[Char]): Unit = rest match {
      case _ if rest.isEmpty || !isAlpha(rest.head)=> scan(rest)
      case _  => skipAlphas(rest.tail)
    }

    def findMatch(rest: Seq[Char],
                  matched: Seq[Char],
                  hasMatch: Boolean,
                  chars: Seq[Char],
                  trie: Option[Trie]): Unit = (rest, matched, hasMatch, chars, trie) match {
      case _ if trie.isEmpty || chars.isEmpty =>
        outputText ++= matched
        if (hasMatch) skipAlphas(rest) else scan(rest)
      case _ if trie.get.value.isEmpty => findMatch(rest, matched, false, chars.tail, trie.get.get(chars.head))
      case _ => findMatch(chars, trie.get.value, true, chars.tail, trie.get.get(chars.head))
    }

    def scan(chars: Seq[Char]): Unit = {
      if (chars.nonEmpty) findMatch(chars.tail, Array(chars.head), false, chars.tail, this.get(chars.head))
    }

    scan(chars)
    outputText.mkString
  }

  override def toString: String = s"Val: $getValueString Rest:\n $trieMap"
}

object Trie {
  def apply(map: Map[String, String]): Trie = {
    new Trie().putAll(map)
  }
}

object TextPreprocessor extends ComplexParamsReadable[TextPreprocessor]

/** <code>TextPreprocessor</code> takes a dataframe and a dictionary
  * that maps (text -> replacement text), scans each cell in the input col
  * and replaces all substring matches with the corresponding value.
  * Priority is given to longer keys and from left to right.
  */
class TextPreprocessor(val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with Wrappable with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("TextPreprocessor"))
  val IDENTITY = """identity"""

  val normFuncs: Map[String, Char => Char] = Map[String, Char => Char] (
    IDENTITY         -> identity,
    """lowerCase"""  -> Character.toLowerCase,
    """upperCase"""  -> Character.toUpperCase
  )

  val map = new MapParam[String, String](this, "map", "Map of substring match to replacement")

  /** @group getParam */
  def getMap: Map[String, String] = get(map).getOrElse(Map())

  /** @group setParam */
  def setMap(value: Map[String, String]): this.type = set("map", value)

  def isValidNormFunc(normFuncName: String): Boolean = normFuncs.contains(normFuncName)

  val normFunc = new Param[String](this, "normFunc", "Name of normalization function to apply", isValidNormFunc _)

  /** @group getParam */
  def getNormFunc: String = get(normFunc).getOrElse("identity")

  /** @group setParam */
  def setNormFunc(value: String): this.type = set("normFunc", value)

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from column selection
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = dataset.sparkSession
    val inputIndex = dataset.columns.indexOf(getInputCol)
    val trie = new Trie(normFunction = normFuncs(getNormFunc)).putAll(getMap)
    val broadcastedTrie = spark.sparkContext.broadcast(trie)

    require(inputIndex != -1, s"Input column $getInputCol does not exist")

    val mapText: String => String = broadcastedTrie.value.mapText
    val textMapper = udf(mapText)
    dataset.withColumn(getOutputCol, textMapper(dataset(getInputCol)).as(getOutputCol))
  }

  def transformSchema(schema: StructType): StructType = {
    schema.add(StructField(getOutputCol, StringType))
  }

  def copy(extra: ParamMap): TextPreprocessor = defaultCopy(extra)

}
