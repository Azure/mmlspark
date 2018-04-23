// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.downloader

import java.io._
import java.net.{URI, URL}
import java.util

import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.{Configuration => HadoopConf}
import org.apache.hadoop.fs.{FileSystem, LocatedFileStatus, Path}
import org.apache.hadoop.io.{IOUtils => HUtils}
import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import spray.json._
import com.microsoft.ml.spark.downloader.SchemaJsonProtocol._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

/** Abstract representation of a repository for future expansion
  *
  * @tparam S an instantiation of the
  */
private[spark] abstract class Repository[S <: Schema] {

  def listSchemas(): Iterable[S]

  def getBytes(schema: S): InputStream

  def addBytes(schema: S, location: URI, bytes: InputStream): S

}

/** Exception returned if a repo cannot find the file
  *
  * @param uri : location of the file
  */
class ModelNotFoundException(uri: URI) extends FileNotFoundException(s"model located at $uri could not be found")

private[spark] class HDFSRepo[S <: Schema](val uri: URI, val hconf: HadoopConf)
                                            (implicit val jsonFormat: JsonFormat[S])
  extends Repository[S] {

  private val rootPath = new Path(uri)

  private val fs = FileSystem.get(uri, hconf)

  if (!fs.exists(rootPath)) {
    fs.mkdirs(rootPath)
  }

  override def listSchemas(): Iterable[S] = {
    val fileIteratorHadoop = fs.listFiles(rootPath, false)
    val fileIterator = new Iterator[LocatedFileStatus] {
      def hasNext: Boolean = fileIteratorHadoop.hasNext

      def next(): LocatedFileStatus = fileIteratorHadoop.next()
    }

    val schemaStrings = fileIterator
      .filter(status =>
        status.isFile & status.getPath.toString.endsWith(".meta"))
      .map(status =>
        IOUtils.toString(fs.open(status.getPath).getWrappedStream))

    schemaStrings.map(s => s.parseJson.convertTo[S]).toList
  }

  override def getBytes(schema: S): InputStream = {
    try {
      fs.open(new Path(schema.uri))
    } catch {
      case _: IOException => throw new ModelNotFoundException(schema.uri)
    }
  }

  override def addBytes(schema: S, location: URI, bytes: InputStream): S = {
    val path = new Path(location)
    val os = fs.create(path)
    try {
      HUtils.copyBytes(bytes, os, hconf)
    } finally {
      os.close()
    }
    val downloadedIs = fs.open(path)
    try {
      schema.assertMatchingHash(downloadedIs)
    } finally {
      downloadedIs.close()
    }

    val newSchema = schema.updateURI(location)
    val schemaPath = new Path(location.getPath + ".meta")
    val osSchema = fs.create(schemaPath)
    val SchemaIs = IOUtils.toInputStream(newSchema.toJson.prettyPrint)
    try {
      HUtils.copyBytes(SchemaIs, osSchema, hconf)
    } finally {
      osSchema.close()
      SchemaIs.close()
    }
    newSchema
  }

}

/** Class to represent repository of models that will eventually be hosted outside
  * the repo.
  */
private[spark] class DefaultModelRepo(val baseURL: URL) extends Repository[ModelSchema] {
  var connectTimeout = 15000
  var readTimeout = 5000

  private def toStream(url: URL) = {
    val urlCon = url.openConnection()
    urlCon.setConnectTimeout(connectTimeout)
    urlCon.setReadTimeout(readTimeout)
    new BufferedInputStream(urlCon.getInputStream)
  }

  private def join(root: URL, file: String) = {
    new Path(new Path(root.toURI), file).toUri.toURL
  }

  override def listSchemas(): Iterable[ModelSchema] = {
    val url = join(baseURL, "MANIFEST")
    val manifestStream = toStream(url)
    try {
      val modelStreams = IOUtils.readLines(manifestStream).map(fn => toStream(join(baseURL, fn)))
      try {
        modelStreams.map(s => IOUtils.toString(s).parseJson.convertTo[ModelSchema])
      } finally {
        modelStreams.foreach(_.close())
      }
    } finally {
      manifestStream.close()
    }
  }

  override def getBytes(schema: ModelSchema): InputStream = {
    try {
      val url = schema.uri.toURL
      val urlCon = url.openConnection()
      urlCon.setConnectTimeout(connectTimeout)
      urlCon.setReadTimeout(readTimeout)
      new BufferedInputStream(urlCon.getInputStream)
    } catch {
      case _: IOException => throw new ModelNotFoundException(schema.uri)
    }
  }

  override def addBytes(schema: ModelSchema, location: URI, bytes: InputStream): ModelSchema =
    throw new IllegalAccessError("Do not have the credentials to write a file to the remote repository")
}

private[spark] abstract class Client {
  var quiet = false

  private def log(s: String): Unit = {
    LogManager.getRootLogger.info(s)
  }

  def repoTransfer[T <: Schema](schema: T, targetLocation: URI,
                                source: Repository[T], target: Repository[T],
                                overwrite: Boolean = false, closeStream: Boolean = true): T = {
    if (target.listSchemas().exists(s =>
      (s.uri == targetLocation) && (s.hash == schema.hash))) {
      log(s"Using model at $targetLocation, skipping download")
      target.listSchemas().find(_.hash == schema.hash).get
    } else {
      log(s"No model found in local repo, writing bytes to $targetLocation")
      val sourceStream = source.getBytes(schema)
      try {
        target.addBytes(schema, targetLocation, sourceStream)
      } finally {
        if (closeStream) sourceStream.close()
      }
    }
  }

}

private[spark] object ModelDownloader {
  private[spark] val defaultURL = new URL("https://mmlspark.azureedge.net/datasets/CNTKModels/")
}

/** Class for downloading models from a server to Local or HDFS
  *
  * @param spark Spark session so that the downloader can save to HDFS
  * @param localPath path to a directory that will store the models (local or HDFS)
  * @param serverURL URL of the server which supplies models ( The default URL is subject to change)
  */
class ModelDownloader(val spark: SparkSession,
                      val localPath: URI,
                      val serverURL: URL = ModelDownloader.defaultURL) extends Client {

  def this(spark: SparkSession, localPath: String, serverURL: String) = {
    this(spark, new URI(localPath), new URL(serverURL))
  }

  private val localModelRepo = new HDFSRepo[ModelSchema](localPath, spark.sparkContext.hadoopConfiguration)

  private val remoteModelRepo = new DefaultModelRepo(serverURL)

  /** Function for querying the local repository for its registered models
    *
    * @return the model schemas found in the downloader's local path
    */
  def localModels: util.Iterator[ModelSchema] = localModelRepo.listSchemas().iterator.asJava

  /** Function for querying the remote server for its registered models
    *
    * @return the model schemas found in remote reposiory accessed through the serverURL
    */
  def remoteModels: util.Iterator[ModelSchema] = remoteModelRepo.listSchemas().iterator.asJava

  /** Method to download a single model
    * @param model the remote model schema
    * @return the new local model schema with a URI that points to the model's location (on HDFS or local)
    */
  def downloadModel(model: ModelSchema): ModelSchema = {
    repoTransfer(model,
      new Path(new Path(localPath), NamingConventions.canonicalModelFilename(model)).toUri,
      remoteModelRepo, localModelRepo)
  }

  def downloadByName(name: String): ModelSchema = {
    val models = remoteModels.filter(_.name == name).toList
    if (models.length != 1) {
      throw new IllegalArgumentException(s"there are ${models.length} models with the same name")
    }
    downloadModel(models.head)
  }

  /** @param models An iterable of remote model schemas
    * @return An list of local model schema whose URI's points to the model's location (on HDFS or local)
    */
  def downloadModels(models: Iterable[ModelSchema] = remoteModels.toIterable): List[ModelSchema] =
  // Call toList so that all models are downloaded when downloadModels are called
    models.map(downloadModel).toList

  /** @param models A java iterator of remote model schemas for in the java api (for python wrapper)
    * @return A java List of local model schema whose URI's points to the model's location (on HDFS or local)
    */
  def downloadModels(models: util.ArrayList[ModelSchema]): util.List[ModelSchema] =
  // Call toList so that all models are downloaded when downloadModels are called
    models.map(downloadModel).toList.asJava

}
