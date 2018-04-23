// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.serialize

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import scala.language.existentials
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

/** This trait allows you to easily add serialization to your Spark
  * Models, assuming that they are completely parameterized by their constructor.
  * The main two fields required ate the <code> TypeTag </code> that allows the
  * writer to inspect the constructor to get the types that need to be serialized,
  * the actual objects that are serialized need to be defined in the field
  * objectsToSave.
  * @tparam S
  */
trait ConstructorWritable[S <: PipelineStage] extends MLWritable {

  val ttag: TypeTag[S]

  val uid: String

  def objectsToSave: List[Any]

  override def write: MLWriter = new ConstructorWriter[S](uid, objectsToSave, ttag) {}
}

abstract class ConstructorWriter[S <: PipelineStage](val uid: String, objectsToSave: List[Any],
                                                     val ttag: TypeTag[S])
  extends MLWriter {

  val typesToSave: List[Type] = Serializer.getConstructorTypes(ttag)

  //TODO Assert that we have gotten the proper types from the constructor

  override protected def saveImpl(path: String): Unit = {
    val baseDir = Serializer.makeQualifiedPath(sc, path)
    // Required in order to allow this to be part of an ML pipeline
    val ctag = ClassTag(Serializer.mirror.runtimeClass(ttag.tpe))
    Serializer.saveMetadata(uid, ctag, new Path(baseDir, "metadata").toString, sc, shouldOverwrite)
    Serializer.writeToHDFS[TypeTag[_]](sc, ttag, new Path(baseDir, "ttag"), shouldOverwrite)

    val objectsToWrite = objectsToSave
    objectsToWrite.zipWithIndex.foreach { case (obj, i) =>
      val path = Serializer.getPath(baseDir, i)
      val serializer = Serializer.typeToSerializer[Any](typesToSave(i), sparkSession)
      serializer.write(obj, path, shouldOverwrite)
    }
  }
}

trait ConstructorReadable[T <: ConstructorWritable[_]] extends MLReadable[T] {

  override def read: MLReader[T] = new ConstructorReader()

}

class ConstructorReader[T]()
  extends MLReader[T] {

  override def load(path: String): T = {
    val baseDir = Serializer.makeQualifiedPath(sc, path)

    val ttag: TypeTag[_] = Serializer.readFromHDFS[TypeTag[_]](sc, new Path(baseDir, "ttag"))

    def types: List[Type] = Serializer.getConstructorTypes(ttag)

    def instantiate[S](ttag: TypeTag[S], ctor: Int = 0)(args: AnyRef*): S = {
      Serializer.mirror.reflectClass(ttag.tpe.typeSymbol.asClass).reflectConstructor(
        ttag.tpe.members.filter(m =>
          m.isMethod && m.asMethod.isConstructor
        ).iterator.toSeq(ctor).asMethod
      )(args: _*).asInstanceOf[S]
    }

    val loadedObjects = types.zipWithIndex.map { case (tpe, i) =>
      val path = Serializer.getPath(baseDir, i)
      val serializer = Serializer.typeToSerializer[AnyRef](tpe, sparkSession)
      serializer.read(path)
    }

    instantiate[T](ttag.asInstanceOf[TypeTag[T]])(loadedObjects: _*)
  }

}
