// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import java.util.UUID

import scala.collection.concurrent.TrieMap
import scala.reflect.ClassTag

/** Holds a variable shared among all workers. Useful to use non-serializable objects in Spark closures.
  *
  * Note this code has been borrowed from:
  * https://www.nicolaferraro.me/2016/02/22/using-non-serializable-objects-in-apache-spark/
  *
  * @author Nicola Ferraro
  */
class SharedVariable[T: ClassTag](constructor: => T) extends AnyRef with Serializable {

  @transient private lazy val instance: T = constructor

  def get: T = instance

}

object SharedVariable {

  def apply[T: ClassTag](constructor: => T): SharedVariable[T] = new SharedVariable[T](constructor)

}

/** Holds a variable shared among all workers that behaves like a local singleton.
  * Useful to use non-serializable objects in Spark closures that maintain state across tasks.
  *
  * @author Nicola Ferraro
  */
class SharedSingleton[T: ClassTag](constructor: => T) extends AnyRef with Serializable {

  val singletonUUID: String = UUID.randomUUID().toString

  @transient private lazy val instance: T = {
    SharedSingleton.singletonPool.synchronized {
      val singletonOption = SharedSingleton.singletonPool.get(singletonUUID)
      if (singletonOption.isEmpty) {
        SharedSingleton.singletonPool.put(singletonUUID, constructor)
      }
    }
    SharedSingleton.singletonPool(singletonUUID).asInstanceOf[T]
  }

  def get: T = instance

}

object SharedSingleton {

  private val singletonPool = new TrieMap[String, Any]()

  def apply[T: ClassTag](constructor: => T): SharedSingleton[T] = new SharedSingleton[T](constructor)

  def poolSize: Int = singletonPool.size

  def poolClear(): Unit = singletonPool.clear()

}
