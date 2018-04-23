// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.IO.http

import com.microsoft.ml.spark.core.env.FileUtilities.File
import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.{BasicResponseHandler, HttpClientBuilder}
import org.apache.spark.sql.functions.{col, length}

import scala.util.parsing.json.JSONObject

class HTTPSuite extends TestBase with WithFreeUrl {

  test("stream from HTTP", TestBase.Extended) {
    val q1 = session.readStream.format(classOf[HTTPSourceProvider].getName)
      .option("host", host)
      .option("port", port.toString)
      .option("name", apiName)
      .load()
      .withColumn("newCol", length(col("value")))
      .writeStream
      .format(classOf[HTTPSinkProvider].getName)
      .option("name", "foo")
      .queryName("foo")
      .option("replyCol", "newCol")
      .option("checkpointLocation", new File(tmpDir.toFile, "checkpoints").toString)
      .start()

    def sendRequest(map: Map[String, Any]): HttpPost = {
      val post = new HttpPost(url)
      val params = new StringEntity(JSONObject(map).toString())
      post.addHeader("content-type", "application/json")
      post.setEntity(params)
      post
    }

    def receiveRequest(post: HttpPost): String = {
      StreamUtilities.using(HttpClientBuilder.create().build()) { client =>
        val response = client.execute(post)
        new BasicResponseHandler().handleResponse(response)
      }.get
    }

    val p1 = sendRequest(Map("foo" -> 1, "bar" -> "here"))
    val p2 = sendRequest(Map("foo" -> 1, "bar" -> "heree"))
    val p3 = sendRequest(Map("foo" -> 1, "bar" -> "hereee"))
    val p4 = sendRequest(Map("foo" -> 1, "bar" -> "hereeee"))
    val posts = List(p1, p2, p3, p4)
    val correctResponses = List(27, 28, 29, 30)

    posts.zip(correctResponses).foreach { p =>
      val recieved = receiveRequest(p._1)
      println((p._1, p._2, recieved))
      assert(recieved === "{\"newCol\":" + p._2 + "}")
    }
    q1.stop()
  }

}
