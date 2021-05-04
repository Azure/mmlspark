// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive.split1

import java.net.URI

import com.microsoft.ml.spark.cognitive.{RESTHelpers, URLEncodingUtils, _}
import com.microsoft.ml.spark.core.env.StreamUtilities.using
import org.apache.commons.io.IOUtils
import org.apache.http.client.methods._
import org.apache.http.entity.StringEntity
import spray.json.DefaultJsonProtocol._
import spray.json._

object FaceUtils extends CognitiveKey {

  import RESTHelpers._

  val BaseURL = "https://eastus.api.cognitive.microsoft.com/face/v1.0/"

  def faceSend(request: HttpRequestBase, path: String,
               params: Map[String, String] = Map()): String = {

    val paramString = if (params.isEmpty) {
      ""
    } else {
      "?" + URLEncodingUtils.format(params)
    }
    request.setURI(new URI(BaseURL + path + paramString))

    retry(List(100, 500, 1000), { () =>
      request.addHeader("Ocp-Apim-Subscription-Key", cognitiveKey)
      request.addHeader("Content-Type", "application/json")
      using(Client.execute(request)) { response =>
        if (!response.getStatusLine.getStatusCode.toString.startsWith("2")) {
          val bodyOpt = request match {
            case er: HttpEntityEnclosingRequestBase => IOUtils.toString(er.getEntity.getContent)
            case _ => ""
          }
          throw new RuntimeException(
            s"Failed: response: $response " +
              s"requestUrl: ${request.getURI}" +
              s"requestBody: $bodyOpt")
        }
        IOUtils.toString(response.getEntity.getContent)
      }.get
    })
  }

  def faceGet(path: String, params: Map[String, String] = Map()): String = {
    faceSend(new HttpGet(), path, params)
  }

  def faceDelete(path: String, params: Map[String, String] = Map()): String = {
    faceSend(new HttpDelete(), path, params)
  }

  def facePost[T](path: String, body: T, params: Map[String, String] = Map())
                 (implicit format: JsonFormat[T]): String = {
    val post = new HttpPost()
    post.setEntity(new StringEntity(body.toJson.compactPrint))
    faceSend(post, path, params)
  }

  def facePut[T](path: String, body: T, params: Map[String, String] = Map())
                (implicit format: JsonFormat[T]): String = {
    val post = new HttpPut()
    post.setEntity(new StringEntity(body.toJson.compactPrint))
    faceSend(post, path, params)
  }

  def facePatch[T](path: String, body: T, params: Map[String, String] = Map())
                  (implicit format: JsonFormat[T]): String = {
    val post = new HttpPatch()
    post.setEntity(new StringEntity(body.toJson.compactPrint))
    faceSend(post, path, params)
  }
}

import com.microsoft.ml.spark.cognitive.split1.FaceUtils._

object FaceListProtocol {
  implicit val PfiEnc: RootJsonFormat[PersistedFaceInfo] = jsonFormat2(PersistedFaceInfo.apply)
  implicit val FlcEnc: RootJsonFormat[FaceListContents] = jsonFormat4(FaceListContents.apply)
  implicit val FliEnc: RootJsonFormat[FaceListInfo] = jsonFormat3(FaceListInfo.apply)
}

object FaceList {

  import FaceListProtocol._

  def add(url: String, faceListId: String,
          userData: Option[String] = None, targetFace: Option[String] = None): Unit = {
    facePost(
      s"facelists/$faceListId/persistedFaces",
      Map("url" -> url),
      List(userData.map("userData" -> _), targetFace.map("targetFace" -> _)).flatten.toMap
    )
    ()
  }

  def create(faceListId: String, name: String,
             userData: Option[String] = None): Unit = {
    facePut(
      s"facelists/$faceListId",
      List(userData.map("userData" -> _), Some("name" -> name)).flatten.toMap
    )
    ()
  }

  def delete(faceListId: String): Unit = {
    faceDelete(s"facelists/$faceListId")
    ()
  }

  def deleteFace(faceListId: String, persistedFaceId: String): Unit = {
    faceDelete(s"facelists/$faceListId/persistedFaces/$persistedFaceId")
    ()
  }

  def get(faceListId: String): FaceListContents = {
    faceGet(s"facelists/$faceListId").parseJson.convertTo[FaceListContents]
  }

  def list(): Seq[FaceListInfo] = {
    faceGet(s"facelists").parseJson.convertTo[Seq[FaceListInfo]]
  }

  def patch(faceListId: String, name: String, userData: String): Unit = {
    facePatch(s"facelists/$faceListId", Map("name" -> name, "userData" -> userData))
    ()
  }

}

object PersonGroupProtocol {
  implicit val PgiEnc: RootJsonFormat[PersonGroupInfo] = jsonFormat3(PersonGroupInfo.apply)
  implicit val PgtsEnc: RootJsonFormat[PersonGroupTrainingStatus] = jsonFormat4(PersonGroupTrainingStatus.apply)
}

object PersonGroup {

  import PersonGroupProtocol._

  def create(personGroupId: String, name: String,
             userData: Option[String] = None): Unit = {
    facePut(
      s"persongroups/$personGroupId",
      List(userData.map("userData" -> _), Some("name" -> name)).flatten.toMap
    )
    ()
  }

  def delete(personGroupId: String): Unit = {
    faceDelete(s"persongroups/$personGroupId")
    ()
  }

  def get(personGroupId: String): Unit = {
    faceGet(s"persongroups/$personGroupId")
    ()
  }

  def list(start: Option[String] = None, top: Option[String] = None): Seq[PersonGroupInfo] = {
    faceGet(s"persongroups",
      List(start.map("start" -> _), top.map("top" -> _)).flatten.toMap
    ).parseJson.convertTo[Seq[PersonGroupInfo]]
  }

  def train(personGroupId: String): Unit = {
    facePost(s"persongroups/$personGroupId/train", body = "")
    ()
  }

  def getTrainingStatus(personGroupId: String): PersonGroupTrainingStatus = {
    faceGet(s"persongroups/$personGroupId/training").parseJson.convertTo[PersonGroupTrainingStatus]
  }

}

object PersonProtocol {
  implicit val PiEnc: RootJsonFormat[PersonInfo] = jsonFormat4(PersonInfo.apply)
}

object Person {

  import PersonProtocol._

  def addFace(url: String, personGroupId: String, personId: String,
              userData: Option[String] = None, targetFace: Option[String] = None): String = {
    facePost(
      s"persongroups/$personGroupId/persons/$personId/persistedFaces",
      Map("url" -> url),
      List(userData.map("userData" -> _), targetFace.map("targetFace" -> _)).flatten.toMap
    ).parseJson.asJsObject().fields("persistedFaceId").convertTo[String]
  }

  def create(name: String, personGroupId: String,
             userData: Option[String] = None): String = {
    facePost(
      s"persongroups/$personGroupId/persons",
      List(Some("name" -> name), userData.map("userData" -> _)).flatten.toMap
    ).parseJson.asJsObject().fields("personId").convertTo[String]
  }

  def delete(personGroupId: String, personId: String): Unit = {
    faceDelete(
      s"persongroups/$personGroupId/persons/$personId"
    )
    ()
  }

  def list(personGroupId: String,
           start: Option[String] = None,
           top: Option[String] = None): Seq[PersonInfo] = {
    faceGet(s"persongroups/$personGroupId/persons",
      List(start.map("start" -> _), top.map("top" -> _)).flatten.toMap
    ).parseJson.convertTo[Seq[PersonInfo]]
  }

  def deleteFace(personGroupId: String, personId: String, persistedFaceId: String): Unit = {
    faceDelete(
      s"persongroups/$personGroupId/persons/$personId/persistedFaces/$persistedFaceId"
    )
    ()
  }

}
