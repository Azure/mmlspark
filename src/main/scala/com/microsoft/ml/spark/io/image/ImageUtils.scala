// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.io.image

import java.awt.color.ColorSpace
import java.awt.image.{BufferedImage, DataBufferByte, Raster}
import java.awt.{Color, Point}
import java.io.ByteArrayInputStream

import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.io.binary.ConfUtils
import javax.imageio.ImageIO
import org.apache.commons.codec.binary.Base64
import org.apache.commons.io.IOUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.ImageInjections
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.{DataFrame, Row}

import scala.util.Try

object ImageUtils {

  import org.apache.spark.ml.image.ImageSchema._

  def channelsToType(channels: Int): Int = channels match {
    case 1 => BufferedImage.TYPE_BYTE_GRAY
    case 3 => BufferedImage.TYPE_3BYTE_BGR
    case 4 => BufferedImage.TYPE_4BYTE_ABGR
    case c => throw new UnsupportedOperationException("Image resize: number of output  " +
      s"channels must be 1, 3, or 4, got $c.")
  }

  def toBufferedImage(row: InternalRow): BufferedImage = {
    toBufferedImage(row.getBinary(5), row.getInt(2), row.getInt(1), row.getInt(3))
  }

  def toBufferedImage(row: Row): BufferedImage = {
    toBufferedImage(getData(row), getWidth(row), getHeight(row), getNChannels(row))
  }

  def toBufferedImage(bytes: Array[Byte], w: Int, h: Int, nChannels: Int): BufferedImage = {
    val img = new BufferedImage(w, h, channelsToType(nChannels))
    img.setData(Raster.createRaster(
      img.getSampleModel,
      new DataBufferByte(bytes, bytes.length),
      new Point()))
    img
  }

  def toSparkImage(img: BufferedImage, path: Option[String] = None): Row = {
    val isGray = img.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY
    val hasAlpha = img.getColorModel.hasAlpha

    val height = img.getHeight
    val width = img.getWidth
    val (nChannels, mode) = if (isGray) {
      (1, ocvTypes("CV_8UC1"))
    } else if (hasAlpha) {
      (4, ocvTypes("CV_8UC4"))
    } else {
      (3, ocvTypes("CV_8UC3"))
    }

    val imageSize = height * width * nChannels
    assert(imageSize < 1e9, "image is too large")
    val decoded = Array.ofDim[Byte](imageSize)

    // Grayscale images in Java require special handling to get the correct intensity
    if (isGray) {
      var offset = 0
      val raster = img.getRaster
      for (h <- 0 until height) {
        for (w <- 0 until width) {
          decoded(offset) = raster.getSample(w, h, 0).toByte
          offset += 1
        }
      }
    } else {
      var offset = 0
      for (h <- 0 until height) {
        for (w <- 0 until width) {
          val color = new Color(img.getRGB(w, h), hasAlpha)
          decoded(offset) = color.getBlue.toByte
          decoded(offset + 1) = color.getGreen.toByte
          decoded(offset + 2) = color.getRed.toByte
          if (hasAlpha) {
            decoded(offset + 3) = color.getAlpha.toByte
          }
          offset += nChannels
        }
      }
    }

    // the internal "Row" is needed, because the image is a single DataFrame column
    Row(Row(path, height, width, nChannels, mode, decoded))
  }

  def safeRead(bytes: Array[Byte]): Option[BufferedImage] = {
    Option(bytes).flatMap(b =>
      Try(Option(ImageIO.read(new ByteArrayInputStream(b))))
        .toOption.flatten)
  }

  def readFromPaths(df: DataFrame, pathCol: String, imageCol: String = "image"): DataFrame = {
    val outputSchema = df.schema.add(imageCol, ImageSchema.columnSchema)
    val encoder = RowEncoder(outputSchema)
    val hconf = ConfUtils.getHConf(df)
    df.mapPartitions { rows =>
      rows.map { row =>
        val path = new Path(row.getAs[String](pathCol))
        val fs = path.getFileSystem(hconf.value)
        val bytes = StreamUtilities.using(fs.open(path)) { is => IOUtils.toByteArray(is) }.get
        val imageRow = ImageInjections.decode(path.toString, bytes)
          .getOrElse(Row(null)) //scalastyle:ignore null
        val ret = Row.merge(Seq(row, imageRow): _*)
        ret
      }
    }(encoder)
  }

  def readFromBytes(df: DataFrame, pathCol: String, bytesCol: String, imageCol: String = "image"): DataFrame = {
    val outputSchema = df.schema.add(imageCol, ImageSchema.columnSchema)
    val encoder = RowEncoder(outputSchema)
    df.mapPartitions { rows =>
      rows.map { row =>
        val path = row.getAs[String](pathCol)
        val bytes = row.getAs[Array[Byte]](bytesCol)
        val imageRow = ImageInjections.decode(path, bytes).getOrElse(Row(null)) //scalastyle:ignore null
        val ret = Row.merge(Seq(row, imageRow): _*)
        ret
      }
    }(encoder)
  }

  def readFromStrings(df: DataFrame,
                      bytesCol: String,
                      imageCol: String = "image",
                      dropPrefix: Boolean = false): DataFrame = {
    val outputSchema = df.schema.add(imageCol, ImageSchema.columnSchema)
    val encoder = RowEncoder(outputSchema)
    df.mapPartitions { rows =>
      rows.map { row =>
        val encoded = row.getAs[String](bytesCol)
        val bytes = new Base64().decode(
          if (dropPrefix) encoded.split(",")(1) else encoded
        )
        val imageRow = ImageInjections.decode(null, bytes).getOrElse(Row(null)) //scalastyle:ignore null
        val ret = Row.merge(Seq(row, imageRow): _*)
        ret
      }
    }(encoder)
  }

}
