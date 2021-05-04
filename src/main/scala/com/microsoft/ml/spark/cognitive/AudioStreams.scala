// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive

import java.io.InputStream

import com.microsoft.cognitiveservices.speech.audio.PullAudioInputStreamCallback

/**
  * Code adapted from https://github.com/Azure-Samples/cognitive-services-speech-sdk
  * /blob/master/samples/java/jre/console/src/com/microsoft/
  * cognitiveservices/speech/samples/console/WavStream.java
  * @param wavStream stream of wav bits with header stripped
  */
class WavStream(val wavStream: InputStream) extends PullAudioInputStreamCallback {

  val stream: InputStream = parseWavHeader(wavStream)

  override def read(dataBuffer: Array[Byte]): Int = {
    Math.max(0, stream.read(dataBuffer, 0, dataBuffer.length))
  }

  override def close(): Unit = {
    stream.close()
  }

  // region Wav File helper functions
  private def readUInt32(inputStream: InputStream) = {
    (0 until 4).foldLeft(0) { case (n, i) => n | inputStream.read << (i * 8) }
  }

  private def readUInt16(inputStream: InputStream) = {
    (0 until 2).foldLeft(0) { case (n, i) => n | inputStream.read << (i * 8) }
  }

  //noinspection ScalaStyle
  def parseWavHeader(reader: InputStream): InputStream = {
    // Tag "RIFF"
    val data = new Array[Byte](4)
    var numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "RIFF".getBytes), "RIFF")

    // Chunk size
    val fileLength = readUInt32(reader)

    numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "WAVE".getBytes), "WAVE")

    numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "fmt ".getBytes), "fmt ")

    val formatSize = readUInt32(reader)
    assert(formatSize >= 16, "formatSize")

    val formatTag = readUInt16(reader)
    val channels = readUInt16(reader)
    val samplesPerSec = readUInt32(reader)
    val avgBytesPerSec = readUInt32(reader)
    val blockAlign = readUInt16(reader)
    val bitsPerSample = readUInt16(reader)
    assert(formatTag == 1, "PCM") // PCM

    assert(channels == 1, "file needs to be single channel")
    assert(samplesPerSec == 16000, "file needs to have 16000 samples per second")
    assert(bitsPerSample == 16, "file needs to have 16 bits per sample")

    // Until now we have read 16 bytes in format, the rest is cbSize and is ignored
    // for now.
    if (formatSize > 16) {
      numRead = reader.read(new Array[Byte]((formatSize - 16).toInt))
      assert(numRead == (formatSize - 16), "could not skip extended format")
    }
    // Second Chunk, data
    // tag: data.
    numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "data".getBytes))

    val dataLength = readUInt32(reader)
    reader
  }
}

class CompressedStream(val stream: InputStream) extends PullAudioInputStreamCallback {

  override def read(dataBuffer: Array[Byte]): Int = {
    Math.max(0, stream.read(dataBuffer, 0, dataBuffer.length))
  }

  override def close(): Unit = {
    stream.close()
  }

}
