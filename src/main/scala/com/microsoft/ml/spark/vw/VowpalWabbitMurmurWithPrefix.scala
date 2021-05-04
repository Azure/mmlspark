// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import org.vowpalwabbit.spark.VowpalWabbitMurmur
import java.nio.charset.StandardCharsets

/**
  * VW style murmur hash with pre-hashing of an initially specified prefix.
  * @param prefix the prefix for each hashed value.
  * @param maxSize maximum size of the string to be hashed.
  */
class VowpalWabbitMurmurWithPrefix(val prefix: String, val maxSize: Int = 2 * 1024) extends Serializable {
  // worst case is 4 bytes per character
  val ys: Array[Byte] = new Array(maxSize * 4)

  val ysStart: Int = {
      // pre-populate the string with the prefix - we could go so-far as keep the intermediate hash state :)
      val prefixBytes = prefix.getBytes(StandardCharsets.UTF_8)
      Array.copy(prefixBytes, 0, ys, 0, prefixBytes.length)

      prefixBytes.length
    }

  def hash(str: String, namespaceHash: Int): Int =
    hash(str, 0, str.length, namespaceHash)

  def hash(str: String, start: Int, end: Int, namespaceHash: Int): Int = {
    if (end - start > maxSize)
      VowpalWabbitMurmur.hash(prefix + str.substring(start, end), namespaceHash)
    else {
      // adapted from https://stackoverflow.com/questions/5513144/converting-char-to-byte/20604909#20604909
      // copy sub part
      var i = start
      var j = ysStart // i for chars; j for bytes
      while (i < end) { // fill ys with bytes
        val c = str.charAt(i)
        if (c < 0x80) {
          ys(j) = c.toByte
          i = i + 1
          j = j + 1
        } else if (c < 0x800) {
          ys(j) = (0xc0 | (c >> 6)).toByte
          ys(j + 1) = (0x80 | (c & 0x3f)).toByte
          i = i + 1
          j = j + 2
        } else if (Character.isHighSurrogate(c)) {
          if (end - i < 2) throw new Exception("overflow") // this is not reachable due to maxSize * 4, so just in case
          val d = str.charAt(i + 1)
          val uc: Int =
            if (Character.isLowSurrogate(d))
              Character.toCodePoint(c, d)
            else
              throw new Exception("malformed")

          ys(j) = (0xf0 | uc >> 18).toByte
          ys(j + 1) = (0x80 | ((uc >> 12) & 0x3f)).toByte
          ys(j + 2) = (0x80 | ((uc >> 6) & 0x3f)).toByte
          ys(j + 3) = (0x80 | (uc & 0x3f)).toByte
          i = i + 2 // 2 chars
          j = j + 4
        } else if (Character.isLowSurrogate(c)) {
          throw new Exception("malformed")
        } else {
          ys(j) = (0xe0 | (c >> 12)).toByte
          ys(j + 1) = (0x80 | ((c >> 6) & 0x3f)).toByte
          ys(j + 2) = (0x80 | (c & 0x3f)).toByte
          i = i + 1
          j = j + 3
        }
      }

      VowpalWabbitMurmur.hash(ys, 0, j, namespaceHash)
    }
  }
}
