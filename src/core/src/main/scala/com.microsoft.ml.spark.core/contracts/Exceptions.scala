// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.contracts

import java.lang.RuntimeException

import org.apache.spark.ml.util.Identifiable

object MMLException {
    implicit class MMLID(val i: Identifiable) extends AnyVal {
        def id: String = i.uid
    }
    def throwEx(msg: String, inner: Throwable = null)(implicit i: MMLID): MMLException = {
        throw new MMLException(i.id, msg, inner)
    }
}
import MMLException._

// The caller must *explicitly* pass null to the source and inner exception
class MMLException(source: String, msg: String, inner: Throwable)
    // suppression = true by default
    // writableStackTrace -> true by design for us
    extends RuntimeException(msg, inner, true, true) {

    // Fix this to be structured for operationalized scenarios?
    // Or will they consume the object?
    override def toString(): String = source + super.toString
}

class FriendlyException(addedInfo: String, inner: Throwable)(implicit aid: MMLID)
    extends MMLException(aid.id, addedInfo, inner)

class ParamException(reason: String)(implicit aid: MMLID)
    extends MMLException(aid.id, reason, null)
