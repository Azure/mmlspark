// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.env

import scala.annotation.StaticAnnotation

/** Generate the internal wrapper for a given class.
  * Used for complicated wrappers, where the basic functionality is auto-generated,
  * and the rest is added in the inherited wrapper.
  */
class InternalWrapper extends StaticAnnotation
