// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.sql

object NamespaceInjections {
  class InjectedAnalysisException(m: String) extends AnalysisException(m: String)
}
