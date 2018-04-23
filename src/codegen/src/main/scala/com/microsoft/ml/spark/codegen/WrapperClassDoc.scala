// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.codegen

import com.microsoft.ml.spark.codegen.Config._
import com.microsoft.ml.spark.core.env.FileUtilities._

/** Provide class level python help documentation for generated classes.
  * Lookup the doc string based on the name of the scala class
  * The text for this help is drawn from the scaladocs explanations in the scala classes.
  *
  * Where possible, there is also sample code to illustrate usage.
  *
  * The default case, TODO + classname will help identify missing class docs as more modules are
  * added to the codebase. When new classes are added, please add a case and docstring here.
  */

object WrapperClassDoc {
  def GenerateWrapperClassDoc(className: String): String = {
    val f = new File(tmpDocDir, className + ".txt")
    if (!f.exists) { println("No class documentation file exists for " + className); "" }
    else readFile(f, _.getLines.toList)
           .map{s => if (s.isEmpty) s else scopeDepth + s}.mkString("\n")
  }

  // The __init__.py file
  def packageHelp(importString: String): String = {
    s"""|$copyrightLines
        |
        |"\""
        |MicrosoftML is a library of Python classes to interface with the
        |Microsoft scala APIs to utilize Apache Spark to create distibuted
        |machine learning models.
        |
        |MicrosoftML simplifies training and scoring classifiers and
        |regressors, as well as facilitating the creation of models using the
        |CNTK library, images, and text.
        |"\""
        |
        |$importString
        |""".stripMargin
  }

}
