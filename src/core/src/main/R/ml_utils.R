# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

#' Calls transform on the Spark model
#'
#' Given a \code{ml_model} fit alongside a data set, transforms
#' a new dataset.
#'
#' @param object,newdata An object coercable to a Spark DataFrame.
#' @param ... Optional arguments; currently unused.
#'
#' @family Spark data frames
#'
#' @export
sdf_transform <- function(object, newdata, ...) {
  sdf <- spark_dataframe(newdata)
  transformed <- invoke(object$model, "transform", sdf)
  sdf_register(transformed)
}
