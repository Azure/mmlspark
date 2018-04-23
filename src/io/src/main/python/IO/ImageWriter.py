# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= '3':
    basestring = str

import pyspark
from pyspark import SparkContext
from pyspark import sql
from pyspark.ml.param.shared import *
from pyspark.sql import DataFrame


def write(df, basePath, pathCol="filenames", imageCol="image", encoding=".png"):
    """
    Reads images from a column of filenames

    Args:
        df (DataFrame): The DataFrame to be processed
        basePath: (str): Where to save the images
        pathCol  (str): The name of the column containing filenames
        imageCol (str): The name of the added column of images
        encoding (str): what openCV encoding to use when saving images

    """
    jvm = SparkContext.getOrCreate()._jvm
    writer = jvm.com.microsoft.ml.spark.IO.image.ImageWriter
    writer.write(df._jdf, basePath, pathCol, imageCol, encoding)
