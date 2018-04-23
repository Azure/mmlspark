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
from pyspark.sql.types import *

BinaryFileFields = ["path", "bytes"]
"""
Names of Binary File Schema field names.
"""

BinaryFileSchema = StructType([
    StructField(BinaryFileFields[0], StringType(),  True),
    StructField(BinaryFileFields[1], BinaryType(), True) ])
"""
Schema for Binary Files.

Schema records consist of BinaryFileFields name, Type, and ??
  path
  bytes
"""

def readBinaryFiles(self, path, recursive = False, sampleRatio = 1.0, inspectZip = True, seed=0):
    """
    Reads the directory of binary files from the local or remote (WASB) source
    This function is attached to SparkSession class.

    :Example:

    >>> spark.readBinaryFiles(path, recursive, sampleRatio = 1.0, inspectZip = True)

    Args:
         path (str): Path to the file directory
         recursive (b (double): Fraction of the files loaded into the dataframe

    Returns:
        DataFrame: DataFrame with a single column "value"; see binaryFileSchema for details

    """
    ctx = SparkContext.getOrCreate()
    reader = ctx._jvm.com.microsoft.ml.spark.IO.binary.BinaryFileReader
    sql_ctx = pyspark.SQLContext.getOrCreate(ctx)
    jsession = sql_ctx.sparkSession._jsparkSession
    jresult = reader.read(path, recursive, jsession, float(sampleRatio), inspectZip, seed)
    return DataFrame(jresult, sql_ctx)

setattr(sql.SparkSession, 'readBinaryFiles', classmethod(readBinaryFiles))

def streamBinaryFiles(self, path, sampleRatio = 1.0, inspectZip = True, seed=0):
    """
    Streams the directory of binary files from the local or remote (WASB) source
    This function is attached to SparkSession class.

    :Example:

    >>> spark.streamBinaryFiles(path, sampleRatio = 1.0, inspectZip = True)

    Args:
         path (str): Path to the file directory

    Returns:
        DataFrame: DataFrame with a single column "value"; see binaryFileSchema for details

    """
    ctx = SparkContext.getOrCreate()
    reader = ctx._jvm.com.microsoft.ml.spark.IO.binary.BinaryFileReader
    sql_ctx = pyspark.SQLContext.getOrCreate(ctx)
    jsession = sql_ctx.sparkSession._jsparkSession
    jresult = reader.stream(path, jsession, float(sampleRatio), inspectZip, seed)
    return DataFrame(jresult, sql_ctx)

setattr(sql.SparkSession, 'streamBinaryFiles', classmethod(streamBinaryFiles))

def isBinaryFile(df, column):
    """
    Returns True if the column contains binary files

    Args:
        df (DataFrame): The DataFrame to be processed
        column (bool): The name of the column being inspected

    Returns:
        bool: True if the colum is a binary files column

    """
    ctx = SparkContext.getOrCreate()
    schema = ctx._jvm.com.microsoft.ml.spark.schema.BinaryFileSchema
    return schema.isBinaryFile(df._jdf, column)
