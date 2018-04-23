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


def streamToPowerBI(df, url, batchInterval=1000):
    jvm = SparkContext.getOrCreate()._jvm
    writer = jvm.com.microsoft.ml.spark.IO.powerbi.PowerBIWriter
    return writer.stream(df.drop("label")._jdf, url, batchInterval)

setattr(pyspark.sql.DataFrame, 'streamToPowerBI', streamToPowerBI)

def writeToPowerBI(df, url):
    jvm = SparkContext.getOrCreate()._jvm
    writer = jvm.com.microsoft.ml.spark.IO.powerbi.PowerBIWriter
    writer.write(df._jdf, url)

setattr(pyspark.sql.DataFrame, 'writeToPowerBI', writeToPowerBI)
