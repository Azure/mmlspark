# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= '3':
    basestring = str

import pyspark
from pyspark import SparkContext
from pyspark.ml.common import inherit_doc
from pyspark.sql.types import *

class HyperparamBuilder(object):
    """
    Specifies the search space for hyperparameters.
    """
    def __init__(self):
        ctx = SparkContext.getOrCreate()
        self.jvm = ctx.getOrCreate()._jvm
        self.hyperparamBuilder = self.jvm.com.microsoft.ml.spark.stages.hyper.HyperparamBuilder()

    def addHyperparam(self, est, param, hyperParam):
        """
        Add a hyperparam to the builder

        Args:
            param (Param): The param to tune
            dist (Dist): Distribution of values

        """
        javaParam = est._java_obj.getParam(param.name)
        self.hyperparamBuilder.addHyperparam(javaParam, hyperParam.get())
        return self

    def build(self):
        """
        Builds the search space of hyperparameters, returns the map of hyperparameters to search through.

        """
        return self.hyperparamBuilder.build()

class DiscreteHyperParam(object):
    """
    Specifies a discrete list of values.
    """
    def __init__(self, values, seed=0):
        ctx = SparkContext.getOrCreate()
        self.jvm = ctx.getOrCreate()._jvm
        self.hyperParam = self.jvm.com.microsoft.ml.spark.stages.hyper.HyperParamUtils.getDiscreteHyperParam(values, seed)

    def get(self):
        return self.hyperParam

class RangeHyperParam(object):
    """
    Specifies a range of values.
    """
    def __init__(self, min, max, seed=0, isDouble=False):
        ctx = SparkContext.getOrCreate()
        self.jvm = ctx.getOrCreate()._jvm
        self.rangeParam = self.jvm.com.microsoft.ml.spark.stages.hyper.HyperParamUtils.getRangeHyperParam(min, max, seed)
    def get(self):
        return self.rangeParam

class GridSpace(object):
    """
    Specifies a predetermined grid of values to search through.
    """
    def __init__(self, paramValues):
        ctx = SparkContext.getOrCreate()
        self.jvm = ctx.getOrCreate()._jvm
        self.gridSpace = self.jvm.com.microsoft.ml.spark.stages.hyper.GridSpace(paramValues)

    def space(self):
        return self.gridSpace

class RandomSpace(object):
    """
    Specifies a random streaming range of values to search through.
    """
    def __init__(self, paramDistributions):
        ctx = SparkContext.getOrCreate()
        self.jvm = ctx.getOrCreate()._jvm
        self.paramSpace = self.jvm.com.microsoft.ml.spark.stages.hyper.RandomSpace(paramDistributions)

    def space(self):
        return self.paramSpace
