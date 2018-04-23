# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= '3':
    basestring = str

from mmlspark.cntk._ImageFeaturizer import _ImageFeaturizer
from pyspark.ml.common import inherit_doc

@inherit_doc
class ImageFeaturizer(_ImageFeaturizer):
    """

    Args:
        SparkSession (SparkSession): The SparkSession that will be used to find the model
        ocation (str): The location of the model, either on local or HDFS
    """
    def setModelLocation(self, sparkSession, location):
        jSpark = sparkSession._jsparkSession
        self._java_obj = self._java_obj.setModelLocation(jSpark, location)
        return self

    def setModel(self, sparkSession, modelSchema):
        jSpark = sparkSession._jsparkSession
        self._java_obj = self._java_obj.setModel(jSpark, modelSchema.toJava(sparkSession))
        return self

    def setMiniBatchSize(self, size):
        self._java_obj = self._java_obj.setMiniBatchSize(size)
        return self
