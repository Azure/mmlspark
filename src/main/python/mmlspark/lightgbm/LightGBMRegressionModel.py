# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys
from pyspark import SQLContext
from pyspark import SparkContext

if sys.version >= '3':
    basestring = str

from mmlspark.lightgbm._LightGBMRegressionModel import _LightGBMRegressionModel
from pyspark import SparkContext
from pyspark.ml.common import inherit_doc
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.wrapper import JavaParams
from mmlspark.core.serialize.java_params_patch import *

@inherit_doc
class LightGBMRegressionModel(_LightGBMRegressionModel):
    def saveNativeModel(self, filename, overwrite=True):
        """
        Save the booster as string format to a local or WASB remote location.
        """
        self._java_obj.saveNativeModel(filename, overwrite)

    @staticmethod
    def loadNativeModelFromFile(filename):
        """
        Load the model from a native LightGBM text file.
        """
        ctx = SparkContext._active_spark_context
        loader = ctx._jvm.com.microsoft.ml.spark.lightgbm.LightGBMRegressionModel
        java_model = loader.loadNativeModelFromFile(filename)
        return JavaParams._from_java(java_model)

    @staticmethod
    def loadNativeModelFromString(model):
        """
        Load the model from a native LightGBM model string.
        """
        ctx = SparkContext._active_spark_context
        loader = ctx._jvm.com.microsoft.ml.spark.lightgbm.LightGBMRegressionModel
        java_model = loader.loadNativeModelFromString(model)
        return JavaParams._from_java(java_model)

    def getFeatureImportances(self, importance_type="split"):
        """
        Get the feature importances as a list.  The importance_type can be "split" or "gain".
        """
        return list(self._java_obj.getFeatureImportances(importance_type))

    def getFeatureShaps(self, vector):
        """
        Get the local shap feature importances.
        """
        if isinstance(vector, DenseVector):
            dense_values = [float(v) for v in vector]
            return list(self._java_obj.getDenseFeatureShaps(dense_values))
        elif isinstance(vector, SparseVector):
            sparse_indices = [int(i) for i in vector.indices]
            sparse_values = [float(v) for v in vector.values]
            return list(self._java_obj.getSparseFeatureShaps(vector.size, sparse_indices, sparse_values))
        else:
            raise TypeError("Vector argument to getFeatureShaps must be a pyspark.linalg sparse or dense vector type")

    def getBoosterBestIteration(self):
        """Get the best iteration from the booster.

        Returns:
            The best iteration, if early stopping was triggered.
        """
        self._java_obj.getBoosterBestIteration()

    def getBoosterNumTotalIterations(self):
        """Get the total number of iterations trained.

        Returns:
            The total number of iterations trained.
        """
        self._java_obj.getBoosterNumTotalIterations()

    def getBoosterNumTotalModel(self):
        """Get the total number of models trained.

        Returns:
            The total number of models.
        """
        self._java_obj.getBoosterNumTotalIterations()

    def getBoosterNumFeatures(self):
        """Get the number of features from the booster.

        Returns:
            The number of features.
        """
        self._java_obj.getBoosterNumTotalIterations()
