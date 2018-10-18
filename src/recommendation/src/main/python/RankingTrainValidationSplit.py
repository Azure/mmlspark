# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.


import sys

if sys.version >= '3':
    basestring = str

from pyspark.ml.param.shared import *
from mmlspark.Utils import *
from pyspark.ml.common import inherit_doc
from pyspark.ml import Model
from pyspark.ml.util import *
from multiprocessing.pool import ThreadPool

import numpy as np
import pyspark
import pyspark.sql.functions as F
import sys
from mmlspark.RankingSplitters import *
from mmlspark.RankingAdapter import *
from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.tuning import ValidatorParams
from pyspark.ml.util import *
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
from pyspark.ml.param.shared import HasParallelism

if sys.version >= '3':
    basestring = str

def _parallelFitTasks(est, train, eva, validation, epm, collectSubModel):
    """
    Creates a list of callables which can be called from different threads to fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.

    :param est: Estimator, the estimator to be fit.
    :param train: DataFrame, training data set, used for fitting.
    :param eva: Evaluator, used to compute `metric`
    :param validation: DataFrame, validation data set, used for evaluation.
    :param epm: Sequence of ParamMap, params maps to be used during fitting & evaluation.
    :param collectSubModel: Whether to collect sub model.
    :return: (int, float, subModel), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)

    def singleTask():
        index, model = next(modelIter)
        metric = eva.evaluate(model.transform(validation, epm[index]))
        return index, metric, model if collectSubModel else None

    return [singleTask] * len(epm)


class HasCollectSubModels(Params):
    """
    Mixin for param collectSubModels: Param for whether to collect a list of sub-models trained during tuning. If set to false, then only the single best sub-model will be available after fitting. If set to true, then all sub-models will be available. Warning: For large models, collecting all sub-models can cause OOMs on the Spark driver.
    """

    collectSubModels = Param(Params._dummy(), "collectSubModels",
                             "Param for whether to collect a list of sub-models trained during tuning. If set to false, then only the single best sub-model will be available after fitting. If set to true, then all sub-models will be available. Warning: For large models, collecting all sub-models can cause OOMs on the Spark driver.",
                             typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasCollectSubModels, self).__init__()
        self._setDefault(collectSubModels=False)

    def setCollectSubModels(self, value):
        """
        Sets the value of :py:attr:`collectSubModels`.
        """
        return self._set(collectSubModels=value)

    def getCollectSubModels(self):
        """
        Gets the value of collectSubModels or its default value.
        """
        return self.getOrDefault(self.collectSubModels)

@inherit_doc
class RankingTrainValidationSplit(Estimator, ValidatorParams, HasCollectSubModels, HasParallelism):
    trainRatio = Param(Params._dummy(), "trainRatio", "Param for ratio between train and\
         validation data. Must be between 0 and 1.", typeConverter=TypeConverters.toFloat)
    userCol = Param(Params._dummy(), "userCol",
                    "userCol: column name for user ids. Ids must be within the integer value range. (default: user)")
    ratingCol = Param(Params._dummy(), "ratingCol", "ratingCol: column name for ratings (default: rating)")

    itemCol = Param(Params._dummy(), "itemCol",
                    "itemCol: column name for item ids. Ids must be within the integer value range. (default: item)")

    def setTrainRatio(self, value):
        """
        Sets the value of :py:attr:`trainRatio`.
        """
        return self._set(trainRatio=value)

    def getTrainRatio(self):
        """
        Gets the value of trainRatio or its default value.
        """
        return self.getOrDefault(self.trainRatio)

    def setItemCol(self, value):
        """

        Args:

            itemCol (str): column name for item ids. Ids must be within the integer value range. (default: item)

        """
        self._set(itemCol=value)
        return self

    def getItemCol(self):
        """

        Returns:

            str: column name for item ids. Ids must be within the integer value range. (default: item)
        """
        return self.getOrDefault(self.itemCol)

    def setRatingCol(self, value):
        """

        Args:

            ratingCol (str): column name for ratings (default: rating)

        """
        self._set(ratingCol=value)
        return self

    def getRatingCol(self):
        """

        Returns:

            str: column name for ratings (default: rating)
        """
        return self.getOrDefault(self.ratingCol)

    def setUserCol(self, value):
        """

        Args:

            userCol (str): column name for user ids. Ids must be within the integer value range. (default: user)

        """
        self._set(userCol=value)
        return self

    def getUserCol(self):
        """

        Returns:

            str: column name for user ids. Ids must be within the integer value range. (default: user)
        """
        return self.getOrDefault(self.userCol)

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, seed=None, trainRatio=0.8):
        """
        __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                 seed=None)
        """
        super(RankingTrainValidationSplit, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, seed=None):
        """
        setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                  seed=None):
        Sets params for cross validator.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies creates a deep copy of
        the embedded paramMap, and copies the embedded and extra parameters over.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        newCV = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newCV.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMaps remain the same
        if self.isSet(self.evaluator):
            newCV.setEvaluator(self.getEvaluator().copy(extra))
        return newCV

    def _create_model(self, java_model):
        model = RankingTrainValidationSplitModel()
        model._java_obj = java_model
        model._transfer_params_from_java()
        return model

    def split(self, dataset, tRatio, userColumn, itemColumn):
        pyspark.sql.DataFrame.min_rating_filter = RankingSplitters.min_rating_filter
        pyspark.sql.DataFrame.stratified_split = RankingSplitters.stratified_split

        temp_train, temp_validation = dataset \
            .dropDuplicates() \
            .withColumnRenamed(userColumn, 'customerID') \
            .withColumnRenamed(itemColumn, 'itemID') \
            .min_rating_filter(min_rating=6, by_customer=True) \
            .stratified_split(min_rating=3, by_customer=True, fixed_test_sample=False, ratio=tRatio)

        train = temp_train \
            .withColumnRenamed('customerID', userColumn) \
            .withColumnRenamed('itemID', itemColumn)

        validation = temp_validation \
            .withColumnRenamed('customerID', userColumn) \
            .withColumnRenamed('itemID', itemColumn)
        return train, validation


    def _fit(self, dataset):

        rating = self.getOrDefault(self.estimator).getRatingCol()
        userColumn = self.getOrDefault(self.estimator).getUserCol()
        itemColumn = self.getOrDefault(self.estimator).getItemCol()

        eva = self.getOrDefault(self.evaluator)

        est = RankingAdapter() \
            .setRecommender(self.getOrDefault(self.estimator)) \
            .setMode("allUsers") \
            .setK(eva) \
            .setUserCol(userColumn) \
            .setItemCol(itemColumn) \
            .setRatingCol(rating)

        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)

        tRatio = self.getOrDefault(self.trainRatio)
        seed = self.getOrDefault(self.seed)

        train, validation = self.split(dataset, tRatio, userColumn, itemColumn)

        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [None for i in range(numModels)]

        tasks = _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam)
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        metrics = [None] * numModels
        for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
            metrics[j] = metric
            if collectSubModelsParam:
                subModels[j] = subModel

        train.unpersist()
        validation.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(RankingTrainValidationSplitModel(bestModel, metrics, subModels))

    # @classmethod
    # def _from_java(cls, java_stage):
    #     """
    #     Given a Java TrainValidationSplit, create and return a Python wrapper of it.
    #     Used for ML persistence.
    #     """
    #
    #     estimator, epms, evaluator = super(TrainValidationSplit, cls)._from_java_impl(java_stage)
    #     trainRatio = java_stage.getTrainRatio()
    #     seed = java_stage.getSeed()
    #     parallelism = java_stage.getParallelism()
    #     collectSubModels = java_stage.getCollectSubModels()
    #     # Create a new instance of this stage.
    #     py_stage = cls(estimator=estimator, estimatorParamMaps=epms, evaluator=evaluator,
    #                    trainRatio=trainRatio, seed=seed, parallelism=parallelism,
    #                    collectSubModels=collectSubModels)
    #     py_stage._resetUid(java_stage.uid())
    #     return py_stage
    #
    # def _to_java(self):
    #     """
    #     Transfer this instance to a Java TrainValidationSplit. Used for ML persistence.
    #     :return: Java object equivalent to this instance.
    #     """
    #
    #     estimator, epms, evaluator = super(TrainValidationSplit, self)._to_java_impl()
    #
    #     _java_obj = JavaParams._new_java_obj("org.apache.spark.ml.tuning.TrainValidationSplit",
    #                                          self.uid)
    #     _java_obj.setEstimatorParamMaps(epms)
    #     _java_obj.setEvaluator(evaluator)
    #     _java_obj.setEstimator(estimator)
    #     _java_obj.setTrainRatio(self.getTrainRatio())
    #     _java_obj.setSeed(self.getSeed())
    #     _java_obj.setParallelism(self.getParallelism())
    #     _java_obj.setCollectSubModels(self.getCollectSubModels())
    #     return _java_obj


@inherit_doc
class RankingTrainValidationSplitModel(Model, ValidatorParams):

    def __init__(self, bestModel, validationMetrics=[], subModels=None):
        super(RankingTrainValidationSplitModel, self).__init__()
        #: best model from cross validation
        self.bestModel = bestModel
        #: evaluated validation metrics
        self.validationMetrics = validationMetrics
        self.subModels = subModels

    def _transform(self, dataset):
        return self.bestModel.transform(dataset)

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying bestModel,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.
        And, this creates a shallow copy of the validationMetrics.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        bestModel = self.bestModel.copy(extra)
        validationMetrics = list(self.validationMetrics)
        return RankingTrainValidationSplitModel(bestModel, validationMetrics)

    def recommendForAllUsers(self, numItems):
        return self.bestModel.recommendForAllUsers(numItems)
        # return self.bestModel._call_java("recommendForAllUsers", numItems)

    def recommendForAllItems(self, numItems):
        return self.bestModel.recommendForAllItems(numItems)

    # @classmethod
    # def _from_java(cls, java_stage):
    #     """
    #     Given a Java TrainValidationSplitModel, create and return a Python wrapper of it.
    #     Used for ML persistence.
    #     """
    #
    #     # Load information from java_stage to the instance.
    #     bestModel = JavaParams._from_java(java_stage.bestModel())
    #     estimator, epms, evaluator = super(TrainValidationSplitModel,
    #                                        cls)._from_java_impl(java_stage)
    #     # Create a new instance of this stage.
    #     py_stage = cls(bestModel=bestModel).setEstimator(estimator)
    #     py_stage = py_stage.setEstimatorParamMaps(epms).setEvaluator(evaluator)
    #
    #     if java_stage.hasSubModels():
    #         py_stage.subModels = [JavaParams._from_java(sub_model)
    #                               for sub_model in java_stage.subModels()]
    #
    #     py_stage._resetUid(java_stage.uid())
    #     return py_stage
    #
    # def _to_java(self):
    #     """
    #     Transfer this instance to a Java TrainValidationSplitModel. Used for ML persistence.
    #     :return: Java object equivalent to this instance.
    #     """
    #
    #     sc = SparkContext._active_spark_context
    #     # TODO: persst validation metrics as well
    #     _java_obj = JavaParams._new_java_obj(
    #         "org.apache.spark.ml.tuning.TrainValidationSplitModel",
    #         self.uid,
    #         self.bestModel._to_java(),
    #         _py2java(sc, []))
    #     estimator, epms, evaluator = super(TrainValidationSplitModel, self)._to_java_impl()
    #
    #     _java_obj.set("evaluator", evaluator)
    #     _java_obj.set("estimator", estimator)
    #     _java_obj.set("estimatorParamMaps", epms)
    #
    #     if self.subModels is not None:
    #         java_sub_models = [sub_model._to_java() for sub_model in self.subModels]
    #         _java_obj.setSubModels(java_sub_models)
    #
    #     return _java_obj
