# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for information.

import sys

if sys.version >= "3":
    basestring = str


from reco_utils.dataset.spark_splitters import spark_random_split
from reco_utils.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation

from reco_utils.common.constants import (
    DEFAULT_PREDICTION_COL,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)


class RecoUtils:
    """Wrapper of Recommender Utilities specific for Spark"""

    @staticmethod
    def spark_random_split(data, ratio=0.75, seed=42):
        """
        Randomly split recommender dataframe.

        :param data: input Spark Dataframe
        :param ratio: Ratio to Split by
        :param seed: Seed to ensure constant splitting results
        """
        return spark_random_split(data, ratio=ratio, seed=seed)

    @staticmethod
    def rating_evaluation(rating_true,
                          rating_pred,
                          col_user=DEFAULT_USER_COL,
                          col_item=DEFAULT_ITEM_COL,
                          col_rating=DEFAULT_RATING_COL,
                          col_prediction=DEFAULT_PREDICTION_COL,
                          ):
        return SparkRatingEvaluation(
            rating_true,
            rating_pred,
            col_user,
            col_item,
            col_rating,
            col_prediction
        )

    @staticmethod
    def ranking_evaluation(
            rating_true,
            rating_pred,
            k=DEFAULT_K,
            relevancy_method="top_k",
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
            threshold=DEFAULT_THRESHOLD,
    ):
        return SparkRankingEvaluation(
            rating_true=rating_true,
            rating_pred=rating_pred,
            k=k,
            relevancy_method=relevancy_method,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction=col_prediction,
            threshold=threshold
        )
