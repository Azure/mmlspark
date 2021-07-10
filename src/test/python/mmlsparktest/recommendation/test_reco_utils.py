# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

from mmlsparktest.spark import *
from mmlspark.recommendation.RecoUtils import RecoUtils
from pyspark.ml.recommendation import ALS

USER_ID = "originalCustomerID"
ITEM_ID = "newCategoryID"
RATING_ID = 'rating'
USER_ID_INDEX = "customerID"
ITEM_ID_INDEX = "itemID"


def get_ratings():
    ratings = spark.createDataFrame([
        (0, 1, 4, 4),
        (0, 3, 1, 1),
        (0, 4, 5, 5),
        (0, 5, 3, 3),
        (0, 7, 3, 3),
        (0, 9, 3, 3),
        (0, 10, 3, 3),
        (1, 1, 4, 4),
        (1, 2, 5, 5),
        (1, 3, 1, 1),
        (1, 6, 4, 4),
        (1, 7, 5, 5),
        (1, 8, 1, 1),
        (1, 10, 3, 3),
        (2, 1, 4, 4),
        (2, 2, 1, 1),
        (2, 3, 1, 1),
        (2, 4, 5, 5),
        (2, 5, 3, 3),
        (2, 6, 4, 4),
        (2, 8, 1, 1),
        (2, 9, 5, 5),
        (2, 10, 3, 3),
        (3, 2, 5, 5),
        (3, 3, 1, 1),
        (3, 4, 5, 5),
        (3, 5, 3, 3),
        (3, 6, 4, 4),
        (3, 7, 5, 5),
        (3, 8, 1, 1),
        (3, 9, 5, 5),
        (3, 10, 3, 3)],
        ["originalCustomerID", "newCategoryID", "rating", "notTime"])
    return ratings


def test_random_split():
    df = get_ratings()
    train, test = RecoUtils.spark_random_split(df)
    als = ALS(userCol=USER_ID, itemCol=ITEM_ID, ratingCol=RATING_ID)
    model = als.fit(train)

    users = train.select(USER_ID).distinct()
    items = train.select(ITEM_ID).distinct()
    user_item = users.crossJoin(items)
    dfs_pred = model.transform(user_item)

    # Remove seen items.
    dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        train.alias("train"),
        (dfs_pred[USER_ID]==train[USER_ID]) & (dfs_pred[ITEM_ID]==train[ITEM_ID]),
        how='outer'
    )
    top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train."+RATING_ID].isNull()) \
        .select("pred."+USER_ID, "pred."+ITEM_ID, "pred.prediction")

    top_all.show(5)

    TOP_K = 10
    cols = {
        'col_user': USER_ID,
        'col_item': ITEM_ID,
        'col_rating': RATING_ID,
        'col_prediction': "prediction",
    }
    RecoUtils.ranking_evaluation(test, top_all, k=TOP_K, **cols)

    prediction = model.transform(test)
    RecoUtils.rating_evaluation(test, prediction, **cols)
